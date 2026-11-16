"""The one way the package talks to the server.

Rules every request follows:

* HTTPS with certificate checks, always. There is no option to turn the
  checks off. Plain ``http://`` is accepted for ``localhost`` and
  ``127.0.0.1`` only (a local test server).
* Redirects are never followed: the Authorization header never leaves the
  configured address.
* Every request has a timeout.
* One shared ``requests`` session (connection reuse).
* A pace below the key's limits (``limits`` of ``GET /api/v1/me``): at most
  90 % of ``requests_per_minute``, and of ``task_submits_per_minute`` for
  queueing tasks.
* Retries, with exponential backoff, jitter and the server's
  ``Retry-After``, only for the answers the server marks "try again"
  (``errors.RETRYABLE_CODES``) and for network failures of requests that
  are safe to repeat. A cap bounds the total waiting; a ``Retry-After``
  beyond it stops at once with a message that says when to try again.
  Anything else, such as ``insufficient_credits``, raises at once.
* An answer that is not the API's JSON (a proxy's HTML page during a
  restart) is judged by its status alone: 502, 503 and 504 are retried, 413
  is ``too_large``, anything else shows its status and first line.
* ``Retry-After`` may be seconds or an HTTP date; a date is measured
  against the answer's own ``Date`` header, never this computer's clock.
* The environment's proxy (``HTTPS_PROXY``) and certificate bundle
  (``REQUESTS_CA_BUNDLE``) apply; ``~/.netrc`` never replaces the key.
* ``User-Agent: nalvera-python/<version>``. Nothing else about this
  computer is sent.
* Nothing raised from inside the HTTP library leaves with its frames: those
  frames hold the request, and the request holds the key. A Ctrl-C is
  raised again as a new ``KeyboardInterrupt``, a broken connection or answer
  becomes ``NetworkError``, anything else ``request_failed`` (its type
  and text kept, never its traceback). The header that carries the key
  shows only the key's prefix in a ``repr``, which is what every display of
  a traceback's variables uses.
* After ``Transport.close`` no request is signed (``signed_out``).

The clock, the sleep and the random source are parameters, so tests run
without waiting.
"""
from __future__ import annotations

import random
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from urllib.parse import urlsplit

import requests
from urllib3.exceptions import NewConnectionError

from . import errors, messages
from ._version import __version__
from .auth import Credentials
from .errors import NalveraError, NetworkError, UnexpectedResponse

DEFAULT_BASE_URL = "https://nalvera.ai"
USER_AGENT = f"nalvera-python/{__version__}"
_LOCAL_HOSTS = ("localhost", "127.0.0.1")

#: (connect, read) seconds.
DEFAULT_TIMEOUT: Tuple[float, float] = (10.0, 60.0)

Timeout = Union[float, Tuple[float, float]]


def validate_base_url(url: str) -> str:
    """``url`` without a trailing slash, or ``errors.ConfigError`` when
    it is not ``https://host[:port]`` (``http://`` only for localhost and
    127.0.0.1)."""
    raw = (url or "").strip()
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        raise errors.ConfigError(messages.text("error.invalid_base_url", url=raw),
                                 code="invalid_base_url") from None
    if (parts.scheme not in ("https", "http") or not host or parts.username or parts.password
            or parts.query or parts.fragment or parts.path not in ("", "/")):
        raise errors.ConfigError(messages.text("error.invalid_base_url", url=raw),
                                 code="invalid_base_url")
    if parts.scheme == "http" and host not in _LOCAL_HOSTS:
        raise errors.ConfigError(messages.text("error.insecure_base_url", url=raw),
                                 code="insecure_base_url")
    netloc = host if port is None else f"{host}:{port}"
    return f"{parts.scheme}://{netloc}"


@dataclass
class RetryPolicy:
    """How long to keep trying. ``max_total_wait`` (seconds) caps the sum
    of all waits for one request."""
    max_attempts: int = 8
    base_delay: float = 1.0
    max_delay: float = 120.0
    max_total_wait: float = 900.0


class Pacer:
    """Spaces requests so they stay below ``per_minute`` (times ``share``).
    Thread-safe: each caller reserves the next free moment."""

    def __init__(self, clock: Callable[[], float], sleep: Callable[[float], None],
                 per_minute: Optional[float] = None, share: float = 0.9):
        self._clock, self._sleep, self._share = clock, sleep, share
        self._lock = threading.Lock()
        self._next = 0.0
        self._interval = 0.0
        self.set_rate(per_minute)

    def set_rate(self, per_minute: Optional[float]) -> None:
        """Requests per minute allowed; None or 0 switches pacing off."""
        # Bounded: an infinite or NaN rate from an answer is the cap, not a crash.
        per_minute = errors.bounded(per_minute, 0.0, 1e6, 0.0)
        allowed = max(1.0, per_minute * self._share) if per_minute else 0.0
        self._interval = 60.0 / allowed if allowed else 0.0

    @property
    def interval(self) -> float:
        """Seconds between two requests."""
        return self._interval

    def wait(self) -> None:
        """Block until this request may go."""
        if not self._interval:
            return
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self._interval
        if slot > now:
            self._sleep(slot - now)


class _HeaderSecret(str):
    """The Authorization header's value. It is sent as it is (``http.client``
    encodes it byte for byte); its ``repr``, which IPython's verbose
    tracebacks, ``rich``, ``pytest -l``, Sentry and a debugger use to show a
    frame's variables, gives the key's prefix only."""

    __slots__ = ()

    def __repr__(self) -> str:
        return repr(errors.redact(str.__str__(self)))

    def __reduce__(self):
        raise TypeError("a header holding an API key cannot be pickled")


class _SignIn(requests.auth.AuthBase):
    """Adds the two sign-in headers to each request. Given as ``auth=``, it
    also keeps ``requests`` from applying a ``.netrc`` entry."""

    def __init__(self, credentials: Credentials):
        self._credentials = credentials

    def __call__(self, request):
        request.headers["Authorization"] = _HeaderSecret(
            "Bearer " + self._credentials.api_key.reveal())
        request.headers["X-Nalvera-Email"] = self._credentials.email
        return request

    def __repr__(self) -> str:
        return f"<sign-in {self._credentials.api_key.prefix}…>"


Notify = Callable[[float, NalveraError], None]


class Cancelled(Exception):
    """A run was interrupted (Ctrl-C): work stops at the next pause."""


class Transport:
    """Signed, paced, retried requests to one Nalvera server.

    ``notify(seconds, error)`` is called before each wait caused by a
    retryable answer, so a caller can tell the user why it pauses. Setting
    ``stop`` (a ``threading.Event``) ends every pause with ``Cancelled``.
    """

    def __init__(self, base_url: str, credentials: Credentials, *,
                 timeout: Timeout = DEFAULT_TIMEOUT, retry: Optional[RetryPolicy] = None,
                 clock: Optional[Callable[[], float]] = None,
                 sleep: Optional[Callable[[float], None]] = None,
                 rand: Optional[Callable[[], float]] = None,
                 notify: Optional[Notify] = None):
        self.base_url = validate_base_url(base_url)
        self.credentials = credentials
        self.timeout = timeout
        self.retry = retry or RetryPolicy()
        # Looked up now (not as default values), so a test can replace them.
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.rand = rand or random.random
        self.notify = notify
        self.stop = threading.Event()
        #: The ``Date`` of the server's latest answer (its clock, not ours).
        self.server_date: Optional[datetime] = None
        self._local = threading.local()
        self._closed = False
        self._auth = _SignIn(credentials)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._pace = Pacer(self.clock, self.sleep)
        self._submit_pace = Pacer(self.clock, self.sleep)

    def __repr__(self) -> str:
        return f"Transport({self.base_url!r}, key={self.credentials.api_key.prefix}…)"

    @property
    def last_attempts(self) -> int:
        """How many times this thread's latest request was sent (1 when it
        was not repeated)."""
        return int(getattr(self._local, "attempts", 1))

    @property
    def last_may_have_run(self) -> bool:
        """Whether an attempt of this thread's latest request, other than
        the one answered, may have run on the server: its connection broke
        after the request went out (an answer cut short or unreadable, a TLS
        error that is not a failed certificate check included), a proxy's
        5xx page without a code came back in place of the answer, the
        platform answered ``server_error``, which may follow a change it
        made, or its 5xx answer said ``"queued": true`` (the task is queued,
        its answer could not be built). Any other refusal with a code was
        not acted on."""
        return bool(getattr(self._local, "may_have_run", False))

    def pause(self, seconds: float) -> None:
        """Wait ``seconds`` in steps of at most one second, so an interrupted
        run stops promptly (``Cancelled``)."""
        remaining = max(0.0, float(seconds))
        while True:
            if self.stop.is_set():
                raise Cancelled()
            if remaining <= 0:
                return
            step = min(1.0, remaining)
            self.sleep(step)
            remaining -= step

    # ── pacing ────────────────────────────────────────────────────────────────
    def set_limits(self, limits: Dict[str, Any]) -> None:
        """Pace by the ``limits`` object of ``GET /api/v1/me``."""
        self._pace.set_rate(limits.get("requests_per_minute"))
        self._submit_pace.set_rate(limits.get("task_submits_per_minute"))

    def _wait_turn(self, method: str, path: str) -> None:
        if method == "POST" and path == "/api/v1/tasks":
            self._submit_pace.wait()
        self._pace.wait()

    # ── retries ───────────────────────────────────────────────────────────────
    def _delay(self, attempt: int, retry_after: Optional[float]) -> float:
        backoff = min(self.retry.max_delay, self.retry.base_delay * (2 ** attempt))
        jittered = backoff * (0.5 + 0.5 * self.rand())
        if retry_after is not None and retry_after > 0:
            # The server's answer is a minimum; a little jitter spreads clients.
            return float(retry_after) + self.rand() * min(1.0, float(retry_after) * 0.1)
        return jittered

    # ── requests ──────────────────────────────────────────────────────────────
    def request(self, method: str, path: str, *, json: Any = None, data: Any = None,
                headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None,
                stream: bool = False, timeout: Optional[Timeout] = None,
                idempotent: Optional[bool] = None,
                retry_codes: Iterable[str] = errors.RETRYABLE_CODES,
                max_total_wait: Optional[float] = None) -> requests.Response:
        """Send one request and return its 2xx answer, or raise the
        ``NalveraError`` for its refusal.

        ``idempotent`` (default: True for GET, HEAD and DELETE) allows
        repeating the request after a network failure. ``retry_codes`` are
        the refusal codes to wait out and repeat.
        """
        url = self.base_url + path
        safe_to_repeat = method in ("GET", "HEAD", "DELETE") if idempotent is None else idempotent
        cap = self.retry.max_total_wait if max_total_wait is None else max_total_wait
        retry_codes = frozenset(retry_codes)
        attempt, waited = 0, 0.0
        self._local.may_have_run = False
        while True:
            self._local.attempts = attempt + 1
            if self._closed:
                raise errors.NalveraError(messages.text("error.signed_out"), code="signed_out")
            self._wait_turn(method, path)
            response, failure = self._send(
                method, url, json=json, data=data, headers=headers, params=params,
                stream=stream, timeout=timeout or self.timeout)
            # Every raise below is outside an ``except`` block: nothing raised
            # from here carries the HTTP library's frames as its context.
            if failure is _STOPPED:
                self._local.may_have_run = True          # it may have gone out
                raise KeyboardInterrupt() from None
            if failure is not None:
                error, sent, repeatable = failure
                if sent:
                    self._local.may_have_run = True
                if not (repeatable and safe_to_repeat):
                    raise error from None
                delay = self._delay(attempt, None)
                if attempt + 1 >= self.retry.max_attempts or waited + delay > cap:
                    raise error from None
                self._announce(delay, error)
                self.pause(delay)
                attempt, waited = attempt + 1, waited + delay
                continue
            self._note_date(response)
            if 300 <= response.status_code < 400:
                response.close()
                raise UnexpectedResponse(
                    messages.text("error.unexpected_redirect", status=response.status_code),
                    code="unexpected_redirect", status=response.status_code)
            if response.status_code < 300:
                return response
            body = _json_body(response)
            coded = isinstance(body, dict) and isinstance(body.get("code"), str)
            if response.status_code >= 500 and (not coded or body.get("code") == "server_error"
                                                or body.get("queued") is True):
                # A proxy's page in place of the answer, the platform's own
                # internal error, which may follow a change it made, or its
                # word that the task is queued although the answer could not
                # be built ("queued": true): the request may have run.
                self._local.may_have_run = True
            retry_after = _retry_after_header(response)
            if isinstance(body, dict) and errors._seconds(body.get("retry_after")) is not None:
                retry_after = errors._seconds(body.get("retry_after"))
            elif retry_after is not None:
                # The header's wait, in the message too ("wait 30 seconds").
                body = dict(body) if isinstance(body, dict) else {}
                body["retry_after"] = retry_after
            error = errors.from_response(response.status_code, body,
                                         None if coded else _text_body(response))
            # A "try again" refusal with a code was not acted on: repeating it
            # is safe. A proxy's error page without one may hide a request
            # that ran.
            if error.code in retry_codes and (coded or safe_to_repeat):
                delay = self._delay(attempt, retry_after)
                if attempt + 1 < self.retry.max_attempts and waited + delay <= cap:
                    self._announce(delay, error)
                    self.pause(delay)
                    attempt, waited = attempt + 1, waited + delay
                    continue
                if retry_after is not None:
                    _say_when(error, retry_after)
            raise error

    def _send(self, method: str, url: str, **kwargs: Any):
        """One attempt: ``(response, None)``, or ``(None, failure)`` where
        failure is ``_STOPPED`` (Ctrl-C) or ``(error, sent, repeatable)``.
        Nothing the HTTP library raises leaves this method, so its frames
        (which hold the prepared request and its headers) are never part of
        a traceback; the error is the package's own, built afresh."""
        try:
            return self._session.request(method, url, allow_redirects=False, verify=True,
                                         auth=self._auth, **kwargs), None
        except KeyboardInterrupt:
            return None, _STOPPED
        except requests.exceptions.SSLError as exc:
            if _certificate_failed(exc):
                return None, (_network_error("certificate"), False, False)
            # A TLS failure that is no failed check (a record broken while
            # reading the answer): like a connection that broke.
            return None, (_network_error("tls"), _sent(exc), True)
        except (requests.ConnectionError, requests.Timeout) as exc:
            reason = "timeout" if isinstance(exc, requests.Timeout) else (
                "proxy" if isinstance(exc, requests.exceptions.ProxyError) else "failed")
            return None, (_network_error(reason), _sent(exc), True)
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ContentDecodingError):
            # The answer started and was cut short or cannot be decoded: the
            # request went out and may have run.
            return None, (_network_error("failed"), True, True)
        except NalveraError as exc:
            # The key was dropped by close() in another thread.
            return None, (exc.with_traceback(None), False, False)
        except Exception as exc:
            cause = f"{type(exc).__name__}: {exc}"
            return None, (NalveraError(messages.text("error.request_failed",
                                                     error=errors.printable(cause)),
                                       code="request_failed",
                                       extra={"cause": type(exc).__name__}), False, False)

    def json(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        """``request`` and the answer's JSON object."""
        response = self.request(method, path, **kwargs)
        body = _json_body(response)
        if not isinstance(body, dict):
            raise UnexpectedResponse(
                messages.text("error.unexpected_response", status=response.status_code),
                status=response.status_code)
        return body

    def _announce(self, seconds: float, error: NalveraError) -> None:
        if self.notify is not None:
            try:
                self.notify(seconds, error)
            except Exception:
                pass

    def _note_date(self, response: requests.Response) -> None:
        when = _http_date(response.headers.get("Date"))
        if when is not None:
            self.server_date = when

    def close(self) -> None:
        """Close the connections and sign out: the key is dropped (only its
        prefix stays) and every later request is refused with
        ``signed_out``."""
        self._closed = True
        self.credentials.api_key.forget()
        self._session.close()


#: ``_send``'s failure for a Ctrl-C while the platform was answering.
_STOPPED = object()


def _certificate_failed(error: BaseException) -> bool:
    """Whether a TLS error is a failed check of the platform's certificate
    (``ssl.SSLCertVerificationError`` anywhere in what it wraps, or
    OpenSSL's "certificate verify failed"), as opposed to a connection that
    broke."""
    todo, seen = [error], set()
    while todo and len(seen) < 32:
        item = todo.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, ssl.SSLCertVerificationError) \
                or "certificate verify failed" in str(item).lower():
            return True
        linked = [getattr(item, "reason", None), item.__cause__, item.__context__,
                  *getattr(item, "args", ())]
        todo += [x for x in linked if isinstance(x, BaseException)]
    return False


def _sent(error: Exception) -> bool:
    """Whether a failed connection may have carried the request to the
    server: everything except a connection that was never made."""
    if isinstance(error, requests.exceptions.ConnectTimeout):
        return False
    cause = error.args[0] if error.args else None
    reason = getattr(cause, "reason", cause)
    return not isinstance(reason, NewConnectionError)


def _network_error(reason: str) -> NetworkError:
    """A failed connection; ``extra["reason"]`` is ``timeout``, ``failed``,
    ``proxy`` or ``certificate`` (a ``net.*`` text)."""
    return NetworkError(messages.text("error.network_error",
                                      reason=messages.text(f"net.{reason}")),
                        extra={"reason": reason})


def _say_when(error: NalveraError, retry_after: float) -> None:
    """Give up on a "try again" answer: keep the wait in ``extra`` and, when
    the error's text does not already name it, say when to try again."""
    error.extra["retry_after"] = retry_after
    template = messages.MESSAGES.get(f"error.{error.code}", "")
    if "{wait}" in template or "{retry_after}" in template:
        # The text names the wait: built again with the header's value.
        error.message = errors.redact(errors._message_for(error.code, error.status,
                                                          error.detail, error.extra))
    else:
        error.message = errors.redact(error.message + " " + messages.text(
            "error.try_again_in", wait=describe_wait(retry_after)))
    error.args = (error.message,)


describe_wait = errors.describe_wait


def _json_body(response: requests.Response) -> Any:
    try:
        if not response.content:
            return None
        return response.json()
    except (ValueError, requests.RequestException):
        return None                       # unreadable, or a streamed body cut short


def _text_body(response: requests.Response) -> str:
    try:
        return response.content[:4096].decode("utf-8", errors="replace")
    except Exception:
        return ""


def _http_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _retry_after_header(response: requests.Response) -> Optional[float]:
    """``Retry-After`` in seconds: a number, or an HTTP date measured against
    the answer's own ``Date`` (the server's clock). None when missing or
    unreadable, or a date without a ``Date`` header to measure it by."""
    value = (response.headers.get("Retry-After") or "").strip()
    if not value:
        return None
    try:
        return errors._seconds(float(value))         # bounded: no "1e400", no NaN
    except (ValueError, OverflowError):
        pass
    when = _http_date(value)
    now = _http_date(response.headers.get("Date"))
    if when is None or now is None:
        return None
    return errors._seconds((when - now).total_seconds())
