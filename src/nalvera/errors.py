"""Errors raised by the nalvera package.

Every error is a ``NalveraError`` with three attributes:

``code``
    A stable string to branch on, such as ``"insufficient_credits"``. Server
    errors keep the server's code; the package adds its own (``no_credentials``,
    ``network_error``, ...). Never branch on the message text: it may change.
``status``
    The HTTP status of the server's answer, or ``None`` for an error raised
    on this computer.
``extra``
    The server's extra fields, such as ``retry_after``, ``budget`` or
    ``required_scope``.

The message (``str(error)``) tells a person what to do next. It never
contains an API key: anything shaped like one is replaced by its first 12
characters followed by ``…[redacted]``.

Catch the subclasses for the cases a script handles differently::

    try:
        tasks = client.run("scan.nii.gz", model="totalsegmentator_ct")
    except nalvera.InsufficientCredits:
        ...  # top up, or pass another budget
    except nalvera.NalveraError as error:
        print(error.code, error)
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Type

from . import messages
from ._version import __version__

# An API key, or anything long enough to be the start of one.
_KEY_RX = re.compile(r"(nvk_[0-9A-Za-z]{8})[0-9A-Za-z]+")
# Control characters in a server's text or a file name could forge extra
# lines or move the cursor: each is shown as "?". C1 controls, and the
# bidirectional and other invisible format characters (which reorder or hide
# what a line seems to say), go too: the soft hyphen, the Mongolian vowel
# separator and the tag characters (U+E0000 to U+E007F) among them.
_CONTROL_RX = re.compile("[\x00-\x1f\x7f-\x9f\u00ad\u061c\u180e\u200b-\u200f"
                         "\u2028-\u202e\u2060-\u2069\ufeff\ufff9-\ufffb"
                         "\U000e0000-\U000e007f]")
#: The longest wait a server's ``retry_after`` (or ``Retry-After``) is taken
#: for, in seconds: a non-finite or larger value counts as this.
MAX_RETRY_AFTER = 86400.0


def redact(value: Any) -> str:
    """``value`` as text with every API key replaced by its 12-character
    prefix and ``…[redacted]``. Applied to every message the package prints
    or raises."""
    return _KEY_RX.sub(r"\1…[redacted]", str(value))


def printable(value: Any) -> str:
    """``value`` as text on one line: control and format characters
    become ``?``."""
    return _CONTROL_RX.sub("?", str(value))


def bounded(value: Any, low: float, high: float, default: Optional[float] = None
            ) -> Optional[float]:
    """A number read from a server's answer, kept between ``low`` and
    ``high``: a value past ``high``, infinite or not a number (NaN) is
    ``high``, one below ``low`` is ``low``. ``default`` when there is none
    or it is not a number at all."""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if math.isnan(number) or number > high:
        return high
    return max(low, number)


class NalveraError(Exception):
    """Base class of every error the package raises. See the module docstring."""

    #: The code used when none is given.
    default_code = "unknown"

    def __init__(self, message: Optional[str] = None, *, code: Optional[str] = None,
                 status: Optional[int] = None, detail: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None):
        self.code: str = code or self.default_code
        self.status: Optional[int] = status
        # The server's text: never a key, never a control character.
        self.detail: Optional[str] = printable(redact(detail)) if detail is not None else None
        self.extra: Dict[str, Any] = dict(extra or {})
        if message is None:
            message = _message_for(self.code, self.status, self.detail, self.extra)
        self.message: str = redact(message)
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, status={self.status!r})"

    @property
    def retry_after(self) -> Optional[float]:
        """Seconds the server asked to wait before trying again, if it said
        (at most ``MAX_RETRY_AFTER``)."""
        return _seconds(self.extra.get("retry_after"))


# ── Problems found on this computer ───────────────────────────────────────────
class NoCredentials(NalveraError):
    """No email or API key was found. The message lists the steps to get a key."""
    default_code = "no_credentials"


class ConfigError(NalveraError, ValueError):
    """A setting is not valid: the key's format, the base address, the budget,
    a label, a generation setting, or the credentials file."""
    default_code = "invalid_setting"


class BudgetChoiceRequired(NalveraError):
    """The account belongs to a lab, no budget was chosen, and nobody can be
    asked. ``options`` lists the budgets to choose from (``"personal"``,
    ``"lab:12"``...) with their balances; ``suggested`` is the one the
    platform suggests (the active lab with the highest balance, as on the
    Submit page). Pass one as ``budget=`` or set ``NALVERA_BUDGET``."""
    default_code = "budget_choice_required"

    def __init__(self, options: List[Dict[str, Any]], message: Optional[str] = None,
                 suggested: Optional[str] = None):
        self.options = options
        listed = [str(o.get("budget")) for o in options]
        self.suggested: str = suggested if suggested in listed else _suggestion(options)
        if message is None:
            lines = "\n".join(
                messages.text("budget.choice_line", budget=printable(o["budget"]),
                              name=printable(o["name"]),
                              available=_credits(o.get("available_credits")),
                              mark=messages.text("budget.mark_suggested")
                              if o.get("budget") == self.suggested else "")
                for o in options)
            message = messages.text("error.budget_choice_required", options=lines,
                                    budget=printable(self.suggested))
        super().__init__(message, code=self.default_code)


def _suggestion(options: List[Dict[str, Any]]) -> str:
    """The platform's rule (``js/submit-pricing.js``): the lab with the
    highest balance, ties by name; with no lab, personal."""
    labs = [o for o in options if str(o.get("budget", "")).startswith("lab:")]
    if not labs:
        return "personal"

    def balance(option: Dict[str, Any]) -> float:
        try:
            return float(option.get("available_credits") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return str(min(labs, key=lambda o: (-balance(o), str(o.get("name") or "")))["budget"])


class PreparationError(NalveraError):
    """A scan could not be anonymized on this computer. ``code`` is the
    anonymizer's (``not_a_scan``, ``unreadable``, ``verification_failed``...);
    nothing of that scan was uploaded."""
    default_code = "preparation_failed"


class NetworkError(NalveraError):
    """The server could not be reached, also after retrying."""
    default_code = "network_error"


class UnexpectedResponse(NalveraError):
    """The server sent an answer this version cannot read, or a redirect."""
    default_code = "unexpected_response"


# ── Refusals by the server ────────────────────────────────────────────────────
class AuthenticationError(NalveraError):
    """The email or key was not accepted, or the key expired or was revoked."""
    default_code = "invalid_credentials"


class PermissionDenied(NalveraError):
    """The key may not do this (``scope_missing``: ``extra["required_scope"]``
    names the permission), or the account, address or lab may not."""
    default_code = "scope_missing"

    @property
    def required_scope(self) -> Optional[str]:
        """The permission the key lacks, for ``scope_missing``."""
        return self.extra.get("required_scope")


class ApiDisabled(NalveraError):
    """API access is switched off for this account, or blocked by an admin."""
    default_code = "api_disabled"


class InsufficientCredits(NalveraError):
    """The chosen budget has too few credits. Not retried: top up, or choose
    another budget. ``extra["budget"]`` names the budget."""
    default_code = "insufficient_credits"


class NoFreeTasks(InsufficientCredits):
    """``budget="free"`` was asked and no free task is left
    (``no_free_tasks``). Nothing was queued or charged; name a budget to
    pay with."""
    default_code = "no_free_tasks"


class CreditCapExceeded(NalveraError):
    """The run would cost more than ``max_credits`` allows
    (``max_credits_exceeded``). ``extra["estimate"]`` is the estimate and
    ``extra["max_credits"]`` the cap."""
    default_code = "max_credits_exceeded"


class ConfirmationRequired(NalveraError):
    """A DICOM folder holds more scans than expected and there is no
    terminal to confirm in (``confirmation_required``). Pass ``yes=True``
    (``--yes``) or ``max_credits`` to go ahead. ``extra["count"]`` and
    ``extra["credits"]`` give the number of scans and the estimate."""
    default_code = "confirmation_required"


class EncryptionError(NalveraError):
    """An upload could not be encrypted as asked, or the account requires
    encrypted uploads. Nothing was uploaded unencrypted. Codes:
    ``encrypt_no_trusted_key``, ``encrypt_unavailable``,
    ``encrypt_required_by_account``, ``encryption_required``,
    ``encryption_unavailable``."""
    default_code = "encrypt_unavailable"


class InvalidRequest(NalveraError):
    """The server refused the request as not valid; the message carries its
    explanation (a generation setting out of range, an unknown task...)."""
    default_code = "invalid_request"


class ScanRejected(NalveraError):
    """The server refused the scan. ``extra["reason"]`` says why."""
    default_code = "scan_rejected"


class TooLarge(ScanRejected):
    """The file is larger than the upload limit (``extra["max_bytes"]``)."""
    default_code = "too_large"


class NotFound(NalveraError):
    """No such task or upload on this account (or it was used already)."""
    default_code = "not_found"


class NotReady(NalveraError):
    """The task has no result yet; ``extra["state"]`` is its state."""
    default_code = "not_ready"


class NotCancellable(NalveraError):
    """The task has finished and cannot be cancelled."""
    default_code = "not_cancellable"


class RateLimited(NalveraError):
    """A limit was reached and stayed reached after waiting: too many
    requests, too many tasks waiting or running, or too many uploads in
    progress. ``retry_after`` says how long to wait."""
    default_code = "rate_limited"


class ServiceUnavailable(NalveraError):
    """The API, the queue or the storage is unavailable for now."""
    default_code = "api_unavailable"


class ServerError(NalveraError):
    """The server had an internal error."""
    default_code = "server_error"


class ClientTooOld(NalveraError):
    """This version of the package is too old for the server. Upgrade with
    ``pip install --upgrade nalvera``."""
    default_code = "client_too_old"


class UploadInterrupted(NalveraError):
    """Internal: an upload part has to be sent again from ``extra["offset"]``."""
    default_code = "offset_mismatch"


class RunInterrupted(KeyboardInterrupt):
    """A run was stopped: the notebook's stop button, Ctrl-C, or
    ``BackgroundRun.cancel()``. ``tasks`` lists the tasks queued before it
    stopped: they keep running on the platform and are in the record. A
    ``KeyboardInterrupt``, so code that stops on Ctrl-C keeps doing so; its
    text says which function follows the tasks, unless ``told`` (a
    notebook's line already named them and that function)."""

    def __init__(self, tasks: Optional[List[Any]] = None, told: bool = False):
        self.tasks = list(tasks or [])
        super().__init__(messages.text("run.interrupted_told" if told else "run.interrupted"))

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class WaitInterrupted(KeyboardInterrupt):
    """``wait`` was stopped (the notebook's stop button, or Ctrl-C).
    ``tasks`` lists the tasks it waited for: they keep running, and
    ``nalvera.wait()`` picks them up again. A ``KeyboardInterrupt``."""

    def __init__(self, tasks: Optional[List[Any]] = None):
        self.tasks = list(tasks or [])
        super().__init__(messages.text("wait.interrupted"))

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


_CLASS_BY_CODE: Dict[str, Type[NalveraError]] = {
    "invalid_request": InvalidRequest,
    "https_required": ConfigError,
    "key_in_url": AuthenticationError,
    "invalid_credentials": AuthenticationError,
    "key_expired": AuthenticationError,
    "key_revoked": AuthenticationError,
    "insufficient_credits": InsufficientCredits,
    "no_free_tasks": NoFreeTasks,
    "encryption_required": EncryptionError,
    "encryption_unavailable": EncryptionError,
    "api_disabled": ApiDisabled,
    "api_blocked": ApiDisabled,
    "email_unverified": PermissionDenied,
    "terms_reaccept_required": PermissionDenied,
    "scope_missing": PermissionDenied,
    "ip_not_allowed": PermissionDenied,
    "lab_not_allowed": PermissionDenied,
    "not_found": NotFound,
    "not_ready": NotReady,
    "not_cancellable": NotCancellable,
    "too_large": TooLarge,
    "scan_rejected": ScanRejected,
    "client_too_old": ClientTooOld,
    "rate_limited": RateLimited,
    "too_many_active_tasks": RateLimited,
    "too_many_uploads": RateLimited,
    "api_unavailable": ServiceUnavailable,
    "queue_full": ServiceUnavailable,
    "storage_full": ServiceUnavailable,
    "temporarily_unavailable": ServiceUnavailable,
    "server_error": ServerError,
    "offset_mismatch": UploadInterrupted,
    "checksum_mismatch": UploadInterrupted,
}

#: Codes the server marks "try again"; the package waits and retries these.
RETRYABLE_CODES = frozenset({
    "rate_limited", "too_many_active_tasks", "too_many_uploads",
    "queue_full", "storage_full", "temporarily_unavailable",
})


def _credits(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "?"


def _message_for(code: str, status: Optional[int], detail: Optional[str],
                 extra: Dict[str, Any]) -> str:
    """The package's own instruction for ``code``; for a code this version
    does not know, the server's ``detail``."""
    key = f"error.{code}"
    if not messages.has(key):
        if detail:
            return printable(detail)
        return messages.text("error.unknown", status=status if status is not None else "-",
                             code=printable(code))
    # Every value from the server's answer is shown on one line.
    fields: Dict[str, Any] = {k: printable(v) if isinstance(v, str) else v
                              for k, v in extra.items() if isinstance(k, str)}
    detail = printable(detail) if detail else detail
    fields.setdefault("detail", detail or "")
    fields.setdefault("reason", extra.get("reason") or detail or "")
    fields.setdefault("version", __version__)
    fields.setdefault("status", status if status is not None else "-")
    if "max_bytes" in extra:
        try:
            fields["max_mb"] = int(extra["max_bytes"]) // (1024 * 1024)
        except (TypeError, ValueError):
            pass
    if code == "too_large" and "max_mb" not in fields:
        # nginx's own refusal names no limit.
        key = "error.too_large_no_limit"
    if code == "not_ready" and isinstance(extra.get("state"), str):
        state_key = f"wait.state.{extra['state']}"
        if messages.has(state_key):
            fields["state"] = messages.text(state_key)
    retry_after = _seconds(extra.get("retry_after"))
    if retry_after is not None:
        fields["retry_after"] = int(round(retry_after))
    fields["wait"] = describe_wait(retry_after) if retry_after is not None \
        else messages.text("time.wait_unknown")
    return messages.text(key, **fields)


def describe_wait(seconds: float) -> str:
    """A wait as words: seconds, minutes or hours, rounded up ("1 s", never
    "1 seconds")."""
    seconds = max(0.0, float(seconds))
    if seconds < 120:
        return messages.text("time.wait_seconds", s=int(math.ceil(seconds)))
    if seconds < 2 * 3600:
        return messages.text("time.wait_minutes", m=int(math.ceil(seconds / 60)))
    return messages.text("time.wait_hours", h=int(math.ceil(seconds / 3600)))


def _seconds(value: Any) -> Optional[float]:
    """A wait in seconds read from an answer, between 0 and
    ``MAX_RETRY_AFTER``: a non-finite or larger value is the cap, a
    negative one 0; None when there is none or it is not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number > MAX_RETRY_AFTER:
        return MAX_RETRY_AFTER
    return max(0.0, number)


def from_response(status: int, body: Any, body_text: Optional[str] = None) -> NalveraError:
    """The error for a refusal ``body`` (``{"detail", "code", ...}``) with
    HTTP ``status``. An unknown code becomes a plain ``NalveraError``
    carrying the server's ``detail``. An answer that is not the API's JSON
    (a proxy's HTML page) gets a code from its status alone; when the status
    says nothing either, the error shows the status and the first line of
    ``body_text``."""
    if not isinstance(body, dict):
        body = {}
    code = body.get("code")
    detail = body.get("detail") if isinstance(body.get("detail"), str) else None
    extra = {k: v for k, v in body.items() if k not in ("code", "detail")}
    if not isinstance(code, str) or not code:
        code = _code_for_status(status)
        if code == "unexpected_status":
            extra["first_line"] = first_line(body_text)
    cls = _CLASS_BY_CODE.get(code, NalveraError)
    if code == "unexpected_status":
        cls = UnexpectedResponse
    return cls(code=code, status=status, detail=detail, extra=extra)


def first_line(text: Optional[str], limit: int = 160) -> str:
    """The first non-empty line of ``text``, without markup, shortened, with
    any key-shaped text scrubbed."""
    for line in (text or "").splitlines():
        stripped = re.sub(r"<[^>]*>", " ", line)
        stripped = " ".join(stripped.split())
        if stripped:
            return printable(redact(stripped[:limit]))
    return "-"


def _code_for_status(status: int) -> str:
    """A code for an answer that carries none (a proxy's error page)."""
    if status in (502, 503, 504):
        return "temporarily_unavailable"
    if status >= 500:
        return "server_error"
    return {400: "invalid_request", 401: "invalid_credentials", 404: "not_found",
            413: "too_large", 429: "rate_limited"}.get(status, "unexpected_status")
