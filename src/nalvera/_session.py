"""The module functions: ``nalvera.login``, ``nalvera.run`` and the others.

One client per process signs in and every module function uses it. It lives
in one private name, guarded by a lock, and is closed by ``logout`` and
when the interpreter exits. With no ``login`` in the process, the first
module function makes the default client (``Client()``: the environment,
then what a login stored), so a script on a cluster needs no key in its
text. Each delegate has the signature and the docstring of the
``nalvera.Client`` method of the same name.
"""
from __future__ import annotations

import atexit
import functools
import inspect
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

from . import auth, messages
from ._http import DEFAULT_BASE_URL
from .client import (
    Client,
    PathLike,
    SavedCopies,
    _remove_stale_workdirs,
    default_record_path,
    read_record,
)
from .client import save_anonymized as _save_anonymized
from .errors import (
    ConfigError,
    NalveraError,
    NoCredentials,
    RunInterrupted,
    WaitInterrupted,
)
from .report import Reporter, cell_token, default_reporter, shows_beside_results

_lock = threading.RLock()
_client: Optional[Client] = None
_closer_registered = False

#: The ``Client`` methods offered as module functions. ``close`` is not one:
#: ``logout`` closes the process's client.
DELEGATED = ("me", "limits", "key_expiry", "encryption", "models", "budgets", "estimate",
             "estimate_run", "run", "run_in_background", "generate", "wait", "download",
             "tasks", "task", "cancel")
#: Functions that work without a sign-in when there is none (they need no
#: key), each with the signature and docstring of its ``Client`` method.
OWN = ("scans", "save_anonymized")
NOT_DELEGATED = ("close",)


def _close_at_exit() -> None:
    global _client
    with _lock:
        current, _client = _client, None
    if current is not None:
        try:
            current.close()
        except Exception:
            pass


def _keep(new: Client) -> None:
    """Make ``new`` the process's client; the one before is closed."""
    global _client, _closer_registered
    with _lock:
        old, _client = _client, new
        if not _closer_registered:
            atexit.register(_close_at_exit)
            _closer_registered = True
    if old is not None and old is not new:
        old.close()


def current() -> Client:
    """The process's client; made from the environment or a stored login
    when no ``login`` was called (raises ``NoCredentials`` with the
    steps to get a key when there is none)."""
    with _lock:
        if _client is None:
            _keep(Client())
        return _client                                   # type: ignore[return-value]


def login(email: Optional[str] = None, api_key: Optional[str] = None, *,
          remember: bool = False, **client_options: Any) -> Client:
    """Sign in to Nalvera.AI for this process; every module function
    (``nalvera.run``, ``nalvera.wait``...) then uses this sign-in.

    Example::

        import nalvera
        nalvera.login("you@university.org", "nvk_...")

    ``email``, ``api_key``
        Default None. With both, one request to Nalvera.AI checks the key.
        A key typed into a notebook cell is saved in the ``.ipynb`` file and
        in IPython's history file: with neither argument, ``login()`` asks
        for the email and the key (hidden) when the notebook or terminal can
        ask, and saves nothing in the notebook. Where nobody can ask, it
        reads ``NALVERA_EMAIL`` and
        ``NALVERA_API_KEY``, then what an earlier ``remember=True`` stored;
        with none of these ``NoCredentials`` lists the steps to get a key.
        One without the other is refused (``ConfigError``, code
        ``login_needs_both``).
    ``remember``
        Default False. True stores the email and the key on this computer
        (in the system keychain with the ``keyring`` extra, else in a file
        only you can read), so a later ``nalvera.login()`` or a script needs
        neither.
    ``client_options``
        Options of ``Client``, such as ``budget="personal"``,
        ``max_credits=50`` or ``mapping=``.

    A failed sign-in leaves an earlier one in place; a successful one
    replaces it. Returns the ``Client``."""
    if (email is None) != (api_key is None):
        raise ConfigError(messages.text("error.login_needs_both"), code="login_needs_both")
    # The start of a session: temporary folders a killed process left go now.
    _remove_stale_workdirs()
    reporter: Reporter = client_options.get("reporter") or default_reporter(
        output=client_options.get("output"), input=client_options.get("input"),
        interactive=client_options.get("interactive"))
    client_options["reporter"] = reporter
    try:
        if email is None and client_options.get("interactive") is not False \
                and reporter.can_ask():
            email = (reporter.ask(messages.text("login.email_prompt")) or "").strip() or None
            if email is None:
                raise NoCredentials()
            api_key = reporter.ask(messages.text("login.key_prompt"), hidden=True)
            api_key = auth.make_key(api_key or "", "prompt").reveal()
        made = Client(email=email, api_key=api_key, **client_options)
    finally:
        api_key = None                                   # out of the traceback's locals
    try:
        me = made.me(refresh=True)
        expires, days = made.key_expiry()
    except BaseException:
        made.close()
        raise
    key = made._transport.credentials.api_key
    if remember:
        directory = Path(client_options["config_dir"]) if client_options.get("config_dir") \
            else None
        where = auth.store(made.email, key, directory=directory, base_url=made.base_url)
        path = auth.credentials_path(directory)
    _keep(made)
    say = reporter.say
    # One confirmation: the account, the key's first characters, its end, and
    # the platform only when it is not the usual one.
    signed_in = {"email": me.get("email") or made.email, "prefix": key.prefix,
                 "expires": str(expires)[:10] if expires else "-"}
    if made.base_url == DEFAULT_BASE_URL:
        say("login.ok", **signed_in)
    else:
        say("login.ok_platform", platform=made.base_url, **signed_in)
    if shows_beside_results(reporter):
        # In IPython the cell ends in the client: its table would say the line again.
        made._signed_in_cell = cell_token()
    if days is not None and days <= _key_warn_days():
        say("whoami.expires_soon", days=max(0, days))
    if remember and where == "keyring":
        say("login.stored_keyring")
    elif remember:
        if auth.keyring_is_plaintext():
            # A keyring backend that keeps keys in a plain file is no keychain.
            say("login.keyring_plaintext")
        say("login.stored_file_windows" if sys.platform == "win32" else "login.stored_file",
            path=str(path))
    if "tasks:write" not in ((me.get("key") or {}).get("scopes") or []):
        say("login.cannot_run")
    return made


def _key_warn_days() -> int:
    from .client import KEY_WARN_DAYS
    return KEY_WARN_DAYS


def logout(*, forget: bool = False) -> bool:
    """Sign out: the process's client is closed and drops the key, so the
    ``Client`` that ``login`` returned sends no request afterwards; the next
    module function signs in again from the environment or a stored login.

    ``forget``
        Default False. True also removes what ``login(..., remember=True)``
        stored on this computer. The key stays valid until you revoke it
        under Settings → API on nalvera.ai.

    Returns True when ``forget`` removed stored credentials."""
    global _client
    with _lock:
        old, _client = _client, None
    if old is not None:
        old.close()
    return bool(auth.forget()) if forget else False


def scans() -> Any:
    """The record on this computer: for each scan its neutral code,
    original name, label, tasks and result files, and each generation;
    one dict per row (a list that a notebook shows as a table). Its
    ``path`` is the record's file. Sends no request.

    Without a login in this process it reads the record of the account
    named by ``NALVERA_EMAIL`` or a stored login, so it needs no key."""
    with _lock:
        signed_in = _client
    if signed_in is not None:
        return signed_in.scans()
    return read_record(default_record_path())


def save_anonymized(paths: Union[PathLike, Sequence[PathLike]], out: PathLike) -> SavedCopies:
    """Write the anonymized copies of scans into the folder ``out``, exactly
    as ``nalvera.run`` would upload them. Sends nothing; needs no key.

    Example::

        copies = nalvera.save_anonymized("ct_scan.nii.gz", out="anonymized")

    ``paths``
        Required. A ``.nii``/``.nii.gz`` file, a folder of NIfTI files, a
        folder of DICOM files, or a list of these.
    ``out``
        Required. The folder; created when missing. It holds the copies and
        nothing else, each named by its new code, so it can be shared. The
        record that links each code to its original name stays on this
        computer: in the signed-in account's record, else in the record of
        the account a login stored (``nalvera.scans()`` shows it).

    Returns the copies (a list that a notebook shows as a table): per scan
    the copy's path, its shape and what was removed; ``failed`` lists scans
    that could not be prepared."""
    with _lock:
        signed_in = _client
    if signed_in is not None:
        return signed_in.save_anonymized(paths, out)
    return _save_anonymized(paths, out, mapping=default_record_path(),
                            reporter=default_reporter())


#: Raised by a module function with the package's own frames dropped: a
#: notebook then shows the cell's line and the error's instruction, not a page
#: of the package's code. ``nalvera.Client`` keeps full tracebacks.
_SHORTENED = (NalveraError, RunInterrupted, WaitInterrupted)


def _short(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def shortened(*args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True                         # noqa: F841 (read by IPython, pytest)
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            # The arguments may hold a key (login): out of the traceback's locals.
            args = kwargs = None                         # noqa: F841
            if isinstance(error, _SHORTENED):
                raise error.with_traceback(None) from None
            raise
    return shortened


login = _short(login)
logout = _short(logout)
scans = _short(scans)
save_anonymized = _short(save_anonymized)


def _delegate(name: str) -> Callable[..., Any]:
    method = getattr(Client, name)

    def function(*args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True                         # noqa: F841 (read by IPython, pytest)
        try:
            return getattr(current(), name)(*args, **kwargs)
        except _SHORTENED as error:
            raise error.with_traceback(None) from None

    signature = inspect.signature(method)
    function.__signature__ = signature.replace(               # type: ignore[attr-defined]
        parameters=list(signature.parameters.values())[1:])
    function.__name__ = function.__qualname__ = name
    function.__doc__ = method.__doc__
    function.__module__ = "nalvera"
    function.__wrapped_method__ = method                       # type: ignore[attr-defined]
    return function


def _adopt(function: Callable[..., Any], name: str) -> None:
    """Give ``function`` the signature and docstring of ``Client.<name>``."""
    method = getattr(Client, name)
    signature = inspect.signature(method)
    function.__signature__ = signature.replace(               # type: ignore[attr-defined]
        parameters=list(signature.parameters.values())[1:])
    function.__doc__ = method.__doc__


FUNCTIONS: Dict[str, Callable[..., Any]] = {name: _delegate(name) for name in DELEGATED}
FUNCTIONS.update(login=login, logout=logout, scans=scans, save_anonymized=save_anonymized)
for _name in OWN:
    _adopt(FUNCTIONS[_name], _name)
for _function in (login, logout, scans, save_anonymized):
    _function.__module__ = "nalvera"
