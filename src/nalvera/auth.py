"""Credentials: the account email and the API key.

Where they come from (each of the two is taken from the first place that
has it):

1. the arguments, ``nalvera.login(email, api_key)`` or
   ``Client(email=..., api_key=...)``;
2. the environment, ``NALVERA_EMAIL`` and ``NALVERA_API_KEY``;
3. the system keychain, when the optional ``keyring`` package is installed
   and a login with ``remember=True`` stored the key there;
4. the credentials file ``credentials.json`` in the config folder
   (``NALVERA_CONFIG_DIR``, default ``~/.config/nalvera``, on Windows
   ``%APPDATA%\\nalvera``), written by a login with ``remember=True``, with
   mode 0600 in a folder with mode 0700. On Windows the file permissions
   are not changed: the file lives in your user profile, which Windows
   restricts to your account.

A key is checked on this computer before any request: ``nvk_`` and 57
letters and digits, the last 6 a checksum of the rest, so a typo is caught
offline. The key is held in an ``ApiKey``, whose ``repr`` and ``str``
show only the 12-character prefix.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from . import messages
from .errors import ConfigError, NoCredentials

KEY_TAG = "nvk_"
KEY_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
KEY_LENGTH = 61
KEY_PREFIX_LENGTH = 12
KEY_CHECK_LENGTH = 6
_KEY_RX = re.compile(r"nvk_[0-9A-Za-z]{57}")

KEYRING_SERVICE = "nalvera"
CREDENTIALS_FILE = "credentials.json"
#: The platform a stored key is for when its record names none
#: (``_http.DEFAULT_BASE_URL``, written out so this module loads no HTTP code).
_DEFAULT = "https://nalvera.ai"


# ── The key ───────────────────────────────────────────────────────────────────
def _base62(number: int, width: int) -> str:
    digits = []
    while number:
        number, rem = divmod(number, 62)
        digits.append(KEY_ALPHABET[rem])
    return "".join(reversed(digits)).rjust(width, "0")


def key_checksum(body: str) -> str:
    """The 6 checksum characters of a key's first 55 characters: CRC-32 of
    their ASCII bytes in base62, as the server computes it."""
    return _base62(zlib.crc32(body.encode("ascii")), KEY_CHECK_LENGTH)


def is_well_formed(key: Any) -> bool:
    """True when ``key`` has an API key's shape and a correct checksum.
    Makes no request."""
    if not isinstance(key, str) or _KEY_RX.fullmatch(key) is None:
        return False
    return key[-KEY_CHECK_LENGTH:] == key_checksum(key[:-KEY_CHECK_LENGTH])


class ApiKey:
    """An API key that never shows itself: ``repr`` and ``str`` give the
    12-character prefix only, and it cannot be pickled or copied into a log
    by accident. ``reveal()`` returns the key for the Authorization header;
    after ``forget()`` it raises (``signed_out``)."""

    __slots__ = ("_value", "_prefix")

    def __init__(self, value: str):
        if not is_well_formed(value):
            value = None                       # out of the traceback's locals
            raise _malformed("argument")
        self._value: Optional[str] = value
        self._prefix = value[:KEY_PREFIX_LENGTH]

    @property
    def prefix(self) -> str:
        """The first 12 characters (``nvk_`` and the key's id): not secret,
        shown in Settings → API."""
        return self._prefix

    def reveal(self) -> str:
        """The full key. Used only to build the Authorization header."""
        if self._value is None:
            raise ConfigError(messages.text("error.signed_out"), code="signed_out")
        return self._value

    def forget(self) -> None:
        """Drop the key: this object holds only its prefix from now on.
        Python cannot wipe a string, so the key's text lasts in memory until
        nothing else refers to it."""
        self._value = None

    def __repr__(self) -> str:
        return f"ApiKey('{self.prefix}…')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ApiKey) and self._value is not None
                and other._value == self._value)

    def __hash__(self) -> int:
        return hash(self.prefix)

    def __reduce__(self):
        raise TypeError("an API key cannot be pickled")


@dataclass(frozen=True)
class Credentials:
    """The email and key a request signs in with; ``source`` says where the
    key came from (``argument``, ``environment``, ``keyring``, ``file``,
    ``prompt``)."""
    email: str
    api_key: ApiKey
    source: str = "argument"

    def __repr__(self) -> str:
        return (f"Credentials(email={self.email!r}, api_key={self.api_key!r}, "
                f"source={self.source!r})")


def _malformed(source: str) -> ConfigError:
    return ConfigError(messages.text("error.invalid_key_format",
                                     source=messages.text(f"source.{source}")),
                       code="invalid_key_format")


def make_key(value: str, source: str) -> ApiKey:
    """``value`` as an ``ApiKey``, or ``ConfigError``
    (``invalid_key_format``) naming where it came from. Surrounding spaces,
    a trailing newline and matching quotes are removed first. Makes no
    request: a mistyped key is caught by its checksum."""
    value = _clean(value) if isinstance(value, str) else value
    if not is_well_formed(value):
        value = None                           # out of the traceback's locals
        raise _malformed(source)
    return ApiKey(value)


# ── Where credentials are stored ──────────────────────────────────────────────
def config_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    """The folder a login stores credentials in: ``NALVERA_CONFIG_DIR``,
    else ``%APPDATA%\\nalvera`` on Windows, else ``$XDG_CONFIG_HOME/nalvera``
    or ``~/.config/nalvera``."""
    env = os.environ if env is None else env
    if env.get("NALVERA_CONFIG_DIR"):
        return Path(env["NALVERA_CONFIG_DIR"]).expanduser()
    if sys.platform == "win32" and env.get("APPDATA"):
        return Path(env["APPDATA"]) / "nalvera"
    base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "nalvera"


def credentials_path(directory: Optional[Path] = None) -> Path:
    """The credentials file inside ``directory`` (default: ``config_dir``)."""
    return Path(directory if directory is not None else config_dir()) / CREDENTIALS_FILE


#: Backends of the keyring library that keep the key in a plain file (the
#: ``keyrings.alt`` file backends): not a keychain, so the credentials file
#: (readable by the user only) is used instead.
_PLAINTEXT_BACKEND_MODULES = ("keyrings.alt.file",)


def is_plaintext_backend(backend: Any) -> bool:
    """True when ``backend`` (a keyring backend, or a chain of them) stores
    keys in a plain file rather than a system keychain."""
    chained = getattr(backend, "backends", None)
    if isinstance(chained, (list, tuple)) and chained:
        return all(is_plaintext_backend(b) for b in chained)
    module = type(backend).__module__ or ""
    return module.startswith(_PLAINTEXT_BACKEND_MODULES)


def _keyring_module():
    """(the keyring module, its backend), or (None, None) without one."""
    try:
        import keyring  # type: ignore[import-not-found]
        from keyring.backends import fail  # type: ignore[import-not-found]
    except Exception:
        return None, None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None, None
    if isinstance(backend, fail.Keyring) or getattr(backend, "priority", 1) <= 0:
        return None, None
    return keyring, backend


def _keyring():
    """The keyring module when it is installed and has a working backend
    that is a real keychain, else None."""
    module, backend = _keyring_module()
    if module is None or is_plaintext_backend(backend):
        return None
    return module


def keyring_is_plaintext() -> bool:
    """True when the keyring library is installed but its backend keeps
    keys in a plain file, so a login stores the key in the credentials file."""
    module, backend = _keyring_module()
    return module is not None and is_plaintext_backend(backend)


def platform_of(url: Any) -> str:
    """The platform an address names, as ``scheme://host[:port]`` in lower
    case (a default port left out): what a stored key is bound to."""
    from urllib.parse import urlsplit
    parts = urlsplit(str(url or "").strip())
    scheme, host = parts.scheme.lower(), (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    if port == {"https": 443, "http": 80}.get(scheme):
        port = None
    return f"{scheme}://{host}" + (f":{port}" if port else "")


def _read_file(path: Path) -> Dict[str, Any]:
    """The stored record, ``{}`` when there is none. Refuses a file other
    users can read (POSIX)."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return {}
    except OSError:
        raise ConfigError(messages.text("error.credentials_file_unreadable", path=str(path)),
                          code="credentials_file_unreadable") from None
    if os.name == "posix" and st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ConfigError(messages.text("error.credentials_file_insecure", path=str(path)),
                          code="credentials_file_insecure")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ConfigError(messages.text("error.credentials_file_unreadable", path=str(path)),
                          code="credentials_file_unreadable") from None
    return data if isinstance(data, dict) else {}


def private_dir(directory: Path, *, levels: int = 1) -> None:
    """Create ``directory`` and each missing folder above it with mode 0700
    (readable by the user only). ``directory``, and the ``levels - 1``
    folders above it, when they existed before, must belong to the user and
    be writable by nobody else: someone who can write there could have put a
    link in place of a file the package writes. Such a folder raises
    ``ConfigError`` (``config_dir_insecure``); an accepted ``directory``
    is set to 0700."""
    directory = Path(directory)
    missing = []
    probe = directory
    while not probe.exists() and probe != probe.parent:
        missing.append(probe)
        probe = probe.parent
    if os.name == "posix":
        checked = directory
        for _ in range(max(1, levels)):
            if checked not in missing:
                _check_private(checked)
            checked = checked.parent
    for folder in reversed(missing):
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError:
            _check_private(folder)
    if os.name == "posix":
        os.chmod(directory, 0o700)


def _check_private(folder: Path) -> None:
    """Refuse a folder another user owns or others can write (POSIX)."""
    if os.name != "posix":
        return
    try:
        st = os.stat(folder)
    except OSError:
        return
    if st.st_uid != os.getuid() or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ConfigError(messages.text("error.config_dir_insecure", path=str(folder)),
                          code="config_dir_insecure")


_private_dir = private_dir


def known_email(env: Optional[Mapping[str, str]] = None,
                directory: Optional[Path] = None) -> Optional[str]:
    """The account's email when it is known without a key: ``NALVERA_EMAIL``,
    else what a login stored; None otherwise. Reads no key."""
    env = os.environ if env is None else env
    found = _clean(env.get("NALVERA_EMAIL"))
    if found is None:
        try:
            found = _clean(_read_file(credentials_path(
                directory if directory is not None else config_dir(env))).get("email"))
        except ConfigError:
            found = None
    return found


RECORDS_FOLDER = "records"


def record_path(email: Optional[str], base_url: str, *,
                directory: Optional[Path] = None) -> Path:
    """The default record of scans and tasks: one CSV file per account and
    platform address in the ``records`` folder of the config folder
    (``directory``, default ``config_dir``). The file name holds the
    platform's host and a digest of the email and the address, never the
    email itself; ``email`` None gives the file for no account."""
    from urllib.parse import urlsplit
    parts = urlsplit(str(base_url or "").strip().rstrip("/"))
    host = (parts.hostname or "platform") + (f"_{parts.port}" if parts.port else "")
    host = re.sub(r"[^A-Za-z0-9._-]+", "_", host)
    identity = f"{(email or '').strip().lower()}\n{parts.scheme.lower()}://{host.lower()}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    stem = f"scans-{host}-{digest}" if email else f"scans-{host}-no-account"
    folder = Path(directory if directory is not None else config_dir()) / RECORDS_FOLDER
    return folder / f"{stem}.csv"


def _write_file(path: Path, record: Dict[str, Any]) -> None:
    """Write ``record`` to ``path`` atomically, readable by the user only."""
    _private_dir(path.parent)
    # A random name, created exclusively and never through a link (mkstemp:
    # O_EXCL, O_NOFOLLOW, mode 0600), then moved into place: nothing planted
    # under a name the package would use can receive the key.
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
            fh.flush()
            os.fsync(fh.fileno())
        if os.name == "posix":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def store(email: str, api_key: ApiKey, *, directory: Optional[Path] = None,
          use_keyring: bool = True, base_url: Optional[str] = None) -> str:
    """Store the credentials for later runs. Returns ``"keyring"`` when the
    key went to the system keychain, else ``"file"``. The email (not secret)
    always goes to the credentials file, so the key can be found again, with
    the platform ``base_url`` (default: https://nalvera.ai) the key is for:
    a stored key is sent to no other address."""
    path = credentials_path(directory)
    record: Dict[str, Any] = {"email": email, "base_url": platform_of(base_url or _DEFAULT)}
    kr = _keyring() if use_keyring else None
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, email, api_key.reveal())
            _write_file(path, {**record, "keyring": True})
            return "keyring"
        except ConfigError:
            raise
        except Exception:
            pass                           # no usable keychain after all: the file
    _write_file(path, {**record, "api_key": api_key.reveal()})
    return "file"


def forget(*, directory: Optional[Path] = None, use_keyring: bool = True) -> bool:
    """Remove the stored credentials (keychain entry and file). Returns
    True when something was removed."""
    path = credentials_path(directory)
    removed = False
    try:
        record = _read_file(path)
    except ConfigError:
        record = {}
    email = record.get("email")
    kr = _keyring() if use_keyring and email else None
    if kr is not None:
        try:
            if kr.get_password(KEYRING_SERVICE, email) is not None:
                kr.delete_password(KEYRING_SERVICE, email)
                removed = True
        except Exception:
            pass
    try:
        path.unlink()
        removed = True
    except FileNotFoundError:
        pass
    return removed


def resolve(email: Optional[str] = None, api_key: Optional[str] = None, *,
            env: Optional[Mapping[str, str]] = None, directory: Optional[Path] = None,
            use_keyring: bool = True, base_url: Optional[str] = None) -> Credentials:
    """The credentials to sign in with, in the order described in the module
    docstring. Raises ``NoCredentials`` (whose message lists the steps
    to get a key) when the email or the key is missing, and
    ``ConfigError`` when a key is malformed or the email holds a
    control character or a character outside ASCII (``invalid_email``).

    ``base_url`` (default None: not checked) is where the key will be sent.
    A stored key (keychain or file) goes only to the platform it was stored
    for (a record without one: https://nalvera.ai): for another address,
    ``NoCredentials`` (``stored_key_other_platform``) says why. A key
    given as an argument or in the environment goes where it is pointed."""
    env = os.environ if env is None else env
    found_email, found_key, source = _clean(email), _clean(api_key), "argument"
    if found_email is None:
        found_email = _clean(env.get("NALVERA_EMAIL"))
    if found_key is None:
        found_key, source = _clean(env.get("NALVERA_API_KEY")), "environment"
    if found_email is None or found_key is None:
        stored = _read_file(credentials_path(directory if directory is not None
                                             else config_dir(env)))
        stored_email = _clean(stored.get("email"))
        if found_email is None:
            found_email = stored_email
        same_account = (found_email is not None and stored_email is not None
                        and found_email.lower() == stored_email.lower())
        if found_key is None and stored.get("keyring") and use_keyring and found_email:
            kr = _keyring()
            if kr is not None:
                try:
                    found_key, source = _clean(kr.get_password(KEYRING_SERVICE, found_email)), \
                        "keyring"
                except Exception:
                    found_key = None
        if found_key is None and same_account and stored.get("api_key"):
            found_key, source = _clean(stored.get("api_key")), "file"
        if found_key is not None and source in ("keyring", "file") and base_url is not None:
            bound = platform_of(stored.get("base_url") or _DEFAULT)
            wanted = platform_of(base_url)
            if bound != wanted:
                found_key = None                   # out of the traceback's locals
                raise NoCredentials(messages.text("error.stored_key_other_platform",
                                                  stored=bound, platform=wanted),
                                    code="stored_key_other_platform")
    if found_email is None or found_key is None:
        raise NoCredentials()
    try:
        check_email(found_email)
        return Credentials(email=found_email, api_key=make_key(found_key, source),
                           source=source)
    except BaseException:
        api_key = found_key = env = None       # out of the traceback's locals
        raise


def check_email(email: str) -> None:
    """Refuse, before any request, an email with a control character or a
    character outside ASCII (``invalid_email``). No account has one (the
    platform accepts ASCII addresses only), and a header holding one would
    fail inside the HTTP library, after the key was put in the request."""
    if not email.isascii() or any(ord(c) < 0x20 or ord(c) == 0x7F for c in email):
        raise ConfigError(messages.text("error.invalid_email"), code="invalid_email")


def _clean(value: Any) -> Optional[str]:
    """A value as pasted into a shell or a .env file: surrounding spaces, a
    trailing newline and one pair of matching quotes are removed."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    return value or None
