"""Encrypted uploads: the NVLT v2 container, a second layer over HTTPS.

HTTPS already encrypts the transfer. An NVLT container adds a layer that
survives a proxy that inspects HTTPS, and keeps the upload encrypted on the
web server until it is processed. The platform decrypts the scan to run the
model and stores it afterwards as it does any other upload.

Format (the Nalvera Privacy Vault's, ``docs/encryption-format.md``): an
ephemeral X25519 key per file, HKDF-SHA256 with a random 16-byte salt,
AES-256-GCM over 1 MiB frames. The 56-byte header is authenticated with
every frame, and the last frame carries a final flag, so tampering,
truncation and reordering are all detected by the platform. Encryption is
randomized: the same scan gives different bytes each time.

The platform's public key comes from ``GET /api/v1/me``. The package uses
only a key whose SHA-256 fingerprint it carries (``TRUSTED_KEY_SHA256``):
a key swapped on the way is never used, and encryption then stops with the
instruction to upgrade the package. A server on ``localhost`` or
``127.0.0.1`` (development, tests) is trusted without a fingerprint.

The streaming encryptor is adapted from the Nalvera Privacy Vault,
``sidecar/vault_sidecar/crypto.py::encrypt_stream`` (Apache-2.0, Nalvera BV).
"""
from __future__ import annotations

import base64
import hashlib
import os
import struct
from typing import Any, BinaryIO, Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit

from . import messages
from .errors import EncryptionError

MAGIC = b"NVLT"
CONTAINER_VERSION = 2
ALGORITHM_ID = 1
HEADER_STRUCT = struct.Struct(">4sBBH32s16s")   # magic, version, algorithm, reserved, key, salt
HEADER_LEN = HEADER_STRUCT.size                # 56
HKDF_INFO = b"nalvera-vault/v2/x25519-aesgcm"
KEY_LEN = 32
SALT_LEN = 16
TAG_LEN = 16
FRAME_PLAINTEXT = 1 << 20                      # 1 MiB of plaintext per frame
FORMAT = "NVLT/2"
SUFFIX = ".nvlt"

#: SHA-256 fingerprints (hex) of the platform public keys this version
#: trusts, current key first. Replacing the platform's key ships its new
#: fingerprint here, in a new package version.
TRUSTED_KEY_SHA256: Tuple[str, ...] = (
    "aaafbb1dea9551e38fad18ae907ce66be6cbf884eeeee09452c9ef55d0b4a701",
)

_LOCAL_HOSTS = ("localhost", "127.0.0.1")


def _crypto():
    """The ``cryptography`` primitives, or ``EncryptionError``
    (``encrypt_unavailable``) when the library cannot be imported."""
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey,
            X25519PublicKey,
        )
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except Exception:
        raise EncryptionError(messages.text("error.encrypt_unavailable"),
                              code="encrypt_unavailable") from None
    return hashes, serialization, X25519PrivateKey, X25519PublicKey, AESGCM, HKDF


def available() -> bool:
    """Whether this computer can encrypt (``cryptography`` imports)."""
    try:
        _crypto()
    except EncryptionError:
        return False
    return True


def fingerprint(public_key: bytes) -> str:
    """The SHA-256 (hex) of a raw 32-byte public key."""
    return hashlib.sha256(public_key).hexdigest()


def _nonce(counter: int) -> bytes:
    # 4 zero bytes and the frame index: unique per frame under a per-file key.
    return b"\x00\x00\x00\x00" + struct.pack(">Q", counter)


def _aad(header: bytes, final: bool) -> bytes:
    return header + (b"\x01" if final else b"\x00")


def encrypt_stream(source: BinaryIO, target: BinaryIO, public_key: bytes, *,
                   _ephemeral: Optional[bytes] = None, _salt: Optional[bytes] = None) -> int:
    """Encrypt ``source`` into ``target`` for the platform's raw 32-byte
    X25519 ``public_key``. Streams 1 MiB frames, holding at most two in
    memory. Returns the number of bytes written. (``_ephemeral`` and
    ``_salt`` fix the random parts for tests of the byte layout; never pass
    them otherwise.)"""
    hashes, serialization, private_cls, public_cls, aesgcm_cls, hkdf_cls = _crypto()
    if len(public_key) != KEY_LEN:
        raise ValueError("the public key must be 32 raw bytes")
    recipient = public_cls.from_public_bytes(public_key)
    ephemeral = (private_cls.from_private_bytes(_ephemeral) if _ephemeral is not None
                 else private_cls.generate())
    ephemeral_public = ephemeral.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    salt = _salt if _salt is not None else os.urandom(SALT_LEN)
    key = hkdf_cls(algorithm=hashes.SHA256(), length=KEY_LEN, salt=salt,
                   info=HKDF_INFO).derive(ephemeral.exchange(recipient))
    aead = aesgcm_cls(key)
    header = HEADER_STRUCT.pack(MAGIC, CONTAINER_VERSION, ALGORITHM_ID, 0, ephemeral_public, salt)
    target.write(header)
    written = len(header)
    counter = 0
    current = source.read(FRAME_PLAINTEXT)
    while True:
        following = source.read(FRAME_PLAINTEXT)
        final = not following
        frame = aead.encrypt(_nonce(counter), current, _aad(header, final))
        target.write(struct.pack(">I", len(frame)))
        target.write(frame)
        written += 4 + len(frame)
        counter += 1
        if final:
            return written
        current = following


def encrypt_file(source: str, target: str, public_key: bytes) -> int:
    """Encrypt the file ``source`` into a new file ``target`` (mode 0600,
    never replacing an existing file). Returns the container's size. On any
    error ``target`` is removed."""
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                 0o600)
    try:
        with os.fdopen(fd, "wb") as out, open(source, "rb") as src:
            size = encrypt_stream(src, out, public_key)
            out.flush()
            os.fsync(out.fileno())
        return size
    except BaseException:
        try:
            os.remove(target)
        except OSError:
            pass
        raise


def is_local(base_url: str) -> bool:
    """Whether ``base_url`` is a server on this computer."""
    try:
        host = (urlsplit(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOCAL_HOSTS


def platform_key(encryption: Optional[Dict[str, Any]], base_url: str, *,
                 trusted: Tuple[str, ...] = TRUSTED_KEY_SHA256) -> bytes:
    """The platform public key to encrypt for, from the ``encryption`` block
    of ``GET /api/v1/me``: the first listed key whose fingerprint is in
    ``trusted`` (for a local server, the first listed key).

    Raises ``EncryptionError``: ``encryption_unavailable`` when the
    platform lists no key, ``encrypt_no_trusted_key`` when none of its keys
    is trusted by this version (upgrade the package)."""
    block = encryption if isinstance(encryption, dict) else {}
    keys = [k for k in block.get("keys") or [] if isinstance(k, dict)]
    if not block.get("available") or not keys:
        raise EncryptionError(messages.text("error.encryption_unavailable",
                                            wait=messages.text("time.wait_unknown")),
                              code="encryption_unavailable")
    local = is_local(base_url)
    for listed in keys:
        try:
            raw = base64.b64decode(str(listed.get("public_key") or ""), validate=True)
        except (ValueError, TypeError):
            continue
        if len(raw) != KEY_LEN:
            continue
        if local or fingerprint(raw) in trusted:
            return raw
    raise EncryptionError(messages.text("error.encrypt_no_trusted_key"),
                          code="encrypt_no_trusted_key")


def decide(asked: Optional[bool], encryption: Optional[Dict[str, Any]], base_url: str,
           *, key_for: Callable[..., bytes] = platform_key) -> Optional[bytes]:
    """Whether a run encrypts, and for which key. ``asked`` is the user's
    choice (True, False, or None for "not said"); ``encryption`` the block
    of ``GET /api/v1/me``.

    * The account requires encryption and the user said False: refused
      (``encrypt_required_by_account``).
    * Asked, or required by the account: the key, or an
      ``EncryptionError`` when it cannot encrypt. Never plaintext.
    * Otherwise: None (no encryption).
    """
    block = encryption if isinstance(encryption, dict) else {}
    required = bool(block.get("required"))
    if required and asked is False:
        raise EncryptionError(messages.text("error.encrypt_required_by_account"),
                              code="encrypt_required_by_account")
    if not (asked or required):
        return None
    _crypto()                                   # fail before anything else is done
    return key_for(block, base_url)


__all__ = ["encrypt_stream", "encrypt_file", "platform_key", "decide", "fingerprint",
           "available", "TRUSTED_KEY_SHA256", "FORMAT", "SUFFIX"]
