"""An NVLT v2 decryptor written for the tests, independently of the package's
encryptor, from the format description (Privacy-Vault docs/encryption-format.md)
and the platform's rules (backend/vault_crypto.py): header as AAD, frame
index in the nonce, a final flag on the last frame, every frame but the last
full. The fake server uses it the way the platform decrypts at ingest."""
from __future__ import annotations

import struct

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

HEADER = struct.Struct(">4sBBH32s16s")
FRAME = 1 << 20
TAG = 16


class NotDecryptable(Exception):
    pass


def new_key_pair():
    """(private raw, public raw) of a fresh X25519 key."""
    private = X25519PrivateKey.generate()
    from cryptography.hazmat.primitives.serialization import NoEncryption, PrivateFormat
    return (private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
            private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def decrypt(data: bytes, private_raw: bytes) -> bytes:
    """The plaintext of an NVLT v2 container, or NotDecryptable."""
    if len(data) < HEADER.size:
        raise NotDecryptable("short")
    header = data[:HEADER.size]
    magic, version, algorithm, _reserved, ephemeral, salt = HEADER.unpack(header)
    if magic != b"NVLT" or version != 2 or algorithm != 1:
        raise NotDecryptable("header")
    private = X25519PrivateKey.from_private_bytes(private_raw)
    shared = private.exchange(X25519PublicKey.from_public_bytes(ephemeral))
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt,
               info=b"nalvera-vault/v2/x25519-aesgcm").derive(shared)
    aead = AESGCM(key)
    frames, at = [], HEADER.size
    while at < len(data):
        if at + 4 > len(data):
            raise NotDecryptable("length")
        (length,) = struct.unpack(">I", data[at:at + 4])
        at += 4
        if length < TAG or at + length > len(data):
            raise NotDecryptable("frame")
        frames.append(data[at:at + length])
        at += length
    if not frames:
        raise NotDecryptable("empty")
    out = []
    for index, frame in enumerate(frames):
        final = index == len(frames) - 1
        if not final and len(frame) != FRAME + TAG:
            raise NotDecryptable("size")
        nonce = b"\x00" * 4 + struct.pack(">Q", index)
        try:
            out.append(aead.decrypt(nonce, frame, header + (b"\x01" if final else b"\x00")))
        except Exception:
            raise NotDecryptable("tag") from None
    return b"".join(out)
