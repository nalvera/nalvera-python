"""Check an anonymized file before it may be uploaded.

The writer (nifti.py) is built to produce a clean file. This module does not
take its word for it: it reopens the result as a stranger would, with its own
gzip reader (zlib, not the gzip module the writer used), and checks

  * the file is ONE gzip member whose header carries no file name, no
    comment, no extra field, no header checksum, and a zero time stamp; its
    trailer's CRC-32 and length are right, and no byte follows it;
  * the NIfTI header is a single-file header of the expected version and
    byte order; every field outside the allow-list is zero, so every text
    field is empty; every allowed field holds exactly the input's value;
  * no header extension follows the header, and the voxel block starts right
    after it;
  * the voxel block is byte for byte the input's (same length, same SHA-256);
  * nothing follows the voxel block;
  * nibabel, reading the written header, finds the input's shape and affine.

A file that fails is deleted and AnonymizeError("verification_failed") is
raised. Nothing that failed this check is ever uploaded.
"""
from __future__ import annotations

import hashlib
import os
import struct
import zlib
from typing import BinaryIO, Tuple

import numpy as np

from .errors import AnonymizeError
from .nifti import (
    ALLOWED_FIELDS,
    CHUNK,
    DATA_OFFSET,
    HEADER_CLASS,
    HEADER_DTYPE,
    HEADER_SIZE,
    MAGIC_AT,
    NIFTI2_EOL,
    SINGLE_MAGIC,
    STRUCTURAL_FIELDS,
    NiftiInfo,
    affine_bytes,
    canonical,
    field_bytes,
)


class _Failed(Exception):
    """One check failed; ``reason`` is a fixed phrase, never file content."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _GzipMember:
    """Reads a file that must be exactly one bare gzip member, returning its
    decompressed bytes in order, and hashes the file's bytes as it goes."""

    def __init__(self, fh: BinaryIO):
        self._fh = fh
        self.sha256 = hashlib.sha256()
        self.size = 0
        head = self._raw(10)
        if len(head) < 10 or head[:2] != b"\x1f\x8b" or head[2] != 8:
            raise _Failed("not a gzip file")
        if head[3] != 0:
            # FTEXT, FHCRC, FEXTRA, FNAME, FCOMMENT or a reserved bit.
            raise _Failed("the gzip header carries a name, comment or extra field")
        if head[4:8] != b"\x00\x00\x00\x00":
            raise _Failed("the gzip header carries a time stamp")
        self._inflate = zlib.decompressobj(-zlib.MAX_WBITS)
        self._pending = b""
        self._buffer = bytearray()
        self._crc = 0
        self._length = 0
        self.ended = False

    def _raw(self, count: int) -> bytes:
        data = self._fh.read(count)
        self.sha256.update(data)
        self.size += len(data)
        return data

    def _pump(self) -> None:
        if not self._pending:
            self._pending = self._raw(CHUNK)
            if not self._pending:
                raise _Failed("the gzip stream is cut short")
        data = self._inflate.decompress(self._pending, CHUNK)
        self._pending = self._inflate.unconsumed_tail
        self._crc = zlib.crc32(data, self._crc)
        self._length += len(data)
        self._buffer += data
        if self._inflate.eof:
            self._finish(self._inflate.unused_data)

    def _finish(self, tail: bytes) -> None:
        while len(tail) < 9:
            more = self._raw(9 - len(tail))
            if not more:
                break
            tail += more
        if len(tail) < 8:
            raise _Failed("the gzip trailer is cut short")
        if len(tail) > 8:
            raise _Failed("bytes follow the gzip stream")
        crc, length = struct.unpack("<II", tail)
        if crc != self._crc or length != self._length & 0xFFFFFFFF:
            raise _Failed("the gzip checksum does not match")
        self.ended = True

    def read(self, count: int) -> bytes:
        """Up to ``count`` decompressed bytes; fewer only at the end."""
        while len(self._buffer) < count and not self.ended:
            self._pump()
        out = bytes(self._buffer[:count])
        del self._buffer[:count]
        return out


def _check_header(block: bytes, expected: NiftiInfo) -> None:
    version, endian = expected.version, expected.endianness
    dtype = HEADER_DTYPE[version].newbyteorder(endian)
    if block[:4] != struct.pack(endian + "i", HEADER_SIZE[version]):
        raise _Failed("the NIfTI version or byte order changed")
    at = MAGIC_AT[version]
    if block[at:at + 4] != SINGLE_MAGIC[version]:
        raise _Failed("not a single-file NIfTI")
    if version == 2 and block[8:12] != NIFTI2_EOL:
        raise _Failed("the NIfTI-2 line-ending check is wrong")
    wanted = np.array(DATA_OFFSET[version], dtype=dtype.fields["vox_offset"][0]).tobytes()
    if field_bytes(block, version, endian, "vox_offset") != wanted:
        raise _Failed("the voxel block does not follow the header")
    same_version = version == expected.source_version
    for name in dtype.names:
        raw = field_bytes(block, version, endian, name)
        if name in STRUCTURAL_FIELDS:      # each checked exactly above
            continue
        if name not in ALLOWED_FIELDS:
            if any(raw):
                raise _Failed(f"the {name} field is set")
            continue
        if same_version:
            matches = raw == expected.raw_fields[name]
        else:
            value = np.frombuffer(raw, dtype=dtype.fields[name][0].base)
            matches = canonical(value) == expected.fields[name]
        if not matches:
            raise _Failed(f"the {name} field changed")

    # nibabel's own reading of the written header must agree.
    header = HEADER_CLASS[version](binaryblock=block, check=False)
    if tuple(int(n) for n in header.get_data_shape()) != expected.shape:
        raise _Failed("the shape changed")
    if int(header.get_data_offset()) != DATA_OFFSET[version]:
        raise _Failed("the voxel block does not follow the header")
    if affine_bytes(block, version) != expected.affine:
        raise _Failed("the affine changed")


def _verify(path: str, expected: NiftiInfo) -> Tuple[str, int]:
    with open(path, "rb") as fh:
        member = _GzipMember(fh)
        size = HEADER_SIZE[expected.version]
        block = member.read(size)
        if len(block) < size:
            raise _Failed("the header is cut short")
        _check_header(block, expected)
        if member.read(4) != b"\x00\x00\x00\x00":
            raise _Failed("a header extension is present")
        digest, left = hashlib.sha256(), expected.voxel_bytes
        while left:
            piece = member.read(min(CHUNK, left))
            if not piece:
                raise _Failed("the voxel block is too short")
            digest.update(piece)
            left -= len(piece)
        if digest.hexdigest() != expected.data_sha256:
            raise _Failed("the voxel data changed")
        if member.read(1):
            raise _Failed("bytes follow the voxel block")
        if not member.ended:
            raise _Failed("the gzip stream did not end")
    return member.sha256.hexdigest(), member.size


def verify_clean(path: str, expected: NiftiInfo) -> Tuple[str, int]:
    """Check the written file at ``path`` against what the writer read from
    the input (``expected``). Returns (SHA-256 of the file, its size in
    bytes). On any failed check, or any error while checking, the file is
    deleted and AnonymizeError(``verification_failed``) is raised."""
    try:
        return _verify(path, expected)
    except _Failed as exc:
        _delete(path)
        raise AnonymizeError("verification_failed", exc.reason) from None
    except (OSError, EOFError, zlib.error, ValueError, ArithmeticError) as exc:
        _delete(path)
        raise AnonymizeError("verification_failed", type(exc).__name__) from None
    except BaseException:
        _delete(path)
        raise


def _delete(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
