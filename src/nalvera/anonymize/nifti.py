"""Rebuild a NIfTI file with a clean header and the same voxels.

What goes in
    A single-file NIfTI-1 or NIfTI-2 (``.nii`` or ``.nii.gz``; gzip is
    recognised by content, never by name, and a stream of several gzip
    members is read through), in either byte order. Header + image pairs
    (``ni1``/``ni2``) and Analyze files are refused.

    As real tools write them: a ``vox_offset`` of 0 (unset) means the voxels
    follow the header; it is accepted only when no extension is flagged and
    the file holds nothing but the header and the voxels, so extension bytes
    are never taken for voxels. A bit depth that disagrees with the data type
    is read by the data type (as nibabel and niftilib do) and kept as stored.
    A scaling slope of 0 or NaN ("no scaling") is kept as stored.

What comes out
    A NIfTI-1 ``.nii.gz`` whose header is NEW: zeros everywhere except the
    fields the format needs (size, magic, voxel offset) and the allow-list
    below, copied as stored. The voxel block is copied byte for byte, in the
    input's byte order, so the stored values, data type, scaling and geometry
    are the input's exactly.

    Kept (ALLOWED_FIELDS): dimensions, data type and bit depth, voxel sizes
    (``pixdim``, whose first value is the qform handedness), scaling slope
    and intercept, units, the qform and sform codes, the stored quaternion,
    offsets and ``srow`` rows (never recomputed from one another), the slice
    timing fields, the intent code.
    Dropped: ``descrip`` (converters write acquisition details and times
    there), ``aux_file``, ``db_name``, ``intent_name``, the unused text
    fields, display ranges, intent parameters, every header extension (a
    DICOM extension can hold the whole DICOM header), and any byte after the
    voxel block.

    The platform accepts NIfTI-1 only. A NIfTI-2 input is written as NIfTI-1
    when every kept field converts exactly and nibabel derives the same
    affine; otherwise it is refused (``nifti2_unsupported``) before any file
    is written, so no upload is spent on a file the platform would reject.

    The gzip stream is written with no file name and a zero time stamp: gzip
    can store both, and a file name can hold a patient name.

Why bytes and not nibabel's ``save``
    Loading an image and saving it again lets the library choose the stored
    type and scaling; with a slope or intercept present that can change the
    stored integers. Copying the block cannot. It also keeps memory flat: the
    voxels pass through in 1 MiB pieces and are never held whole, so the
    peak is a few MiB (one piece, the header, and zlib's buffers) whatever
    the scan's size.

Pixel data is not changed. A face in a head scan and text burned into the
image stay as they are.
"""
from __future__ import annotations

import gzip
import hashlib
import math
import os
import struct
import zlib
from dataclasses import dataclass
from typing import BinaryIO, Dict, NamedTuple, Optional, Tuple

import numpy as np
from nibabel.nifti1 import Nifti1Header
from nibabel.nifti1 import header_dtype as _NIFTI1_DTYPE
from nibabel.nifti2 import Nifti2Header
from nibabel.nifti2 import header_dtype as _NIFTI2_DTYPE

from .errors import AnonymizeError

HEADER_SIZE = {1: 348, 2: 540}
# Where the voxels start in a file with no extension: the header, then the
# 4-byte extension flag.
DATA_OFFSET = {1: 352, 2: 544}
SINGLE_MAGIC = {1: b"n+1\x00", 2: b"n+2\x00"}
MAGIC_AT = {1: 344, 2: 4}
NIFTI2_EOL = b"\r\n\x1a\n"
HEADER_DTYPE = {1: _NIFTI1_DTYPE, 2: _NIFTI2_DTYPE}
HEADER_CLASS = {1: Nifti1Header, 2: Nifti2Header}
# The version every output is written in (the platform accepts no other).
OUTPUT_VERSION = 1

# Technical fields copied into the new header. None of them is text.
ALLOWED_FIELDS = (
    "dim_info", "dim", "intent_code", "datatype", "bitpix",
    "slice_start", "pixdim", "scl_slope", "scl_inter", "slice_end", "slice_code",
    "xyzt_units", "slice_duration", "toffset",
    "qform_code", "sform_code",
    "quatern_b", "quatern_c", "quatern_d", "qoffset_x", "qoffset_y", "qoffset_z",
    "srow_x", "srow_y", "srow_z",
)
# Fields the format itself needs; this module writes their values.
STRUCTURAL_FIELDS = ("sizeof_hdr", "magic", "eol_check", "vox_offset")

# Bytes per voxel of each NIfTI data type code. float128 and complex256 are
# left out: numpy cannot represent them on every platform, so neither the
# check after writing nor the platform could read them.
_ITEMSIZE = {2: 1, 4: 2, 8: 4, 16: 4, 32: 8, 64: 8, 128: 3, 256: 1, 512: 2,
             768: 4, 1024: 8, 1280: 8, 1792: 16, 2304: 4}

GZIP_MAGIC = b"\x1f\x8b"
CHUNK = 1024 * 1024
_GZIP_LEVEL = 4          # level 9 costs ~40% more time for under 1% on CT/MR data
# A header may declare any size. Refuse nonsense before a hostile file makes
# the copy loop run for hours (the platform's own cap is far lower).
_MAX_VOXEL_BYTES = 64 * 1024 ** 3
# Extensions are skipped without being parsed; bound the skip the same way.
_MAX_EXTENSION_BYTES = 256 * 1024 ** 2


@dataclass(frozen=True)
class NiftiInfo:
    """What the writer learned about a scan, for the check after writing.
    Nothing here is identifying: field values are the technical allow-list."""
    version: int                  # NIfTI version of the written file
    source_version: int           # NIfTI version of the input
    endianness: str               # "<" or ">", the input's and the output's
    shape: Tuple[int, ...]
    datatype: int
    voxel_bytes: int              # length of the voxel block
    data_sha256: str              # of the voxel block, as stored
    source_sha256: str            # of the input file's bytes as found on disk
    raw_fields: Dict[str, bytes]  # each kept field as stored in the input
    fields: Dict[str, bytes]      # each kept field's value, version-independent
    affine: bytes                 # nibabel's affine of the input, float64 little-endian


@dataclass(frozen=True)
class HeaderSummary:
    """What a NIfTI file's header says, read without its voxels.

    ``shape`` is the image's dimensions, ``version`` its NIfTI version (1 or
    2), and ``problem`` the AnonymizeError code ``write_clean`` would raise
    for a file with this header (``unreadable``, ``nifti2_unsupported``), or
    None when the header gives no reason to refuse it.
    """
    shape: Tuple[int, ...]
    version: int
    problem: Optional[str] = None


class _Layout(NamedTuple):
    shape: Tuple[int, ...]
    datatype: int
    voxel_bytes: int          # length of the voxel block, from the data type
    offset: int               # where the voxels start in the input
    offset_unset: bool        # vox_offset was 0: the voxels follow the header


def is_gzip(path: str) -> bool:
    """True when the file at ``path`` starts with the gzip signature (its
    content decides, never its name). False when it cannot be read."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == GZIP_MAGIC
    except OSError:
        return False


def looks_like_nifti(path: str) -> bool:
    """True when the file at ``path`` is gzip (a possible ``.nii.gz``) or
    starts with a NIfTI-1 or NIfTI-2 header size in either byte order.
    Content only; False when it cannot be read."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return False
    return head[:2] == GZIP_MAGIC or _version_of(head) is not None


def canonical(value) -> bytes:
    """A field's value as little-endian float64 or int64 bytes, so the same
    value compares equal across NIfTI versions and byte orders. Exact: every
    float32 and every integer field converts without loss."""
    array = np.asarray(value)
    if array.dtype.kind == "f":
        return array.astype("<f8").tobytes()
    if array.dtype.kind in "iub":
        return array.astype("<i8").tobytes()
    return array.tobytes()


def affine_bytes(block: bytes, version: int) -> bytes:
    """nibabel's affine for the header bytes ``block`` of NIfTI ``version``
    (1 or 2), as float64 little-endian bytes. Raises ValueError when nibabel
    cannot compute one (a broken quaternion)."""
    header = HEADER_CLASS[version](binaryblock=block, check=False)
    return np.asarray(header.get_best_affine(), dtype="<f8").tobytes()


class _HashingReader:
    """Reads a file and hashes every byte read, so the original file's
    SHA-256 comes out of the same pass that copies it."""

    def __init__(self, fh: BinaryIO):
        self._fh = fh
        self.sha256 = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._fh.read(size)
        self.sha256.update(data)
        return data

    def drain(self) -> None:
        """Read (and hash) the rest of the file."""
        while self.read(CHUNK):
            pass


def _read(stream: BinaryIO, count: int) -> bytes:
    try:
        return stream.read(count)
    except (OSError, EOFError, zlib.error) as exc:
        # A truncated or damaged gzip stream, or a disk error, on the INPUT.
        raise AnonymizeError("unreadable", type(exc).__name__) from None


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    pieces, left = [], count
    while left:
        piece = _read(stream, min(left, CHUNK))
        if not piece:
            break
        pieces.append(piece)
        left -= len(piece)
    return b"".join(pieces)


def _version_of(first: bytes) -> Optional[Tuple[int, str]]:
    for version, size in HEADER_SIZE.items():
        if first == struct.pack("<i", size):
            return version, "<"
        if first == struct.pack(">i", size):
            return version, ">"
    return None


def field_bytes(block: bytes, version: int, endian: str, name: str) -> bytes:
    """The field ``name`` of the header bytes ``block`` (NIfTI ``version`` 1
    or 2, byte order ``endian`` "<" or ">") exactly as stored."""
    field_type, offset = HEADER_DTYPE[version].newbyteorder(endian).fields[name][:2]
    return block[offset:offset + field_type.itemsize]


def read_header(stream: BinaryIO) -> Tuple[np.void, int, str, bytes]:
    """(fields, version, byte order, header bytes) of the fixed header at the
    start of an open stream. Reads the fixed header only, never the
    extensions after it. ``fields`` gives values (numpy handles the byte
    order); ``field_bytes`` gives a field's stored bytes."""
    first = _read_exact(stream, 4)
    found = _version_of(first) if len(first) == 4 else None
    if found is None:
        raise AnonymizeError("not_a_scan", "no NIfTI header")
    version, endian = found
    size = HEADER_SIZE[version]
    block = first + _read_exact(stream, size - 4)
    if len(block) < size:
        raise AnonymizeError("unreadable", "the header is cut short")
    at = MAGIC_AT[version]
    if block[at:at + 4] != SINGLE_MAGIC[version]:
        # "ni1"/"ni2" is a header + image pair; anything else is Analyze.
        raise AnonymizeError("not_a_scan", "not a single-file NIfTI")
    if version == 2 and block[8:12] not in (NIFTI2_EOL, b"\x00" * 4):
        # The line-ending check bytes: a text-mode copy rewrote them, and
        # with them, bytes in the voxel block.
        raise AnonymizeError("unreadable", "the file was altered by a text-mode copy")
    return _parse(block, version, endian), version, endian, block


def _layout(fields: np.void, version: int) -> _Layout:
    """Where the voxels are and how many bytes they take, as the header says."""
    dim = [int(v) for v in fields["dim"]]
    ndim = dim[0]
    if not 1 <= ndim <= 7:
        raise AnonymizeError("unreadable", "the header's dimensions are not valid")
    shape = tuple(dim[1:ndim + 1])
    if any(n < 1 for n in shape):
        raise AnonymizeError("unreadable", "the header's dimensions are not valid")
    datatype = int(fields["datatype"])
    # The data type alone sets the voxel size; bitpix is not consulted, as in
    # nibabel and niftilib (tools write it wrong, and the platform reads the
    # file with nibabel).
    itemsize = _ITEMSIZE.get(datatype)
    if itemsize is None:
        raise AnonymizeError("unreadable", "the data type is not supported")
    voxel_bytes = itemsize
    for n in shape:
        voxel_bytes *= n           # Python integers: no overflow
        if voxel_bytes > _MAX_VOXEL_BYTES:
            raise AnonymizeError("too_large", "the header declares more voxels than a scan holds")

    raw_offset = fields["vox_offset"]
    if version == 1:
        value = float(raw_offset)
        if not math.isfinite(value) or value != int(value):
            raise AnonymizeError("unreadable", "the voxel offset is not valid")
        offset = int(value)
    else:
        offset = int(raw_offset)
    if offset == 0:
        # Unset: the standard puts the voxels right after the header (nibabel
        # would read them from byte 0, the header itself). The writer then
        # requires no flagged extension and nothing after the voxels, so
        # extension bytes can never pass for voxels.
        return _Layout(shape, datatype, voxel_bytes, DATA_OFFSET[version], True)
    if offset < DATA_OFFSET[version]:
        raise AnonymizeError("unreadable", "the voxel offset points inside the header")
    if offset - HEADER_SIZE[version] > _MAX_EXTENSION_BYTES:
        raise AnonymizeError("unreadable", "the voxel offset is past any real extension")
    return _Layout(shape, datatype, voxel_bytes, offset, False)


def _check_plain_size(layout: _Layout, size: int) -> None:
    end = layout.offset + layout.voxel_bytes
    if size < end:
        raise AnonymizeError("unreadable", "the file ends before its last voxel")
    if layout.offset_unset and size != end:
        raise AnonymizeError("unreadable", "the voxel offset is unset and the file holds more "
                                           "than a header and its voxels")


def build_header(fields: np.void, version: int, endian: str) -> bytes:
    """A NEW header of NIfTI ``version`` in byte order ``endian``: zeros,
    the structural fields, and ALLOWED_FIELDS copied from ``fields``. The
    voxels follow it after a 4-byte "no extension" flag."""
    out = np.zeros((), dtype=HEADER_DTYPE[version].newbyteorder(endian))
    out["sizeof_hdr"] = HEADER_SIZE[version]
    out["magic"] = SINGLE_MAGIC[version]
    if version == 2:
        out["eol_check"] = np.frombuffer(NIFTI2_EOL, dtype="i1")
    out["vox_offset"] = DATA_OFFSET[version]
    with np.errstate(over="ignore", invalid="ignore"):
        for name in ALLOWED_FIELDS:
            out[name] = fields[name]
    return out.tobytes()


def _parse(block: bytes, version: int, endian: str) -> np.void:
    return np.frombuffer(block, dtype=HEADER_DTYPE[version].newbyteorder(endian), count=1)[0]


def _source_affine(fields: np.void, version: int, endian: str) -> bytes:
    try:
        return affine_bytes(build_header(fields, version, endian), version)
    except (ValueError, ArithmeticError):
        raise AnonymizeError("unreadable", "the orientation is not valid") from None


def _output_header(fields: np.void, version: int, endian: str, source_affine: bytes) -> bytes:
    """The new NIfTI-1 header. A NIfTI-2 input converts only when every kept
    field converts exactly and nibabel derives the same affine; otherwise
    AnonymizeError("nifti2_unsupported"): the platform accepts NIfTI-1 only."""
    candidate = None
    try:
        candidate = build_header(fields, OUTPUT_VERSION, endian)
        if version == OUTPUT_VERSION:
            return candidate
        parsed = _parse(candidate, OUTPUT_VERSION, endian)
        exact = all(canonical(parsed[name]) == canonical(fields[name])
                    for name in ALLOWED_FIELDS)
        if exact and affine_bytes(candidate, OUTPUT_VERSION) == source_affine:
            return candidate
    except (OverflowError, ValueError, TypeError):
        pass
    raise AnonymizeError("nifti2_unsupported")


def _create(path: str) -> BinaryIO:
    """A new file only this user may read, created atomically with that mode:
    O_EXCL refuses an existing file or a planted link."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.fdopen(os.open(path, flags, 0o600), "wb")


def write_clean(source: str, target: str, max_bytes: Optional[int] = None) -> NiftiInfo:
    """Write the anonymized copy of the NIfTI file ``source`` to the new file
    ``target`` (gzip, mode 0600) and return what the check after writing
    needs.

    ``max_bytes`` (default None: no limit) refuses an output larger than the
    upload limit as soon as it passes it.

    Raises AnonymizeError (``not_found``, ``not_a_scan``, ``unreadable``,
    ``too_large``, ``nifti2_unsupported``); an OSError from writing
    ``target`` (a full disk) passes through. Every refusal the header can
    tell is made before ``target`` is created; on any later failure
    ``target`` is removed.
    """
    return _rebuild(source, target, max_bytes)


def read_clean(source: str) -> NiftiInfo:
    """Everything ``write_clean`` checks and learns about the NIfTI file
    ``source``, reading it whole but writing nothing: the same refusals, and
    the same NiftiInfo (``content_key`` of it identifies the file
    ``write_clean`` would write)."""
    return _rebuild(source, None, None)


def _rebuild(source: str, target: Optional[str], max_bytes: Optional[int]) -> NiftiInfo:
    if not os.path.isfile(source):
        raise AnonymizeError("not_found")
    created = False
    try:
        compressed = is_gzip(source)
        try:
            raw_in = open(source, "rb")
        except OSError as exc:
            raise AnonymizeError("unreadable", type(exc).__name__) from None
        with raw_in:
            tee = _HashingReader(raw_in)
            stream = gzip.GzipFile(fileobj=tee, mode="rb") if compressed else tee
            fields, version, endian, source_header = read_header(stream)
            layout = _layout(fields, version)
            if not compressed:
                _check_plain_size(layout, os.fstat(raw_in.fileno()).st_size)
            source_affine = _source_affine(fields, version, endian)
            header = _output_header(fields, version, endian, source_affine)

            # The extension flag, then any extensions: read and discarded.
            flag = _read_exact(stream, 4)
            if len(flag) < 4:
                raise AnonymizeError("unreadable", "the file ends inside its extensions")
            if layout.offset_unset and flag[:1] != b"\x00":
                raise AnonymizeError("unreadable", "the voxel offset is unset and an "
                                                   "extension follows the header")
            left = layout.offset - HEADER_SIZE[version] - 4
            while left:
                piece = _read(stream, min(CHUNK, left))
                if not piece:
                    raise AnonymizeError("unreadable", "the file ends inside its extensions")
                left -= len(piece)

            digest = hashlib.sha256()
            if target is None:
                _copy_voxels(stream, layout.voxel_bytes, digest, None, None, None)
            else:
                raw_out = _create(target)
                created = True
                with raw_out:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, mtime=0,
                                       compresslevel=_GZIP_LEVEL) as out:
                        out.write(header)
                        out.write(b"\x00\x00\x00\x00")        # "no extension follows"
                        _copy_voxels(stream, layout.voxel_bytes, digest, out, raw_out,
                                     max_bytes)
                    raw_out.flush()
                    os.fsync(raw_out.fileno())
                if max_bytes is not None and os.path.getsize(target) > max_bytes:
                    raise AnonymizeError("too_large")
            if layout.offset_unset and _read(stream, 1):
                raise AnonymizeError("unreadable", "the voxel offset is unset and bytes "
                                                   "follow the voxels")
            # Bytes after the voxel block are never copied; they are only
            # hashed, for the original file's SHA-256.
            try:
                tee.drain()
            except OSError as exc:
                raise AnonymizeError("unreadable", type(exc).__name__) from None
    except BaseException:
        if created:
            _remove(target)
        raise
    return NiftiInfo(
        version=OUTPUT_VERSION, source_version=version, endianness=endian, shape=layout.shape,
        datatype=layout.datatype, voxel_bytes=layout.voxel_bytes,
        data_sha256=digest.hexdigest(), source_sha256=tee.sha256.hexdigest(),
        raw_fields={name: field_bytes(source_header, version, endian, name)
                    for name in ALLOWED_FIELDS},
        fields={name: canonical(fields[name]) for name in ALLOWED_FIELDS},
        affine=source_affine)


def _copy_voxels(stream: BinaryIO, length: int, digest, out, raw_out,
                 max_bytes: Optional[int]) -> None:
    left = length
    while left:
        piece = _read(stream, min(CHUNK, left))
        if not piece:
            raise AnonymizeError("unreadable", "the file ends before its last voxel")
        digest.update(piece)
        left -= len(piece)
        if out is not None:
            out.write(piece)
            if max_bytes is not None and raw_out.tell() > max_bytes:
                raise AnonymizeError("too_large")


def describe(path: str) -> HeaderSummary:
    """What the header of the NIfTI file at ``path`` says, read without its
    voxels (a few hundred bytes, decompressed when the file is gzip).

    Raises AnonymizeError (``not_found``, ``not_a_scan``, ``unreadable``,
    ``too_large``) when the header gives no shape. A header that gives a
    shape but that ``write_clean`` would refuse is returned with its
    ``problem`` set. What only the whole file can show (a gzip stream cut
    short) is not checked here.
    """
    if not os.path.isfile(path):
        raise AnonymizeError("not_found")
    compressed = is_gzip(path)
    try:
        raw_in = open(path, "rb")
    except OSError as exc:
        raise AnonymizeError("unreadable", type(exc).__name__) from None
    with raw_in:
        stream = gzip.GzipFile(fileobj=raw_in, mode="rb") if compressed else raw_in
        fields, version, endian, _block = read_header(stream)
        layout = _layout(fields, version)
        try:
            if not compressed:
                _check_plain_size(layout, os.fstat(raw_in.fileno()).st_size)
            _output_header(fields, version, endian, _source_affine(fields, version, endian))
            if layout.offset_unset and _read_exact(stream, 4)[:1] != b"\x00":
                raise AnonymizeError("unreadable", "an extension follows the header")
        except AnonymizeError as error:
            return HeaderSummary(layout.shape, version, error.code)
    return HeaderSummary(layout.shape, version, None)


def content_key(info: NiftiInfo) -> Tuple:
    """A key of ``info`` (a NiftiInfo from ``write_clean`` or
    ``read_clean``), equal for two of them exactly when their anonymized
    files are identical: the same byte order, the same kept header fields and
    the same voxel bytes."""
    return (info.version, info.endianness, info.datatype, info.shape,
            tuple(info.fields[name] for name in ALLOWED_FIELDS), info.data_sha256)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
