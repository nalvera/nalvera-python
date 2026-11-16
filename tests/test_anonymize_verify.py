"""The check after writing (nalvera.anonymize.verify).

Each test writes a clean file with the real writer, corrupts it in one way,
and asserts the check refuses it and deletes it. The corrupted file is
rebuilt with a bare gzip header unless the gzip header is the corruption,
and an untouched rebuild passes, so a refusal is never the rebuild's fault.
"""
from __future__ import annotations

import hashlib
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize_helpers as h  # noqa: E402
from nalvera.anonymize import AnonymizeError  # noqa: E402
from nalvera.anonymize.nifti import write_clean  # noqa: E402
from nalvera.anonymize.verify import verify_clean  # noqa: E402

AFFINE = np.array([[0.0, -0.8, 0.0, 120.5],
                   [0.9, 0.0, 0.0, -98.25],
                   [0.0, 0.0, 2.5, 33.0],
                   [0.0, 0.0, 0.0, 1.0]])
# Every value exact in float32 and a quaternion of 0 and 1: the NIfTI-2 input
# is written as NIfTI-1, so the check compares across versions.
AFFINE_EXACT = np.array([[-0.75, 0.0, 0.0, 120.5],
                         [0.0, 0.875, 0.0, -98.25],
                         [0.0, 0.0, 2.5, 33.0],
                         [0.0, 0.0, 0.0, 1.0]])


@pytest.fixture(params=[(1, "<"), (1, ">"), (2, "<")], ids=["nifti1", "big-endian", "nifti2"])
def clean(request, tmp_path):
    """(path of a clean output, what the writer expects, its payload)."""
    version, endian = request.param
    affine = AFFINE if version == 1 else AFFINE_EXACT
    data = np.random.default_rng(3).integers(0, 4000, size=(10, 9, 4)).astype(np.int16)
    payload, _ = h.nifti_bytes(data, version=version, endian=endian, qform=affine,
                               sform=affine, slope=1.0, inter=-1024.0,
                               fields={"descrip": b"Hendrickx^Marieke"})
    source = tmp_path / "in.nii.gz"
    source.write_bytes(h.gzip_bytes(payload, name=b"Hendrickx.nii"))
    target = str(tmp_path / "out.nii.gz")
    info = write_clean(str(source), target)
    with open(target, "rb") as fh:
        raw = h.read_gzip_members(fh.read())
    return target, info, raw


def _header_size(info) -> int:
    return 348 if info.version == 1 else 540


def _rewrite(path: str, payload: bytes, **gzip_args) -> None:
    with open(path, "wb") as fh:
        fh.write(h.gzip_bytes(payload, **gzip_args))


def _field(info, name: str):
    """(offset, numpy type) of a header field in the written file."""
    from nalvera.anonymize.nifti import HEADER_DTYPE
    dtype = HEADER_DTYPE[info.version].newbyteorder(info.endianness)
    field_type, offset = dtype.fields[name][:2]
    return offset, field_type


def _set(raw: bytes, info, name: str, value) -> bytes:
    offset, field_type = _field(info, name)
    data = np.array(value, dtype=field_type).tobytes()
    return raw[:offset] + data + raw[offset + len(data):]


def _refused(path: str, info) -> AnonymizeError:
    with pytest.raises(AnonymizeError) as caught:
        verify_clean(path, info)
    assert caught.value.code == "verification_failed"
    assert not os.path.exists(path)
    return caught.value


def test_clean_file_passes_and_is_described(clean):
    path, info, raw = clean
    digest, size = verify_clean(path, info)
    with open(path, "rb") as fh:
        content = fh.read()
    assert digest == hashlib.sha256(content).hexdigest()
    assert size == len(content)


def test_written_as_nifti1(clean):
    """Every output is NIfTI-1, a NIfTI-2 input included (the platform
    accepts NIfTI-1 only)."""
    path, info, raw = clean
    assert info.version == 1
    assert raw[:4] in (b"\x5c\x01\x00\x00", b"\x00\x00\x01\x5c")


def test_untouched_rebuild_passes(clean):
    path, info, raw = clean
    _rewrite(path, raw)
    verify_clean(path, info)


@pytest.mark.parametrize("name, value", [
    ("descrip", b"Hendrickx^Marieke"),
    ("aux_file", b"HDX"),
    ("intent_name", b"x"),
])
def test_text_field_set(clean, name, value):
    path, info, raw = clean
    _rewrite(path, _set(raw, info, name, value))
    _refused(path, info)


def test_version_specific_text_field_set(clean):
    path, info, raw = clean
    name = "db_name" if info.version == 1 else "unused_str"
    _rewrite(path, _set(raw, info, name, b"Hendrickx"))
    _refused(path, info)


@pytest.mark.parametrize("name, value", [
    ("cal_max", 3000.0), ("intent_p1", 1.0), ("cal_min", -5.0),
])
def test_unlisted_field_set(clean, name, value):
    path, info, raw = clean
    _rewrite(path, _set(raw, info, name, value))
    _refused(path, info)


@pytest.mark.parametrize("name", ["srow_x", "pixdim", "quatern_b", "scl_inter", "dim_info",
                                  "slice_duration", "qform_code"])
def test_allowed_field_changed(clean, name):
    path, info, raw = clean
    offset, field_type = _field(info, name)
    current = np.frombuffer(raw[offset:offset + field_type.itemsize], dtype=field_type.base)
    changed = current.copy()
    changed.flat[0] = changed.flat[0] + 1
    _rewrite(path, _set(raw, info, name, changed.reshape(field_type.shape)))
    _refused(path, info)


def test_header_extension_appended(clean):
    path, info, raw = clean
    size = _header_size(info)
    endian = info.endianness
    extension = struct.pack(endian + "ii", 32, 6) + b"Hendrickx^Marieke".ljust(24, b"\x00")
    patched = _set(raw, info, "vox_offset", size + 4 + len(extension))
    patched = patched[:size] + b"\x01\x00\x00\x00" + extension + patched[size + 4:]
    _rewrite(path, patched)
    _refused(path, info)


def test_extension_flag_set(clean):
    path, info, raw = clean
    size = _header_size(info)
    _rewrite(path, raw[:size] + b"\x01\x00\x00\x00" + raw[size + 4:])
    _refused(path, info)


def test_voxel_byte_flipped(clean):
    path, info, raw = clean
    flipped = bytearray(raw)
    flipped[-100] ^= 0x01
    _rewrite(path, bytes(flipped))
    _refused(path, info)


def test_voxel_block_short(clean):
    path, info, raw = clean
    _rewrite(path, raw[:-1])
    _refused(path, info)


def test_bytes_after_voxels(clean):
    path, info, raw = clean
    _rewrite(path, raw + b"Hendrickx")
    _refused(path, info)


def test_magic_changed(clean):
    path, info, raw = clean
    at = 344 if info.version == 1 else 4
    _rewrite(path, raw[:at] + b"ni1\x00" + raw[at + 4:])
    _refused(path, info)


@pytest.mark.parametrize("gzip_args", [
    dict(name=b"Hendrickx.nii"),
    dict(comment=b"Hendrickx"),
    dict(extra=b"Hendrickx"),
    dict(mtime=1700000000),
], ids=["file name", "comment", "extra field", "time stamp"])
def test_gzip_header_fields(clean, gzip_args):
    path, info, raw = clean
    _rewrite(path, raw, **gzip_args)
    _refused(path, info)


def test_header_crc_flag(clean):
    path, info, raw = clean
    with open(path, "rb") as fh:
        content = bytearray(fh.read())
    content[3] |= 0x02
    with open(path, "wb") as fh:
        fh.write(bytes(content))
    _refused(path, info)


def test_bytes_after_gzip_stream(clean):
    path, info, raw = clean
    with open(path, "ab") as fh:
        fh.write(b"Hendrickx^Marieke")
    _refused(path, info)


def test_empty_member_with_a_name_appended(clean):
    """A second gzip member holding no data but a file name: the gzip module
    reads straight through it, this check does not."""
    path, info, raw = clean
    with open(path, "ab") as fh:
        fh.write(h.gzip_bytes(b"", name=b"Hendrickx_Marieke.nii"))
    _refused(path, info)


def test_trailer_crc_wrong(clean):
    path, info, raw = clean
    with open(path, "rb") as fh:
        content = bytearray(fh.read())
    content[-8] ^= 0xFF
    with open(path, "wb") as fh:
        fh.write(bytes(content))
    _refused(path, info)


def test_truncated_file(clean):
    path, info, raw = clean
    with open(path, "rb") as fh:
        content = fh.read()
    with open(path, "wb") as fh:
        fh.write(content[:-20])
    _refused(path, info)


def test_not_gzip(clean):
    path, info, raw = clean
    with open(path, "wb") as fh:
        fh.write(raw)
    _refused(path, info)


def test_missing_file(clean):
    path, info, raw = clean
    os.remove(path)
    _refused(path, info)


def test_reason_never_holds_file_content(clean):
    path, info, raw = clean
    _rewrite(path, _set(raw, info, "descrip", b"Hendrickx^Marieke"))
    error = _refused(path, info)
    assert h.leaks(str(error).encode() + (error.detail or "").encode(), h.all_tokens()) == []
