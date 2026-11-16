"""NIfTI files as real tools write them, and NIfTI-2.

Real tools: concatenated gzip members; ``vox_offset`` 0, which means the voxels
follow the header; a scaling slope of 0 or NaN (no scaling), whose stored
bytes and fields stay as they are; a bit depth that disagrees with the data
type (the data type decides, as in nibabel); 2-D and 4-D files, passed on
unchanged for the server to decide.
NIfTI-2: the platform accepts NIfTI-1 only. A NIfTI-2 file is written as
NIfTI-1 when every kept field converts exactly; otherwise it is refused here
(``nifti2_unsupported``) before anything is written.
"""
from __future__ import annotations

import hashlib
import os
import struct
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize_helpers as h  # noqa: E402
from nalvera.anonymize import AnonymizeError, nifti, prepare  # noqa: E402

DATA = np.random.default_rng(15).integers(-1024, 3000, size=(8, 7, 5)).astype(np.int16)
# Every value exact in float32, and a quaternion of 0 and 1 only: a NIfTI-2
# header with this geometry converts to NIfTI-1 exactly.
EXACT = np.array([[-0.75, 0.0, 0.0, 120.5],
                  [0.0, 0.875, 0.0, -98.25],
                  [0.0, 0.0, 2.5, 33.0],
                  [0.0, 0.0, 0.0, 1.0]])
SHEAR = np.array([[0.75, 0.1, 0.0, -90.0],
                  [0.0, 0.75, 0.2, -126.0],
                  [0.05, 0.0, 1.25, -72.0],
                  [0.0, 0.0, 0.0, 1.0]])


def _write(folder, name: str, data: bytes) -> str:
    path = os.path.join(str(folder), name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _decompressed(path: str) -> bytes:
    with open(path, "rb") as fh:
        return h.read_gzip_members(fh.read())


def _fields(block: bytes, version: int = 1, endian: str = "<") -> np.void:
    dtype = nifti.HEADER_DTYPE[version].newbyteorder(endian)
    return np.frombuffer(block[:nifti.HEADER_SIZE[version]], dtype=dtype, count=1)[0]


def _refused(path: str, work, code: str) -> AnonymizeError:
    with pytest.raises(AnonymizeError) as caught:
        prepare(path, workdir=str(work))
    assert caught.value.code == code
    assert os.listdir(work) == []
    for message in (str(caught.value), repr(caught.value), caught.value.detail or ""):
        assert h.leaks(message.encode(), h.all_tokens()) == []
    return caught.value


# ---- concatenated gzip members ------------------------------------------------------

def test_concatenated_gzip_members(tmp_path):
    payload, voxels = h.nifti_bytes(DATA, qform=EXACT, slope=1.0, inter=-1024.0)
    # Boundaries inside the header, at the voxels and inside the voxels, an
    # empty member, and a name in each member's header.
    cuts = [0, 100, 352, 400, len(payload)]
    members = [h.gzip_bytes(payload[a:b], name=b"Hendrickx_part.nii") for a, b in
               zip(cuts, cuts[1:])]
    members.insert(2, h.gzip_bytes(b"", comment=b"Hendrickx"))
    original = b"".join(members)
    path = _write(tmp_path, "members.nii.gz", original)
    [scan] = prepare(path, workdir=str(tmp_path / "work"))
    with open(scan.path, "rb") as fh:
        written = fh.read()
    assert h.leaks(written, ["Hendrickx"]) == []
    raw = h.read_gzip_members(written)
    assert raw[352:] == voxels
    assert np.array_equal(np.asanyarray(nib.load(scan.path).dataobj.get_unscaled()), DATA)
    assert scan.sha256_original == hashlib.sha256(original).hexdigest()


# ---- vox_offset 0 --------------------------------------------------------------------

def _offset_zero(version: int = 1, extension: bytes = b"", after: bytes = b"") -> tuple:
    header = h.make_header(DATA.shape, DATA.dtype, version=version, qform=EXACT,
                           sform=EXACT, slope=1.0, inter=-1024.0)
    header["vox_offset"] = 0
    flag = b"\x01\x00\x00\x00" if extension else b"\x00\x00\x00\x00"
    voxels = np.asarray(DATA, dtype="<i2").tobytes(order="F")
    return header.binaryblock + flag + extension + voxels + after, voxels


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("compress", [True, False], ids=["gzip", "plain"])
def test_vox_offset_zero_means_right_after_the_header(tmp_path, version, compress):
    payload, voxels = _offset_zero(version)
    path = _write(tmp_path, "offset0.nii", h.gzip_bytes(payload) if compress else payload)
    [scan] = prepare(path, workdir=str(tmp_path / "work"))
    raw = _decompressed(scan.path)
    assert raw[352:] == voxels
    assert float(_fields(raw)["vox_offset"]) == 352.0      # written, never 0
    out = nib.load(scan.path)
    assert out.dataobj.offset == 352
    assert np.array_equal(np.asanyarray(out.dataobj.get_unscaled()), DATA)
    assert np.array_equal(out.affine, EXACT)


_EXTENSION = h.extension_bytes([(6, h.PLANTS["comment_ext"].encode())], "<")


@pytest.mark.parametrize("case, compress, before_writing", [
    ("extension", True, True),
    # The length looks right: the voxels are short by the extension's size.
    # Read from byte 352, the extension's text would become voxels.
    ("extension, voxels short by its size", True, True),
    ("extension, voxels short by its size", False, True),
    ("bytes after the voxels", True, False),
    ("bytes after the voxels", False, True),
    ("voxels cut short", False, True),
])
def test_vox_offset_zero_that_is_ambiguous_is_refused(tmp_path, monkeypatch, case, compress,
                                                      before_writing):
    """With no offset, extension bytes or a shifted block could pass for
    voxels: only a file holding exactly the header and its voxels, with no
    extension flagged, passes. What the header and the file size show is
    refused before any file is written."""
    if case.startswith("extension"):
        payload, _ = _offset_zero(extension=_EXTENSION)
        if "short" in case:
            payload = payload[:-len(_EXTENSION)]
    elif case == "bytes after the voxels":
        payload, _ = _offset_zero(after=h.PLANTS["after_voxels"].encode())
    else:
        payload, _ = _offset_zero()
        payload = payload[:-10]
    if before_writing:
        def never(path):
            raise AssertionError("an output file was created")

        monkeypatch.setattr(nifti, "_create", never)
    path = _write(tmp_path, "offset0.nii.gz", h.gzip_bytes(payload) if compress else payload)
    _refused(path, tmp_path / "work", "unreadable")


# ---- scl_slope 0 or NaN -------------------------------------------------------------

@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("slope", [0.0, float("nan")], ids=["zero", "nan"])
def test_scaling_slope_zero_or_nan_is_kept_as_stored(tmp_path, slope, version):
    payload, voxels = h.nifti_bytes(DATA, version=version, qform=EXACT, sform=EXACT,
                                    slope=slope, inter=5.0)
    path = _write(tmp_path, "slope.nii.gz", h.gzip_bytes(payload))
    [scan] = prepare(path, workdir=str(tmp_path / "work"))
    raw = _decompressed(scan.path)
    assert raw[352:] == voxels
    before, after = _fields(payload, version), _fields(raw, 1)
    for name in ("scl_slope", "scl_inter"):
        assert nifti.canonical(before[name]) == nifti.canonical(after[name]), name
    if version == 1:
        assert raw[112:120] == payload[112:120]             # scl_slope, scl_inter as stored
    a, b = nib.load(path), nib.load(scan.path)
    assert (a.dataobj.slope, a.dataobj.inter) == (b.dataobj.slope, b.dataobj.inter) == (1.0, 0.0)
    assert np.asanyarray(a.dataobj).tobytes() == np.asanyarray(b.dataobj).tobytes()


# ---- a bit depth that disagrees with the data type ---------------------------------

@pytest.mark.parametrize("bitpix", [0, 8, 32])
def test_bit_depth_disagreeing_with_the_data_type(tmp_path, bitpix):
    header = h.make_header(DATA.shape, DATA.dtype, qform=EXACT)       # int16
    _, voxels = h.nifti_bytes(DATA, header=header)
    header["bitpix"] = bitpix
    payload = header.binaryblock + b"\x00\x00\x00\x00" + voxels
    path = _write(tmp_path, "bitpix.nii", payload)
    [scan] = prepare(path, workdir=str(tmp_path / "work"))
    raw = _decompressed(scan.path)
    assert raw[352:] == voxels                    # the data type sets the voxel size
    assert int(_fields(raw)["bitpix"]) == bitpix   # the field is kept as stored
    assert scan.shape == DATA.shape
    assert np.array_equal(np.asanyarray(nib.load(scan.path).dataobj.get_unscaled()), DATA)


# ---- 2-D and 4-D --------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(32, 30), (7, 6, 5, 3), (7, 6, 5, 1), (4, 4, 3, 2, 2)])
def test_2d_and_4d_pass_unchanged(tmp_path, shape):
    data = np.random.default_rng(4).integers(0, 4000, size=shape).astype(np.int16)
    payload, voxels = h.nifti_bytes(data)
    path = _write(tmp_path, "nd.nii.gz", h.gzip_bytes(payload))
    [scan] = prepare(path, workdir=str(tmp_path / "work"))
    assert scan.shape == shape
    assert _decompressed(scan.path)[352:] == voxels
    assert nib.load(scan.path).shape == shape


# ---- NIfTI-2 ------------------------------------------------------------------------

@pytest.mark.parametrize("endian", ["<", ">"])
def test_nifti2_that_fits_becomes_nifti1(tmp_path, endian):
    payload, voxels = h.nifti_bytes(DATA, version=2, endian=endian, qform=EXACT, sform=EXACT,
                                    slope=1.0, inter=-1024.0)
    path = _write(tmp_path, "v2.nii.gz", h.gzip_bytes(payload))
    [scan] = prepare(path, workdir=str(tmp_path / "work"))
    raw = _decompressed(scan.path)
    assert raw[:4] == struct.pack(endian + "i", 348)
    assert raw[344:348] == b"n+1\x00"
    assert raw[352:] == voxels
    a, b = nib.load(path), nib.load(scan.path)
    assert a.affine.tobytes() == b.affine.tobytes()
    assert np.asanyarray(a.dataobj).tobytes() == np.asanyarray(b.dataobj).tobytes()


INEXACT = {
    "slope": dict(slope=0.7, inter=0.0),
    "intercept": dict(slope=1.0, inter=-1024.1),
    "voxel size": dict(zooms=(0.7, 0.7, 2.0)),
    "sform with shear": dict(sform=SHEAR),
    "rotated qform": dict(qform=np.array([[0.0, -0.8, 0.0, 120.5], [0.9, 0.0, 0.0, -98.25],
                                          [0.0, 0.0, 2.5, 33.0], [0.0, 0.0, 0.0, 1.0]])),
}


@pytest.mark.parametrize("case", sorted(INEXACT) + ["dimension above 32767"])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_nifti2_that_cannot_be_nifti1_is_refused_before_writing(tmp_path, monkeypatch, case,
                                                                endian):
    if case == "dimension above 32767":
        data = np.arange(40000, dtype=np.int8).reshape(40000, 1, 1)
        payload, _ = h.nifti_bytes(data, version=2, endian=endian)
    else:
        payload, _ = h.nifti_bytes(DATA, version=2, endian=endian, **INEXACT[case])

    def never(path):
        raise AssertionError("an output file was created")

    monkeypatch.setattr(nifti, "_create", never)
    path = _write(tmp_path, h.PLANTS["file_name"] + ".nii.gz", h.gzip_bytes(payload))
    _refused(path, tmp_path / "work", "nifti2_unsupported")


def test_nifti2_refusal_names_nothing_from_the_file(tmp_path):
    header = h.make_header(DATA.shape, DATA.dtype, version=2, sform=SHEAR,
                           fields={"descrip": h.PLANTS["descrip"].encode(),
                                   "intent_name": h.PLANTS["intent_name"].encode()})
    payload, _ = h.nifti_bytes(DATA, version=2, header=header,
                               extensions=[(6, h.PLANTS["comment_ext"].encode())])
    path = _write(tmp_path, h.PLANTS["file_name"] + ".nii.gz",
                  h.gzip_bytes(payload, name=h.PLANTS["gzip_name"].encode()))
    _refused(path, tmp_path / "work", "nifti2_unsupported")
