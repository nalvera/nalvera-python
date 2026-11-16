"""The NIfTI path: same voxels and geometry, nothing identifying, hostile input.

Each input is assembled byte by byte (anonymize_helpers), run through
``prepare``, and compared with nibabel's reading of the input and with the
input's own voxel bytes.
"""
from __future__ import annotations

import gzip
import os
import sys
import time
import tracemalloc

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize_helpers as h  # noqa: E402
from nalvera.anonymize import AnonymizeError, nifti, prepare  # noqa: E402

RNG = np.random.default_rng(20261115)
ORTHO = np.array([[0.0, -0.8, 0.0, 120.5],
                  [0.9, 0.0, 0.0, -98.25],
                  [0.0, 0.0, 2.5, 33.0],
                  [0.0, 0.0, 0.0, 1.0]])
SHEAR = np.array([[0.75, 0.1, 0.0, -90.0],
                  [0.0, 0.75, 0.2, -126.0],
                  [0.05, 0.0, 1.25, -72.0],
                  [0.0, 0.0, 0.0, 1.0]])
# A rotated geometry whose values are all exact in float32: a NIfTI-2 sform
# with it converts to NIfTI-1 exactly.
ORTHO_EXACT = np.array([[0.0, -0.75, 0.0, 120.5],
                        [0.875, 0.0, 0.0, -98.25],
                        [0.0, 0.0, 2.5, 33.0],
                        [0.0, 0.0, 0.0, 1.0]])


def _write(folder, name, data: bytes) -> str:
    path = os.path.join(str(folder), name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _decompressed(path: str) -> bytes:
    with open(path, "rb") as fh:
        return h.read_gzip_members(fh.read())


def assert_same_scan(original: str, scan, voxels: bytes) -> None:
    """The anonymized file holds the input's voxels and geometry exactly."""
    a, b = nib.load(original), nib.load(scan.path)
    # The stored voxel block, from the files themselves.
    raw = _decompressed(scan.path)
    offset = 352 if raw[:4] in (b"\x5c\x01\x00\x00", b"\x00\x00\x01\x5c") else 544
    assert raw[offset:] == voxels
    assert raw[offset - 4:offset] == b"\x00\x00\x00\x00"
    # The stored array as nibabel reads it: same type, byte order and bytes.
    stored_a = np.asanyarray(a.dataobj.get_unscaled())
    stored_b = np.asanyarray(b.dataobj.get_unscaled())
    assert stored_a.dtype == stored_b.dtype
    assert stored_a.tobytes() == stored_b.tobytes()
    # The scaled values.
    scaled_a, scaled_b = np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)
    assert scaled_a.dtype == scaled_b.dtype
    assert scaled_a.tobytes() == scaled_b.tobytes()
    assert a.get_data_dtype() == b.get_data_dtype()
    assert np.array_equal(a.affine, b.affine)
    assert a.affine.tobytes() == b.affine.tobytes()
    assert a.header.get_zooms() == b.header.get_zooms()
    assert a.shape == b.shape == scan.shape
    assert np.array_equal(a.header.get_qform(), b.header.get_qform())
    assert np.array_equal(a.header.get_sform(), b.header.get_sform())
    for getter in ("get_qform", "get_sform"):
        (xa, ca), (xb, cb) = (getattr(x.header, getter)(coded=True) for x in (a, b))
        assert ca == cb
        assert (xa is None) == (xb is None) and (xa is None or np.array_equal(xa, xb))
    for attribute in ("slope", "inter"):
        va, vb = getattr(a.dataobj, attribute), getattr(b.dataobj, attribute)
        assert np.asarray(va).tobytes() == np.asarray(vb).tobytes()
    for name in ("pixdim", "quatern_b", "quatern_c", "quatern_d", "qoffset_x", "qoffset_y",
                 "qoffset_z", "srow_x", "srow_y", "srow_z", "scl_slope", "scl_inter",
                 "xyzt_units", "qform_code", "sform_code", "dim"):
        assert nifti.canonical(a.header[name]) == nifti.canonical(b.header[name]), name


def _ct_data(shape=(20, 18, 7)):
    return RNG.integers(0, 4096, size=shape).astype(np.int16)


def _float_data(shape=(9, 8, 7)):
    data = RNG.normal(0, 1000, size=shape).astype(np.float32)
    flat = data.reshape(-1)
    flat[:6] = [np.nan, np.inf, -np.inf, -0.0, 1e-42, np.finfo(np.float32).max]
    return data


CASES = {
    # name: (data, header arguments, compress)
    "int16 CT, slope 1, intercept -1024": (
        _ct_data(), dict(qform=ORTHO, sform=ORTHO, slope=1.0, inter=-1024.0), True),
    "uint16 with a non-trivial slope": (
        RNG.integers(0, 65535, size=(16, 15, 6)).astype(np.uint16),
        dict(sform=ORTHO, slope=0.3701, inter=12.5), True),
    "float32 with NaN, inf, -0 and denormals": (_float_data(), dict(sform=ORTHO), True),
    "qform and sform differ": (
        _ct_data(), dict(qform=ORTHO, qcode=1, sform=SHEAR, scode=2), True),
    "affine with shear": (
        _ct_data(), dict(sform=SHEAR, scode=4, qform=np.diag([0.75, 0.75, 1.25, 1.0]),
                         qcode=0), True),
    "4D volume": (
        RNG.integers(-500, 500, size=(7, 6, 5, 3)).astype(np.int16),
        dict(zooms=(1.5, 1.5, 3.0, 2.2), qform=np.diag([1.5, 1.5, 3.0, 1.0])), True),
    "NIfTI-2 that fits NIfTI-1": (
        _ct_data(), dict(version=2, qform=np.diag([0.5, 0.5, 2.0, 1.0]),
                         sform=np.diag([0.5, 0.5, 2.0, 1.0]), slope=1.0, inter=-1024.0), True),
    "big-endian": (
        _ct_data(), dict(endian=">", qform=ORTHO, sform=ORTHO, slope=1.0, inter=-1024.0), True),
    "big-endian NIfTI-2": (
        _float_data(), dict(version=2, endian=">", sform=ORTHO_EXACT), True),
    "uncompressed .nii": (_ct_data(), dict(qform=ORTHO, sform=ORTHO), False),
    "uint8": (RNG.integers(0, 255, size=(8, 8, 8)).astype(np.uint8), dict(), True),
    "int8": (RNG.integers(-128, 127, size=(8, 8, 8)).astype(np.int8), dict(), True),
    "int32": (RNG.integers(-2**31, 2**31 - 1, size=(8, 8, 8)).astype(np.int32), dict(), True),
    "uint32": (RNG.integers(0, 2**32 - 1, size=(8, 8, 8)).astype(np.uint32), dict(), True),
    "int64": (RNG.integers(-2**62, 2**62, size=(8, 8, 8)).astype(np.int64), dict(), True),
    "float64": (RNG.normal(size=(8, 8, 8)), dict(sform=SHEAR), True),
    "complex64": ((RNG.normal(size=(6, 6, 6)) + 1j * RNG.normal(size=(6, 6, 6)))
                  .astype(np.complex64), dict(), True),
    "2D image": (RNG.integers(0, 4096, size=(32, 30)).astype(np.int16), dict(), True),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_bit_identical(tmp_path, case):
    data, header_args, compress = CASES[case]
    payload, voxels = h.nifti_bytes(data, **header_args)
    name = "input.nii.gz" if compress else "input.nii"
    original = _write(tmp_path, name, h.gzip_bytes(payload) if compress else payload)
    [scan] = prepare(original, workdir=str(tmp_path / "work"))
    assert_same_scan(original, scan, voxels)


def test_nifti2_becomes_nifti1_only_when_exact(tmp_path):
    data, args, _ = CASES["NIfTI-2 that fits NIfTI-1"]
    payload, _ = h.nifti_bytes(data, **args)
    [scan] = prepare(_write(tmp_path, "exact.nii.gz", h.gzip_bytes(payload)),
                     workdir=str(tmp_path / "work"))
    assert nib.load(scan.path).header["sizeof_hdr"] == 348
    # One that needs float64 is refused: the platform accepts NIfTI-1 only.
    payload, _ = h.nifti_bytes(_ct_data(), version=2, qform=ORTHO, sform=SHEAR, slope=0.7,
                               inter=-1024.1)
    with pytest.raises(AnonymizeError) as caught:
        prepare(_write(tmp_path, "inexact.nii.gz", h.gzip_bytes(payload)),
                workdir=str(tmp_path / "work2"))
    assert caught.value.code == "nifti2_unsupported"
    assert os.listdir(tmp_path / "work2") == []


def test_misleading_names(tmp_path):
    """gzip is found by content: a gzip file without .gz, a plain file with .gz."""
    data, args, _ = CASES["int16 CT, slope 1, intercept -1024"]
    payload, voxels = h.nifti_bytes(data, **args)
    zipped = _write(tmp_path, "scan.nii", h.gzip_bytes(payload))
    plain = _write(tmp_path, "scan.nii.gz", payload)
    other = _write(tmp_path, "scan.dat", h.gzip_bytes(payload))
    for original in (zipped, plain, other):
        [scan] = prepare(original, workdir=str(tmp_path / "work"))
        raw = _decompressed(scan.path)
        assert raw[352:] == voxels
        assert np.array_equal(np.asanyarray(nib.load(scan.path).dataobj.get_unscaled()), data)


# ---- nothing identifying -----------------------------------------------------

def _planted(version: int, endian: str, *, compress: bool = True,
             args=None) -> tuple:
    """(file bytes, voxels, file name) of a scan with a name in every place."""
    data = _ct_data((12, 11, 5))
    p = h.PLANTS
    text = {
        "descrip": (p["descrip"] + " " + p["descrip_utf8"]).encode("utf-8"),
        "aux_file": p["aux_file"].encode(),
        "intent_name": p["intent_name"].encode(),
    }
    if version == 1:
        text["db_name"] = p["db_name"].encode()
        text["data_type"] = p["data_type"].encode()
    else:
        text["unused_str"] = p["unused_str"].encode()
    header_args = dict(qform=ORTHO, sform=ORTHO, slope=1.0, inter=-1024.0)
    header_args.update(args or {})
    header = h.make_header(data.shape, data.dtype, version=version, endian=endian,
                           fields=text, **header_args)
    extensions = [
        (6, p["comment_ext"].encode()),
        (2, h.dicom_extension(p["dicom_ext"])),
        (4, p["utf16_ext"].encode("utf-16-le")),
    ]
    after = p["after_voxels"].encode() + p["after_voxels"].encode("utf-16-be")
    payload, voxels = h.nifti_bytes(data, version=version, endian=endian,
                                    extensions=extensions, after=after, header=header)
    if compress:
        payload = h.gzip_bytes(payload, name=p["gzip_name"].encode(),
                               comment=p["gzip_comment"].encode(),
                               extra=p["gzip_extra"].encode())
    name = p["file_name"] + (".nii.gz" if compress else ".nii")
    return payload, voxels, name


EXACT = np.diag([0.5, 0.5, 2.0, 1.0])


@pytest.mark.parametrize("version, endian, compress, args, written", [
    (1, "<", True, None, 1),
    (1, ">", True, None, 1),
    (1, "<", False, None, 1),
    (2, "<", True, dict(qform=EXACT, sform=EXACT), 1),      # fits NIfTI-1: converted
    (2, ">", True, dict(qform=EXACT, sform=ORTHO_EXACT), 1),
])
def test_nothing_identifying(tmp_path, version, endian, compress, args, written):
    payload, voxels, name = _planted(version, endian, compress=compress, args=args)
    original = _write(tmp_path, name, payload)
    # The test only means something if every plant is in the input.
    input_bytes = payload + (h.read_gzip_members(payload) if compress else b"") + name.encode()
    expected_plants = [k for k in h.PLANTS
                       if not (version == 1 and k == "unused_str")
                       and not (version == 2 and k in ("db_name", "data_type"))
                       and not (not compress and k.startswith("gzip_"))]
    for key in expected_plants:
        assert h.leaks(input_bytes, [h.PLANTS[key]]), key

    [scan] = prepare(original, workdir=str(tmp_path / "work"))
    with open(scan.path, "rb") as fh:
        compressed = fh.read()
    decompressed = h.read_gzip_members(compressed)
    assert h.leaks(compressed, h.all_tokens()) == []
    assert h.leaks(decompressed, h.all_tokens()) == []
    for public in (scan.path, scan.filename, scan.code):
        assert h.leaks(os.path.basename(public).encode(), h.all_tokens()) == []
    assert set(os.listdir(tmp_path / "work")) == {scan.filename}
    assert nib.load(scan.path).header["sizeof_hdr"] == (348 if written == 1 else 540)
    # The voxels are still the input's, and are the file's last bytes.
    assert decompressed.endswith(voxels)
    assert len(decompressed) == (352 if written == 1 else 544) + len(voxels)


def test_unlisted_numeric_fields_are_zeroed(tmp_path):
    data = _ct_data((6, 6, 6))
    header = h.make_header(data.shape, data.dtype, qform=ORTHO, fields={
        "cal_max": 3000.0, "cal_min": -1000.0, "intent_p1": 7.0, "intent_p2": 8.0,
        "intent_p3": 9.0, "glmax": 1234, "glmin": -55, "extents": 16384,
        "session_error": 3, "regular": b"r"})
    payload, _ = h.nifti_bytes(data, header=header)
    original = _write(tmp_path, "in.nii.gz", h.gzip_bytes(payload))
    [scan] = prepare(original, workdir=str(tmp_path / "work"))
    out = nib.load(scan.path).header
    for name in ("cal_max", "cal_min", "intent_p1", "intent_p2", "intent_p3", "glmax",
                 "glmin", "extents", "session_error"):
        assert out[name] == 0, name
    assert out["regular"] == b""
    for name in ("descrip", "aux_file", "db_name", "intent_name", "data_type"):
        assert out[name] == b"", name


def test_gzip_header_is_bare(tmp_path):
    payload, _, name = _planted(1, "<")
    [scan] = prepare(_write(tmp_path, name, payload), workdir=str(tmp_path / "work"))
    with open(scan.path, "rb") as fh:
        head = fh.read(10)
    assert head[:3] == b"\x1f\x8b\x08"
    assert head[3] == 0                         # no name, comment, extra or header CRC
    assert head[4:8] == b"\x00\x00\x00\x00"     # zero time stamp


def test_output_is_reproducible(tmp_path):
    payload, _, name = _planted(1, "<")
    original = _write(tmp_path, name, payload)
    first = prepare(original, workdir=str(tmp_path / "a"))[0]
    second = prepare(original, workdir=str(tmp_path / "b"))[0]
    assert first.code != second.code
    assert first.sha256 == second.sha256


# ---- hostile input -------------------------------------------------------------

def _header_bytes(version=1, endian="<", shape=(4, 4, 4), dtype=np.int16, **fields):
    header = h.make_header(shape, dtype, version=version, endian=endian,
                           fields={"descrip": h.PLANTS["descrip"].encode()})
    for name, value in fields.items():
        header[name] = value
    return header


def _hostile_cases():
    data = _ct_data((4, 4, 4))
    good, voxels = h.nifti_bytes(data)
    cases = {}

    payload, _ = h.nifti_bytes(_ct_data((64, 64, 64)))
    cases["more voxels than the file holds (gzip)"] = (
        h.gzip_bytes(payload[:352 + 1000]), "unreadable")
    cases["more voxels than the file holds (plain)"] = (payload[:352 + 1000], "unreadable")

    header = _header_bytes()
    header["dim"] = [3, 32767, 32767, 32767, 1, 1, 1, 1]
    cases["absurd dimensions"] = (h.gzip_bytes(header.binaryblock + b"\x00" * 4), "too_large")

    header = _header_bytes(version=2)
    header["dim"] = [3, 2 ** 40, 2 ** 40, 7, 1, 1, 1, 1]
    cases["absurd NIfTI-2 dimensions"] = (h.gzip_bytes(header.binaryblock + b"\x00" * 4),
                                          "too_large")

    for label, dim in (("negative dimension", [3, 4, -4, 4, 1, 1, 1, 1]),
                       ("zero dimensions", [0, 4, 4, 4, 1, 1, 1, 1]),
                       ("eight dimensions", [8, 4, 4, 4, 1, 1, 1, 1])):
        header = _header_bytes()
        header["dim"] = dim
        cases[label] = (h.gzip_bytes(header.binaryblock + b"\x00" * 4 + voxels), "unreadable")

    # An offset of 0 means "right after the header": test_anonymize_oddities.
    for label, offset in (("voxel offset inside the header", 100),
                          ("voxel offset not a whole number", 352.5),
                          ("voxel offset NaN", float("nan"))):
        header = _header_bytes()
        block = h.nifti_bytes(data, header=header)[0]
        header["vox_offset"] = offset
        cases[label] = (h.gzip_bytes(header.binaryblock + block[348:]), "unreadable")

    header = _header_bytes()
    header["vox_offset"] = 1e9
    cases["voxel offset far past the end (plain)"] = (header.binaryblock + b"\x00" * 4,
                                                      "unreadable")
    header = _header_bytes()
    header["vox_offset"] = 100 * 1024 * 1024
    cases["voxel offset far past the end (gzip)"] = (
        h.gzip_bytes(header.binaryblock + b"\x00" * 4 + voxels), "unreadable")

    zipped = h.gzip_bytes(good)
    cases["truncated gzip stream"] = (zipped[:len(zipped) // 2], "unreadable")
    damaged = bytearray(zipped)
    damaged[10:40] = b"\xff" * 30          # an invalid deflate block type
    cases["damaged deflate data"] = (bytes(damaged), "unreadable")
    cases["zero-length file"] = (b"", "not_a_scan")
    cases["text file named .nii.gz"] = (b"Patient: Hendrickx^Marieke\n" * 40, "not_a_scan")
    cases["gzip of a text file"] = (h.gzip_bytes(b"Hendrickx^Marieke\n" * 40), "not_a_scan")
    cases["header cut short"] = (good[:200], "unreadable")

    header = _header_bytes()
    header["magic"] = b"ni1"
    cases["NIfTI pair header (ni1)"] = (header.binaryblock, "not_a_scan")
    header = _header_bytes()
    header["magic"] = b""
    cases["Analyze header"] = (header.binaryblock + voxels, "not_a_scan")

    # A bit depth that disagrees with a known data type is read by the data
    # type, as nibabel does: test_anonymize_oddities.
    header = _header_bytes()
    header["datatype"] = 999
    cases["unknown data type"] = (h.gzip_bytes(h.nifti_bytes(data, header=header)[0]),
                                  "unreadable")

    header = _header_bytes(shape=(512, 512, 512))
    cases["gzip bomb header with no voxels"] = (h.gzip_bytes(header.binaryblock + b"\x00" * 4),
                                                "unreadable")
    return cases


HOSTILE = _hostile_cases()


@pytest.mark.parametrize("case", sorted(HOSTILE))
def test_hostile_input(tmp_path, case):
    payload, code = HOSTILE[case]
    original = _write(tmp_path, h.PLANTS["file_name"] + ".nii.gz", payload)
    work = tmp_path / "work"
    tracemalloc.start()
    started = time.monotonic()
    try:
        with pytest.raises(AnonymizeError) as caught:
            prepare(original, workdir=str(work))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert time.monotonic() - started < 10
    assert peak < 32 * 1024 * 1024
    assert caught.value.code == code
    assert os.listdir(work) == []
    # The error never repeats what the file holds.
    for message in (str(caught.value), repr(caught.value), caught.value.detail or ""):
        assert h.leaks(message.encode(), h.all_tokens()) == []


def test_missing_file(tmp_path):
    with pytest.raises(AnonymizeError) as caught:
        nifti.write_clean(str(tmp_path / "absent.nii.gz"), str(tmp_path / "out.nii.gz"))
    assert caught.value.code == "not_found"


def test_existing_target_is_never_overwritten_or_removed(tmp_path):
    payload, _ = h.nifti_bytes(_ct_data((4, 4, 4)))
    original = _write(tmp_path, "in.nii", payload)
    target = _write(tmp_path, "taken.nii.gz", b"someone else's file")
    with pytest.raises(FileExistsError):
        nifti.write_clean(original, target)
    with open(target, "rb") as fh:
        assert fh.read() == b"someone else's file"


def test_memory_stays_flat(tmp_path):
    """A 64 MiB volume passes through with a peak of a few MiB."""
    data = np.tile(np.arange(4096, dtype=np.int16), 256 * 32).reshape(256, 256, 512)
    payload, _ = h.nifti_bytes(data, slope=1.0, inter=-1024.0)
    original = _write(tmp_path, "big.nii.gz", gzip.compress(payload, compresslevel=1))
    del payload, data
    tracemalloc.start()
    try:
        [scan] = prepare(original, workdir=str(tmp_path / "work"))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert scan.shape == (256, 256, 512)
    assert peak < 16 * 1024 * 1024, peak
