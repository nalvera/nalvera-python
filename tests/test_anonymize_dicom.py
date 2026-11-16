"""DICOM in (nalvera.anonymize.dicom, and prepare on DICOM).

Synthetic CT series are written with pydicom, with patient details planted
in every usual attribute, and converted with the real dcm2niix (the binary
the ``dcm2niix`` Python package installs). A stand-in converter (a small
script, chosen with NALVERA_DCM2NIIX) tests what happens around the
conversion: the command line, the private folder, several volumes, failure.
"""
from __future__ import annotations

import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize_helpers as h  # noqa: E402
import nalvera.anonymize as anonymize  # noqa: E402
from nalvera.anonymize import AnonymizeError, dicom, prepare  # noqa: E402
from nalvera.anonymize.nifti import write_clean  # noqa: E402

needs_dcm2niix = pytest.mark.skipif(
    os.environ.get("NALVERA_DCM2NIIX") is None and dicom.find_dcm2niix() is None,
    reason="dcm2niix is not installed (pip install dcm2niix)")

@pytest.fixture
def standin(tmp_path, monkeypatch):
    """Use the stand-in converter; returns a function reading its log."""
    return h.install_standin(tmp_path, monkeypatch)


@pytest.fixture
def series_folder(tmp_path):
    """A folder named after the patient holding one planted CT series."""
    folder = tmp_path / "Hendrickx_Marieke_CT"
    folder.mkdir()
    pixels = h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)
    return folder, pixels


def _snapshot(folder) -> list:
    found = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            stat = os.stat(path)
            found.append((os.path.relpath(path, folder), stat.st_size, stat.st_mtime_ns))
    return sorted(found)


def _assert_clean(scan) -> None:
    with open(scan.path, "rb") as fh:
        compressed = fh.read()
    for data in (compressed, h.read_gzip_members(compressed)):
        assert h.leaks(data, h.DICOM_TOKENS) == []
    assert h.leaks(scan.filename.encode(), h.DICOM_TOKENS) == []
    header = nib.load(scan.path).header
    for name in ("descrip", "aux_file", "db_name", "intent_name"):
        assert header[name] == b"", name
    assert len(header.extensions) == 0


# ---- finding the converter ---------------------------------------------------------

def test_finds_the_package_binary(monkeypatch):
    package = pytest.importorskip("dcm2niix")
    monkeypatch.delenv("NALVERA_DCM2NIIX", raising=False)
    found = dicom.find_dcm2niix()
    assert found == str(package.bin)
    assert os.access(found, os.X_OK)
    # It is the real converter.
    done = subprocess.run([found, "-h"], capture_output=True, timeout=60)
    assert b"dcm2niiX version" in done.stdout


def test_falls_back_to_path(tmp_path, monkeypatch):
    folder = tmp_path / "bin"
    folder.mkdir()
    tool = folder / "dcm2niix"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    monkeypatch.delenv("NALVERA_DCM2NIIX", raising=False)
    monkeypatch.setitem(sys.modules, "dcm2niix", None)       # the package is absent
    monkeypatch.setenv("PATH", str(folder))
    assert dicom.find_dcm2niix() == str(tool)


def test_override_must_be_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("NALVERA_DCM2NIIX", str(tmp_path / "absent"))
    assert dicom.find_dcm2niix() is None


def test_tool_missing(series_folder, tmp_path, monkeypatch):
    folder, _ = series_folder
    monkeypatch.delenv("NALVERA_DCM2NIIX", raising=False)
    monkeypatch.setitem(sys.modules, "dcm2niix", None)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(folder), workdir=str(work))
    assert caught.value.code == "dicom_tool_missing"
    assert os.listdir(work) == []


# ---- real conversions --------------------------------------------------------------

@needs_dcm2niix
def test_single_series(series_folder, tmp_path):
    folder, pixels = series_folder
    before = _snapshot(folder)
    work = tmp_path / "work"
    [scan] = prepare(str(folder), workdir=str(work), label="P012 baseline")
    assert scan.source == "dicom"
    assert scan.series == "3"
    assert scan.original_name == "Hendrickx_Marieke_CT"
    assert scan.original_path == str(folder)
    assert scan.label == "P012 baseline"
    assert scan.sha256_original is None
    assert scan.shape == (16, 16, 4)
    _assert_clean(scan)
    # Every stored value is the DICOM's (the converter may reorder voxels).
    stored = np.asanyarray(nib.load(scan.path).dataobj.get_unscaled())
    assert np.array_equal(np.sort(stored.ravel()), np.sort(pixels.ravel()))
    image = nib.load(scan.path)
    assert image.dataobj.slope == 1.0 and image.dataobj.inter == -1024.0
    # The DICOM folder is untouched; the work folder holds the scan only.
    assert _snapshot(folder) == before
    assert os.listdir(work) == [scan.filename]


@needs_dcm2niix
def test_converter_output_is_bit_identical_after_cleaning(series_folder, tmp_path):
    """The converter itself writes identifying text (so the cleaning
    matters), and the cleaning keeps its voxels and geometry exactly."""
    folder, _ = series_folder
    private = tmp_path / "private"
    private.mkdir()
    [volume] = dicom.convert(dicom.list_dicom_files(str(folder)), str(private))
    raw = nib.load(volume.path)
    assert raw.header["descrip"] != b""
    assert b"Hendrickx" in bytes(raw.header["aux_file"])
    target = str(tmp_path / "clean.nii.gz")
    write_clean(volume.path, target)
    clean = nib.load(target)
    assert np.asanyarray(raw.dataobj.get_unscaled()).tobytes() == \
        np.asanyarray(clean.dataobj.get_unscaled()).tobytes()
    assert np.array_equal(raw.affine, clean.affine)
    assert raw.header.get_zooms() == clean.header.get_zooms()
    assert clean.header["descrip"] == b"" and clean.header["aux_file"] == b""


@needs_dcm2niix
def test_two_series_in_one_folder(tmp_path):
    folder = tmp_path / "Hendrickx_study"
    (folder / "second").mkdir(parents=True)
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)
    h.write_dicom_series(str(folder / "second"), series_number=7, plants=h.DICOM_PLANTS, seed=9)
    work = tmp_path / "work"
    scans = prepare(str(folder), workdir=str(work))
    assert [s.series for s in scans] == ["3", "7"]
    assert len({s.code for s in scans}) == 2
    assert {s.original_name for s in scans} == {"Hendrickx_study"}
    for scan in scans:
        _assert_clean(scan)
    assert sorted(os.listdir(work)) == sorted(s.filename for s in scans)


@needs_dcm2niix
def test_single_slice_file_is_left_out(series_folder, tmp_path):
    """One single-frame DICOM file is a 2-D image, which the converter
    leaves out (-i y): no scan."""
    folder, _ = series_folder
    first = sorted(os.listdir(folder))[0]
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(folder / first), workdir=str(work))
    assert caught.value.code == "no_volume"
    assert caught.value.left_out == 1
    assert os.listdir(work) == []


def test_single_dicom_file(series_folder, tmp_path, standin):
    """A single DICOM file (a multi-frame volume) goes to the converter on
    its own and is named after the file."""
    folder, _ = series_folder
    first = sorted(os.listdir(folder))[0]
    [scan] = prepare(str(folder / first), workdir=str(tmp_path / "work"))
    assert scan.source == "dicom"
    assert scan.original_name == first
    assert [e["target"] for e in standin()["entries"]] == [os.path.realpath(folder / first)]
    _assert_clean(scan)


def test_folder_without_dicom(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "patient.txt").write_text("Hendrickx^Marieke")
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(folder), workdir=str(tmp_path / "work"))
    assert caught.value.code == "not_a_scan"


# ---- around the conversion (stand-in converter) ------------------------------------

def test_command_line_and_private_folder(series_folder, tmp_path, standin):
    folder, _ = series_folder
    (folder / "notes.txt").write_text("Patient Hendrickx^Marieke")
    (folder / ".DS_Store").write_bytes(b"\x00" * 64)
    work = tmp_path / "work"
    [scan] = prepare(str(folder), workdir=str(work))
    log = standin()
    args = log["args"]
    for flag, value in (("-g", "i"), ("-b", "n"), ("-z", "i"), ("-i", "y"), ("-f", "vol_s%s"),
                        ("-w", "2")):
        assert args[args.index(flag) + 1] == value
    out = args[args.index("-o") + 1]
    # The private folder is in the system's temporary folder, never in workdir.
    private = os.path.dirname(out)
    import tempfile
    assert os.path.basename(private).startswith("nalvera-dicom-")
    assert os.path.dirname(private) == tempfile.gettempdir()
    assert not os.path.exists(private)
    assert all(mode == "0o700" for mode in log["modes"].values())
    # The converter sees neutral links to the DICOM files only.
    originals = sorted(os.path.realpath(folder / n) for n in os.listdir(folder)
                       if n.startswith("IM"))
    assert sorted(e["target"] for e in log["entries"]) == originals
    assert all(e["link"] for e in log["entries"])
    assert all(e["name"].endswith(".dcm") and e["name"][:6].isdigit() for e in log["entries"])
    # The private folder is gone; only the scan remains.
    assert os.listdir(work) == [scan.filename]
    _assert_clean(scan)


def test_several_volumes_from_one_series(series_folder, tmp_path, standin, monkeypatch):
    folder, _ = series_folder
    monkeypatch.setenv("STANDIN_NAMES", "vol_s12.nii,vol_s5_e2.nii.gz,vol_s5.nii.gz")
    scans = prepare(str(folder), workdir=str(tmp_path / "work"))
    assert [s.series for s in scans] == ["5", "5_e2", "12"]
    for scan in scans:
        _assert_clean(scan)
        assert scan.shape == (6, 5, 4)
    assert len({s.sha256 for s in scans}) == 3


def test_conversion_failed_keeps_converter_output_out(series_folder, tmp_path, standin,
                                                      monkeypatch):
    folder, _ = series_folder
    monkeypatch.setenv("STANDIN_NAMES", "")
    monkeypatch.setenv("STANDIN_EXIT", "4")
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(folder), workdir=str(work))
    assert caught.value.code == "conversion_failed"
    for message in (str(caught.value), repr(caught.value), caught.value.detail or ""):
        assert h.leaks(message.encode(), h.DICOM_TOKENS) == []
    assert os.listdir(work) == []


def test_failure_after_conversion_cleans_up(series_folder, tmp_path, standin, monkeypatch):
    folder, _ = series_folder
    monkeypatch.setenv("STANDIN_NAMES", "vol_s5.nii.gz,vol_s6.nii.gz")
    real = anonymize.verify_clean
    calls = []

    def second_fails(path, info):
        calls.append(path)
        if len(calls) == 2:
            os.remove(path)
            raise AnonymizeError("verification_failed", "test")
        return real(path, info)

    monkeypatch.setattr(anonymize, "verify_clean", second_fails)
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(folder), workdir=str(work))
    assert caught.value.code == "verification_failed"
    assert len(calls) == 2
    assert os.listdir(work) == []


def test_interrupt_cleans_up(series_folder, tmp_path, standin, monkeypatch):
    folder, _ = series_folder

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(anonymize, "write_clean", interrupted)
    work = tmp_path / "work"
    with pytest.raises(KeyboardInterrupt):
        prepare(str(folder), workdir=str(work))
    assert os.listdir(work) == []


def test_timeout(series_folder, tmp_path, standin, monkeypatch):
    folder, _ = series_folder

    def too_slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(dicom.subprocess, "run", too_slow)
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(folder), workdir=str(work))
    assert caught.value.code == "conversion_failed"
    assert os.listdir(work) == []


# ---- recognising DICOM files -------------------------------------------------------

def test_is_dicom_file(series_folder, tmp_path):
    from pydicom.dataset import Dataset
    from pydicom.uid import CTImageStorage, generate_uid

    folder, _ = series_folder
    part10 = folder / sorted(os.listdir(folder))[0]
    bare = []
    for implicit in (True, False):
        path = tmp_path / f"bare-{implicit}"
        ds = Dataset()
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.PatientName = "Hendrickx^Marieke"
        h.save_bare(ds, str(path), implicit=implicit)
        assert path.read_bytes()[:2] == b"\x08\x00"
        bare.append(path)
    no_sop = tmp_path / "no-sop"
    ds = Dataset()
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.PatientName = "Hendrickx^Marieke"
    h.save_bare(ds, str(no_sop), implicit=True)
    text = tmp_path / "text.dcm"
    text.write_text("Hendrickx^Marieke")
    lookalike = tmp_path / "lookalike"
    lookalike.write_bytes(b"\x08\x00" + b"\xff" * 300)
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    nifti_file = tmp_path / "scan.nii"
    nifti_file.write_bytes(h.nifti_bytes(np.zeros((4, 4, 4), np.int16))[0])
    assert dicom.is_dicom_file(str(part10))
    for path in bare:
        assert dicom.is_dicom_file(str(path)), path
    for path in (no_sop, text, lookalike, empty, nifti_file, tmp_path / "absent"):
        assert not dicom.is_dicom_file(str(path)), path


def test_list_dicom_files_follows_file_links_once(series_folder, tmp_path):
    folder, _ = series_folder
    first = sorted(os.listdir(folder))[0]
    os.symlink(folder / first, folder / "link-to-first")
    (folder / "loop").symlink_to(folder, target_is_directory=True)
    files = dicom.list_dicom_files(str(folder))
    assert len(files) == 4
    assert len(set(files)) == 4
