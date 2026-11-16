"""inspect(): what a path holds, learned without writing an anonymized file,
so the package can ask the cost and the budget before uploading; and
folders of NIfTI files.

A NIfTI file is read up to its header. A DICOM folder needs the converter to
know its scans: ``convert=False`` (the default) only counts its files;
``convert=True`` converts it in a private temporary folder, applies the
rules of ``prepare`` (corrected copies, identical volumes, files left out),
and removes the folder.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize_helpers as h  # noqa: E402
import nalvera.anonymize as anonymize  # noqa: E402
from nalvera.anonymize import AnonymizeError, FoundScan, Inspection, dicom, prepare  # noqa: E402

needs_dcm2niix = pytest.mark.skipif(
    os.environ.get("NALVERA_DCM2NIIX") is None and dicom.find_dcm2niix() is None,
    reason="dcm2niix is not installed (pip install dcm2niix)")

SHEAR = np.array([[0.75, 0.1, 0.0, -90.0],
                  [0.0, 0.75, 0.2, -126.0],
                  [0.05, 0.0, 1.25, -72.0],
                  [0.0, 0.0, 0.0, 1.0]])


def _nifti(folder, name: str, shape=(6, 5, 4), compress: bool = True, **args) -> str:
    data = np.random.default_rng(len(name)).integers(0, 3000, size=shape).astype(np.int16)
    payload, _ = h.nifti_bytes(data, **args)
    path = os.path.join(str(folder), name)
    with open(path, "wb") as fh:
        fh.write(h.gzip_bytes(payload) if compress else payload)
    return path


def _snapshot(folder) -> list:
    found = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            stat = os.stat(os.path.join(root, name))
            found.append((os.path.relpath(os.path.join(root, name), folder), stat.st_size,
                          stat.st_mtime_ns))
    return sorted(found)


# ---- NIfTI ---------------------------------------------------------------------------

def test_inspect_a_nifti_file(tmp_path):
    path = _nifti(tmp_path, "Hendrickx.nii.gz", shape=(10, 9, 5))
    before = _snapshot(tmp_path)
    found = anonymize.inspect(path)
    assert isinstance(found, Inspection)
    assert (found.kind, found.nifti_files, found.dicom_files) == ("nifti", 1, 0)
    assert found.scans == (FoundScan(path=path, name="Hendrickx.nii.gz", source="nifti",
                                     shape=(10, 9, 5)),)
    assert not found.converted and not found.needs_conversion
    assert found.dropped == () and found.left_out is None
    assert _snapshot(tmp_path) == before                     # nothing written


def test_inspect_agrees_with_prepare(tmp_path):
    for name, args in (("a.nii.gz", {}), ("b.nii", dict(compress=False)),
                       ("c.nii.gz", dict(shape=(7, 6, 5, 3))),
                       ("d.nii.gz", dict(version=2, qform=np.diag([0.5, 0.5, 2.0, 1.0])))):
        path = _nifti(tmp_path, name, **args)
        [scan] = prepare(path, workdir=str(tmp_path / "work"))
        assert anonymize.inspect(path).scans[0].shape == scan.shape


def test_inspect_says_what_prepare_would_refuse(tmp_path):
    folder = tmp_path / "cohort"
    folder.mkdir()
    _nifti(folder, "a.nii.gz", shape=(6, 5, 4))
    _nifti(folder, "b.nii", shape=(7, 6, 5), compress=False)
    _nifti(folder, "c.nii.gz", shape=(6, 5, 4), version=2, sform=SHEAR)
    (folder / "d.nii.gz").write_bytes(h.gzip_bytes(b"Hendrickx^Marieke\n" * 40))
    payload, _ = h.nifti_bytes(np.zeros((40, 40, 40), np.int16))
    (folder / "e.nii").write_bytes(payload[:2000])          # cut short
    (folder / "notes.txt").write_text("Hendrickx")
    (folder / ".hidden.nii.gz").write_bytes(b"")
    (folder / "sub").mkdir()
    _nifti(folder / "sub", "deeper.nii.gz")
    found = anonymize.inspect(str(folder))
    assert found.kind == "nifti_folder"
    assert found.nifti_files == 5 and found.dicom_files == 0
    assert [(s.name, s.shape, s.error) for s in found.scans] == [
        ("a.nii.gz", (6, 5, 4), None),
        ("b.nii", (7, 6, 5), None),
        ("c.nii.gz", (6, 5, 4), "nifti2_unsupported"),
        ("d.nii.gz", None, "not_a_scan"),
        ("e.nii", (40, 40, 40), "unreadable"),
    ]
    assert all(s.path == os.path.join(str(folder), s.name) for s in found.scans)


def test_prepare_a_folder_of_nifti_files(tmp_path):
    folder = tmp_path / "cohort"
    folder.mkdir()
    a = _nifti(folder, "patient01.nii.gz")
    b = _nifti(folder, "patient02.nii", compress=False, shape=(7, 6, 5))
    (folder / "sub").mkdir()
    _nifti(folder / "sub", "ignored.nii.gz")                 # top level only
    work = tmp_path / "work"
    scans = prepare(str(folder), workdir=str(work), label="cohort A")
    assert [(s.original_path, s.original_name, s.shape, s.label) for s in scans] == [
        (a, "patient01.nii.gz", (6, 5, 4), "cohort A"),
        (b, "patient02.nii", (7, 6, 5), "cohort A")]
    assert all(s.source == "nifti" and s.sha256_original for s in scans)
    assert sorted(os.listdir(work)) == sorted(s.filename for s in scans)


def test_a_folder_of_nifti_files_is_prepared_whole_or_not_at_all(tmp_path):
    folder = tmp_path / "cohort"
    folder.mkdir()
    _nifti(folder, "a.nii.gz")
    _nifti(folder, "b.nii.gz", version=2, sform=SHEAR)
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(folder), workdir=str(work))
    assert caught.value.code == "nifti2_unsupported"
    assert os.listdir(work) == []


def test_a_folder_with_nifti_and_dicom(tmp_path):
    folder = tmp_path / "both"
    folder.mkdir()
    _nifti(folder, "a.nii.gz")
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)
    found = anonymize.inspect(str(folder))
    assert (found.kind, found.nifti_files, found.dicom_files) == ("mixed", 1, 4)
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(folder), workdir=str(work))
    assert caught.value.code == "mixed_folder"
    assert os.listdir(work) == []


def test_nifti_below_a_dicom_folder_is_ignored(tmp_path, monkeypatch):
    folder = tmp_path / "study"
    (folder / "derived").mkdir(parents=True)
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)
    _nifti(folder / "derived", "old_output.nii.gz")
    found = anonymize.inspect(str(folder))
    assert (found.kind, found.nifti_files, found.dicom_files) == ("dicom", 0, 4)


def test_empty_folder_and_other_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    other = tmp_path / "report.pdf"
    other.write_bytes(b"%PDF-1.4 Hendrickx")
    for path, code in ((empty, "not_a_scan"), (other, "not_a_scan"),
                       (tmp_path / "absent", "not_found")):
        for call in (lambda p: anonymize.inspect(p),
                     lambda p: prepare(p, workdir=str(tmp_path / "work"))):
            with pytest.raises(AnonymizeError) as caught:
                call(str(path))
            assert caught.value.code == code


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="file modes do not stop this user")
def test_a_folder_that_cannot_be_read(tmp_path):
    folder = tmp_path / "locked"
    folder.mkdir()
    _nifti(folder, "a.nii.gz")
    folder.chmod(0)
    try:
        for call in (lambda: anonymize.inspect(str(folder)),
                     lambda: prepare(str(folder), workdir=str(tmp_path / "work"))):
            with pytest.raises(AnonymizeError) as caught:
                call()
            assert caught.value.code == "unreadable"
    finally:
        folder.chmod(0o700)


# ---- DICOM ---------------------------------------------------------------------------

def test_inspect_dicom_without_converting(tmp_path, monkeypatch):
    folder = tmp_path / "series"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)

    def refuse(*args, **kwargs):
        raise AssertionError("the converter was started")

    monkeypatch.setattr(dicom.subprocess, "run", refuse)
    found = anonymize.inspect(str(folder))
    assert (found.kind, found.dicom_files, found.nifti_files) == ("dicom", 4, 0)
    assert found.scans == ()
    assert found.needs_conversion and not found.converted
    assert found.left_out is None and found.converter_status is None


@needs_dcm2niix
def test_inspect_dicom_with_converting(tmp_path):
    folder = tmp_path / "Hendrickx_study"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS,
                         positions=h.tilted_positions(6), orientation=h.tilted_orientation(),
                         extra={"GantryDetectorTilt": 15.0})
    h.write_dicom_series(str(folder), series_number=4, plants=h.DICOM_PLANTS, prefix="A", seed=8)
    h.write_dicom_series(str(folder), series_number=4, plants=h.DICOM_PLANTS, prefix="B", seed=8)
    h.write_secondary_capture(str(folder), plants=h.DICOM_PLANTS, text="Hendrickx^Marieke")
    private = tmp_path / "private"
    before = _snapshot(folder)
    found = anonymize.inspect(str(folder), convert=True, workdir=str(private))
    assert found.kind == "dicom" and found.converted and not found.needs_conversion
    assert found.dicom_files == 15
    assert os.listdir(private) == []                         # nothing left behind
    assert _snapshot(folder) == before
    scans = prepare(str(folder), workdir=str(tmp_path / "work"))
    assert [(s.series, s.shape, s.corrections) for s in found.scans] == \
        [(s.series, s.shape, s.corrections) for s in scans]
    assert all(s.source == "dicom" and s.name == "Hendrickx_study" and s.error is None
               for s in found.scans)
    assert list(found.dropped) == list(scans.dropped)
    assert [d.reason for d in found.dropped] == ["uncorrected", "duplicate"]
    assert found.left_out == scans.left_out == 1
    assert found.converter_status == scans.converter_status == 0


@needs_dcm2niix
def test_inspect_a_screenshot_folder(tmp_path):
    folder = tmp_path / "screens"
    folder.mkdir()
    h.write_secondary_capture(str(folder), plants=h.DICOM_PLANTS, text="Hendrickx^Marieke")
    with pytest.raises(AnonymizeError) as caught:
        anonymize.inspect(str(folder), convert=True)
    assert caught.value.code == "no_volume"
    assert caught.value.left_out == 1
