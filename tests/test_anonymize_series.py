"""One DICOM folder, several converter outputs.

Several volumes: dcm2niix writes a corrected copy beside the original (``_Tilt``,
``_Eq``), echoes, phase images and same-numbered series. Each scan becomes a
paid task, so a corrected copy replaces its original, identical volumes
become one scan, and every volume left out is reported in the result.
Pictures: screenshots, dose reports and scouts carry the patient's name in their
pixels, which the anonymizer does not change: the converter leaves them out
(``-i y``) and the result says how many files that was.
A failure: a converter that fails part-way: the volumes it wrote are used and the
failure is reported; no volume at all is ``conversion_failed``.

Real-converter tests write synthetic series with pydicom; the rule on names
is also tested on its own, without the converter.
"""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize_helpers as h  # noqa: E402
from nalvera.anonymize import (  # noqa: E402
    AnonymizeError,
    DroppedVolume,
    PreparedScans,
    dicom,
    prepare,
)
from nalvera.anonymize.nifti import write_clean  # noqa: E402

needs_dcm2niix = pytest.mark.skipif(
    os.environ.get("NALVERA_DCM2NIIX") is None and dicom.find_dcm2niix() is None,
    reason="dcm2niix is not installed (pip install dcm2niix)")


def _assert_clean(scan) -> None:
    with open(scan.path, "rb") as fh:
        compressed = fh.read()
    for data in (compressed, h.read_gzip_members(compressed)):
        assert h.leaks(data, h.DICOM_TOKENS) == []
    header = nib.load(scan.path).header
    for name in ("descrip", "aux_file", "db_name", "intent_name"):
        assert header[name] == b"", name
    assert len(header.extensions) == 0


def _converted(folder, tmp_path) -> dict:
    """The converter's own outputs for ``folder``, by series label."""
    private = tmp_path / "direct"
    private.mkdir()
    volumes = dicom.convert(dicom.list_dicom_files(str(folder)), str(private))
    return {v.series: v.path for v in volumes}


def _clean_sha256(path: str, tmp_path) -> str:
    """The SHA-256 the anonymized copy of one converter output has (the
    writer is deterministic, so equal hashes mean the same scan)."""
    import hashlib

    target = str(tmp_path / f"clean-{os.path.basename(path)}")
    write_clean(path, target)
    with open(target, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---- the rule on names (no converter) ------------------------------------------------

def _volumes(names):
    return [dicom.volume_of(f"/private/out/{name}") for name in names]


def test_corrected_copy_replaces_its_original():
    kept, dropped = dicom.choose(_volumes([
        "vol_s3.nii.gz", "vol_s3_Tilt_1.nii.gz",
        "vol_s3a.nii.gz", "vol_s3a_Tilt_1.nii.gz",
        "vol_s4.nii.gz", "vol_s4_Eq_1.nii.gz",
        "vol_s6.nii.gz", "vol_s6_Tilt_Eq_1.nii.gz",
        "vol_s7.nii.gz", "vol_s7_Tilt_1.nii.gz", "vol_s7_Eq_1.nii.gz", "vol_s7_Tilt_Eq_1.nii.gz",
        "vol_s5_e1.nii.gz", "vol_s5_e2.nii.gz", "vol_s5_e2_ph.nii.gz", "vol_s8.nii",
    ]))
    assert [(v.series, v.corrections) for v in kept] == [
        ("3_Tilt_1", ("gantry_tilt",)),
        ("3a_Tilt_1", ("gantry_tilt",)),
        ("4_Eq_1", ("slice_spacing",)),
        ("5_e1", ()), ("5_e2", ()), ("5_e2_ph", ()),
        ("6_Tilt_Eq_1", ("gantry_tilt", "slice_spacing")),
        ("7_Tilt_Eq_1", ("gantry_tilt", "slice_spacing")),
        ("8", ()),
    ]
    by_path = {v.path: v for v in kept}
    assert sorted((os.path.basename(gone.path), os.path.basename(instead.path))
                  for gone, instead in dropped) == [
        ("vol_s3.nii.gz", "vol_s3_Tilt_1.nii.gz"),
        ("vol_s3a.nii.gz", "vol_s3a_Tilt_1.nii.gz"),
        ("vol_s4.nii.gz", "vol_s4_Eq_1.nii.gz"),
        ("vol_s6.nii.gz", "vol_s6_Tilt_Eq_1.nii.gz"),
        ("vol_s7.nii.gz", "vol_s7_Tilt_Eq_1.nii.gz"),
        ("vol_s7_Eq_1.nii.gz", "vol_s7_Tilt_Eq_1.nii.gz"),
        ("vol_s7_Tilt_1.nii.gz", "vol_s7_Tilt_Eq_1.nii.gz"),
    ]
    assert all(instead.path in by_path for _gone, instead in dropped)


def test_two_different_corrections_are_both_kept():
    """A tilt copy and a spacing copy with no combined one are two different
    volumes: both stay, under their own names; the original goes."""
    kept, dropped = dicom.choose(_volumes(["vol_s9.nii.gz", "vol_s9_Tilt_1.nii.gz",
                                           "vol_s9_Eq_1.nii.gz"]))
    assert sorted(v.series for v in kept) == ["9_Eq_1", "9_Tilt_1"]
    assert [os.path.basename(gone.path) for gone, _ in dropped] == ["vol_s9.nii.gz"]


def test_names_without_corrections_are_untouched():
    names = ["vol_s2.nii.gz", "vol_s2a.nii.gz", "vol_s10_e2.nii.gz", "vol_s10_e1.nii.gz"]
    kept, dropped = dicom.choose(_volumes(names))
    assert [v.series for v in kept] == ["2", "2a", "10_e1", "10_e2"]
    assert dropped == []


# ---- several outputs, with the real converter -----------------------------------------

@needs_dcm2niix
@pytest.mark.parametrize("case, corrections", [
    ("tilt", ("gantry_tilt",)),
    ("uneven spacing", ("slice_spacing",)),
    ("tilt and uneven spacing", ("gantry_tilt", "slice_spacing")),
])
def test_one_scan_from_a_corrected_series(tmp_path, case, corrections):
    folder = tmp_path / "Hendrickx_CT"
    folder.mkdir()
    zs = [0.0, 2.5, 5.0, 7.5, 12.5, 15.0, 17.5, 20.0]
    positions = [(-6.0, -6.0, z) for z in zs] if "uneven" in case else h.tilted_positions(8)
    orientation = h.tilted_orientation(15.0) if "tilt" in case else (1, 0, 0, 0, 1, 0)
    extra = {"GantryDetectorTilt": 15.0} if "tilt" in case else {}
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS,
                         positions=positions, orientation=orientation, extra=extra)
    outputs = _converted(folder, tmp_path)
    assert len(outputs) == 2, outputs                # the original and its corrected copy
    label = next(label for label in outputs if label != "3")
    assert label.startswith("3_")

    work = tmp_path / "work"
    scans = prepare(str(folder), workdir=str(work))
    assert isinstance(scans, PreparedScans) and isinstance(scans, list)
    [scan] = scans
    assert scan.series == label
    assert scan.corrections == corrections
    assert scan.sha256 == _clean_sha256(outputs[label], tmp_path)     # the corrected copy
    _assert_clean(scan)
    assert scans.dropped == [DroppedVolume(series="3", reason="uncorrected", kept=label,
                                           shape=(16, 16, 8))]
    assert scans.left_out == 0
    assert scans.converter_status == 0
    assert os.listdir(work) == [scan.filename]


@needs_dcm2niix
def test_echoes_and_phase_images_are_scans_of_their_own(tmp_path):
    from pydicom.uid import generate_uid

    folder = tmp_path / "Hendrickx_MR"
    folder.mkdir()
    study, frame = generate_uid(), generate_uid()
    echo_series, phase_series = generate_uid(), generate_uid()
    for echo, te in ((1, 5.0), (2, 10.0)):
        h.write_dicom_series(str(folder), series_number=5, plants=h.DICOM_PLANTS, modality="MR",
                             positions=[(-6.0, -6.0, 3.0 * k) for k in range(4)],
                             extra={"EchoNumbers": echo, "EchoTime": te,
                                    "RepetitionTime": 500.0},
                             prefix=f"E{echo}", seed=echo, first_instance=10 * echo,
                             series_uid=echo_series, frame_uid=frame, study_uid=study)
    for part, image_type in (("M", "M"), ("P", "P")):
        h.write_dicom_series(str(folder), series_number=6, plants=h.DICOM_PLANTS, modality="MR",
                             positions=[(-6.0, -6.0, 3.0 * k) for k in range(4)],
                             image_type=["ORIGINAL", "PRIMARY", image_type, "ND"],
                             extra={"EchoTime": 5.0, "RepetitionTime": 500.0,
                                    "Manufacturer": "SIEMENS"},
                             prefix=part, seed=7 if part == "M" else 8,
                             first_instance=1 if part == "M" else 101,
                             series_uid=phase_series, frame_uid=frame, study_uid=study)
    scans = prepare(str(folder), workdir=str(tmp_path / "work"))
    assert [s.series for s in scans] == ["5_e1", "5_e2", "6", "6_ph"]
    assert scans.dropped == []
    assert len({s.sha256 for s in scans}) == 4
    for scan in scans:
        _assert_clean(scan)


@needs_dcm2niix
def test_identical_volumes_become_one_scan(tmp_path):
    """The same images exported twice under new UIDs: the converter writes
    ``3`` and ``3a``, which hold the same voxels and geometry."""
    folder = tmp_path / "Hendrickx_export"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS, prefix="A", seed=4)
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS, prefix="B", seed=4)
    assert sorted(_converted(folder, tmp_path)) == ["3", "3a"]
    work = tmp_path / "work"
    scans = prepare(str(folder), workdir=str(work))
    [scan] = scans
    assert scan.series == "3"
    assert scans.dropped == [DroppedVolume(series="3a", reason="duplicate", kept="3",
                                           shape=(16, 16, 4))]
    assert os.listdir(work) == [scan.filename]


@needs_dcm2niix
def test_same_number_with_different_images_gives_two_scans(tmp_path):
    folder = tmp_path / "Hendrickx_two"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS, prefix="A", seed=4)
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS, prefix="B", seed=5)
    scans = prepare(str(folder), workdir=str(tmp_path / "work"))
    assert [s.series for s in scans] == ["3", "3a"]
    assert scans.dropped == []


def test_dropped_volumes_reported_with_a_stand_in(series_folder_plain, tmp_path, monkeypatch):
    """The rule runs on whatever the converter writes (no real converter)."""
    h.install_standin(tmp_path, monkeypatch,
                      names="vol_s5.nii.gz,vol_s5_Tilt_1.nii.gz,vol_s6_e2.nii.gz")
    work = tmp_path / "work"
    scans = prepare(str(series_folder_plain), workdir=str(work))
    assert [(s.series, s.corrections) for s in scans] == [("5_Tilt_1", ("gantry_tilt",)),
                                                           ("6_e2", ())]
    assert [(d.series, d.reason, d.kept) for d in scans.dropped] == [
        ("5", "uncorrected", "5_Tilt_1")]
    assert scans.dropped[0].shape == (6, 5, 4)
    assert sorted(os.listdir(work)) == sorted(s.filename for s in scans)


@pytest.fixture
def series_folder_plain(tmp_path):
    folder = tmp_path / "Hendrickx_Marieke_CT"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)
    return folder


# ---- screenshots and scouts -----------------------------------------------------------

@needs_dcm2niix
def test_secondary_capture_alone_gives_no_scan(tmp_path):
    folder = tmp_path / "Hendrickx_dose_report"
    folder.mkdir()
    pixels = h.write_secondary_capture(str(folder), plants=h.DICOM_PLANTS,
                                       text="Hendrickx^Marieke 19610423")
    assert b"Hendrickx^Marieke" in pixels                    # the name is in its pixels
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(folder), workdir=str(work))
    assert caught.value.code == "no_volume"
    assert caught.value.left_out == 1
    assert os.listdir(work) == []
    for message in (str(caught.value), repr(caught.value), caught.value.detail or ""):
        assert h.leaks(message.encode(), h.DICOM_TOKENS) == []


@needs_dcm2niix
def test_screenshots_and_scouts_beside_a_series_are_left_out(tmp_path):
    folder = tmp_path / "Hendrickx_study"
    folder.mkdir()
    h.write_secondary_capture(str(folder), plants=h.DICOM_PLANTS,
                              text="Hendrickx^Marieke 19610423")
    h.write_localizer(str(folder), plants=h.DICOM_PLANTS)
    pixels = h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)
    work = tmp_path / "work"
    scans = prepare(str(folder), workdir=str(work))
    [scan] = scans
    assert scan.series == "3"
    assert scans.left_out == 4                   # the screenshot and three scout images
    assert scans.dropped == []
    _assert_clean(scan)
    stored = np.asanyarray(nib.load(scan.path).dataobj.get_unscaled())
    assert np.array_equal(np.sort(stored.ravel()), np.sort(pixels.ravel()))
    assert os.listdir(work) == [scan.filename]


def test_converter_is_asked_to_leave_out_derived_and_2d_images(series_folder_plain, tmp_path,
                                                               monkeypatch):
    log = h.install_standin(tmp_path, monkeypatch)
    prepare(str(series_folder_plain), workdir=str(tmp_path / "work"))
    args = log()["args"]
    assert args[args.index("-i") + 1] == "y"


def test_left_out_counts_files_in_no_volume(series_folder_plain, tmp_path, monkeypatch):
    """Four DICOM files go in; the converter says it made a volume of three."""
    h.install_standin(tmp_path, monkeypatch, names="vol_s3.nii.gz")
    monkeypatch.setenv("STANDIN_COUNTS", "3")
    scans = prepare(str(series_folder_plain), workdir=str(tmp_path / "work"))
    assert scans.left_out == 1


def test_left_out_unknown_without_the_converters_counts(series_folder_plain, tmp_path,
                                                        monkeypatch):
    h.install_standin(tmp_path, monkeypatch, names="vol_s3.nii.gz")
    scans = prepare(str(series_folder_plain), workdir=str(tmp_path / "work"))
    assert scans.left_out is None


# ---- a partial conversion -------------------------------------------------------------

def test_partial_conversion_keeps_what_was_written(series_folder_plain, tmp_path, monkeypatch):
    h.install_standin(tmp_path, monkeypatch, names="vol_s3.nii.gz")
    monkeypatch.setenv("STANDIN_EXIT", "8")
    monkeypatch.setenv("STANDIN_COUNTS", "2")
    work = tmp_path / "work"
    scans = prepare(str(series_folder_plain), workdir=str(work))
    [scan] = scans
    assert scans.converter_status == 8
    assert scans.partial
    assert scans.left_out == 2
    _assert_clean(scan)
    assert os.listdir(work) == [scan.filename]


def test_failed_conversion_without_a_volume(series_folder_plain, tmp_path, monkeypatch):
    h.install_standin(tmp_path, monkeypatch, names="")
    monkeypatch.setenv("STANDIN_EXIT", "8")
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(str(series_folder_plain), workdir=str(work))
    assert caught.value.code == "conversion_failed"
    assert os.listdir(work) == []


@needs_dcm2niix
def test_real_partial_conversion(tmp_path):
    """A series whose compressed pixel data cannot be decoded beside one
    that can: the converter exits with status 8 and writes one volume."""
    from pydicom.encaps import encapsulate
    from pydicom.uid import JPEG2000Lossless

    folder = tmp_path / "Hendrickx_partial"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)
    damaged = folder / "damaged"
    damaged.mkdir()
    h.write_dicom_series(str(damaged), series_number=9, plants=h.DICOM_PLANTS, seed=3)
    import pydicom

    for name in os.listdir(damaged):
        path = str(damaged / name)
        ds = pydicom.dcmread(path)
        ds.file_meta.TransferSyntaxUID = JPEG2000Lossless
        ds.PixelData = encapsulate([b"\xff\x4f\xff\x51" + bytes(range(64))])
        ds["PixelData"].VR = "OB"
        h.save_part10(ds, path)
    work = tmp_path / "work"
    scans = prepare(str(folder), workdir=str(work))
    [scan] = scans
    assert scan.series == "3"
    assert scans.converter_status not in (None, 0)
    assert scans.partial
    assert scans.left_out == 4
    _assert_clean(scan)
    assert os.listdir(work) == [scan.filename]


# ---- what a NIfTI input reports -------------------------------------------------------

def test_nifti_result_has_nothing_to_report(tmp_path):
    payload, _ = h.nifti_bytes(np.zeros((4, 4, 4), np.int16))
    source = tmp_path / "scan.nii"
    source.write_bytes(payload)
    scans = prepare(str(source), workdir=str(tmp_path / "work"))
    assert isinstance(scans, PreparedScans)
    assert scans.dropped == [] and scans.left_out is None and scans.converter_status is None
    assert not scans.partial
    assert scans[0].corrections == ()


def test_discard_removes_the_files(tmp_path):
    payload, _ = h.nifti_bytes(np.zeros((4, 4, 4), np.int16))
    source = tmp_path / "scan.nii"
    source.write_bytes(payload)
    work = tmp_path / "work"
    scans = prepare(str(source), workdir=str(work))
    assert os.listdir(work) == [scans[0].filename]
    scans.discard()
    assert os.listdir(work) == []
    scans.discard()                                          # twice is harmless
