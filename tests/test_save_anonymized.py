"""``nalvera.save_anonymized`` with the real anonymizer: a DICOM folder, and
NIfTI-2 inputs (one that converts to NIfTI-1 exactly, one that is refused).
The folder named ``out`` receives the copies and nothing else."""
from __future__ import annotations

import os

import nibabel as nib
import pytest

import anonymize_helpers as h
from test_anonymize_nifti import CASES, ORTHO, SHEAR, _ct_data
from test_mcp_support import dicom_ready

needs_dicom = pytest.mark.skipif(not dicom_ready(), reason="needs pydicom and dcm2niix")


def _only_copies(folder) -> list:
    names = sorted(os.listdir(folder))
    assert names and all(n.startswith("scan-") and n.endswith(".nii.gz") for n in names), names
    return names


@needs_dicom
def test_a_dicom_folder_leaves_only_the_copies(env, tmp_path):
    import nalvera
    from nalvera import messages
    study = tmp_path / "Hendrickx_Marieke_1961"
    study.mkdir()
    h.write_dicom_series(str(study), series_number=3, plants=h.DICOM_PLANTS, slices=4, seed=2)
    out = tmp_path / "share"
    copies = nalvera.save_anonymized(study, out)
    assert len(copies) == 1 and not copies.failed
    [name] = _only_copies(out)
    copy = copies[0]
    assert copy.path == out / name and copy.series == "3"
    assert copy.removed == [messages.text("anonymized.removed_name"),
                            messages.text("anonymized.removed_dicom")]
    data = (out / name).read_bytes()
    assert h.leaks(h.read_gzip_members(data), h.all_tokens()) == []
    rows = nalvera.scans()
    assert [r["original_name"] for r in rows] == ["Hendrickx_Marieke_1961"]
    assert str(out) not in str(rows.path)


def test_a_nifti2_input_that_converts_exactly_says_what_was_removed(env, tmp_path):
    import nalvera
    from nalvera import messages
    data, args, _ = CASES["NIfTI-2 that fits NIfTI-1"]
    payload, _ = h.nifti_bytes(data, **{**args, "fields": {"descrip": b"Patient X"}})
    original = tmp_path / "nifti2.nii.gz"
    original.write_bytes(h.gzip_bytes(payload))
    out = tmp_path / "share"
    copies = nalvera.save_anonymized(original, out)
    [name] = _only_copies(out)
    assert nib.load(out / name).header["sizeof_hdr"] == 348          # NIfTI-1
    assert messages.text("anonymized.removed_field", field="descrip") in copies[0].removed
    assert b"Patient X" not in h.read_gzip_members((out / name).read_bytes())


def test_a_nifti2_input_that_cannot_convert_is_reported_and_writes_nothing(env, tmp_path):
    import nalvera
    payload, _ = h.nifti_bytes(_ct_data(), version=2, qform=ORTHO, sform=SHEAR, slope=0.7,
                               inter=-1024.1)
    original = tmp_path / "inexact.nii.gz"
    original.write_bytes(h.gzip_bytes(payload))
    out = tmp_path / "share"
    copies = nalvera.save_anonymized(original, out)
    assert list(copies) == [] and [f.code for f in copies.failed] == ["nifti2_unsupported"]
    assert os.listdir(out) == []
