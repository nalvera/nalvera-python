"""Anonymize a scan without uploading it, then show what was removed and what was kept.

No key, no request. OUT receives the copy, named by its code, and nothing else; the record
that links the code to the original stays on this computer. Run: python anonymize_only.py
"""
import nibabel as nib
import numpy as np

import nalvera

SCAN = "ct_scan.nii.gz"
OUT = "anonymized"

copies = nalvera.save_anonymized(SCAN, OUT)
copy = copies[0]
print(copy.original_name, "->", copy.path, "shape", copy.shape)
print("removed:", ", ".join(copy.removed))
before, after = nib.load(SCAN), nib.load(copy.path)
for field in ("descrip", "aux_file", "intent_name", "db_name"):
    print(f"{field}: {before.header[field].item()!r} -> {after.header[field].item()!r}")
print("header extensions:", len(before.header.extensions), "->", len(after.header.extensions))
print("voxels unchanged:", np.array_equal(np.asarray(before.dataobj), np.asarray(after.dataobj)))
print("geometry unchanged:", np.array_equal(before.affine, after.affine))
print("record:", copies.record)
# The anonymizer removes nothing from the image: a face in a head scan and text burned in stay.
