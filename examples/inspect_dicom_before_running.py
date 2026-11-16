"""See what a DICOM folder would become before anything is uploaded: its scans and shapes.

Converts in a private temporary folder and removes it. Needs no key; spends nothing.
With prices: nalvera.estimate_run("dicom_study", model="totalsegmentator_ct").
Run: python inspect_dicom_before_running.py
"""
from nalvera.anonymize import inspect

FOLDER = "dicom_study"

found = inspect(
    FOLDER,
    convert=True,           # default: False (counts the DICOM files only). Options: True | False
)
print(found.kind, "folder with", found.dicom_files, "DICOM files and", len(found.scans), "scans")
for scan in found.scans:
    print("series", scan.series, "shape", scan.shape, "corrected:", scan.corrections or "-")
for volume in found.dropped:
    print("left out: series", volume.series, f"({volume.reason}); used instead:", volume.kept)
if found.left_out:
    print(found.left_out, "files are in no scan: screenshots, scouts, derived images")
