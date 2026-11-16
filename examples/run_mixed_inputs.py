"""Run one model on a mix of inputs at once: NIfTI files, a NIfTI folder, a DICOM folder.

One folder holding both NIfTI and DICOM files is refused (mixed_folder): pass the parts as
separate paths. Spends credits: one task per scan, budget "personal".
Run: python run_mixed_inputs.py
"""
import nalvera

PATHS = ["ct_scan.nii.gz", "ct_scan_2.nii", "scans", "dicom_folder"]

tasks = nalvera.run(
    PATHS,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    yes=True,                      # default: False (asks, or stops, when a DICOM folder holds
                                   # several scans). Options: True | False
)
nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
