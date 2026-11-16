"""Run a folder with one DICOM subfolder per subject, so each result carries its subject.

Each subfolder is given as its own path: results are named <subfolder>_<series>__... .
Spends credits: one task per series, budget "personal".
Run: python run_dicom_subjects_in_subfolders.py
"""
from pathlib import Path

import nalvera

FOLDER = Path("dicom_subjects")     # dicom_subjects/subject_01/, dicom_subjects/subject_02/

subjects = sorted(p for p in FOLDER.iterdir() if p.is_dir())
tasks = nalvera.run(
    subjects,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    yes=True,                      # default: False (asks, or stops, when a DICOM folder holds
                                   # several scans). Options: True | False
)
nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
