"""Run TotalSegmentator on a DICOM folder holding one series.

The series is converted to NIfTI and anonymized on this computer: no DICOM tag is uploaded.
Spends credits: one task, budget "personal". Run: python run_dicom_folder.py
"""
import nalvera

FOLDER = "dicom_folder"            # its subfolders are searched too

tasks = nalvera.run(
    FOLDER,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
for task in tasks:
    print(task.original_name, "is", task.code, "on the platform")
nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
# Saved as results/dicom_folder_<series>__totalsegmentator_ct__total.nii.gz
