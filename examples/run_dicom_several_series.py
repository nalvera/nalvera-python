"""Run TotalSegmentator on a DICOM folder holding several series: one task per series.

Spends credits: one task per series, budget "personal". inspect_dicom_before_running.py shows
the series first.
"""
import nalvera

FOLDER = "dicom_study"

tasks = nalvera.run(
    FOLDER,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    yes=True,                      # default: False (asks, or stops, when a DICOM folder holds
                                   # several scans). Options: True | False
)
print(len(tasks), "series queued")
nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
# Saved as results/dicom_study_<series>__totalsegmentator_ct__total.nii.gz
