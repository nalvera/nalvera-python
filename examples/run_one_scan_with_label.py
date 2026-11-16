"""Run one scan with a label, which the platform shows in place of the scan's code.

The label leaves this computer: choose one that does not identify a patient. Spends credits:
one task, budget "personal". Run: python run_one_scan_with_label.py
"""
import nalvera

SCAN = "ct_scan.nii.gz"

tasks = nalvera.run(
    SCAN,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    label="S012 baseline",         # default: None (the platform shows the scan's code).
                                   # 1 to 60 letters, digits, spaces, dots, dashes, underscores
)
print("The platform shows:", tasks[0].name)         # S012_baseline.nii.gz
nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
