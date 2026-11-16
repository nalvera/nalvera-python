"""Segment an MRI scan with TotalSegmentator MRI: the full body (default task) and the vertebrae.

Spends credits: two tasks, budget "personal". Lists the model's tasks first.
Run: python segment_mr_totalsegmentator.py
"""
import nalvera

SCAN = "mr_scan.nii.gz"
MODEL = "totalsegmentator_mr"
TASKS = ["total_mr", "vertebrae_mr"]   # "total_mr" is the default task

model = next(m for m in nalvera.models() if m["id"] == MODEL)
for task in model["tasks"]:
    print(task["id"], "(default)" if task["default"] else "", task["name"])

for task in TASKS:
    tasks = nalvera.run(
        SCAN,
        model=MODEL,               # required. See: nalvera.models()
        task=task,                 # default: None (the model's default task).
                                   # Options: the tasks nalvera.models() lists
        budget="personal",         # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    )
    nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
