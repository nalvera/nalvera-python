"""Segment an MRI scan with MRSegmentator: the anatomy (default task) and the body composition.

Spends credits: two tasks, budget "personal". For a CT scan use the task "anatomy_ct".
Run: python segment_mr_mrsegmentator.py
"""
import nalvera

SCAN = "mr_scan.nii.gz"
MODEL = "mrsegmentator"
TASKS = ["anatomy", "body_comp"]       # MRI tasks; "anatomy" is the default task

model = next(m for m in nalvera.models() if m["id"] == MODEL)
for task in model["tasks"]:
    print(task["id"], "/".join(task["modality"]), "(default)" if task["default"] else "")

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
