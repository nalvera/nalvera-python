"""Segment a CT scan with MOOSE: the organs (default task) and the lung lobes.

Spends credits: two tasks, budget "personal". Lists the model's tasks first.
Run: python segment_ct_moose.py
"""
import nalvera

SCAN = "ct_scan.nii.gz"
MODEL = "moose_v3_clinical"
TASKS = ["clin_ct_organs", "clin_ct_lungs"]   # "clin_ct_organs" is the default task

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
