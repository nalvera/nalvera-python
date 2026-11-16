"""Segment the muscles of a whole-body MRI scan with MuscleMap (one task: wholebody).

Spends credits: one task, budget "personal".
"""
import nalvera

SCAN = "mr_scan.nii.gz"

tasks = nalvera.run(
    SCAN,
    model="musclemap",             # required. See: nalvera.models()
    task="wholebody",              # default: None (the model's default task).
                                   # Options: the tasks nalvera.models() lists
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
