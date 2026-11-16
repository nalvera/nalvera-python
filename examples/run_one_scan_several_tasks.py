"""Run several tasks of one model on the same scan.

One run per task: an upload serves one task, so the scan is anonymized and uploaded once per
task. Spends credits: one task each, budget "personal". Run: python run_one_scan_several_tasks.py
"""
import nalvera

SCAN = "ct_scan.nii.gz"
MODEL = "totalsegmentator_ct"
TASKS = ["total", "lung_vessels", "body"]

queued = []
for task in TASKS:
    queued += nalvera.run(
        SCAN,
        model=MODEL,               # required. See: nalvera.models()
        task=task,                 # default: None (the model's default task).
                                   # Options: the tasks nalvera.models() lists
        budget="personal",         # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    )
finished = nalvera.wait(queued)
nalvera.download(finished, out="results")    # default: "results"
