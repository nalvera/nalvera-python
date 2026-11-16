"""Run a cohort and write summary.csv: original name, code, task id, state, result file.

Spends credits from the budget "personal" and queues no task that would pass MAX_CREDITS. The
codes are what the platform shows; the original names stay on this computer.
Run: python run_cohort_with_summary_csv.py
"""
import csv

import nalvera

FOLDER = "cohort"
MAX_CREDITS = 50

result = nalvera.run(
    FOLDER,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    max_credits=MAX_CREDITS,       # default: None (NALVERA_MAX_CREDITS, else no cap)
)
with open("summary.csv", "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["original_name", "code", "task_id", "state", "result_path"])
    for task in nalvera.wait(result):
        saved = []
        if task.state == "completed":
            saved = nalvera.download(task, out="results")  # default: "results"
        writer.writerow([task.original_name, task.code, task.id, task.state,
                         saved[0] if saved else ""])
    for scan in result.failed + result.not_queued:
        writer.writerow([scan.original_name, "", "", scan.code, ""])
