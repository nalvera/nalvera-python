"""Run TotalSegmentator on many scans with a named budget and a spending cap.

Spends credits from the budget "personal" and queues no task that would pass MAX_CREDITS: a run
whose estimate passes the cap does not start. Two scans upload at a time.
Run: python run_many_scans.py
"""
from pathlib import Path

import nalvera

SCANS = sorted(Path("cohort").glob("**/*.nii.gz"))     # any list of paths
MAX_CREDITS = 50

result = nalvera.run(
    SCANS,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    max_credits=MAX_CREDITS,       # default: None (NALVERA_MAX_CREDITS, else no cap)
)
print(len(result), "queued,", result.charged_credits, "credits charged")
for scan in result.failed + result.not_queued:
    print("Not queued:", scan.original_name, scan.code)
nalvera.download(nalvera.wait(result), out="results")  # default: "results"
# Two scans with one name are saved with their folder's name in front.
