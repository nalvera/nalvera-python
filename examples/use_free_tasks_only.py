"""Use free tasks only: queue what they pay for and stop when they run out. Charges nothing.

Scans the free tasks do not cover are in result.not_queued; run them later with a budget.
"""
import sys

import nalvera

FOLDER = "scans"

try:
    result = nalvera.run(
        FOLDER,
        model="totalsegmentator_ct",   # required. See: nalvera.models()
        budget="free",                 # default: None (free tasks first, then personal, or
                                       # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
    )
except nalvera.NoFreeTasks:
    sys.exit("No free task is left: nothing was uploaded or charged.")
print(len(result), "queued for free;", len(result.not_queued), "wait for a budget")
for scan in result.not_queued:
    print("Not queued:", scan.original_name)
nalvera.download(nalvera.wait(result), out="results")  # default: "results"
