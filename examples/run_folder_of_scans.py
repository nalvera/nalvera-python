"""Run TotalSegmentator on every NIfTI file at the top level of a folder: one task each.

Spends credits: one task per scan, budget "personal". A scan that cannot be read is listed in
result.failed and the others go on.
"""
import nalvera

FOLDER = "scans"

result = nalvera.run(
    FOLDER,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
for failure in result.failed:
    print("Not queued:", failure.original_name, failure.code)
nalvera.download(nalvera.wait(result), out="results")  # default: "results"
