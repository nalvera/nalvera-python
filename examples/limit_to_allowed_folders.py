"""Let a client read scans and save results only inside folders you allow (a tool, a service).

A path outside them is refused before anything is read (error code path_not_allowed).
Spends credits: one task, budget "personal". Run: python limit_to_allowed_folders.py
"""
from pathlib import Path

import nalvera

ALLOWED = [Path("data")]
OUTSIDE = "other/ct_scan.nii.gz"

client = nalvera.Client(
    allowed_folders=ALLOWED,       # default: None (any folder)
)
tasks = client.run(
    "data/ct_scan.nii.gz",
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
client.download(client.wait(tasks), out="data/results")  # default: "results"
try:
    client.estimate_run(OUTSIDE, "totalsegmentator_ct")
except nalvera.ConfigError as error:
    print("Refused:", error.code)  # path_not_allowed
