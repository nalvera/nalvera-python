"""Keep the record, which says whose scan each code is, in a private file you choose.

By default it is in the package's configuration folder; keep it off shared drives. Spends
credits: one task, budget "personal". For every script: set NALVERA_MAPPING.
"""
from pathlib import Path

import nalvera

SCAN = "ct_scan.nii.gz"
MAPPING = Path.home() / "private" / "nalvera_scans.csv"

client = nalvera.Client(
    mapping=MAPPING,               # default: None (NALVERA_MAPPING, else one file per
                                   # account in the configuration folder)
)
tasks = client.run(
    SCAN,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
client.download(client.wait(tasks), out="results")  # default: "results"
print("Codes and original names are in", client.mapping_path)
