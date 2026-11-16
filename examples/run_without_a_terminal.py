"""Run unattended (a cluster, CI) with an exit status that a scheduler can act on.

Credentials come from the environment; statuses: 1 a scan failed, 3 sign-in, 4 credits or
budget, 5 limits or network. Spends credits: one task per scan, budget "personal".
"""
import sys

import nalvera
from nalvera.cli import exit_status

FOLDER = "scans"

try:
    # No login call: NALVERA_EMAIL and NALVERA_API_KEY come from a secret store.
    result = nalvera.run(
        FOLDER,
        model="totalsegmentator_ct",  # required. See: nalvera.models()
        budget="personal",            # default: None (free tasks first, then personal,
                                      # or asks).
                                      # Options: "personal" | "lab:<id>" | "free" | "ask"
        yes=True,                     # default: False (stops when a DICOM folder holds
                                      # several scans). Options: True | False
    )
    finished = nalvera.wait(result)
    nalvera.download(finished, out="results")  # default: "results"
except nalvera.NalveraError as error:
    print(error, file=sys.stderr)
    sys.exit(exit_status(error))       # 3 sign-in, 4 credits or budget, 5 limits or network
sys.exit(1 if result.failed or any(task.state != "completed" for task in finished) else 0)
