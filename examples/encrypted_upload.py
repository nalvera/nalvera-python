"""Encrypt each anonymized scan on this computer before upload, a second layer on top of HTTPS.

The platform decrypts it to run the task. When the package cannot encrypt, nothing is uploaded.
Spends credits: one task, budget "personal".
"""
import nalvera

SCAN = "ct_scan.nii.gz"

print(nalvera.encryption())         # available, required by your account, used by default
tasks = nalvera.run(
    SCAN,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    encrypt=True,                  # default: None (NALVERA_ENCRYPT, else on only when your
                                   # account requires it). Options: True | False
)
nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
