"""Run a scan stored as an uncompressed .nii file.

The anonymized copy is always compressed (.nii.gz) on this computer before upload. Spends
credits: one task, budget "personal". Run: python run_uncompressed_nii.py
"""
import nalvera

SCAN = "ct_scan.nii"

tasks = nalvera.run(
    SCAN,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
