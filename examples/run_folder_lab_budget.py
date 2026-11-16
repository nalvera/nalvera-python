"""Run a folder and charge the lab the website suggests: your active lab with the most credits.

Free tasks are still used first. Spends credits from that lab; with no active lab it stops
before uploading. Run: python run_folder_lab_budget.py
"""
import sys

import nalvera

FOLDER = "scans"

suggested = nalvera.budgets()["suggested"]       # "lab:<id>", or "personal" without a lab
if not suggested.startswith("lab:"):
    sys.exit("This account has no active lab: see pay_from_personal_budget.py.")
result = nalvera.run(
    FOLDER,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget=suggested,              # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
print(len(result), "queued,", result.charged_credits, "credits charged to", suggested)
nalvera.download(nalvera.wait(result), out="results")  # default: "results"
