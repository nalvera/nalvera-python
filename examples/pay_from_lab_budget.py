"""Charge a lab's budget. Free tasks are still used first.

Spends credits from the lab's budget; the lab's manager may set a limit per member.
"""
import nalvera

FOLDER = "scans"
LAB = "lab:12"                     # your labs and their budget names: nalvera.budgets()

for lab in nalvera.budgets()["labs"]:
    print(f"lab:{lab['id']}", lab["name"], lab["balance"], "credits,", lab["member_status"])
result = nalvera.run(
    FOLDER,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget=LAB,                    # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
print(result.charged_credits, "credits charged to", LAB)
nalvera.download(nalvera.wait(result), out="results")  # default: "results"
