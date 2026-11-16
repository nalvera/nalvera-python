"""Pay from your personal credits. The platform still uses your free tasks first.

Spends credits from the budget "personal".
"""
import nalvera

FOLDER = "scans"

result = nalvera.run(
    FOLDER,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
)
for task in result:
    print(task.original_name, "paid by", task.charged["source"], task.charged["credits"])
print(result.charged_credits, "credits charged in all")
nalvera.download(nalvera.wait(result), out="results")  # default: "results"
