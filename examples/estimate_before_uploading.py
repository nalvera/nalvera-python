"""Learn what a run would cost before anything is uploaded: per scan, in total, per budget.

The scans are anonymized in a temporary folder and priced by their size; nothing is uploaded.
Needs a key; spends nothing.
"""
import nalvera

FOLDER = "scans"

estimate = nalvera.estimate_run(
    FOLDER,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    task="total",                  # default: None (the model's default task).
                                   # Options: the tasks nalvera.models() lists
)
for scan in estimate.scans:
    print(scan.original_name, scan.shape, scan.credits, "credits")
print("Total:", estimate.credits, "credits;", estimate.free_count, "paid by free tasks;",
      estimate.paid_credits, "credits for the rest")
for option in estimate.budgets:
    mark = " (suggested)" if option["budget"] == estimate.suggested else ""
    print(f'budget="{option["budget"]}":', option["available_credits"], "credits" + mark)
