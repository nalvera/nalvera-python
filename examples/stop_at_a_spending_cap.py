"""Cap what a run may spend: a run whose estimate passes the cap does not start.

Spends credits from the budget "personal"; queueing stops before a task would pass MAX_CREDITS.
For every run: NALVERA_MAX_CREDITS. Run: python stop_at_a_spending_cap.py
"""
import sys

import nalvera

FOLDER = "scans"
MAX_CREDITS = 10

try:
    result = nalvera.run(
        FOLDER,
        model="totalsegmentator_ct",   # required. See: nalvera.models()
        budget="personal",             # default: None (free tasks first, then personal, or
                                       # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
        max_credits=MAX_CREDITS,       # default: None (NALVERA_MAX_CREDITS, else no cap)
    )
except nalvera.CreditCapExceeded as error:
    sys.exit(f"Not started: it would cost {error.extra['estimate']} credits, "
             f"more than the cap of {error.extra['max_credits']}.")
print(len(result), "queued,", result.charged_credits, "credits charged")
for scan in result.not_queued:                 # left out to stay under the cap
    print("Not queued:", scan.original_name, scan.code)
