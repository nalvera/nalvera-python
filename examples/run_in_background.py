"""Start a run in the background and keep working: poll it, read its progress, collect it.

For a notebook or an app. Nothing is asked in the background, so the budget is named.
Spends credits: one task per scan, budget "personal". Run: python run_in_background.py
"""
import time

import nalvera

FOLDER = "scans"

progress = nalvera.ListReporter()      # keeps each message as data; quiet=True drops them
client = nalvera.Client(
    reporter=progress,                 # default: None (prints to the terminal)
)
run = client.run_in_background(
    FOLDER,
    model="totalsegmentator_ct",       # required. See: nalvera.models()
    budget="personal",                 # default: None (free tasks first, then personal, or
                                       # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
)
while not run.done():                  # run.cancel() stops it at its next pause
    time.sleep(0.5)
    if progress.events:
        print("Latest:", progress.events[-1].text)
result = run.result()                  # the RunResult, or the run's error raised here
print(len(result), "queued")
