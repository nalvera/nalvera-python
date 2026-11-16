"""Start a run again after it stopped (Ctrl-C, a lost connection, a reboot).

A scan already queued, running or done comes back as its earlier task, so nothing is queued or
charged twice. Spends credits (budget "personal") for the scans not queued yet.
Run: python resume_an_interrupted_run.py, and the same again after a stop
"""
import nalvera

FOLDER = "scans"

result = nalvera.run(
    FOLDER,
    model="totalsegmentator_ct",   # required. See: nalvera.models()
    budget="personal",             # default: None (free tasks first, then personal, or asks).
                                   # Options: "personal" | "lab:<id>" | "free" | "ask"
    force=False,                   # default: False (returns the earlier task of a scan already
                                   # queued, running or done). Options: True | False
)
for task in result:
    print(task.original_name, task.id, "queued before" if task.earlier else "queued now")
finished = nalvera.wait(result)
nalvera.download(finished, out="results")  # default: "results"
