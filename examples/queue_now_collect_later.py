"""Queue scans now and collect the results later, in a new process or on another day.

The first run uploads, queues and writes the task ids to QUEUED; the next run waits for them
and downloads the results. Spends credits on the first run (budget "personal").
Run: python queue_now_collect_later.py (once now, once later)
"""
import json
from pathlib import Path

import nalvera

FOLDER = "scans"
QUEUED = Path("queued_tasks.json")

if not QUEUED.exists():
    tasks = nalvera.run(
        FOLDER,
        model="totalsegmentator_ct",   # required. See: nalvera.models()
        budget="personal",             # default: None (free tasks first, then personal, or
                                       # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
    )
    QUEUED.write_text(json.dumps([task.id for task in tasks]))
    print(len(tasks), "queued. Run this script again later to collect the results.")
else:
    finished = nalvera.wait(json.loads(QUEUED.read_text()))   # ids work as well as tasks
    nalvera.download(finished, out="results")  # default: "results"
    QUEUED.unlink()                     # names come from the record on this computer
