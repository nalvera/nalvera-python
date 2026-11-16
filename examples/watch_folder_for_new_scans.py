"""Queue each new scan in a folder once. Run it on a schedule, for example every 10 minutes.

Each run spends credits (budget "personal") on at most MAX_TASKS scans and queues no task that
would pass MAX_CREDITS; STATE lists what was queued. A file still being copied fails this run and
is tried in the next. Run: python watch_folder_for_new_scans.py
"""
import json
from pathlib import Path

import nalvera

INBOX = Path("inbox")
STATE = Path("inbox_state.json")
MAX_TASKS = 20
MAX_CREDITS = 25

done = set(json.loads(STATE.read_text())) if STATE.exists() else set()
new = [p for p in sorted(INBOX.glob("*.nii.gz")) if p.name not in done][:MAX_TASKS]
if new:
    result = nalvera.run(
        new,
        model="totalsegmentator_ct",   # required. See: nalvera.models()
        budget="personal",             # default: None (free tasks first, then personal, or
                                       # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
        max_credits=MAX_CREDITS,       # default: None (NALVERA_MAX_CREDITS, else no cap)
    )
    done.update(Path(scan.original_path).name for scan in result.skipped)
    done.update(task.original_name for task in result)
    STATE.write_text(json.dumps(sorted(done)))
    print(len(result), "queued. nalvera.wait() follows them.")
