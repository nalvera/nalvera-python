"""Run a scan and react to each kind of error a script should handle differently.

Branch on error.code; the wording can change. Spends credits: one task, budget "personal".
Exit statuses: 1 scan refused, 3 sign-in, 4 credits, 5 try later.
Run: python handle_errors.py
"""
import sys

import nalvera

SCAN = "ct_scan.nii.gz"

try:
    tasks = nalvera.run(
        SCAN,
        model="totalsegmentator_ct",   # required. See: nalvera.models()
        budget="personal",             # default: None (free tasks first, then personal, or
                                       # asks). Options: "personal" | "lab:<id>" | "free" | "ask"
    )
except nalvera.RunInterrupted as stop:             # Ctrl-C
    sys.exit(f"Stopped. Tasks queued: {len(stop.tasks)}. They keep running.")
except nalvera.NalveraError as error:
    print(error, file=sys.stderr)                  # says what to do next
    if error.code in ("insufficient_credits", "no_free_tasks", "max_credits_exceeded",
                      "budget_choice_required"):
        sys.exit(4)                                # top up, or name another budget
    if error.code in ("no_credentials", "invalid_credentials", "key_expired", "key_revoked",
                      "scope_missing", "api_disabled", "api_blocked", "client_too_old"):
        sys.exit(3)                                # a new key, a permission, API access on
    if error.code in ("scan_rejected", "too_large", "not_a_scan", "unreadable"):
        sys.exit(1)                                # this scan cannot run
    if error.code in ("rate_limited", "too_many_active_tasks", "too_many_uploads", "queue_full",
                      "storage_full", "temporarily_unavailable", "api_unavailable",
                      "network_error"):
        sys.exit(5)    # the package waited as long as it may: run again later (retry_after)
    raise
nalvera.download(nalvera.wait(tasks), out="results")  # default: "results"
