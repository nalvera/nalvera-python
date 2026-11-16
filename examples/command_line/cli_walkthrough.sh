#!/usr/bin/env bash
# Every nalvera command once, in the order a new user needs them. Spends credits: two scan
# tasks, and one generation that is cancelled at once (budget personal). Needs ct_scan.nii.gz
# and ct_scan_2.nii.gz here. Run: bash cli_walkthrough.sh
set -euo pipefail

nalvera --version
nalvera --help                  # every command, setting and exit status

# Sign in once per computer: asks for the email and the key (hidden input), checks and
# stores them.
# --email  default: none (asked for)
nalvera login
nalvera whoami                  # account, key, permissions, end date, free tasks, budgets
nalvera limits                  # the numbers of your key's limits
nalvera models                  # every model and task you may run, with prices
# --model  default: none (every model)
nalvera models --model totalsegmentator_ct

# What would leave this computer: the anonymized copy, in anonymized/.
# Nothing is sent and no key is needed.
# --out  default: none (required)
nalvera anonymize ct_scan.nii.gz --out anonymized

# What a run would cost. The scan is anonymized in a temporary folder; nothing is uploaded.
# --task  default: none (the model's default task)
nalvera estimate ct_scan.nii.gz --model totalsegmentator_ct --task total

# Anonymize, upload, queue, wait, and save the result in results/.
# --budget  default: none (NALVERA_BUDGET, else free tasks first, then personal or a question)
#   Options: personal | lab:<id> | free | ask
# --out  default: results
nalvera run ct_scan.nii.gz --model totalsegmentator_ct --task total --budget personal --out results

# Queue and return at once; keep the task id to follow and collect it later, from any process.
# --no-wait  default: off
# --ids  default: off (only the queued task ids on standard output, one per line)
task_id=$(nalvera run ct_scan_2.nii.gz --model totalsegmentator_ct --budget personal --no-wait --ids)
nalvera status                  # waiting and running tasks: position, time waited, time running
nalvera wait "$task_id"         # until it finishes
# --all  default: off (waiting and running tasks only)
# --limit  default: 50
nalvera status --all --limit 10   # finished tasks too, newest first, with their ids
nalvera scans                   # this computer's record: code, original name, tasks, results
# Without a task id: each result of this computer's record that was not saved yet.
nalvera download --out results

# A generation needs no scan. Every setting: nalvera models --model nv_generate_ctmr
# What it would cost from each budget; nothing is queued.
# --estimate  default: off
nalvera generate --model nv_generate_ctmr --task rflow-ct --param mode=image-only \
  --param "output_size=[256,256,128]" --param "spacing=[1.5,1.5,2.0]" --estimate
# --param  default: none (each setting as name=value; lists in quotes)
generation_id=$(nalvera generate --model nv_generate_ctmr --task rflow-ct --param mode=image-only \
  --param "output_size=[256,256,128]" --param "spacing=[1.5,1.5,2.0]" --budget personal \
  --no-wait --ids)
nalvera cancel "$generation_id"   # a running task stops at the worker's next check

# Remove the stored key from this computer.
nalvera logout
