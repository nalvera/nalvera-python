# The nalvera command line: every command and option

Generated from the package's own parser by `tests/examples_helpers.py`; a test fails when this file is out of date. `nalvera run --help`, and the same for each command, shows it in the terminal.

- `nalvera --help`: Show this help and exit.
- `nalvera --version`: Show the version and exit.

## `nalvera login`

Check your email and API key and store them on this computer.

- `--email EMAIL`: Email address of your Nalvera.AI account. Asked for when left out. Default: none.

## `nalvera logout`

Remove the stored email and key from this computer.

## `nalvera whoami`

Show your account, plan, key permissions, end date, free tasks and budgets.

## `nalvera limits`

Show the limits that apply to your key.

## `nalvera models`

List the models you can run, with their tasks and prices.

- `--model ID`: Show only this model, with the settings a generation accepts. Default: none.

## `nalvera anonymize PATH... --out DIR`

Write the anonymized copies of scans, as a run prepares them, into a folder. The folder holds the copies only. Nothing is sent and no key is needed.

- `PATH`: NIfTI file (.nii or .nii.gz), folder of NIfTI files, or folder of DICOM files. Several paths may be given. Required.
- `--out DIR`: Folder for the copies. Created when missing; no file in it is replaced. Required.

## `nalvera estimate PATH... --model M`

Show what a run would cost, per scan and in total, with free tasks and budgets. Scans are anonymized in a temporary folder; nothing is uploaded.

- `PATH`: NIfTI file (.nii or .nii.gz), folder of NIfTI files, or folder of DICOM files. Several paths may be given. Required.
- `--model M`: Model to price, for example totalsegmentator_ct. See nalvera models. Required.
- `--task T`: Task of the model. Without it the model's default task is priced. Default: none.
- `--force`: Price scans that are already queued, running or done with this model and task too. Default: off.

## `nalvera run PATH... --model M`

Anonymize scans on this computer, upload them, queue one task for each, wait and save the results.

- `PATH`: NIfTI file (.nii or .nii.gz), folder of NIfTI files, or folder of DICOM files. Several paths may be given. Required.
- `--model M`: Model to run, for example totalsegmentator_ct. See nalvera models. Required.
- `--task T`: Task of the model. Without it the model's default task runs. Default: none.
- `--label L`: Label that Nalvera.AI shows for the scans in place of the scan code. 1 to 60 letters, digits, spaces, dots, dashes or underscores. Default: none.
- `--text TEXT`: Free-text query for tasks that need one. Default: none.
- `--budget BUDGET`: Budget that pays beyond the free tasks: personal, lab:ID, free (free tasks only, nothing charged) or ask. Default: NALVERA_BUDGET, else free tasks first, then personal without a lab, or a question with a lab.
- `--max-credits N`: Spending cap in credits: a run that would cost more does not start, and queueing stops before a task passes it. Default: NALVERA_MAX_CREDITS, else no cap.
- `--yes`: Use the suggested budget without asking, and go ahead when a DICOM folder holds several scans. Default: off.
- `--encrypt`: Encrypt each upload on this computer (NVLT, on top of HTTPS). Default: NALVERA_ENCRYPT, else only when your account requires it.
- `--no-encrypt`: Do not add the NVLT layer. Refused when your account requires encrypted uploads.
- `--force`: Do not leave out scans that are already queued, running or done with this model and task. Default: off.
- `--no-wait`: Queue the tasks and return at once. Follow them later with nalvera status or nalvera wait. Default: off.
- `--ids`: Write only the queued task ids to standard output, one per line, for a script; every other line goes to standard error. Default: off.
- `--out DIR`: Folder for the result files. Default: results.

## `nalvera generate --model M --task T`

Create a synthetic scan with a generative model, wait and save the result. No input scan is needed.

- `--model M`: Generative model, for example nv_generate_ctmr. See nalvera models. Required.
- `--task T`: Task of the model, for example rflow-ct or rflow-mr-brain. Required.
- `--param NAME=VALUE`: One setting as name=value, for example mode=image-only. Repeat the option for each setting. Lists need quotes: "spacing=[1.5,1.5,2.0]". Settings per task: nalvera models --model ID. Default: none.
- `--label L`: Label that Nalvera.AI shows for the task. The result file name starts with it. 1 to 60 letters, digits, spaces, dots, dashes or underscores. Default: none.
- `--estimate`: Show what the generation would cost from each budget, and whether a free task pays for it. Nothing is queued; no budget is needed. Default: off.
- `--force`: Queue it although a task with the same label, model, task and settings exists. Default: off.
- `--budget BUDGET`: Budget that pays beyond the free tasks: personal, lab:ID, free (free tasks only, nothing charged) or ask. Default: NALVERA_BUDGET, else a free task first, then personal without a lab, or a question with a lab.
- `--max-credits N`: Spending cap in credits: a generation that would cost more is not queued. Default: NALVERA_MAX_CREDITS, else no cap.
- `--yes`: Use the suggested budget without asking. Default: off.
- `--no-wait`: Queue the generation and return at once. Follow it later with nalvera status or nalvera wait. Default: off.
- `--ids`: Write only the queued task id to standard output, for a script; every other line goes to standard error. Default: off.
- `--out DIR`: Folder for the result files. Default: results.

## `nalvera status TASK_ID...`

Show the state of tasks: queue position, time waited and time running.

- `TASK_ID`: Task IDs to show. Without IDs, all tasks waiting or running.
- `--watch`: Keep updating until the tasks finish. Default: off.
- `--all`: List finished tasks too (completed, failed, cancelled), newest first, with their task IDs. Default: off.
- `--limit N`: With --all: how many tasks to list, 1 to 100. Default: 50.

## `nalvera wait TASK_ID...`

Wait until tasks finish, showing queue position, time waited and time running.

- `TASK_ID`: Task IDs to wait for. Without IDs, all tasks waiting or running.

## `nalvera download TASK_ID...`

Download the results of finished tasks.

- `TASK_ID`: Task IDs to download. Without IDs, each completed task in this computer's record (nalvera scans) whose result is not saved yet.
- `--out DIR`: Folder for the result files. Default: results.
- `--part PART`: One file of a result that has several files, such as image or mask. Default: none.

## `nalvera cancel TASK_ID...`

Cancel waiting or running tasks.

- `TASK_ID`: Task IDs to cancel. Required.

## `nalvera scans`

Show the scans recorded on this computer: code, original name, label, tasks and result files.

## `nalvera mcp`

Run the MCP server for an AI assistant on this computer. The assistant's app starts it and speaks with it on standard input and output.

## Exit statuses

- `0`: all done
- `1`: some scans failed or were rejected, or a task did not complete
- `2`: wrong usage: an option, a path or a setting is not valid
- `3`: sign-in or permission: no key, a key refused, expired or revoked, a permission the key lacks, API access off, an upload that cannot be encrypted as required, or a package too old
- `4`: credits or budget: too few credits, no free task left, a budget choice or a confirmation needed, or the spending cap reached
- `5`: limits or the platform unavailable after retries, or no connection
- `130`: interrupted (Ctrl-C)
