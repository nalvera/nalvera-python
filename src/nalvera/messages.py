"""Every sentence the package shows to a person, in one place.

``MESSAGES`` maps a stable key to its text. Code never builds a sentence by
joining pieces: it picks a key and fills the placeholders named in the
comment beside it (``{name}`` style; ``text()`` does the filling). Keys stay
stable, so wording can change here without touching code or tests.

Key groups
    error.<code>     an error, by its stable code (server codes and the package's own)
    anonymize.<code> a scan the anonymizer could not prepare (read by nalvera.anonymize too)
    mapping.notice   the notice file written beside the mapping (read by nalvera.anonymize)
    help.*           command-line help, including help.exit.<status>
    setting.<NAME>   what an environment setting does, and setting.<NAME>.default
    net.* reason.*   why a connection failed, why a request is tried again
    run.* upload.* budget.* confirm.* wait.* download.* estimate.* login.* whoami.*
    limits.* models.* scans.* status.* cancel.* scope.* time.* table.*

One source, two dialects. Where a text names something a person must type,
it holds a fragment (``{%login}``) rendered for the interface in use: in
Python ``nalvera.login(email, api_key)``, on the command line
``nalvera login`` (see ``FRAGMENTS``). The library renders Python; the
``nalvera`` command sets the command line for its run (``set_dialect``).
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

MESSAGES: Dict[str, str] = {
    # ── Errors from the server, by code ───────────────────────────────────────
    # {detail}: the server's own explanation, shown as it is.
    "error.invalid_request": "The platform did not accept the request. {detail}",
    # no placeholders
    "error.https_required": "The platform accepts HTTPS only. Check NALVERA_BASE_URL.",
    # no placeholders
    "error.key_in_url":
        "A key appeared in a web address. The platform revokes a real key at once, so create "
        "a new key under Settings → API at https://nalvera.ai, then sign in with it: {%login}",
    # no placeholders
    "error.invalid_credentials":
        "The email or the API key is not valid. Check both, then sign in again: {%login}. A "
        "new key is created under Settings → API at https://nalvera.ai.",
    # no placeholders
    "error.key_expired":
        "This key has passed its end date. Create a new key under Settings → API at "
        "https://nalvera.ai, then sign in with it: {%login}",
    # no placeholders
    "error.key_revoked":
        "This key was revoked. Create a new key under Settings → API at "
        "https://nalvera.ai, then sign in with it: {%login}",
    # {budget}: the budget that was charged, such as personal or lab:12
    "error.insufficient_credits":
        "The budget {budget} has too few credits for this task. Add credits at "
        "https://nalvera.ai or choose another budget with {%budget}, then try again.",
    # no placeholders
    "error.api_disabled":
        "API access is switched off for this account. Switch it on under Settings → API at "
        "https://nalvera.ai.",
    # no placeholders
    "error.api_blocked": "API access for this account is blocked. Contact contact@nalvera.ai.",
    # no placeholders
    "error.email_unverified":
        "Verify your email address at https://nalvera.ai before using the API.",
    # no placeholders
    "error.terms_reaccept_required":
        "Sign in at https://nalvera.ai and accept the current Terms and Privacy Policy, then "
        "try again.",
    # {required_scope}: the permission the key lacks, such as tasks:write
    "error.scope_missing":
        "This key may not do that. It needs the permission {required_scope}. Create a key "
        "with it under Settings → API at https://nalvera.ai, then sign in with it: {%login}",
    # no placeholders
    "error.ip_not_allowed":
        "This key is not allowed from this IP address. Create a key that allows it under "
        "Settings → API at https://nalvera.ai, then sign in with it: {%login}",
    # no placeholders (the server names no budget in this refusal)
    "error.lab_not_allowed":
        "You cannot use this lab's budget. Ask a lab manager, or choose another budget with "
        "{%budget}.",
    # no placeholders
    "error.not_found": "Nothing was found for that ID. {%tasks} lists your tasks.",
    # {state}: the task's state, such as queued or failed
    "error.not_ready":
        "This is not ready yet (state: {state}). Follow it with {%follow}.",
    # {max_mb}: the upload limit in MB
    "error.too_large": "The file is larger than the upload limit of {max_mb} MB.",
    # no placeholders (the platform's front server refused the size without naming the limit)
    "error.too_large_no_limit":
        "The file is larger than the platform accepts. {%limits} shows the limit.",
    # {reason}: the server's reason; the scan's name comes from run.scan_failed
    "error.scan_rejected": "The platform rejected the scan: {reason}",
    # {version}: this package's version; {minimum_version}: the oldest version the server accepts
    "error.client_too_old":
        "This version of nalvera ({version}) is no longer supported. The minimum version is "
        "{minimum_version}. Upgrade it: pip install --upgrade nalvera",
    # {wait}: the wait the server asked for, a time.wait_* text
    "error.rate_limited": "Too many requests. Wait {wait}, then try again.",
    # {limit}: the most API tasks that may wait or run at once
    "error.too_many_active_tasks":
        "This account already has {limit} API tasks waiting or running. Wait for some to "
        "finish, then try again.",
    # {limit}: the most uploads that may be in progress at once; {wait}: until the oldest
    # expires, a time.wait_* text
    "error.too_many_uploads":
        "This account already has {limit} uploads in progress, from a run that is still going "
        "or left by one that was stopped. A slot is free when the running one ends, or in "
        "{wait}, when the oldest upload expires.",
    # no placeholders
    "error.api_unavailable": "The API is not available at the moment. Try again later.",
    # {wait}: the wait the server asked for, a time.wait_* text
    "error.queue_full": "The queue is full. Try again in {wait}.",
    # {wait}: the wait the server asked for, a time.wait_* text
    "error.storage_full": "Storage on the platform is full for now. Try again in {wait}.",
    # no placeholders (a proxy's answer carries no wait time when this text is built)
    "error.temporarily_unavailable":
        "The platform could not handle the request right now. Try again shortly.",
    # no placeholders
    "error.server_error":
        "The platform reported an internal error. Try again later. If it continues, contact "
        "contact@nalvera.ai.",
    # no placeholders (the package resumes the upload by itself)
    "error.offset_mismatch":
        "The upload continues at another position. The package resumes from there.",
    # no placeholders (the package sends the part again by itself)
    "error.checksum_mismatch":
        "A part of the upload arrived damaged. The package sends it again.",
    # no placeholders
    "error.not_cancellable": "This task can no longer be cancelled.",
    # {status}: HTTP status; {code}: the server's code (an answer this version does not know)
    "error.unknown": "The platform answered with status {status} ({code}).",
    # {status}: HTTP status; {first_line}: the first line of the answer, key-shaped text removed
    "error.unexpected_status":
        "The platform answered with status {status} and a page this version cannot read: "
        "{first_line}",
    # no placeholders (the server sends budget "free")
    "error.no_free_tasks":
        "No free task is left, so nothing was queued or charged. To pay with credits, try "
        "again with {%budget_personal} or {%budget_lab}.",
    # no placeholders
    "error.encryption_required":
        "Your account requires encrypted uploads, and this upload was not encrypted. Try "
        "again with {%encrypt}, or upgrade the package: pip install --upgrade nalvera",
    # {wait}: a time.wait_* text (the platform lists no key: asked for here, or required by
    # the account; uploads wait for it first)
    "error.encryption_unavailable":
        "The platform has no encryption key at the moment, so nothing was uploaded. Try again "
        "in {wait}, or upload without the extra layer: leave out {%encrypt} and, if it is on, "
        "switch off Encrypt uploads under Settings → API at https://nalvera.ai.",

    # ── Errors of the package itself ──────────────────────────────────────────
    # no placeholders: the steps to get a key
    "error.no_credentials":
        "No API key is set or stored on this computer. To get one:\n"
        "1. Sign in at https://nalvera.ai (create an account there if you have none).\n"
        "2. Open the menu (⋮) at the bottom left, then Settings → API.\n"
        "3. Switch API access on and create a key that may run tasks.\n"
        "4. Sign in with the email and the key: {%login}",
    # {source}: where the key came from, a value of the source.* keys
    "error.invalid_key_format":
        "The key from {source} is not an API key. A key starts with nvk_ and has 61 "
        "characters. Check that you copied all of it.",
    # {url}: the address given
    "error.insecure_base_url":
        "The address {url} is refused. NALVERA_BASE_URL must start with https://. http:// is "
        "accepted only for localhost and 127.0.0.1.",
    # {url}: the address given
    "error.invalid_base_url":
        "The address {url} is not valid. Give the site's address only, such as "
        "https://nalvera.ai",
    # {path}: the credentials file
    "error.credentials_file_insecure":
        "The credentials file {path} can be read by other users. Make it readable by you "
        "only: {%chmod}",
    # {path}: the credentials file
    "error.credentials_file_unreadable":
        "The credentials file {path} could not be read. Sign in again to store the key: "
        "{%login_remember}",
    # {status}: HTTP status
    "error.unexpected_redirect":
        "The platform answered with a redirect ({status}). The API key is never sent to "
        "another address. Check NALVERA_BASE_URL.",
    # {status}: HTTP status
    "error.unexpected_response":
        "The platform sent an answer this version cannot read ({status}).",
    # {reason}: what failed, a value of the net.* keys
    "error.network_error":
        "Could not reach Nalvera.AI: {reason}. Check your internet connection and "
        "NALVERA_BASE_URL.",
    # {error}: the type and text of what failed inside the HTTP library
    "error.request_failed": "The request to Nalvera.AI failed on this computer: {error}",
    # no placeholders (after nalvera.logout() or Client.close())
    "error.signed_out": "This client is signed out. Sign in again: {%login}",
    # no placeholders
    "error.invalid_email":
        "The email address holds a control character or a character outside ASCII, which no "
        "Nalvera.AI account has. Check it, then sign in again: {%login}",
    # {budget}: the value given
    "error.invalid_budget":
        "The budget {budget} is not valid. Use personal, lab:<id>, free or ask. {%budgets} "
        "lists your labs.",
    # {estimate}: credits the run would charge; {max_credits}: the cap
    "error.max_credits_exceeded":
        "This run would charge about {estimate} credits, more than the cap of {max_credits}. "
        "Nothing was uploaded. Raise the cap ({%max_credits} or NALVERA_MAX_CREDITS) or run "
        "fewer scans.",
    # {count}: scans found; {credits}: their estimated cost
    "error.confirmation_required":
        "The DICOM folders hold {count} scans, one task each (estimated {credits} credits). "
        "Nothing was uploaded. Pass {%yes} or {%max_credits} to go ahead.",
    # no placeholders
    "error.confirmation_declined": "Stopped before uploading. Nothing was queued.",
    # no placeholders
    "error.encrypt_no_trusted_key":
        "This version of nalvera does not know the platform's encryption key, so nothing was "
        "uploaded. Upgrade it: pip install --upgrade nalvera",
    # no placeholders
    "error.encrypt_unavailable":
        "Uploads cannot be encrypted on this computer: the cryptography library is missing. "
        "Install it: pip install cryptography",
    # no placeholders
    "error.encrypt_required_by_account":
        "Your account requires encrypted uploads, so {%no_encrypt} (NALVERA_ENCRYPT=0) "
        "cannot be used. Leave it out and every upload is encrypted.",
    # {name}: the setting; {value}: the value found
    "error.invalid_flag": "{name}={value} is not valid. Use 1 or 0.",
    # {value}: the value given
    "error.invalid_max_credits":
        "The spending cap {value} is not valid. Give a number of credits, such as 25.",
    # {path}: the folder
    "error.mixed_folder":
        "The folder {path} holds NIfTI files and DICOM files. Keep them in separate folders, "
        "or pass the NIfTI files one by one.",
    # {path}: the folder
    "error.no_scans_in_folder":
        "The folder {path} holds no NIfTI file (.nii or .nii.gz) and no DICOM file.",
    # {path}: the file
    "error.path_outside_folder":
        "{path} is a link to a file outside its folder. Pass the file itself.",
    # {path}: the path
    "error.path_not_allowed":
        "{path} is outside the allowed folders ({%allowed_folders}).",
    # {path}: the mapping file
    "error.mapping_not_writable":
        "The scan record {path} cannot be written. Choose a writable file with "
        "{%mapping}.",
    # {path}: the result folder
    "error.output_not_writable":
        "Results cannot be saved in {path}. Choose a writable folder with {%out}.",
    # {path}: the temporary folder
    "error.temp_not_writable":
        "The temporary folder {path} cannot be written. Set TMPDIR to a writable folder.",
    # {path}: the temporary folder; {need_mb}, {free_mb}: megabytes
    "error.temp_space_low":
        "The temporary folder {path} has {free_mb} MB free; the anonymized copies need about "
        "{need_mb} MB. Free some space or set TMPDIR to another folder.",
    # {wait}: a time.wait_* text
    "error.try_again_in": "Try again in {wait}.",
    # {budget}: a lab budget that is not among the account's active labs
    "error.lab_not_listed":
        "The budget {budget} is not one of your active labs. {%budgets} lists your "
        "budgets.",
    # {options}: one line per budget (budget.choice_line); {budget}: the suggested budget
    "error.budget_choice_required":
        "More than one budget can pay for this run, and none is chosen:\n{options}\nTo use "
        "the suggested one, pass {%budget_value} or set NALVERA_BUDGET={budget}.",
    # no placeholders
    "error.invalid_label":
        "A label has 1 to 60 letters, digits, spaces, dots, dashes or underscores.",
    # {param}: the --param value given
    "error.invalid_param": "The setting {param} is not valid. Write it as name=value.",
    # {name}: the setting's name
    "error.duplicate_param": "The setting {name} is given more than once.",
    # {model}: the model id given
    "error.unknown_model":
        "There is no model {model} that this account may run. {%models} lists them.",
    # {model}: model id; {task}: task id given; {tasks}: the task ids the model has
    "error.unknown_task": "The model {model} has no task {task}. Its tasks: {tasks}",
    # {model}: model id
    "error.model_is_generative":
        "The model {model} makes new images from settings and takes no scan. Use "
        "{%generate}.",
    # {model}: model id
    "error.model_needs_scan":
        "The model {model} needs a scan. Use {%run}.",
    # {model}: model id; {task}: task id
    "error.text_required":
        "The task {task} of {model} needs a text query: add {%text} with the query.",
    # no placeholders
    "error.no_paths": "Give at least one scan: a .nii or .nii.gz file, or a DICOM folder.",
    # {value}: what was given; {low}, {high}: the range
    "error.invalid_limit": "The limit {value} is not valid. Give a number from {low} to {high}.",
    # {path}: the mapping file
    "error.no_local_record":
        "{path} does not exist, so this computer has no record of tasks. Name the tasks, or "
        "choose the record of your runs with {%mapping}.",
    # no placeholders
    "error.no_task_ids": "Give at least one task id.",
    # {seconds}: how long it waited
    "error.wait_timeout":
        "Stopped waiting after {seconds} s. The tasks keep running; {%wait} picks them up again.",
    # {name}: original scan name; {reason}: why
    "error.upload_failed": "{name}: the upload failed: {reason}",
    # no placeholders
    "error.upload_gave_up": "the platform kept refusing the upload parts",
    # no placeholders: every scan of the run failed
    "error.nothing_queued": "No task was queued: every scan failed.",

    # ── Why a connection failed ({reason} of error.network_error) ─────────────
    "net.timeout": "the connection timed out",
    "net.certificate":
        "the platform's certificate could not be verified. Behind a network that inspects "
        "HTTPS, set REQUESTS_CA_BUNDLE to your organization's certificate bundle",
    "net.failed": "the connection failed",
    "net.tls": "the encrypted connection failed",
    "net.proxy": "the proxy refused the connection. Check HTTPS_PROXY",

    # ── Why a request is tried again ({reason} of run.retry_wait) ─────────────
    "reason.rate_limited": "too many requests",
    "reason.too_many_uploads": "too many uploads in progress",
    "reason.queue_full": "the queue is full",
    "reason.storage_full": "storage on the platform is full",
    "reason.temporarily_unavailable": "the platform is restarting or busy",
    "reason.encryption_unavailable": "the platform's encryption key is not available",

    # ── Waits, in words ({wait} of error.try_again_in) ───────────────────────
    # {s} seconds, {m} minutes, {h} hours
    "time.wait_seconds": "{s} s",
    "time.wait_minutes": "{m} min",
    "time.wait_hours": "{h} h",
    # no placeholders: the server named no wait
    "time.wait_unknown": "a little while",

    # ── Where a key came from ({source} above) ────────────────────────────────
    "source.argument": "the api_key argument",
    "source.environment": "the variable NALVERA_API_KEY",
    "source.keyring": "the system keychain",
    "source.file": "the credentials file",
    "source.prompt": "your input",

    # ── The anonymizer, by its code (shown after the scan's original name) ────
    # no placeholders, except anonymize.other: {code}: the anonymizer's code
    "anonymize.not_found": "No file or folder was found at that path. Check the path.",
    "anonymize.not_a_scan":
        "This is not a NIfTI file (.nii or .nii.gz) or a folder of DICOM files.",
    "anonymize.unreadable":
        "The file could not be read as a NIfTI scan. Check that the file is complete.",
    "anonymize.too_large":
        "The scan is larger than the upload limit. {%limits} shows the limit.",
    "anonymize.dicom_tool_missing":
        "The DICOM converter dcm2niix was not found. Install it separately with: pip install "
        "dcm2niix (or from https://github.com/rordenlab/dcm2niix). NIfTI files work without "
        "it.",
    "anonymize.mixed_folder":
        "The folder holds NIfTI files and DICOM files. Keep them in separate folders, or pass "
        "the NIfTI files one by one.",
    "anonymize.no_volume":
        "The DICOM folder holds no image that can become a scan. Screenshots, dose reports, "
        "scouts, derived images and single images are left out.",
    "anonymize.nifti2_unsupported":
        "The platform accepts NIfTI-1 only, and this NIfTI-2 file cannot be converted to "
        "NIfTI-1 exactly. Save it as NIfTI-1, then try again.",
    # no placeholders (nalvera.anonymize shows this text as it is)
    "anonymize.mapping_unreadable":
        "The scan record is not one this package wrote, so it was left unchanged. "
        "Choose another file.",
    # {path}: the mapping file
    "error.mapping_unreadable":
        "The scan record {path} is not one this package wrote, so it was left unchanged and "
        "nothing was uploaded. Choose another file with {%mapping}.",
    # {name}: the mapping file's name (the notice file written beside it)
    "mapping.notice":
        "{name} links each scan code shown on Nalvera.AI to the original file on this "
        "computer. With it, anyone can tell whose scan a code is. Keep it on this computer: "
        "do not upload, email or share it.\n",
    "anonymize.conversion_failed":
        "The DICOM folder could not be converted to a NIfTI file. Check that its files are "
        "complete.",
    "anonymize.verification_failed":
        "The anonymized file did not pass its check and was deleted. Nothing was uploaded.",
    "anonymize.invalid_label":
        "A label has 1 to 60 letters, digits, spaces, dots, dashes or underscores.",
    "anonymize.other": "Could not be prepared ({code}).",

    # ── A run ─────────────────────────────────────────────────────────────────
    # {count}: free tasks left
    "run.free_tasks_left": "Free tasks left: {count}. They are used first.",
    # no placeholders
    "run.free_period": "Tasks are free at the moment.",
    # {name}: original file or folder name; {series}: the DICOM series
    "run.series_name": "{name} (series {series})",
    # {name}: original name
    "run.preparing": "Removing identifying data from {name} on this computer...",
    # {name}: original name; {code}: the neutral code the platform sees
    "run.prepared": "{name} is ready as {code}.",
    # {name}: original name
    "run.checking": "Checking {name} on the platform...",
    # {name}: original name; {reason}: why it failed
    "run.scan_failed": "{name}: skipped. {reason}",
    # {name}: original name; {task_id}: task id; {charge}: one of the run.charge_* texts
    "run.queued": "Queued {name} as task {task_id} ({charge}).",
    # {name}: original name; {task_id}: task id; {state}: wait.state.*
    "run.already_queued": "{name} is already queued as task {task_id} ({state}).",
    # {name}: original name; {task_id}: task id; {state}: wait.state.*
    "run.unexpected_replay":
        "The platform answered for {name} with task {task_id} ({state}), queued by an earlier "
        "request with the same key. Nothing was charged again.",
    # no placeholders: printed once per run, after the first scan is prepared
    "run.pixel_notice":
        "The anonymizer removes nothing from the image: faces in head scans and burned-in text "
        "stay.",
    # no placeholders
    "run.encrypting": "Each upload is encrypted on this computer, on top of HTTPS.",
    # {path}: the folder; {count}: NIfTI files found at its top level
    "run.folder_nifti": "{path}: {count} NIfTI files found, one task each.",
    # {name}: the DICOM folder; {count}: scans (series) found in it
    "run.dicom_found": "{name}: {count} scans found.",
    # {name}: the DICOM folder; {series}: the volume left out; {kept}: the one used;
    # {reason}: a dropped.* text
    "run.dicom_dropped": "{name}: volume {series} is not uploaded ({reason}); {kept} is used.",
    # no placeholders
    "dropped.uncorrected": "a corrected copy replaces it",
    "dropped.duplicate": "the same image as another volume",
    # {name}: the DICOM folder; {count}: DICOM files that are in no scan
    "run.dicom_left_out":
        "{name}: {count} DICOM files belong to no scan (screenshots, dose reports, scouts, "
        "derived or single images, duplicates).",
    # {name}: the DICOM folder; {count}: scans converted before the converter stopped
    "run.dicom_partial":
        "{name}: the converter stopped part-way. {count} scans were converted; the other "
        "files were not.",
    # {name}: original name
    "run.duplicate_in_run": "{name}: left out, the same scan is already in this run.",
    # {name}: original name; {task_id}: the earlier task; {state}: wait.state.*
    "run.already_done":
        "{name}: already sent as task {task_id} ({state}); that task is returned. {%force} "
        "runs it again.",
    # {name}: original name; {task_id}: task found by its name after an earlier run stopped
    "run.recovered": "{name}: found task {task_id} from an earlier run that stopped.",
    # {count}: tasks in the run; {credits}: estimated credits before free tasks
    "run.plan": "Tasks: {count}. Estimated cost: {credits} credits.",
    # {free}: tasks paid by free tasks; {count}: tasks in the run
    "run.plan_free": "Free tasks pay for {free} of the {count} tasks.",
    # {budget}: the chosen budget; {covers}: tasks it pays for; {paid}: tasks it has to pay
    "run.plan_covers":
        "The budget {budget} pays for {covers} of the {paid} paid tasks. The run stops when "
        "the budget runs out.",
    # {free}: tasks queued as free; {left}: scans left out
    "run.free_only_partial": "Free tasks only: {free} queued, {left} left out.",
    # {budget}: the budget used from now on
    "run.free_ran_out": "No free task is left. The remaining tasks are paid from {budget}.",
    # {count}: scans; {credits}: estimated credits
    "run.confirm_dicom":
        "The DICOM folders hold {count} scans, one task each (estimated {credits} credits). "
        "Upload and queue them?",
    # {name}: original name; {reason}: a run.left_out.* text
    "run.left_out": "{name}: not queued. {reason}",
    # no placeholders
    "run.left_out.no_free_tasks": "No free task was left for it.",
    # no placeholders
    "run.left_out.max_credits_exceeded": "It would pass the spending cap ({%max_credits}).",
    # {count}: tasks queued before the run stopped
    "run.stopped_queued": "Tasks queued before the run stopped: {count}. They keep running.",
    # {count}: tasks a call queued before it raised or was stopped; {ids}: their ids, joined by
    # list.separator; {credits}: what they charged (a notebook keeps this one line of the call)
    "run.stopped_with_tasks":
        "Tasks queued before the stop: {count} ({ids}), {credits} credits charged. They keep "
        "running; follow them with {%follow}.",
    # no placeholders
    "run.charge_free": "free task",
    # {credits}: credits charged
    "run.charge_personal": "{credits} credits, personal",
    # {credits}: credits charged; {lab_id}: the lab's id
    "run.charge_lab": "{credits} credits, lab {lab_id}",
    # {items}: the run.summary.* counts that are not zero, joined by list.sentence
    "run.summary": "{items}.",
    # {count}: tasks queued
    "run.summary.queued": "Tasks queued: {count}",
    # {count}: scans that failed
    "run.summary.failed": "Failed: {count}",
    # {count}: repeats left out
    "run.summary.skipped": "Already sent: {count}",
    # {count}: scans left out by the budget or the cap
    "run.summary.not_queued": "Not queued: {count}",
    # between the counts of run.summary
    "list.sentence": ". ",
    # {limit}: active task cap; {seconds}: wait before the next try
    "run.active_cap_wait":
        "{limit} API tasks are waiting or running. The next one is sent when one finishes "
        "(next try in {seconds} s).",
    # {ids}: task ids separated by spaces
    "run.no_wait_hint": "Follow them with {%wait_ids}.",
    # {budget}: the budget sent with each task
    "run.budget": "Budget: {budget}",
    # no placeholders
    "run.interrupted":
        "Stopped. Temporary anonymized files were removed. Tasks already queued keep "
        "running; follow them with {%follow}.",
    # no placeholders (a notebook's run.stopped_with_tasks line names the tasks and the call)
    "run.interrupted_told": "Stopped. Temporary anonymized files were removed.",
    # {seconds}: wait in seconds; {reason}: the text of the error that caused the wait
    "run.retry_wait": "Trying again in {seconds} s. Reason: {reason}",

    # ── Uploads ───────────────────────────────────────────────────────────────
    # {name}: original name; {percent}: 0-100; {done_mb}, {total_mb}: megabytes
    "upload.progress": "Uploading {name}: {percent}% ({done_mb} of {total_mb} MB)",
    # {name}: original name
    "upload.done": "Upload complete: {name}",
    # {name}: original name; {offset_mb}: where it continues, in MB
    "upload.resume": "Connection lost while uploading {name}. Continuing at {offset_mb} MB.",

    # ── Choosing a budget ─────────────────────────────────────────────────────
    # {count}: tasks in the run; {credits}: estimated credits for the whole run
    "budget.question":
        "Choose the budget that pays for this run (tasks: {count}, cost: {credits} credits):",
    # {free}: tasks paid by free tasks; {count}: the other tasks; {credits}: their cost
    "budget.question_free":
        "Free tasks pay for {free} of the tasks. Choose the budget that pays for the other "
        "{count} (cost: {credits} credits):",
    # {count}: free tasks left
    "budget.free_note": "Free tasks left: {count}. They are used first.",
    # no placeholders
    "budget.free_period_note": "Tasks are free at the moment; no budget is charged.",
    # {number}: choice number; {name}: budget name; {available}: credits;
    # {mark}: a budget.mark_* text or ""
    "budget.option": "  {number}. {name}: {available} credits available{mark}",
    # no placeholders
    "budget.mark_suggested": " (suggested)",
    # no placeholders
    "budget.mark_short": " (not enough for this run)",
    # {default}: the number chosen when Enter is pressed
    "budget.prompt": "Number [{default}]: ",
    # {max}: the highest choice number
    "budget.invalid_choice": "Type a number from 1 to {max}.",
    # {budget}: the chosen budget, such as lab:12
    "budget.skip_hint":
        "To skip this question next time, pass {%budget_value} or set "
        "NALVERA_BUDGET={budget}.",
    # no placeholders
    "budget.personal_name": "Personal",
    # {budget}: budget value; {name}: budget name; {available}: credits;
    # {mark}: budget.mark_suggested or ""
    "budget.choice_line": "  {budget}: {name}, {available} credits available{mark}",
    # no placeholders: added after a yes/no question
    "confirm.prompt": " [y/N]: ",

    # ── Waiting ───────────────────────────────────────────────────────────────
    # {name}: original name; {state}: wait.state.*; {position}: wait.position or "";
    # {waited}, {running}: durations (time.*)
    "wait.line": "{name}: {state}{position}, waited {waited}, running {running}",
    # {position}: place in the queue
    "wait.position": ", queue position {position}",
    # no placeholders
    "wait.none": "-",
    "wait.state.queued": "waiting",
    "wait.state.processing": "running",
    "wait.state.completed": "completed",
    "wait.state.failed": "failed",
    "wait.state.cancelled": "cancelled",
    "wait.state.cancelling": "cancelling",
    # {items}: the wait.summary.* counts that are not zero, joined by list.separator
    "wait.summary": "{items}.",
    # {count}: tasks in each state
    "wait.summary.completed": "{count} completed",
    "wait.summary.failed": "{count} failed",
    "wait.summary.cancelled": "{count} cancelled",
    # {name}: original name; {error}: the platform's error text
    "wait.failed_detail": "{name}: {error}",
    # no placeholders
    "wait.interrupted": "Stopped waiting. The tasks keep running; {%wait} picks them up again.",
    # {count}: tasks not finished when wait raised; {ids}: their ids, joined by list.separator
    "wait.stopped":
        "Tasks still waiting or running: {count} ({ids}). {%wait} follows them again.",
    # no placeholders
    "wait.nothing": "No tasks to wait for.",

    # ── Durations ─────────────────────────────────────────────────────────────
    # {h} hours, {m} minutes, {s} seconds
    "time.hours": "{h} h {m:02d} min",
    "time.minutes": "{m} min {s:02d} s",
    "time.seconds": "{s} s",

    # ── Downloads ─────────────────────────────────────────────────────────────
    # {path}: where it was saved (the file name starts with the original scan name)
    "download.saved": "Saved {path}",
    # {path}: the new name; {existing}: the file kept
    "download.renamed": "{existing} already exists. Saved as {path}",
    # {name}: original name or task name; {state}: wait.state.*
    "download.skipped": "{name}: no result to download ({state})",
    # {path}: the mapping file
    "download.not_recorded": "The result was saved. Recording it in {path} failed.",
    # {path}: the mapping file (nalvera download with no task id)
    "download.nothing_new": "No new results to download for the tasks in {path}.",
    # {count}: tasks of the local record still waiting or running
    "download.still_running":
        "Tasks still waiting or running: {count}. {%download} saves their results later.",

    # ── estimate ──────────────────────────────────────────────────────────────
    # {name}: original name; {credits}: its cost; {shape}: its voxel grid
    "estimate.scan": "{name}: {credits} credits ({shape})",
    # {count}: tasks; {credits}: total before free tasks
    "estimate.total": "Tasks: {count}. Estimated cost: {credits} credits.",
    # {free}: tasks paid by free tasks; {paid}: credits for the rest
    "estimate.free": "Free tasks pay for {free} of them; the rest costs {paid} credits.",
    # no placeholders: one generation, paid by a free task
    "estimate.free_one": "A free task pays for it.",
    # {budget}: budget value; {name}: its name; {available}: credits; {mark}: budget.mark_* or ""
    "estimate.budget": "  {budget}: {name}, {available} credits available{mark}",

    # ── login / logout ────────────────────────────────────────────────────────
    "login.email_prompt": "Email address of your Nalvera.AI account: ",
    "login.key_prompt": "API key (the input is hidden): ",
    # no placeholders
    "login.checking": "Checking the key with the platform...",
    # {email}: account; {prefix}: key prefix; {expires}: end date
    "login.ok": "Signed in as {email} with the key {prefix}... The key ends on {expires}.",
    # {platform}: the platform's address, when it is not https://nalvera.ai; {email}: account;
    # {prefix}: key prefix; {expires}: end date
    "login.ok_platform":
        "Signed in to {platform} as {email} with the key {prefix}... The key ends on "
        "{expires}.",
    # no placeholders
    "login.stored_keyring": "The key is stored in the system keychain.",
    # {path}: the credentials file
    "login.stored_file": "The key is stored in {path}, readable by you only.",
    # {path}: the credentials file
    "login.stored_file_windows":
        "The key is stored in {path}, inside your Windows user profile, which Windows "
        "restricts to your account.",
    # no placeholders
    "login.env_overrides":
        "NALVERA_API_KEY or NALVERA_EMAIL is set and takes precedence over the stored key.",
    # no placeholders
    "login.cannot_run":
        "This key may not run tasks. Create a key with 'Run tasks and spend credits' under "
        "Settings → API at https://nalvera.ai, then sign in with it: {%login}",
    # no placeholders
    "logout.done":
        "Signed out. The email and the key were removed from this computer. The key stays "
        "valid until you revoke it under Settings → API.",
    # no placeholders
    "logout.nothing": "No stored key was found.",

    # ── whoami ────────────────────────────────────────────────────────────────
    # {email}: account; {plan}: plan name
    "whoami.account": "Account:         {email} ({plan})",
    # {prefix}: key prefix; {name}: key name
    "whoami.key": "Key:             {name} ({prefix}...)",
    # {expires}: end date
    "whoami.expires": "Key ends:        {expires}",
    # {days}: whole days left
    "whoami.expires_soon":
        "Days left on this key: {days}. Create a new key under Settings → API at "
        "https://nalvera.ai, then sign in with it: {%login}",
    # no placeholders
    "whoami.encryption_required":
        "Uploads:         encrypted on top of HTTPS (your account requires it)",
    "whoami.encryption_on": "Uploads:         encrypted on top of HTTPS (NALVERA_ENCRYPT)",
    "whoami.encryption_off":
        "Uploads:         HTTPS only ({%encrypt} adds a second layer)",
    "whoami.encryption_unavailable":
        "Uploads:         HTTPS only (the platform has no encryption key)",
    # {scopes}: the scope.* texts, joined
    "whoami.scopes": "This key may:    {scopes}",
    # {count}: free tasks left
    "whoami.free": "Free tasks left: {count}",
    # no placeholders
    "whoami.free_period": "Free tasks:      tasks are free at the moment",
    # {available}: credits; {monthly}: left this month; {topup}: top-up credits
    "whoami.personal":
        "Personal:        {available} credits ({monthly} this month, {topup} top-up)",
    # {name}: lab name; {budget}: lab:<id>; {available}: balance; {status}: membership status
    "whoami.lab":
        "Lab:             {name} ({budget}): {available} credits, membership {status}",
    # {budget}: the suggested budget
    "whoami.suggested": "Suggested:       {budget}",
    # no placeholders
    "whoami.terms":
        "Sign in at https://nalvera.ai and accept the current Terms and Privacy Policy "
        "before running tasks.",
    # ", " between listed items
    "list.separator": ", ",

    # ── What a scope allows ───────────────────────────────────────────────────
    "scope.tasks:read": "see tasks",
    "scope.results:read": "download results",
    "scope.tasks:write": "run tasks and spend credits",

    # ── limits ────────────────────────────────────────────────────────────────
    # {value}: the number in each line
    "limits.requests_per_minute": "Requests per minute:                 {value}",
    "limits.task_submits_per_minute": "Tasks queued per minute:             {value}",
    "limits.max_active_tasks": "Tasks waiting or running at once:    {value}",
    "limits.max_uploads_in_progress": "Uploads at once:                     {value}",
    "limits.max_file_bytes": "Largest scan file:                   {value} MB",
    "limits.chunk_bytes": "Upload part size:                    {value} MB",
    "limits.min_poll_seconds": "Shortest wait between status checks: {value} seconds",

    # ── models ────────────────────────────────────────────────────────────────
    # {id}: model id; {name}: model name; {version}: version; {modality}: CT, MR…
    "models.model": "{id}  {name} {version} ({modality})",
    # {id}: task id; {name}: task name; {mark}: models.default_mark or ""; {price}: EUR per Mvoxel
    "models.task": "    {%task_value}  {name}{mark}, {price} EUR per million voxels",
    # {id}: task id; {name}: task name; {mark}: models.default_mark or "" (no price given)
    "models.task_no_price": "    {%task_value}  {name}{mark}",
    # no placeholders
    "models.default_mark": " (default)",
    # no placeholders
    "models.needs_text": "      needs {%text}",
    # no placeholders
    "models.generation_intro": "      settings for {%generate_params}:",
    # {setting}: setting name; {value}: allowed values
    "models.generation_item": "        {setting}: {value}",
    # no placeholders
    "models.generative_mark":
        "    makes images from settings, no scan needed. Use {%generate_any}",

    # ── scans ─────────────────────────────────────────────────────────────────
    # {path}: the mapping file
    "scans.none": "No scans are recorded in {path} yet.",
    # {path}: the mapping file
    "scans.header":
        "Scan codes and original names in {path}. This file stays on this computer:",
    # {code}: neutral code; {name}: original name; {label}: label or ""
    "scans.scan": "{code}  {name}{label}",
    # {label}: the label
    "scans.label": "  label: {label}",
    # {task_id}: task id; {model}: model; {task}: task
    "scans.task": "    task {task_id}  {model} {task}",
    # {path}: result file
    "scans.result": "    result {path}",

    # ── anonymize ─────────────────────────────────────────────────────────────
    # {name}: original name; {path}: the anonymized copy; {shape}: its voxel grid
    "anonymized.copy": "{name} -> {path} ({shape})",
    # {removed}: anonymized.removed_* texts, joined
    "anonymized.removed":
        "  Removed: {removed}. Checked: voxel values and geometry are those of the source "
        "volume.",
    # no placeholders
    "anonymized.removed_name": "the file name",
    # no placeholders
    "anonymized.removed_dicom": "every DICOM tag",
    # {field}: a NIfTI header text field, such as descrip
    "anonymized.removed_field": "the header text field {field}",
    # {count}: NIfTI header extensions
    "anonymized.removed_extensions": "header extensions ({count})",
    # {count}: copies written; {path}: the output folder; {mapping}: the mapping file
    "anonymized.done":
        "Anonymized copies: {count}, in {path}. Nothing was sent. The record that links each "
        "code to its original file is {mapping}, on this computer.",

    # ── status / cancel ───────────────────────────────────────────────────────
    # no placeholders
    "status.none": "No tasks are waiting or running.",
    # no placeholders (nalvera status --all)
    "status.none_at_all": "This account has no tasks yet.",
    # {task_id}: task id; {line}: a wait.line text
    "status.listed": "{task_id}  {line}",
    # {task_id}: task id; {state}: wait.state.* (a running task stops at the worker's next check)
    "cancel.done": "Cancel requested for task {task_id}. State: {state}",

    # ── Command-line help ─────────────────────────────────────────────────────
    "help.description":
        "Run AI tasks on Nalvera.AI from your computer. For research use only.",
    "help.epilog": "Documentation: https://nalvera.ai/command-line",
    "help.settings_title": "settings (read from the environment):",
    # {name}: the variable; {meaning}: setting.<NAME>; {default}: setting.<NAME>.default
    "help.setting_line": "  {name}\n      {meaning} Default: {default}",
    "help.exit_title": "exit status:",
    # {status}: the number; {meaning}: a help.exit.* text
    "help.exit_line": "  {status:<4} {meaning}",
    "help.exit.0": "all done",
    "help.exit.1": "some scans failed or were rejected, or a task did not complete",
    "help.exit.2": "wrong usage: an option, a path or a setting is not valid",
    "help.exit.3":
        "sign-in or permission: no key, a key refused, expired or revoked, a permission the "
        "key lacks, API access off, an upload that cannot be encrypted as required, or a "
        "package too old",
    "help.exit.4":
        "credits or budget: too few credits, no free task left, a budget choice or a "
        "confirmation needed, or the spending cap reached",
    "help.exit.5":
        "limits or the platform unavailable after retries, or no connection",
    "help.exit.130": "interrupted (Ctrl-C)",
    "help.version": "Show the version and exit.",
    "help.help": "Show this help and exit.",
    "help.commands": "commands",
    "help.options": "options",
    "help.login": "Check your email and API key and store them on this computer.",
    "help.login.email": "Email address of your Nalvera.AI account. Asked for when left out.",
    "help.logout": "Remove the stored email and key from this computer.",
    "help.whoami":
        "Show your account, plan, key permissions, end date, free tasks and budgets.",
    "help.limits": "Show the limits that apply to your key.",
    "help.models": "List the models you can run, with their tasks and prices.",
    "help.models.model": "Show only this model, with the settings a generation accepts.",
    "help.anonymize":
        "Write the anonymized copies of scans, as a run prepares them, into a folder. The "
        "folder holds the copies only. Nothing is sent and no key is needed.",
    "help.anonymize.out":
        "Folder for the copies. Created when missing; no file in it is replaced.",
    "help.run":
        "Anonymize scans on this computer, upload them, queue one task for each, wait and "
        "save the results.",
    "help.run.paths":
        "NIfTI file (.nii or .nii.gz), folder of NIfTI files, or folder of DICOM files. "
        "Several paths may be given.",
    "help.run.model": "Model to run, for example totalsegmentator_ct. See nalvera models.",
    "help.run.task": "Task of the model. Without it the model's default task runs.",
    "help.run.label":
        "Label that Nalvera.AI shows for the scans in place of the scan code. 1 to 60 "
        "letters, digits, spaces, dots, dashes or underscores.",
    "help.run.text": "Free-text query for tasks that need one.",
    "help.run.budget":
        "Budget that pays beyond the free tasks: personal, lab:ID, free (free tasks only, "
        "nothing charged) or ask. Default: NALVERA_BUDGET, else free tasks first, then "
        "personal without a lab, or a question with a lab.",
    "help.run.max_credits":
        "Spending cap in credits: a run that would cost more does not start, and queueing "
        "stops before a task passes it. Default: NALVERA_MAX_CREDITS, else no cap.",
    "help.run.encrypt":
        "Encrypt each upload on this computer (NVLT, on top of HTTPS). Default: "
        "NALVERA_ENCRYPT, else only when your account requires it.",
    "help.run.no_encrypt":
        "Do not add the NVLT layer. Refused when your account requires encrypted uploads.",
    "help.run.force":
        "Do not leave out scans that are already queued, running or done with this model and "
        "task.",
    "help.run.no_wait":
        "Queue the tasks and return at once. Follow them later with nalvera status or "
        "nalvera wait.",
    "help.run.out": "Folder for the result files. Default: results.",
    "help.run.ids":
        "Write only the queued task ids to standard output, one per line, for a script; "
        "every other line goes to standard error.",
    "help.run.yes":
        "Use the suggested budget without asking, and go ahead when a DICOM folder holds "
        "several scans.",
    "help.estimate":
        "Show what a run would cost, per scan and in total, with free tasks and budgets. "
        "Scans are anonymized in a temporary folder; nothing is uploaded.",
    "help.generate":
        "Create a synthetic scan with a generative model, wait and save the result. No input "
        "scan is needed.",
    "help.generate.model":
        "Generative model, for example nv_generate_ctmr. See nalvera models.",
    "help.generate.task": "Task of the model, for example rflow-ct or rflow-mr-brain.",
    "help.generate.param":
        "One setting as name=value, for example mode=image-only. Repeat the option for each "
        "setting. Lists need quotes: \"spacing=[1.5,1.5,2.0]\". Settings per task: "
        "nalvera models --model ID.",
    "help.generate.label":
        "Label that Nalvera.AI shows for the task. The result file name starts with it. 1 to "
        "60 letters, digits, spaces, dots, dashes or underscores.",
    "help.generate.force":
        "Queue it although a task with the same label, model, task and settings exists.",
    "help.generate.budget":
        "Budget that pays beyond the free tasks: personal, lab:ID, free (free tasks only, "
        "nothing charged) or ask. Default: NALVERA_BUDGET, else a free task first, then "
        "personal without a lab, or a question with a lab.",
    "help.generate.max_credits":
        "Spending cap in credits: a generation that would cost more is not queued. Default: "
        "NALVERA_MAX_CREDITS, else no cap.",
    "help.generate.yes": "Use the suggested budget without asking.",
    "help.generate.no_wait":
        "Queue the generation and return at once. Follow it later with nalvera status or "
        "nalvera wait.",
    "help.generate.ids":
        "Write only the queued task id to standard output, for a script; every other line "
        "goes to standard error.",
    "help.generate.out": "Folder for the result files. Default: results.",
    "help.estimate.paths":
        "NIfTI file (.nii or .nii.gz), folder of NIfTI files, or folder of DICOM files. "
        "Several paths may be given.",
    "help.estimate.model": "Model to price, for example totalsegmentator_ct. See nalvera models.",
    "help.estimate.task": "Task of the model. Without it the model's default task is priced.",
    "help.estimate.force":
        "Price scans that are already queued, running or done with this model and task too.",
    "help.anonymize.paths":
        "NIfTI file (.nii or .nii.gz), folder of NIfTI files, or folder of DICOM files. "
        "Several paths may be given.",
    "help.generate.estimate":
        "Show what the generation would cost from each budget, and whether a free task pays "
        "for it. Nothing is queued; no budget is needed.",
    "help.status": "Show the state of tasks: queue position, time waited and time running.",
    "help.status.ids": "Task IDs to show. Without IDs, all tasks waiting or running.",
    "help.status.watch": "Keep updating until the tasks finish.",
    "help.status.all":
        "List finished tasks too (completed, failed, cancelled), newest first, with their "
        "task IDs.",
    "help.status.limit": "With --all: how many tasks to list, 1 to 100. Default: 50.",
    "help.wait":
        "Wait until tasks finish, showing queue position, time waited and time running.",
    "help.wait.ids": "Task IDs to wait for. Without IDs, all tasks waiting or running.",
    "help.download": "Download the results of finished tasks.",
    "help.download.ids":
        "Task IDs to download. Without IDs, each completed task in this computer's record "
        "(nalvera scans) whose result is not saved yet.",
    "help.download.out": "Folder for the result files. Default: results.",
    "help.download.part":
        "One file of a result that has several files, such as image or mask.",
    "help.cancel": "Cancel waiting or running tasks.",
    "help.cancel.ids": "Task IDs to cancel.",
    "help.scans":
        "Show the scans recorded on this computer: code, original name, label, tasks and "
        "result files.",
    "help.mcp":
        "Run the MCP server for an AI assistant on this computer. The assistant's app starts "
        "it and speaks with it on standard input and output.",
    # no placeholders
    "error.mcp_sdk": "{%mcp} needs the MCP extra: pip install \"nalvera[mcp]\"",

    # ── General ───────────────────────────────────────────────────────────────
    # {message}: an error's text
    "cli.error": "Error: {message}",
    # no placeholders
    "cli.interrupted": "Stopped.",
    # no placeholders
    "cli.interrupted_twice": "Stopped at once. Temporary files are removed at the next run.",

    # ── Signing in from Python, running again ─────────────────────────────────
    # no placeholders (nalvera.login with only one of the two)
    "error.login_needs_both":
        "Give both the email and the API key, or neither. With neither, the package asks for "
        "them, reads NALVERA_EMAIL and NALVERA_API_KEY, or uses what an earlier login stored.",
    # {path}: the record of scans and tasks on this computer (said once per run)
    "run.record": "Scan codes and original names are recorded in {path}, on this computer.",
    # {name}: the scan's original name; {reason}: why it could not be queued (the error's text)
    "error.scan_named": "{name}: {reason}",
    # {stored}: the platform a stored key is for; {platform}: the one this program reaches
    "error.stored_key_other_platform":
        "The key stored on this computer is for {stored}, and this program reaches "
        "{platform} (NALVERA_BASE_URL), so the stored key is not sent. To use {platform}, "
        "sign in for it: {%login}",
    # {path}: the folder
    "error.config_dir_insecure":
        "The folder {path} belongs to another user or others can write to it, so no key and "
        "no record is stored in it. Make it yours only ({%chmod_folder}), or choose another "
        "folder with {%config_dir}.",
    # no placeholders
    "login.keyring_plaintext":
        "The keyring library on this computer keeps keys in a plain file, so it is not used.",
    # no placeholders (the label repeats a scan's file or folder name)
    "error.label_is_file_name":
        "The label repeats a scan's file name, and a label is sent to the platform. Choose a "
        "label that does not identify a patient.",
    # {path}: the folder's name; {count}: the most files and folders checked
    "error.folder_too_large":
        "{path} holds more than {count} files and folders, too many to check. Give a smaller "
        "folder.",
    # {position}: the scan's place in the run, from 1 (an assistant sees no file name)
    "mcp.scan_position": "scan {position}",
    # no placeholders (estimate_cost was given a path together with shape or params)
    "mcp.path_alone": "path: give one of path, shape and params",
    # {name}: original name; {corrections}: corrected.* texts, joined
    "run.dicom_corrected":
        "{name}: the converter resampled this series to correct {corrections}. Its voxel "
        "values and geometry are those of the resampled volume.",
    "corrected.gantry_tilt": "a gantry tilt",
    "corrected.slice_spacing": "uneven slice spacing",
    # {label}: the generation's label; {task_id}: the earlier task; {state}: wait.state.*
    "generate.earlier":
        "{label}: already queued as task {task_id} ({state}) with these settings; that task is "
        "returned. {%force} queues it again.",
    # no placeholders (a generation without a label whose answer was lost: it may be queued)
    "generate.may_be_queued": "The generation may be queued: check {%tasks} before trying again.",
    # {name}: original name or task name; {path}: the file an earlier download saved
    "download.already_saved": "{name}: already saved as {path}",

    # ── Tables a notebook shows (column titles and notes) ─────────────────────
    "table.task_id": "task id",
    "table.name": "name",
    "table.model": "model",
    "table.task": "task",
    "table.state": "state",
    "table.queue": "queue position",
    "table.waited": "waited",
    "table.running": "running",
    "table.default": "default",
    "table.model_name": "model name",
    "table.task_name": "task name",
    "table.modality": "modality",
    "table.price": "EUR per million voxels",
    "table.budget": "budget",
    "table.budget_name": "name",
    "table.available": "available",
    "table.limit": "limit",
    "table.value": "value",
    "table.code": "code",
    "table.original_name": "original name",
    "table.label": "label",
    "table.result": "result file",
    "table.scan": "scan",
    "table.note": "note",
    "table.charge": "charged",
    "table.file": "file",
    "table.copy": "anonymized copy",
    "table.shape": "voxel grid",
    "table.removed": "removed",
    "table.credits": "credits",
    "table.yes": "yes",
    "table.empty": "(none)",
    "table.free_tasks": "free tasks left",
    "table.free_period": "free at the moment",
    "table.account": "account",
    "table.platform": "platform",
    "table.plan": "plan",
    "table.key": "key",
    "table.key_name": "key name",
    "table.permissions": "this key may",
    "table.key_ends": "key ends",
    "table.earlier": "none: queued earlier",
    # no placeholders: under a table with a task queued earlier
    "table.force_note": "To queue again what was queued earlier, pass {%force}.",
    # {budget}: the suggested budget
    "table.suggested": "Suggested budget: {budget}",
    # {credits}: credits charged by this call (under a table of several charged tasks)
    "table.charged": "Charged by this call: {credits} credits.",
    # {path}: the record's file
    "table.record": "Record on this computer: {path}",
    # {task_id}: task id; {state}: wait.state.*
    "table.no_result": "{task_id}: no result ({state})",
    "table.limit.requests_per_minute": "requests per minute",
    "table.limit.task_submits_per_minute": "tasks queued per minute",
    "table.limit.max_active_tasks": "tasks waiting or running at once",
    "table.limit.max_uploads_in_progress": "uploads at once",
    "table.limit.max_file_bytes": "largest scan file, bytes",
    "table.limit.chunk_bytes": "upload part size, bytes",
    "table.limit.min_poll_seconds": "shortest wait between status checks, seconds",
    # why a scan was left out (SkippedScan.reason), shown in a run's table
    "skipped.duplicate_in_run": "the same scan is already in this run",
    "skipped.already_queued": "already queued",
    "skipped.already_processing": "already running",
    "skipped.already_completed": "already done",

    # ── Settings read from the environment (nalvera.settings.SETTINGS) ────────
    "setting.NALVERA_EMAIL": "Email address of your Nalvera.AI account.",
    "setting.NALVERA_EMAIL.default": "what an earlier login stored",
    "setting.NALVERA_API_KEY":
        "API key, for a cluster or a shared script. Set it from a secret store.",
    "setting.NALVERA_API_KEY.default": "what an earlier login stored",
    "setting.NALVERA_BUDGET": "Budget beyond the free tasks: personal, lab:<id>, free or ask.",
    "setting.NALVERA_BUDGET.default":
        "free tasks first, then personal without a lab, or a question with a lab",
    "setting.NALVERA_MAX_CREDITS": "Spending cap per run, in credits.",
    "setting.NALVERA_MAX_CREDITS.default": "no cap",
    "setting.NALVERA_ENCRYPT": "1 encrypts every upload (NVLT, on top of HTTPS); 0 does not.",
    "setting.NALVERA_ENCRYPT.default": "only when your account requires it",
    "setting.NALVERA_MAPPING":
        "CSV file on this computer that records each scan's code and original name, its "
        "tasks and their result files.",
    "setting.NALVERA_MAPPING.default":
        "one file per account and platform, in the folder of NALVERA_CONFIG_DIR",
    "setting.NALVERA_BASE_URL": "Address of the platform.",
    "setting.NALVERA_BASE_URL.default": "https://nalvera.ai",
    "setting.NALVERA_CONFIG_DIR":
        "Folder for what a login stored and for the record of scans and tasks.",
    "setting.NALVERA_CONFIG_DIR.default":
        "~/.config/nalvera (%APPDATA%\\nalvera on Windows)",
    "setting.NALVERA_DCM2NIIX": "Path of the dcm2niix program that converts DICOM.",
    "setting.NALVERA_DCM2NIIX.default": "the one installed with the package, else PATH",
    "setting.NALVERA_MCP_ALLOWED_DIRS":
        "Folders the MCP server may read scans from and save results in, "
        "separated by : (; on Windows).",
    "setting.NALVERA_MCP_ALLOWED_DIRS.default":
        "the folder the server starts in, unless that is the root or the home folder",
    "setting.NALVERA_MCP_SHOW_NAMES":
        "1 lets the MCP server tell the assistant the scans' original names and the saved "
        "results' paths; the assistant's provider then sees them.",
    "setting.NALVERA_MCP_SHOW_NAMES.default": "off: scan codes, positions and folders only",
    "setting.XDG_CONFIG_HOME": "Base folder for settings on Linux and macOS.",
    "setting.XDG_CONFIG_HOME.default": "~/.config",
    "setting.APPDATA": "Base folder for settings on Windows.",
    "setting.APPDATA.default": "set by Windows",
    "setting.HTTPS_PROXY": "Proxy for HTTPS connections (read by requests).",
    "setting.HTTPS_PROXY.default": "no proxy",
    "setting.REQUESTS_CA_BUNDLE":
        "Certificate bundle for a network that inspects HTTPS (read by requests).",
    "setting.REQUESTS_CA_BUNDLE.default": "the certificates installed with requests",
}


# ── What a person types, in each interface ────────────────────────────────────
#: The two interfaces a text is rendered for. The library speaks Python;
#: ``nalvera.cli.main()`` switches to the command line for its own run.
PYTHON, CLI = "python", "cli"
DIALECTS = (PYTHON, CLI)

#: A text names something the person must type as ``{%name}``; it is
#: rendered as (Python, command line). A rendering may hold placeholders of
#: the text's own fields. Texts under ``help.`` are command-line help and
#: name commands as they are.
FRAGMENTS: Dict[str, Tuple[str, str]] = {
    "login": ('nalvera.login("you@university.org", "nvk_...")', "nalvera login"),
    "login_remember": ("nalvera.login(remember=True)", "nalvera login"),
    "models": ("nalvera.models()", "nalvera models"),
    "limits": ("nalvera.limits()", "nalvera limits"),
    "budgets": ("nalvera.budgets()", "nalvera whoami"),
    "tasks": ("nalvera.tasks()", "nalvera status --all"),
    "follow": ("nalvera.wait()", "nalvera status"),
    "wait": ("nalvera.wait()", "nalvera wait"),
    "wait_ids": ("nalvera.wait(tasks)", "nalvera wait {ids}"),
    "download": ("nalvera.download()", "nalvera download"),
    "run": ('nalvera.run(paths, model="{model}")', "nalvera run PATH --model {model}"),
    "generate": ('nalvera.generate(model="{model}", task=task, params=params)',
                 "nalvera generate --model {model}"),
    "generate_any": ("nalvera.generate()", "nalvera generate"),
    "generate_params": ("params= of nalvera.generate()", "nalvera generate --param name=value"),
    "budget": ("budget=", "--budget"),
    "budget_personal": ('budget="personal"', "--budget personal"),
    "budget_lab": ('budget="lab:<id>"', "--budget lab:<id>"),
    "budget_value": ('budget="{budget}"', "--budget {budget}"),
    "yes": ("yes=True", "--yes"),
    "max_credits": ("max_credits=", "--max-credits"),
    "force": ("force=True", "--force"),
    "encrypt": ("encrypt=True", "--encrypt"),
    "no_encrypt": ("encrypt=False", "--no-encrypt"),
    "text": ("text=", "--text"),
    "task_value": ('task="{id}"', "--task {id}"),
    "out": ("out=", "--out"),
    "mapping": ("mapping=", "NALVERA_MAPPING"),
    "allowed_folders": ("allowed_folders=", "NALVERA_MCP_ALLOWED_DIRS"),
    "chmod": ('os.chmod("{path}", 0o600)', "chmod 600 {path}"),
    "chmod_folder": ('os.chmod("{path}", 0o700)', "chmod 700 {path}"),
    "config_dir": ("config_dir=", "NALVERA_CONFIG_DIR"),
    "mcp": ("python -m nalvera.mcp", "nalvera mcp"),
}

_FRAGMENT_RX = re.compile(r"\{%(\w+)\}")
_dialect = PYTHON


def dialect() -> str:
    """The interface texts are rendered for: ``"python"`` or ``"cli"``."""
    return _dialect


def set_dialect(name: str) -> str:
    """Render texts for ``name`` (``"python"`` or ``"cli"``) from now on;
    returns the one used before."""
    global _dialect
    if name not in DIALECTS:
        raise ValueError(f"unknown dialect {name!r}")
    previous, _dialect = _dialect, name
    return previous


@contextmanager
def using(name: str) -> Iterator[None]:
    """Render texts for ``name`` inside the ``with`` block."""
    previous = set_dialect(name)
    try:
        yield
    finally:
        set_dialect(previous)


def fragment(name: str, dialect_: Optional[str] = None) -> str:
    """The rendering of the fragment ``name`` (unfilled) for ``dialect_``
    (default: the current one)."""
    return FRAGMENTS[name][DIALECTS.index(dialect_ or _dialect)]


def template(key: str, dialect_: Optional[str] = None) -> Optional[str]:
    """The text for ``key`` with its fragments rendered for ``dialect_``
    (default: the current one) and its placeholders still open; None for an
    unknown key."""
    raw = MESSAGES.get(key)
    if raw is None:
        return None
    index = DIALECTS.index(dialect_ or _dialect)
    return _FRAGMENT_RX.sub(lambda m: FRAGMENTS[m.group(1)][index], raw)


class _Blank(dict):
    """Fills a placeholder the caller did not supply with an empty string,
    so a missing field can never crash the printing of an error."""

    def __missing__(self, key: str) -> str:
        return ""


def render(key: str, dialect_: Optional[str], /, **fields: Any) -> str:
    """The text for ``key`` rendered for ``dialect_`` (None: the current
    one), its placeholders filled from ``fields``."""
    shown = template(key, dialect_)
    if shown is None:
        return key
    try:
        return shown.format_map(_Blank(fields))
    except (ValueError, TypeError, IndexError):
        # A field of the wrong type for a format spec such as {m:02d}.
        return shown


def text(key: str, /, **fields: Any) -> str:
    """The text for ``key`` with its placeholders filled from ``fields``, in
    the current dialect.

    An unknown key returns the key itself, so a missing entry shows up
    plainly instead of raising.
    """
    return render(key, None, **fields)


def has(key: str) -> bool:
    """Whether ``key`` has a text."""
    return key in MESSAGES
