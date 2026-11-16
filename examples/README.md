# Examples

A Python script or a notebook for each thing the `nalvera` package does. Each script names its inputs as constants at the top (`SCAN = "ct_scan.nii.gz"`): put your own files there. Sign in first with `nalvera.login(email, api_key)` (see Sign in). A script that spends credits says so in its header and names a budget.

Every setting a script passes carries its default and its options in one form:

```python
budget="personal",   # default: None (free tasks first, then personal, or asks).
                     # Options: "personal" | "lab:<id>" | "free" | "ask"
```

The table at the end lists every setting once.

## Notebooks

| You want to | Notebook |
|---|---|
| Make your first run: sign in, see the models, run one scan, wait, save and open the result | [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) |
| Run a folder of scans with a cost estimate, a budget and a spending cap | [notebooks/many_scans_and_budgets.ipynb](notebooks/many_scans_and_budgets.ipynb) |
| Generate a synthetic CT and open it | [notebooks/generate_synthetic_ct.ipynb](notebooks/generate_synthetic_ct.ipynb) |

## Sign in

| You want to | Script |
|---|---|
| Sign in with the email and the key, have the key asked for, store it, or read it from the environment | [set_api_key.py](set_api_key.py) |

## Start

| You want to | Script |
|---|---|
| Check the sign-in: account, plan, key permissions, end date | [sign_in_and_check_account.py](sign_in_and_check_account.py) |
| See the models and tasks you may run, with prices | [list_models_and_tasks.py](list_models_and_tasks.py) |
| See your free tasks, personal credits and labs | [check_budgets_and_free_tasks.py](check_budgets_and_free_tasks.py) |
| See the limits of your key | [show_limits.py](show_limits.py) |

## One scan

| You want to | Script |
|---|---|
| Segment one `.nii.gz` scan and save the result | [run_one_scan.py](run_one_scan.py) |
| Run an uncompressed `.nii` file | [run_uncompressed_nii.py](run_uncompressed_nii.py) |
| Give a scan a label that the platform shows | [run_one_scan_with_label.py](run_one_scan_with_label.py) |
| Run several tasks on the same scan | [run_one_scan_several_tasks.py](run_one_scan_several_tasks.py) |
| Run files, a NIfTI folder and a DICOM folder in one run | [run_mixed_inputs.py](run_mixed_inputs.py) |

## DICOM

| You want to | Script |
|---|---|
| Run a DICOM folder holding one series | [run_dicom_folder.py](run_dicom_folder.py) |
| Run a DICOM folder holding several series, one task each | [run_dicom_several_series.py](run_dicom_several_series.py) |
| Run one DICOM subfolder per subject, results named by subject | [run_dicom_subjects_in_subfolders.py](run_dicom_subjects_in_subfolders.py) |
| See what a DICOM folder becomes before anything is uploaded | [inspect_dicom_before_running.py](inspect_dicom_before_running.py) |

## Many scans

| You want to | Script |
|---|---|
| Run every NIfTI file in a folder | [run_folder_of_scans.py](run_folder_of_scans.py) |
| Run many scans with a named budget and a spending cap | [run_many_scans.py](run_many_scans.py) |
| Run a cohort and write a CSV: original name, code, task, state, result | [run_cohort_with_summary_csv.py](run_cohort_with_summary_csv.py) |
| Queue each new scan in a folder once, on a schedule, with caps | [watch_folder_for_new_scans.py](watch_folder_for_new_scans.py) |

## Each model

| You want to | Script |
|---|---|
| Segment CT with TotalSegmentator (full body, lung vessels), and list its tasks | [segment_ct_totalsegmentator.py](segment_ct_totalsegmentator.py) |
| Segment MRI with TotalSegmentator MRI (full body, vertebrae) | [segment_mr_totalsegmentator.py](segment_mr_totalsegmentator.py) |
| Segment CT with MOOSE (organs, lung lobes) | [segment_ct_moose.py](segment_ct_moose.py) |
| Segment the muscles of a whole-body MRI with MuscleMap | [segment_mr_muscles_musclemap.py](segment_mr_muscles_musclemap.py) |
| Segment CT with NV-Segment-CT (117 structures, airway) | [segment_ct_nv_segment.py](segment_ct_nv_segment.py) |
| Segment MRI with MRSegmentator (anatomy, body composition) | [segment_mr_mrsegmentator.py](segment_mr_mrsegmentator.py) |
| Ask Merlin about a CT: findings as text, a text query | [ct_findings_and_text_query_merlin.py](ct_findings_and_text_query_merlin.py) |
| Get a structured CT report from NV-Reason-CT and read it | [ct_report_nv_reason.py](ct_report_nv_reason.py) |

## Generation

| You want to | Script |
|---|---|
| Generate a synthetic CT | [generate_ct.py](generate_ct.py) |
| Generate a CT with its anatomy label map | [generate_ct_with_mask.py](generate_ct_with_mask.py) |
| Generate a synthetic brain MRI | [generate_mr.py](generate_mr.py) |
| Generate one brain MRI in several contrasts | [generate_mr_cross_contrast.py](generate_mr_cross_contrast.py) |
| Generate several volumes in one task | [generate_several_volumes.py](generate_several_volumes.py) |
| Download one file of a generation (the mask) | [download_one_part_of_a_generation.py](download_one_part_of_a_generation.py) |
| Learn what a generation costs before queueing it | [estimate_a_generation.py](estimate_a_generation.py) |

## Paying

| You want to | Script |
|---|---|
| Learn what a run costs before anything is uploaded | [estimate_before_uploading.py](estimate_before_uploading.py) |
| Use free tasks only and stop when they run out | [use_free_tasks_only.py](use_free_tasks_only.py) |
| Use free tasks first, then personal credits | [pay_from_personal_budget.py](pay_from_personal_budget.py) |
| Charge a lab's budget | [pay_from_lab_budget.py](pay_from_lab_budget.py) |
| Charge the lab the website suggests | [run_folder_lab_budget.py](run_folder_lab_budget.py) |
| Cap what a run may spend | [stop_at_a_spending_cap.py](stop_at_a_spending_cap.py) |

## Queue

| You want to | Script |
|---|---|
| Queue now and collect the results later, from another process | [queue_now_collect_later.py](queue_now_collect_later.py) |
| Follow the queue: position, time waited, time running | [follow_queue.py](follow_queue.py) |
| Cancel tasks | [cancel_tasks.py](cancel_tasks.py) |
| List every task of the account, page by page | [list_platform_tasks.py](list_platform_tasks.py) |
| Run in the background and keep working, with progress as data | [run_in_background.py](run_in_background.py) |

## Results

| You want to | Script |
|---|---|
| Download results; running it again downloads nothing twice | [download_results.py](download_results.py) |
| See which code is which scan (the record on this computer) | [list_my_scans_and_tasks.py](list_my_scans_and_tasks.py) |
| Measure each label of a result in milliliters | [organ_volumes_from_result.py](organ_volumes_from_result.py) |

## Privacy

| You want to | Script |
|---|---|
| Anonymize without uploading, and see what is removed and what is kept | [anonymize_only.py](anonymize_only.py) |
| Encrypt each upload on your computer, on top of HTTPS | [encrypted_upload.py](encrypted_upload.py) |
| Keep the record of scans in a file you choose | [keep_the_mapping_elsewhere.py](keep_the_mapping_elsewhere.py) |
| Allow a client to read and write only some folders | [limit_to_allowed_folders.py](limit_to_allowed_folders.py) |

## Automation

| You want to | Script |
|---|---|
| Run without anyone watching (a cluster, CI), with exit statuses | [run_without_a_terminal.py](run_without_a_terminal.py) |
| Start a stopped run again without queueing or paying twice | [resume_an_interrupted_run.py](resume_an_interrupted_run.py) |

## Errors

| You want to | Script |
|---|---|
| Handle no credits, a rejected scan, a limit, a lost connection, an expired or revoked key, API access off | [handle_errors.py](handle_errors.py) |

## AI assistants

| You want to | File |
|---|---|
| See what each tool does and costs, and set it up | [mcp/README.md](mcp/README.md) |
| Use Nalvera.AI from Claude Code in a project | [mcp/claude_code.mcp.json](mcp/claude_code.mcp.json) |
| Use it from Claude Desktop | [mcp/claude_desktop_config.json](mcp/claude_desktop_config.json) |
| Use it from Cursor | [mcp/cursor_mcp.json](mcp/cursor_mcp.json) |
| Use it from VS Code | [mcp/vscode_mcp.json](mcp/vscode_mcp.json) |
| Use it from Codex | [mcp/codex_config.toml](mcp/codex_config.toml) |
| Start the server and list its tools from Python | [mcp/list_tools.py](mcp/list_tools.py) |

## Settings, defaults and options

<!-- Generated by tests/examples_helpers.py from the package's code and the server's listing; a test fails when it is out of date. -->

| Argument | Environment variable | Default | Options |
|---|---|---|---|
| `allowed_folders` (`Client`) |  | `None`: any folder |  |
| `api_key` (`login`, `Client`) | `NALVERA_API_KEY` | `None`: NALVERA_API_KEY, else the keychain or the file a login with remember=True wrote |  |
| `avoid` (`prepare`) |  | `None`: no code is avoided | scan codes, such as those the record holds |
| `base_url` (`Client`) | `NALVERA_BASE_URL` | `None`: NALVERA_BASE_URL, else https://nalvera.ai | an `https://` address (`http://` only for localhost) |
| `budget` (`Client`, `run`, `generate`, `run_in_background`) | `NALVERA_BUDGET` | `None`: NALVERA_BUDGET; without one, free tasks first, then personal; with a lab a question when the notebook or terminal can ask, else BudgetChoiceRequired before any upload | `personal` \| `lab:<id>` \| `free` \| `ask` |
| `check` (`prepare`) |  | `None`: every file is read | a function of a file's real path that raises to refuse it |
| `clock` (`Client`) |  | `None`: time.monotonic (for tests) |  |
| `code` (`find`) |  | `None`: any code | a scan code such as `scan-7F3K2Q9A` |
| `config_dir` (`Client`) | `NALVERA_CONFIG_DIR` | `None`: NALVERA_CONFIG_DIR, else ~/.config/nalvera (%APPDATA%\nalvera on Windows) |  |
| `convert` (`inspect`) |  | `False`: counts DICOM files without converting |  |
| `email` (`login`, `Client`) | `NALVERA_EMAIL` | `None`: NALVERA_EMAIL, else what a login with remember=True stored |  |
| `encrypt` (`Client`, `run`, `run_in_background`) | `NALVERA_ENCRYPT` | `None`: NALVERA_ENCRYPT, else on only when the account requires it |  |
| `force` (`run`, `estimate_run`, `generate`, `run_in_background`) |  | `False`: returns the earlier task of a scan already queued, running or done with the same model and task (generate: of the same label, model, task and settings) |  |
| `forget` (`logout`) |  | `False` |  |
| `input` (`Client`) |  | `None`: sys.stdin |  |
| `interactive` (`Client`) |  | `None`: asks when the notebook or the terminal can |  |
| `label` (`run`, `generate`, `prepare`, `run_in_background`) |  | `None`: the platform shows the scan's code | 1 to 60 letters, digits, spaces, dots, dashes, underscores |
| `limit` (`tasks`) |  | `50` | 1 to 100 |
| `mapping` (`Client`) | `NALVERA_MAPPING` | `None`: NALVERA_MAPPING, else one file per account in the configuration folder |  |
| `max_bytes` (`prepare`) |  | `None`: no size limit |  |
| `max_credits` (`Client`, `run`, `generate`, `run_in_background`) | `NALVERA_MAX_CREDITS` | `None`: NALVERA_MAX_CREDITS, else no cap | a number of credits, 0 or more |
| `model` (`run`, `estimate_run`, `run_in_background`, `generate`, `estimate`) |  | required | `nalvera.models()` |
| `name` (`tasks`) |  | `None`: any name |  |
| `offset` (`tasks`) |  | `0` |  |
| `original` (`find`) |  | `None`: any file |  |
| `out` (`save_anonymized`, `download`) |  | required (save_anonymized); `'results'` (download) |  |
| `output` (`Client`) |  | `None`: sys.stdout |  |
| `overwrite` (`download`) |  | `False`: a result saved before is returned, not fetched again; a new file gets a number |  |
| `params` (`generate`, `estimate`) |  | `None`: no settings | the generation settings below |
| `part` (`download`) |  | `None`: every file of the result (several: one .zip) | the task's `parts`: `image`, `mask`, `image2`, ... |
| `path` (`prepare`, `inspect`, `Mapping`) |  | required | prepare, inspect: a scan file or folder; Mapping: the CSV file |
| `paths` (`save_anonymized`, `run`, `estimate_run`, `run_in_background`) |  | required | `.nii`/`.nii.gz` files, folders of them, DICOM folders |
| `poll_seconds` (`wait`) |  | `None`: the key's min_poll_seconds |  |
| `quiet` (`Client`) |  | `False`: messages go to the reporter |  |
| `refresh` (`me`, `models`) |  | `False`: the answer read earlier |  |
| `remember` (`login`) |  | `False`: nothing is stored |  |
| `reporter` (`Client`) |  | `None`: a NotebookReporter in a notebook, else a StreamReporter (prints); a NullReporter with quiet=True |  |
| `shape` (`estimate`) |  | `None` | `[X, Y, Z]` |
| `sleep` (`Client`) |  | `None`: time.sleep (for tests) |  |
| `state` (`tasks`) |  | `'all'`: all tasks | `active` \| `finished` \| `all` |
| `task` (`run`, `estimate_run`, `run_in_background`, `generate`, `estimate`) |  | `None` (run, estimate_run, run_in_background, estimate): the model's default task; required (generate) | the tasks `nalvera.models()` lists |
| `task_id` (`task`, `find`) |  | required (task); `None` (find): any task | a Task or its id |
| `tasks` (`wait`, `download`, `cancel`) |  | `None` (wait, download): every task waiting or running (wait), every result of the record not saved yet (download); required (cancel) | Task objects or task ids |
| `text` (`run`, `run_in_background`) |  | `None`: no text; required by a task with needs_text |  |
| `timeout` (`wait`) |  | `None`: no limit |  |
| `upload_id` (`estimate`) |  | `None` |  |
| `workdir` (`prepare`, `inspect`) |  | required (prepare); `None` (inspect): the system's temporary folder | a private folder (created with mode 0700) |
| `yes` (`run`, `generate`, `run_in_background`) |  | `False`: asks, or stops, for a budget choice or a DICOM folder with several scans |  |
| Path of the dcm2niix program that converts DICOM. | `NALVERA_DCM2NIIX` | the one installed with the package, else PATH |  |
| Folders the MCP server may read scans from and save results in, separated by : (; on Windows). | `NALVERA_MCP_ALLOWED_DIRS` | the folder the server starts in, unless that is the root or the home folder |  |
| 1 lets the MCP server tell the assistant the scans' original names and the saved results' paths; the assistant's provider then sees them. | `NALVERA_MCP_SHOW_NAMES` | off: scan codes, positions and folders only |  |
| Base folder for settings on Linux and macOS. | `XDG_CONFIG_HOME` | ~/.config |  |
| Base folder for settings on Windows. | `APPDATA` | set by Windows |  |
| Proxy for HTTPS connections (read by requests). | `HTTPS_PROXY` | no proxy |  |
| Certificate bundle for a network that inspects HTTPS (read by requests). | `REQUESTS_CA_BUNDLE` | the certificates installed with requests |  |
| `params["mode"]` (rflow-ct; `generate`, `estimate`) |  | required | `"image-only"` \| `"image-mask"` |
| `params["output_size"]` (rflow-ct; `generate`, `estimate`) |  | required | `[X, X, Z]`, X from 128 \| 256 \| 384 \| 512, Z from 128 \| 256 |
| `params["spacing"]` (rflow-ct; `generate`, `estimate`) |  | required | `[S, S, Sz]` in mm, each 0.5 to 5.0 |
| `params["body_region"]` (rflow-ct; `generate`, `estimate`) |  | `[]`; required for image-mask | `"head"` \| `"chest"` \| `"thorax"` \| `"abdomen"` \| `"pelvis"` \| `"lower"` |
| `params["anatomy_list"]` (rflow-ct; `generate`, `estimate`) |  | `[]` | tumors of the region: head: none; chest: lung tumor, bone lesion; thorax: lung tumor, bone lesion; abdomen: hepatic tumor, pancreatic tumor, colon cancer primaries, bone lesion; pelvis: colon cancer primaries, bone lesion; lower: colon cancer primaries, bone lesion |
| `params["controllable_anatomy_size"]` (rflow-ct; `generate`, `estimate`) |  | `[]` | accepted and not used: the platform leaves it empty |
| `params["num_output_samples"]` (rflow-ct; `generate`, `estimate`) |  | `1` | 1 to 64 |
| `params["seed"]` (rflow-ct; `generate`, `estimate`) |  | `0` | 0 to 2147483647 |
| `params["num_inference_steps"]` (rflow-ct; `generate`, `estimate`) |  | `30` | 1 to 1000 |
| rules (rflow-ct) |  |  | output_size is [X, Y, Z] with X equal to Y; spacing is [X, Y, Z] in mm with X equal to Y; X times its spacing must be at least 256 mm; image-mask needs one body_region; anatomy_list may name tumors available in that region |
| `params["mode"]` (rflow-mr-brain; `generate`, `estimate`) |  | required | `"image-only"` \| `"cross-contrast"` |
| `params["output_size"]` (rflow-mr-brain; `generate`, `estimate`) |  | required | `[X, X, Z]`, X from 128 \| 256 \| 384 \| 512, Z from 128 \| 256 |
| `params["spacing"]` (rflow-mr-brain; `generate`, `estimate`) |  | required | `[S, S, Sz]` in mm, each 0.5 to 5.0 |
| `params["contrasts"]` (rflow-mr-brain; `generate`, `estimate`) |  | required (one for image-only, two or more for cross-contrast) | `"T1"` \| `"T2"` \| `"FLAIR"` \| `"SWI"` |
| `params["skull_stripped"]` (rflow-mr-brain; `generate`, `estimate`) |  | `False` | `False` \| `True` |
| `params["num_output_samples"]` (rflow-mr-brain; `generate`, `estimate`) |  | `1` | 1 to 64 |
| `params["seed"]` (rflow-mr-brain; `generate`, `estimate`) |  | `0` | 0 to 2147483647 |
| `params["num_inference_steps"]` (rflow-mr-brain; `generate`, `estimate`) |  | `30` | 1 to 1000 |
| `params["cfg_guidance_scale"]` (rflow-mr-brain; `generate`, `estimate`) |  | `2.0` | 0.0 to 20.0 |
| rules (rflow-mr-brain) |  |  | output_size is [X, Y, Z] with X equal to Y; spacing is [X, Y, Z] in mm with X equal to Y; image-only takes exactly one contrast; cross-contrast at least two; at Z 256 the size must be 256 x 256 x 256 |
