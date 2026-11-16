# Nalvera.AI in an AI assistant (MCP)

`python -m nalvera.mcp` starts a server that an AI assistant's app runs on your computer. Through it the assistant lists models, checks budgets and costs, runs tasks on your scans, generates synthetic scans, follows tasks and saves results. Scans are anonymized on your computer before upload, as everywhere in the package. For research use only.

## Set up once

1. `pip install "nalvera[mcp]"`.
2. Store your key on this computer once, from Python: `nalvera.login(remember=True)`, which asks for it. The server finds it there, and no file here holds a key ([set_api_key.py](../set_api_key.py) shows every way to give one). Keep keys out of an app's settings and out of chats; revoke a key that was written down, on nalvera.ai under Settings → API.
3. Add the snippet for your app:

| App | Snippet | Where it goes |
|---|---|---|
| Claude Code | [claude_code.mcp.json](claude_code.mcp.json) | `.mcp.json` in the project folder |
| Claude Desktop | [claude_desktop_config.json](claude_desktop_config.json) | Settings → Developer → Edit Config; set your scans folder |
| Cursor | [cursor_mcp.json](cursor_mcp.json) | `.cursor/mcp.json` in the project folder |
| VS Code | [vscode_mcp.json](vscode_mcp.json) | `.vscode/mcp.json` in the project folder |
| Codex | [codex_config.toml](codex_config.toml) | `~/.codex/config.toml`; set your scans folder |
| Check that it works | [list_tools.py](list_tools.py) | `python list_tools.py` lists the tools and your free tasks |

`command` must be the Python that has the package installed: when the app finds another one, put the full path there (in that Python, `import sys; print(sys.executable)` shows it).

## Folders

The server reads scans and saves results only inside its allowed folders, symbolic links included:

```text
NALVERA_MCP_ALLOWED_DIRS  default: unset (the folder the app starts the server in; none when that is the root or the home folder).
                          Options: folders separated by : (; on Windows)
NALVERA_MAPPING           default: unset (one file per account in the configuration folder)
```

Results go to the folder `results` inside the first allowed folder, unless the assistant names another allowed folder. The record that links each scan code to your file name stays in the package's configuration folder on your computer, outside the allowed folders.

## What the assistant is told

```text
NALVERA_MCP_SHOW_NAMES  default: unset (scan codes, positions and folders only)
                        Options: 1 (also the original file names and the paths of saved results; the assistant's provider then sees them)
```

## The tools

| Tool | What it does | Cost |
|---|---|---|
| `list_models` | Models and tasks you may run, with prices and generation settings | free |
| `get_budgets` | Free tasks left, personal credits, each lab's balance | free |
| `estimate_cost` | What a task costs: for a scan file or folder, a voxel grid, or generation settings | free |
| `generate` | Queues a synthetic CT or brain MRI volume (MAISI-v2) | spends credits |
| `list_tasks` | Your tasks, newest first, with queue position and times | free |
| `get_task` | One task's state, queue position, times and error | free |
| `wait_for_task` | Waits up to 25 seconds for a task to change | free |
| `cancel_task` | Cancels a waiting or running task | refunds follow the website's rules |
| `get_text_result` | The text of a report task, up to 64 KB, marked as AI model output | free |
| `run_scan` | Anonymizes scans here, uploads them and queues a task per scan, in the background | spends credits |
| `get_run` | Progress of a `run_scan`: per scan, percent uploaded, task id, reason, credits | free |
| `download_result` | Saves a result in an allowed folder; never replaces a file | free |

## Spending

- `run_scan` and `generate` need a budget (`personal`, `lab:<id>` or `free`) and `max_credits`. Above the cap nothing is uploaded or queued, and the estimate comes back for the assistant to show you.
- A repeated call charges once: `generate` takes a `request_id`, and `run_scan` returns a scan already queued with the same model and task.
- A DICOM folder with several series is listed first, each series with its cost; the assistant then calls again with the series you choose.
