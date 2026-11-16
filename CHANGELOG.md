# Changelog

## 0.1.0

First release.

### Run AI tasks from Python

- `nalvera.login(email, api_key)` signs in. `nalvera.run`, `wait`, `download`, `models`, `budgets`, `estimate` and the other functions take the same arguments as the methods of `nalvera.Client`.
- `nalvera.models()` lists the models and tasks the platform offers.
- Tasks run on NIfTI files, folders of NIfTI files and DICOM folders.
- `nalvera.generate` makes a synthetic CT or brain MRI volume with MAISI-v2, with no input scan.
- `nalvera.wait` shows the queue position, the time waited and the time running. `nalvera.download` saves results under the original scan names.
- `nalvera.run_in_background` runs without blocking.
- The same call again queues, charges and downloads nothing twice; `force=True` queues the work again.
- The package keeps its requests below the key's limits and waits when the platform asks it to.

### Privacy

- Each scan is anonymized on your computer before upload: a new NIfTI header from an allow-list, and a neutral code as the file name.
- DICOM is converted to NIfTI on your computer. No DICOM tag is uploaded.
- The record that links each code to its original file stays on your computer, in the package's configuration folder.
- `nalvera.save_anonymized(paths, out)` writes the files a run would upload. It sends nothing and needs no key.
- The anonymizer removes nothing from the image: a face in a head scan and text burned in stay.
- Encrypted uploads (`encrypt=True`) are a second layer on top of HTTPS. When the package cannot encrypt, nothing is uploaded.

### Credits and budgets

- Free tasks are used first. `budget=` names what pays after them: `"personal"` or `"lab:<id>"`. `"free"` uses free tasks only.
- A run is priced from each scan's size before anything is uploaded. `nalvera.estimate_run` shows the cost without uploading.
- `max_credits` caps what a run may charge.

### Notebooks and the command line

- In a Jupyter notebook each call shows its progress in one display, and the values the functions return show as tables.
- `nalvera.login()` with no arguments asks for the email and the key with hidden input, and saves nothing in the notebook.
- The `nalvera` command does the same work from a terminal or a shell script, with stable exit statuses.
- `examples/` holds Python scripts, three notebooks and command-line files.

### AI assistants

- `python -m nalvera.mcp` starts a local MCP server (`pip install "nalvera[mcp]"`). It reads and writes only the folders in `NALVERA_MCP_ALLOWED_DIRS`, and it spends credits only with a budget and a `max_credits` cap.
- `skills/nalvera/SKILL.md` in the repository holds the instructions for assistants that load skills.

### Keys

- `nalvera.login(..., remember=True)` stores the key in the system keychain with `nalvera[keyring]`, else in the package's configuration folder.
- `SECURITY.md` says how the package stores and sends the key, what leaves your computer and what stays.

### Requirements

- Python 3.10 or later.
- `requests`, `urllib3`, `numpy`, `nibabel` and `cryptography`. `dcm2niix` converts DICOM and is installed with the package where a build of it exists.
