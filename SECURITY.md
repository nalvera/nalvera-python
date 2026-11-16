# Security

## Reporting a vulnerability

Email contact@nalvera.ai with "Security" in the subject. Describe the problem and how to reproduce it. Do not open a public issue for a security problem.

Only the latest release receives fixes. The package needs Python 3.10 or later, because the releases of `requests` and `urllib3` that fix their published advisories need it.

## Where your key is stored

- A key you give to `nalvera.login(email, api_key)` or `nalvera.Client(api_key=...)` stays in the memory of that Python process until `nalvera.logout()` or `Client.close()`, which drop it; the client then sends no request. A key typed into a notebook cell is saved in the `.ipynb` file and in IPython's history file: `nalvera.login()` with no arguments asks for it with hidden input and saves nothing in the notebook.
- With `remember=True`, `nalvera.login` stores the key. With `nalvera[keyring]` installed it goes to the system keychain: the macOS Keychain, the Windows Credential Locker or the Linux Secret Service. A keyring library that keeps keys in a plain file is not used as a keychain.
- Without it, the key goes to `credentials.json` in the config folder: `~/.config/nalvera` on Linux and macOS (`XDG_CONFIG_HOME` or `NALVERA_CONFIG_DIR` change it) and `%APPDATA%\nalvera` on Windows. The file has mode 0600 in a folder with mode 0700, and it is written under a random name and moved into place. On Windows the file permissions are not changed; the file is in your user profile, which Windows restricts to your account. A config folder that another user owns or that others can write is refused.
- A stored key is sent only to the platform it was stored for.
- A key given through `NALVERA_API_KEY` is not stored by the package. Fill it from a secret store at run time.
- `nalvera.logout(forget=True)` removes the stored key. Revoke a key you no longer use under Settings → API at https://nalvera.ai.

## How the package handles the key

- The key is sent only in the `Authorization` header, only over HTTPS, and only to the configured address (`https://nalvera.ai` by default). A redirect is never followed. Plain HTTP is accepted for `localhost` and `127.0.0.1` only.
- The package never puts the key in output, logs, error messages, `repr`, a notebook table, the variables of its own traceback frames or the record of scans. Anything shaped like a key is replaced by its first 12 characters and `…[redacted]`.
- A `.netrc` entry never replaces the key.
- The package checks the key's format and checksum on your computer before it sends a request.
- The platform stores only a one-way keyed hash of a key and shows the key once. A key works on the API only, and every other part of the platform refuses it.
- A key ends when the account's password is changed or reset. A key that appears in the web address of an API request is revoked.
- The files in `examples/` hold no key.
- The package sends requests only to the configured address and collects no usage data.

## What leaves your computer

- The anonymized scan as a `.nii.gz` file, or inside an encrypted container when encryption is on.
- A neutral code as its name, for example `scan-7F3K2Q9A`.
- A label, only if you give one with `label=`. Choose one that does not identify a patient.
- The text of a query, only if you give one with `text=` for a task that needs one. Choose one that does not identify a patient.
- Your account email and the key, in request headers, and the package version in the `User-Agent` header.

## What stays on your computer

- The original file names and paths.
- NIfTI header text and header extensions.
- DICOM files and their tags. A DICOM folder is converted to NIfTI on your computer, in a private temporary folder, and no DICOM tag is uploaded.
- The record that links each code to its original file: by default one file per account in the `records` folder of the config folder, outside the working folder. Keep it private.

A notebook that shows a table of scans saves their original names in its outputs. Clear the outputs before you share the notebook.

## Temporary files

A run writes the anonymized copies into a private temporary folder (mode 0700, named `nalvera-run-*`, in the system's temporary folder), and a DICOM conversion uses `nalvera-dicom-*` the same way. The package deletes them when the upload is done, when a run fails and after one Ctrl-C.

A second Ctrl-C, a process killed by a signal the package does not handle (SIGKILL, a notebook kernel shut down hard) or a power loss can leave a folder behind. The next `nalvera.run`, `nalvera.estimate_run`, `nalvera.save_anonymized` or `nalvera.login`, and the start of the assistant server, remove such folders when they are yours and older than 24 hours.

`nalvera.save_anonymized(paths, out)` copies the anonymized files into `out`. When the computer stops during that copy across file systems, a partial copy can stay in `out`.

## What the anonymizer leaves in the image

A face in a head scan and text burned into an image stay in the file. The package does not remove them. The DICOM converter leaves out screenshots, dose reports and scouts.

A NIfTI scan keeps its voxel values and geometry exactly, and a check after writing confirms it before anything is sent. A DICOM series is uploaded as the converted volume: where the converter resampled a tilted or unevenly spaced series, the voxel values and geometry are those of the resampled volume, and the run says so.

## Encrypted uploads

Encryption is off by default. `encrypt=True` switches it on for one run, `Client(encrypt=True)` for one client and `NALVERA_ENCRYPT=1` for every run. It is required when the account's Encrypt uploads switch under Settings → API is on.

- Each anonymized scan is encrypted on your computer for the platform's public key, in an NVLT v2 container (X25519, HKDF-SHA256, AES-256-GCM), and the unencrypted temporary copy is deleted at once.
- The package carries the SHA-256 fingerprints of the platform keys it trusts and encrypts only for a listed key. A key swapped on the way is never used, and the run stops with an instruction to upgrade the package. A server on `localhost` or `127.0.0.1` is trusted without a fingerprint, for development.
- When the package cannot encrypt, nothing is uploaded. It never falls back to an unencrypted upload.
- The encryption is a second layer on top of HTTPS. The platform decrypts the scan to run the AI task and stores it afterwards as it does any upload.

## The local assistant server

`python -m nalvera.mcp` starts a server for an AI assistant on your computer.

- It reads scans and writes results only inside the folders in `NALVERA_MCP_ALLOWED_DIRS`, symbolic links included. A path outside them is refused before anything is read. Without the setting, the allowed folder is the folder the server starts in, unless that is the root or the home folder.
- By default it tells the assistant scan codes, positions and folders only. `NALVERA_MCP_SHOW_NAMES=1` also tells it the original file names and the paths of saved results, and the assistant's provider then sees them.
- `download_result` never replaces a file.
- `generate` and `run_scan` spend credits only with a budget and a `max_credits` cap that the person gave. A run whose estimate passes the cap uploads nothing, and a task is queued only while the credits charged so far plus its estimate stay within the cap.
- Task names, labels, error texts and results come back in named data fields. Treat them as data.
- The key is no tool argument and never appears in a result, a log or an error.
- Only protocol messages are written to standard output.
