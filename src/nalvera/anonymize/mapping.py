"""The record on this computer: scan codes, original files, tasks, results.

The platform only ever sees a scan's code (``scan-7F3K2Q9A``). This CSV file,
on the user's computer, records which original file each code came from,
which tasks ran on it and where their results were saved, so the package can
show and save everything under the original names. It also records each
generation (its label, model, task and a digest of its settings) and each
saved result's part and size, so that running a notebook again queues and
downloads nothing twice. It is the key that re-identifies the scans: it
stays on the user's computer (by default in the package's configuration
folder, one file per account and platform), and a notice file beside it
says so.

One row per scan, task or result file. Columns: the scan's ``code``,
``original_path``, ``original_name``, ``source``, ``series``, ``label``,
``sha256`` and ``sha256_original``, ``created_at``, then ``task_id``,
``model``, ``task``, a saved result's ``result_path``, then ``kind``
(``scan`` or ``generation``; empty in rows of older files, meaning
``scan``), a generation's ``params_sha256``, and the saved result's
``result_part`` and ``result_size`` (bytes), the ``text_sha256`` of a
task's text query (a scan with another query is another task), and the
``corrections`` the DICOM converter made to a series (``gantry_tilt``,
``slice_spacing``; its voxels were resampled). A person can open it in a
spreadsheet.

Safety
  * Created with mode 0600 (owner only), as is every replacement.
  * Every change runs under an exclusive lock on a companion lock file, then
    writes a temporary file and atomically replaces the CSV, so two
    ``nalvera`` processes in one folder neither corrupt the file nor lose
    each other's rows, and a reader never sees a half-written file.
  * A cell starting with ``=``, ``+``, ``-``, ``@``, a tab or a carriage
    return is written with a leading ``'`` so a file name cannot become a
    spreadsheet formula (a leading ``'`` is doubled); ``rows()`` returns the
    original text.
  * Anything shaped like an API key is replaced by its prefix: the key never
    reaches this file.
"""
from __future__ import annotations

import contextlib
import csv
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

from .errors import AnonymizeError, text

COLUMNS = (
    "code", "original_path", "original_name", "source", "series", "label",
    "sha256", "sha256_original", "created_at",
    "task_id", "model", "task", "result_path",
    "kind", "params_sha256", "result_part", "result_size",
    "text_sha256", "corrections",
)
_SCAN_COLUMNS = COLUMNS[:9]
_TASK_COLUMNS = ("task_id", "model", "task")
# OWASP's list of characters that start a formula in common spreadsheets,
# plus the quote that marks an escaped cell.
_ESCAPE_FIRST = ("=", "+", "-", "@", "\t", "\r", "'")
_KEY_RX = re.compile(r"nvk_[0-9A-Za-z]{8}[0-9A-Za-z]+")
_ENCODING = "utf-8-sig"          # spreadsheets read non-ASCII names correctly

NOTICE = (
    "{name} links each scan code shown on Nalvera.AI to the original file on this "
    "computer. With it, anyone can tell whose scan a code is. Keep it on this computer: "
    "do not upload, email or share it.\n"
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_cell(value: object) -> str:
    cell = "" if value is None else str(value)
    cell = _KEY_RX.sub(lambda m: m.group(0)[:12] + "…[redacted]", cell)
    return "'" + cell if cell.startswith(_ESCAPE_FIRST) else cell


def _from_cell(cell: Optional[str]) -> str:
    cell = cell or ""
    return cell[1:] if cell.startswith("'") else cell


def _replace(source: str, target: str) -> None:
    # Windows refuses to replace a file another process has open (a reader
    # of this mapping, a spreadsheet); readers hold it for milliseconds.
    for _ in range(50):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt":
                raise
            time.sleep(0.1)
    os.replace(source, target)


class Mapping:
    """The mapping CSV at ``path`` (the package's default name is
    ``nalvera_scans.csv``). The file and its notice are created on the first
    record, not before; reading a missing file gives no rows."""

    def __init__(self, path: str):
        self.path = os.path.abspath(os.fspath(path))
        folder, name = os.path.split(self.path)
        self._lock_path = os.path.join(folder, "." + name + ".lock")
        stem = name[:-4] if name.lower().endswith(".csv") else name
        self.notice_path = os.path.join(folder, stem + ".README.txt")

    def __repr__(self) -> str:
        return f"Mapping({self.path!r})"

    # ---- reading -----------------------------------------------------------

    def rows(self) -> List[Dict[str, str]]:
        """Every row, oldest first, as dicts keyed by column name."""
        return self._read()[1]

    def find(self, code: Optional[str] = None, task_id: Optional[str] = None,
             original: Optional[str] = None) -> List[Dict[str, str]]:
        """The rows matching every criterion given (each default None: any):
        ``code`` a scan code, ``task_id`` a task id, ``original`` an
        original file (its name, or its path in any spelling)."""
        wanted = os.path.abspath(original) if original else None
        found = []
        for row in self.rows():
            if code is not None and row.get("code") != code:
                continue
            if task_id is not None and row.get("task_id") != task_id:
                continue
            if original is not None and original not in (row.get("original_name"),
                                                         row.get("original_path")) \
                    and wanted != row.get("original_path"):
                continue
            found.append(row)
        return found

    # ---- recording ---------------------------------------------------------

    def record_scan(self, prepared) -> Dict[str, str]:
        """Add a row for ``prepared`` (a ``PreparedScan``): its code,
        original path and name, source, series, label, the SHA-256 of the
        uploaded file and of the original file, and the time. Recording the
        same code twice keeps the first row."""
        row = {
            "kind": "scan",
            "code": prepared.code,
            "original_path": prepared.original_path,
            "original_name": prepared.original_name,
            "source": prepared.source,
            "series": getattr(prepared, "series", None),
            "label": getattr(prepared, "label", None),
            "sha256": prepared.sha256,
            "sha256_original": getattr(prepared, "sha256_original", None),
            "created_at": _now(),
            "corrections": ";".join(getattr(prepared, "corrections", None) or ()),
        }

        def change(rows: List[Dict[str, str]]) -> Dict[str, str]:
            for existing in rows:
                if existing.get("code") == prepared.code:
                    return existing
            rows.append(self._row(row))
            return rows[-1]
        return self._update(change)

    def record_task(self, code: Optional[str], task_id: str, model: str,
                    task: Optional[str], text_sha256: Optional[str] = None) -> Dict[str, str]:
        """Link the task ``task_id`` (of ``model`` and its ``task``, which may
        be None for the model's default) to the scan ``code``, with
        ``text_sha256``, the digest of its text query (default None: none).
        The scan's first task fills its row; each further task gets a row of
        its own. ``code`` may be None (a generation has no scan). Recording
        a task twice keeps the first."""
        def change(rows: List[Dict[str, str]]) -> Dict[str, str]:
            for existing in rows:
                if existing.get("task_id") == task_id:
                    return existing
            scans = [r for r in rows if code and r.get("code") == code]
            for existing in scans:
                if not existing.get("task_id"):
                    existing.update(self._row({"task_id": task_id, "model": model,
                                               "task": task, "text_sha256": text_sha256},
                                              base=existing))
                    return existing
            base = {k: scans[0].get(k, "") for k in _SCAN_COLUMNS + ("corrections",)} \
                if scans else {"code": code or "", "created_at": _now()}
            base.update({"task_id": task_id, "model": model, "task": task,
                         "text_sha256": text_sha256})
            rows.append(self._row(base))
            return rows[-1]
        return self._update(change)

    def record_generation(self, task_id: str, model: str, task: str, label: Optional[str],
                          params_sha256: str) -> Dict[str, str]:
        """Add a row for a generation: the task ``task_id`` of ``model`` and
        ``task``, its ``label`` (may be None) and ``params_sha256``, the
        SHA-256 of its settings. Recording a task twice keeps the first
        row."""
        def change(rows: List[Dict[str, str]]) -> Dict[str, str]:
            for existing in rows:
                if existing.get("task_id") == task_id:
                    return existing
            rows.append(self._row({"kind": "generation", "label": label, "task_id": task_id,
                                   "model": model, "task": task,
                                   "params_sha256": params_sha256, "created_at": _now()}))
            return rows[-1]
        return self._update(change)

    def record_result(self, task_id: str, result_path: str, part: Optional[str] = None,
                      size: Optional[int] = None) -> Dict[str, str]:
        """Record ``result_path``, a result file saved for the task
        ``task_id``: one ``part`` of it (None: the whole result) of ``size``
        bytes. A task's first result fills its row; each further file
        (another part) gets a row of its own; the same file again updates
        its part and size."""
        result_path = os.path.abspath(os.fspath(result_path))
        details = {"result_part": part or "", "result_size": "" if size is None else str(size)}

        def change(rows: List[Dict[str, str]]) -> Dict[str, str]:
            mine = [r for r in rows if r.get("task_id") == task_id]
            for existing in mine:
                if existing.get("result_path") == result_path:
                    existing.update(details)
                    return existing
            for existing in mine:
                if not existing.get("result_path"):
                    existing["result_path"] = result_path
                    existing.update(details)
                    return existing
            base = dict(mine[0]) if mine else {"task_id": task_id, "created_at": _now()}
            base["result_path"] = result_path
            base.update(details)
            rows.append(self._row(base))
            return rows[-1]
        return self._update(change)

    # ---- storage -----------------------------------------------------------

    @staticmethod
    def _row(values: Dict[str, object], base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        row = dict(base or {})
        for name in COLUMNS:
            row.setdefault(name, "")
        for name, value in values.items():
            row[name] = "" if value is None else str(value)
        return row

    def _read(self):
        """(column names, rows) of the file; an absent file has none."""
        try:
            with open(self.path, encoding=_ENCODING, newline="") as fh:
                reader = csv.DictReader(fh)
                names = list(reader.fieldnames or [])
                rows = [{k: _from_cell(v) for k, v in raw.items() if k is not None}
                        for raw in reader]
        except FileNotFoundError:
            return list(COLUMNS), []
        except (OSError, UnicodeDecodeError, csv.Error):
            raise AnonymizeError("mapping_unreadable") from None
        if "code" not in names or "task_id" not in names:
            # Not a mapping this package wrote: never overwrite it.
            raise AnonymizeError("mapping_unreadable")
        # Columns a user added in a spreadsheet are kept, after ours.
        extra = [n for n in names if n not in COLUMNS]
        return list(COLUMNS) + extra, rows

    def _update(self, change):
        folder = os.path.dirname(self.path)
        os.makedirs(folder, mode=0o700, exist_ok=True)
        with self._locked():
            names, rows = self._read()
            result = change(rows)
            self._write(names, rows)
        self._write_notice()
        return dict(result)

    def _write(self, names: List[str], rows: List[Dict[str, str]]) -> None:
        folder, name = os.path.split(self.path)
        fd, temp = tempfile.mkstemp(prefix="." + name + ".", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding=_ENCODING, newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: _to_cell(row.get(k, "")) for k in names})
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(temp, 0o600)
            _replace(temp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(temp)
            raise

    def _write_notice(self) -> None:
        if os.path.exists(self.notice_path):
            return
        name = os.path.basename(self.path)
        notice = text("mapping.notice", NOTICE)
        try:
            body = notice.format(name=name)
        except (KeyError, IndexError, ValueError):
            body = NOTICE.format(name=name)
        with contextlib.suppress(OSError):
            with open(self.notice_path, "x", encoding="utf-8") as fh:
                fh.write(body)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "nt":
                import msvcrt
                while True:
                    try:
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)   # waits up to 10 s per try
                        break
                    except OSError:
                        continue
                try:
                    yield
                finally:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
