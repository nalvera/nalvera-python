"""A stand-in for ``nalvera.anonymize`` with the same interface.

The client's tests replace the real module with this one, so they test the
client alone. Behaviour:

* a file whose bytes start with ``NOTASCAN`` raises ``AnonymizeError("not_a_scan")``;
* a missing path raises ``AnonymizeError("not_found")``;
* a folder gives one scan per file inside it (a "DICOM folder", one series each);
* any other file is copied unchanged to ``<workdir>/<code>.nii.gz``;
* ``SHAPE=AxBxC;`` inside a file sets its shape (default (4, 4, 4));
* ``sha256`` is the hash of the bytes (the same file always gives the same
  hash, as the real module's deterministic output does); ``sha256_original``
  is that hash for a file, None for a "DICOM" scan.

``prepared`` lists every PreparedScan made, so a test can check that the
temporary copies were removed. ``dicom.list_dicom_files`` finds ``.dcm``
files, standing in for the real module's finder.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
COLUMNS = ["code", "original_path", "original_name", "source", "series", "label", "sha256",
           "sha256_original", "created_at", "task_id", "model", "task", "result_path",
           "kind", "params_sha256", "result_part", "result_size", "text_sha256",
           "corrections"]


class AnonymizeError(Exception):
    def __init__(self, code: str, detail: Optional[str] = None):
        self.code = code
        self.detail = detail
        super().__init__(code)


@dataclass(frozen=True)
class PreparedScan:
    code: str
    filename: str
    path: str
    size: int
    sha256: str
    shape: Tuple[int, ...]
    original_path: str
    original_name: str
    source: str
    series: Optional[str] = None
    label: Optional[str] = None
    sha256_original: Optional[str] = None
    corrections: Tuple[str, ...] = ()


def _new_code() -> str:
    return "scan-" + "".join(secrets.choice(_ALPHABET) for _ in range(8))


def _shape(data: bytes) -> Tuple[int, ...]:
    match = re.search(rb"SHAPE=(\d+)x(\d+)x(\d+);", data)
    return tuple(int(g) for g in match.groups()) if match else (4, 4, 4)


def _one(source: Path, workdir: str, label: Optional[str], series: Optional[str],
         original_name: str, original_path: str, avoid=None) -> PreparedScan:
    data = source.read_bytes()
    if data.startswith(b"NOTASCAN"):
        raise AnonymizeError("not_a_scan")
    code = _new_code()
    for _ in range(100):
        if not avoid or code not in avoid:
            break
        code = _new_code()
    out = Path(workdir) / f"{code}.nii.gz"
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    scan = PreparedScan(code=code, filename=out.name, path=str(out), size=len(data),
                        sha256=digest, shape=_shape(data), original_path=original_path,
                        original_name=original_name, source="dicom" if series else "nifti",
                        series=series, label=label,
                        sha256_original=None if series else digest)
    prepared.append(scan)
    return scan


prepared: List[PreparedScan] = []


def prepare(path: Any, *, workdir: str, label: Optional[str] = None,
            max_bytes: Optional[int] = None, avoid=None, check=None) -> List[PreparedScan]:
    source = Path(path)
    if not source.exists():
        raise AnonymizeError("not_found")
    if check is not None:
        for item in ([f for f in sorted(source.iterdir()) if f.is_file()]
                     if source.is_dir() else [source]):
            check(os.path.realpath(item))
    if source.is_dir():
        return [_one(f, workdir, label, f.stem.split(".")[0], source.name, str(source),
                     avoid=avoid)
                for f in sorted(source.iterdir()) if f.is_file()]
    return [_one(source, workdir, label, None, source.name, str(source), avoid=avoid)]


def list_dicom_files(folder: str) -> List[str]:
    out = []
    for root, _dirs, files in os.walk(folder):
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".dcm")]
    return out


class Mapping:
    """One row per scan; a task or a result adds a row copying the scan's
    columns (the real module decides the real layout). Like the real
    one, the file is created on the first record, and a CSV it did not
    write is refused with ``mapping_unreadable``."""

    _lock = threading.Lock()

    def __init__(self, path: str):
        self.path = Path(path)

    def _append(self, row: Dict[str, Any]) -> None:
        with self._lock:
            new = not self.path.exists()
            fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS)
                if new:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") or "" for k in COLUMNS})

    def rows(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock, open(self.path, newline="") as fh:
            reader = csv.DictReader(fh)
            if "code" not in (reader.fieldnames or []) or "task_id" not in (reader.fieldnames
                                                                          or []):
                raise AnonymizeError("mapping_unreadable")
            return [dict(r) for r in reader]

    def record_scan(self, scan: PreparedScan) -> None:
        if any(r["code"] == scan.code for r in self.rows()):
            return
        self._append({"code": scan.code, "original_path": scan.original_path,
                      "original_name": scan.original_name, "source": scan.source,
                      "series": scan.series or "", "label": scan.label or "",
                      "sha256": scan.sha256, "sha256_original": scan.sha256_original or "",
                      "created_at": "2026-11-15T10:00:00Z",
                      "corrections": ";".join(getattr(scan, "corrections", ()) or ())})

    def record_task(self, code: str, task_id: str, model: str, task: str,
                    text_sha256: Optional[str] = None) -> None:
        rows = self.rows()
        if any(r["task_id"] == task_id for r in rows):
            return
        base = next((r for r in rows if r["code"] == code), {"code": code})
        self._append({**base, "task_id": task_id, "model": model, "task": task,
                      "result_path": "", "text_sha256": text_sha256 or ""})

    def record_generation(self, task_id: str, model: str, task: str, label: Optional[str],
                          params_sha256: str) -> None:
        if any(r["task_id"] == task_id for r in self.rows()):
            return
        self._append({"kind": "generation", "task_id": task_id, "model": model, "task": task,
                      "label": label or "", "params_sha256": params_sha256,
                      "created_at": "2026-11-15T10:00:00Z"})

    def record_result(self, task_id: str, result_path: str, part: Optional[str] = None,
                      size: Optional[int] = None) -> None:
        base = next((r for r in self.rows() if r["task_id"] == task_id), {"task_id": task_id})
        self._append({**base, "result_path": str(os.path.abspath(result_path)),
                      "result_part": part or "", "result_size": "" if size is None else size})

    def find(self, code: Optional[str] = None, task_id: Optional[str] = None,
             original: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for row in self.rows():
            if code is not None and row["code"] != code:
                continue
            if task_id is not None and row["task_id"] != task_id:
                continue
            if original is not None and original not in (row["original_name"],
                                                         row["original_path"]):
                continue
            out.append(row)
        return out


def as_module() -> ModuleType:
    """This stand-in as a module object named ``nalvera.anonymize``."""
    module = ModuleType("nalvera.anonymize")
    for name in ("AnonymizeError", "PreparedScan", "prepare", "Mapping", "prepared"):
        setattr(module, name, globals()[name])
    module.dicom = SimpleNamespace(list_dicom_files=list_dicom_files)
    return module
