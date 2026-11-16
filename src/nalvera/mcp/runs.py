"""``run_scan`` and ``get_run``: the package's own pipeline in the background.

A run is one ``run_scan`` call. Its thread does what ``Client.run`` does
(anonymize, verify, optionally encrypt, upload, queue, with the re-run
protection, the spending cap and the write-ahead mapping of the package),
through a ``RunClient`` that adds four things the server needs:

* the decision comes back fast: the run is anonymized and priced first, and
  the tool call that started it returns as soon as the run is over the cap,
  needs a confirmation, or starts uploading;
* a DICOM folder with several scans queues nothing until the series are
  confirmed (``series``), and then only those series;
* uploads of different runs go one run at a time, so one assistant cannot
  fill the account's upload slots;
* the run's state is written to a small file (``<config>/mcp-runs``), and
  the scans' codes link it to the mapping file, so ``get_run`` still answers
  after the server restarted.

Nothing here prints or reads a terminal: progress arrives through the
client's reporter. A run file holds what ``get_run`` answers (each scan's
code and position, paths relative to the allowed folder; original names
only with ``NALVERA_MCP_SHOW_NAMES``), never a key.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .. import budget as budget_mod
from .. import client as client_mod
from ..errors import ConfigError, CreditCapExceeded, NalveraError
from ..report import Event, Reporter

RUN_ID_RX = re.compile(r"run-[0-9a-f]{12}")
ACTIVE_STATES = frozenset({"preparing", "waiting", "uploading"})
#: Run files older than this are removed when the server starts.
KEEP_RUN_FILES_SECONDS = 30 * 24 * 3600
_CONFIRM, _OVER_CAP, _IN_PROGRESS = "series_to_confirm", "over_max_credits", "in_progress"

ErrorFields = Callable[[BaseException], Dict[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round(value: Any) -> float:
    try:
        return round(float(value or 0.0), 2)
    except (TypeError, ValueError):
        return 0.0


class _ProgressReporter(Reporter):
    """Keeps the upload percentage of each scan by its shown name; shows
    nothing and never asks."""

    def __init__(self) -> None:
        self.percent: Dict[str, int] = {}
        self._lock = threading.Lock()

    def message(self, event: Event) -> None:
        name = event.fields.get("name")
        if not isinstance(name, str):
            return
        with self._lock:
            if event.key == "upload.progress":
                try:
                    self.percent[name] = max(self.percent.get(name, 0),
                                             int(event.fields.get("percent") or 0))
                except (TypeError, ValueError):
                    pass
            elif event.key == "upload.done":
                self.percent[name] = 100

    def get(self, name: str) -> Optional[int]:
        with self._lock:
            return self.percent.get(name)


class RunClient(client_mod.Client):
    """The package's ``nalvera.Client`` with the hooks of one run.
    The overrides pass every argument through, so they follow changes of
    the methods they wrap."""

    def __init__(self, *args: Any, run: Run, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._mcp_run = run

    def _new_job(self, *args: Any, **kwargs: Any):
        job = super()._new_job(*args, **kwargs)
        self._mcp_run.attach(job)
        return job

    def _prepare_all(self, job: Any, *args: Any, **kwargs: Any) -> None:
        super()._prepare_all(job, *args, **kwargs)
        self._mcp_run.prepared(self, job)

    def _pay(self, job: Any, *args: Any, **kwargs: Any) -> None:
        self._mcp_run.priced(job)
        super()._pay(job, *args, **kwargs)
        self._mcp_run.paid(self)


class Run:
    """One ``run_scan`` call and what became of it."""

    def __init__(self, manager: RunManager, request: Dict[str, Any],
                 paths: List[str], shown: Dict[str, str]):
        self.manager = manager
        self.id = "run-" + secrets.token_hex(6)
        self.created_at = _now()
        self.request = request
        self.key = json.dumps(request, sort_keys=True)
        self.paths = paths                       # resolved, inside the allowed folders
        self.shown = shown                       # resolved path -> as the assistant gave it
        self.state = "preparing"
        self.reason: Optional[str] = _IN_PROGRESS
        self.error: Optional[Dict[str, Any]] = None
        self.estimate: Optional[Dict[str, Any]] = None
        self.series_to_confirm: Optional[List[Dict[str, Any]]] = None
        self.scans: List[Dict[str, Any]] = []
        self.charged_credits = 0.0
        self.decided = threading.Event()
        self.finished = threading.Event()
        self.progress = _ProgressReporter()
        self.client: Optional[RunClient] = None
        self.thread: Optional[threading.Thread] = None
        self._jobs: List[Any] = []
        self._job: Any = None
        self._surveying = False                  # the estimate before a confirmation
        self._holds_slot = False
        self._lock = threading.RLock()

    # ── hooks called from the client (run thread) ─────────────────────────────
    def attach(self, job: Any) -> None:
        with self._lock:
            self._jobs.append(job)
            self._job = job

    def prepared(self, client: RunClient, job: Any) -> None:
        """After anonymizing: keep only the confirmed series, then record the
        scans' codes (the link to the mapping file after a restart)."""
        if self._surveying:
            return
        wanted = self.request.get("series")
        if wanted is not None:
            present = {str(e.prepared.series) for e in job.entries
                       if e.ready and e.source_kind == "dicom_folder"
                       and e.prepared.series is not None}
            missing = sorted(set(wanted) - present)
            if missing:
                raise ConfigError(self.manager.texts["errors"]["series_not_found"],
                                  code="series_not_found", detail=", ".join(missing))
            kept = []
            for entry in job.entries:
                scan = entry.prepared
                if entry.ready and entry.source_kind == "dicom_folder" \
                        and str(scan.series) not in wanted:
                    client._remove_prepared(entry)
                    continue
                kept.append(entry)
            job.entries[:] = kept
        self.save()

    def priced(self, job: Any) -> None:
        """Every scan is priced: the estimate the answer shows."""
        ready = [e for e in job.entries if e.ready]
        costs = [float(e.credits or 0.0) for e in ready]
        free = budget_mod.free_cover(client_mod._free_state({}, job.estimate), len(ready))
        budget = self.request["budget"]
        with self._lock:
            self.estimate = {
                "scans": len(ready), "credits": _round(sum(costs)), "free_tasks": free,
                "paid_credits": 0.0 if budget == budget_mod.FREE else _round(sum(costs[free:])),
                "budget": budget, "max_credits": float(self.request["max_credits"])}

    def paid(self, client: RunClient) -> None:
        """The run is within the cap: uploads start once no other run of
        this server is uploading."""
        self._set_state("waiting")
        self.decided.set()
        slot = self.manager.upload_slot
        while not slot.acquire(timeout=0.2):
            client._transport.pause(0)                # raises Cancelled once stopped
        self._holds_slot = True
        self._set_state("uploading")

    # ── the thread ────────────────────────────────────────────────────────────
    def work(self) -> None:
        client = self.client
        assert client is not None
        request = self.request
        try:
            if request.get("series") is None and self.manager.has_dicom(self.paths):
                if self._needs_confirmation(client):
                    return
            client.run(self.paths, request["model"], request.get("task"),
                       budget=request["budget"], label=request.get("label"),
                       text=request.get("text"), yes=True, encrypt=None,
                       max_credits=float(request["max_credits"]),
                       force=bool(request.get("force")))
            self._end("done")
        except CreditCapExceeded:
            self._end("over_max_credits", reason=_OVER_CAP)
        except KeyboardInterrupt:                     # RunInterrupted: stopped on shutdown
            self._end("interrupted")
        except BaseException as error:
            self._end("failed", error=self.manager.error_fields(error))
        finally:
            if self._holds_slot:
                self._holds_slot = False
                self.manager.upload_slot.release()
            self.decided.set()
            self.finished.set()
            try:
                client.close()
            except Exception:
                pass

    def _needs_confirmation(self, client: RunClient) -> bool:
        """A DICOM folder that gives several scans is listed with each one's
        cost; nothing is uploaded until series are chosen."""
        self._surveying = True
        try:
            found = client.estimate_run(self.paths, self.request["model"],
                                        self.request.get("task"),
                                        force=bool(self.request.get("force")))
        finally:
            self._surveying = False
            self._job = None
        per_folder: Dict[str, int] = {}
        for scan in found.scans:
            if scan.series is not None:
                per_folder[scan.original_path] = per_folder.get(scan.original_path, 0) + 1
        if not any(n > 1 for n in per_folder.values()):
            return False
        budget = self.request["budget"]
        listed = []
        for s in found.scans:
            if s.series is None:
                continue
            item = {"path": self.manager.relative(s.original_path), "series": str(s.series),
                    "shape": [int(n) for n in s.shape], "credits": _round(s.credits)}
            if self.manager.show_names:
                item["name"] = s.original_name
            listed.append(item)
        with self._lock:
            self.series_to_confirm = listed
            self.estimate = {
                "scans": len(found.scans), "credits": _round(found.credits),
                "free_tasks": int(found.free_count),
                "paid_credits": 0.0 if budget == budget_mod.FREE else _round(found.paid_credits),
                "budget": budget, "max_credits": float(self.request["max_credits"])}
            self.scans = [self._skipped_view(s, n) for n, s in enumerate(found.skipped, 1)] + \
                [self._failure_view(f, n) for n, f in enumerate(found.failed,
                                                                len(found.skipped) + 1)]
        self._end("needs_confirmation", reason=_CONFIRM)
        return True

    def _set_state(self, state: str) -> None:
        with self._lock:
            self.state = state
        self.save()

    def _end(self, state: str, *, reason: Optional[str] = None,
             error: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if self._job is not None:
                self.scans = self._live_scans()
                self.charged_credits = _round(getattr(self._job, "charged", 0.0))
            # Scans the run never queued, by why it ended.
            leftover = {"interrupted": ("interrupted", None),
                        "failed": ("not_queued", (error or {}).get("code")),
                        "over_max_credits": ("not_queued", "max_credits_exceeded")}.get(state)
            if leftover is not None:
                for scan in self.scans:
                    if scan["state"] in ("ready", "uploading"):
                        scan.update(state=leftover[0], reason=leftover[1], percent_uploaded=None)
            self.state, self.reason, self.error = state, reason, error
        self.save()

    # ── views ─────────────────────────────────────────────────────────────────
    def _live_scans(self) -> List[Dict[str, Any]]:
        job = self._job
        if job is None:
            return list(self.scans)
        entries = sorted(list(job.entries), key=lambda e: e.order)
        return [self._entry_view(e, n) for n, e in enumerate(entries, 1)]

    def _name(self, original: Any, code: Any, position: int) -> str:
        """How a scan is named to the assistant: its code (else its position
        in the run); the original name only with NALVERA_MCP_SHOW_NAMES."""
        if self.manager.show_names and original:
            return os.path.basename(str(original).rstrip(os.sep)) or str(original)
        return str(code) if code else self.manager.texts_position(position)

    def _entry_view(self, entry: Any, position: int) -> Dict[str, Any]:
        scan = entry.prepared
        code = getattr(scan, "code", None)
        view: Dict[str, Any] = {
            "name": self._name(getattr(scan, "original_name", None) or entry.path, code,
                               position),
            "position": position,
            "code": code, "series": _text(getattr(scan, "series", None)),
            "state": "ready", "percent_uploaded": None, "task_id": None, "reason": None,
            "detail": None, "charged": None}
        if entry.failure is not None:
            view.update(state="failed", reason=entry.failure.code,
                        detail=self._detail(entry.failure))
        elif entry.skipped is not None:
            view.update(state="skipped", reason=entry.skipped.reason,
                        task_id=entry.skipped.task_id)
        elif entry.not_queued is not None:
            view.update(state="not_queued", reason=entry.not_queued.code)
        elif entry.task is not None:
            task = entry.task
            view.update(state="queued", task_id=task.id)
            if task.charged_now:               # also a replay of this run's own lost answer
                view["charged"] = _charged(task.charged)
            else:
                view["reason"] = "already_queued"
        elif entry.upload_path or entry.upload_id:
            view.update(state="uploading", percent_uploaded=self.progress.get(entry.name) or 0)
        return view

    def _skipped_view(self, skipped: Any, position: int) -> Dict[str, Any]:
        return {"name": self._name(skipped.original_name, None, position),
                "position": position, "code": None,
                "series": None, "state": "skipped", "percent_uploaded": None,
                "task_id": skipped.task_id, "reason": skipped.reason, "detail": None,
                "charged": None}

    def _failure_view(self, failure: Any, position: int) -> Dict[str, Any]:
        return {"name": self._name(failure.original_name, None, position),
                "position": position, "code": None,
                "series": None, "state": "failed", "percent_uploaded": None, "task_id": None,
                "reason": failure.code, "detail": self._detail(failure), "charged": None}

    def _detail(self, failure: Any) -> Optional[str]:
        detail = _failure_detail(failure)
        return self.manager.scrub(detail) if detail else None

    def view(self) -> Dict[str, Any]:
        """The answer of ``run_scan`` and ``get_run`` (the manifest's schema)."""
        with self._lock:
            active = self.state in ACTIVE_STATES
            live = active and self._job is not None and not self._surveying
            scans = self._live_scans() if live else [dict(s) for s in self.scans]
            charged = _round(getattr(self._job, "charged", 0.0)) if live \
                else self.charged_credits
            reason = _IN_PROGRESS if active else self.reason
            return {"run_id": self.id, "state": self.state, "reason": reason,
                    "hint": self.manager.texts["reasons"].get(reason) if reason else None,
                    "error": self.error, "estimate": self.estimate,
                    "series_to_confirm": self.series_to_confirm, "scans": scans,
                    "charged_credits": charged}

    def record(self) -> Dict[str, Any]:
        """What the run file holds."""
        data = self.view()
        data.update(created_at=self.created_at, boot=self.manager.boot, pid=os.getpid(),
                    model=self.request["model"], task=self.request.get("task"))
        return data

    def save(self) -> None:
        self.manager.store.save(self.id, self.record())

    def workdirs(self) -> List[str]:
        with self._lock:
            return [str(job.workdir) for job in self._jobs if getattr(job, "workdir", None)]


class RunStore:
    """Run files in ``folder`` (mode 0700), one JSON file per run (mode
    0600), each written to a temporary name and renamed."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._lock = threading.Lock()

    def _path(self, run_id: str) -> Path:
        if not RUN_ID_RX.fullmatch(run_id or ""):
            raise KeyError(run_id)
        return self.folder / f"{run_id}.json"

    def save(self, run_id: str, record: Dict[str, Any]) -> None:
        with self._lock:
            try:
                self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
                handle, temp = tempfile.mkstemp(dir=str(self.folder), prefix=".run-")
                try:
                    with os.fdopen(handle, "w", encoding="utf-8") as fh:
                        json.dump(record, fh, ensure_ascii=False)
                    os.chmod(temp, 0o600)
                    os.replace(temp, self._path(run_id))
                except BaseException:
                    try:
                        os.remove(temp)
                    except OSError:
                        pass
                    raise
            except OSError:
                pass                       # a run goes on without its file

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        try:
            with open(self._path(run_id), encoding="utf-8") as fh:
                data = json.load(fh)
        except (KeyError, OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def prune(self, max_age: float = KEEP_RUN_FILES_SECONDS) -> None:
        try:
            names = os.listdir(self.folder)
        except OSError:
            return
        now = time.time()
        for name in names:
            path = self.folder / name
            try:
                if now - path.stat().st_mtime > max_age:
                    path.unlink()
            except OSError:
                pass


class RunManager:
    """Every run of this server process."""

    def __init__(self, *, store: RunStore, mapping: Path, texts: Dict[str, Any],
                 make_client: Callable[[Run], RunClient], error_fields: ErrorFields,
                 has_dicom: Callable[[Sequence[str]], bool], decision_wait: float = 20.0,
                 scrub: Callable[[str], str] = str, show_names: bool = False,
                 relative: Callable[[str], str] = os.path.basename):
        self.store = store
        self.mapping = Path(mapping)
        self.texts = texts
        self.make_client = make_client
        self.error_fields = error_fields
        self.has_dicom = has_dicom
        self.scrub = scrub
        self.show_names = show_names
        self.relative = relative
        self.decision_wait = decision_wait
        self.boot = secrets.token_hex(8)
        self.upload_slot = threading.Lock()
        self.runs: Dict[str, Run] = {}
        self._lock = threading.Lock()
        store.prune()

    def texts_position(self, position: int) -> str:
        """The name of a scan known by its position only (no code yet)."""
        from .. import messages
        return messages.text("mcp.scan_position", position=position)

    def start(self, request: Dict[str, Any], paths: List[str],
              shown: Dict[str, str]) -> Dict[str, Any]:
        """Start a run, or return the active run made by the same request (a
        retry), then wait up to ``decision_wait`` seconds for its decision."""
        run = Run(self, request, paths, shown)
        with self._lock:
            same = next((r for r in self.runs.values()
                         if r.key == run.key and r.state in ACTIVE_STATES), None)
            if same is None:
                run.client = self.make_client(run)           # may raise: nothing started
                self.runs[run.id] = run
                run.save()
                run.thread = threading.Thread(target=run.work, name=f"nalvera-mcp-{run.id}",
                                              daemon=True)
                run.thread.start()
            else:
                run = same
        run.decided.wait(self.decision_wait)
        return run.view()

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self.runs.get(run_id)
        if run is not None:
            return run.view()
        record = self.store.load(run_id)
        if record is None:
            return None
        if record.get("state") in ACTIVE_STATES and record.get("boot") != self.boot \
                and not _alive(record.get("pid")):
            # Its server stopped. The file is left as it is: it belongs to
            # that process.
            record = self._recover(record)
        return {key: record.get(key) for key in (
            "run_id", "state", "reason", "hint", "error", "estimate", "series_to_confirm",
            "scans", "charged_credits")}

    def _recover(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """A run the server was stopped in the middle of: its tasks queued
        before the stop are found in the mapping file by the scans' codes;
        the other scans were not queued."""
        rows: List[Dict[str, Any]] = []
        try:
            rows = client_mod._anonymizer().Mapping(str(self.mapping)).rows()
        except Exception:
            rows = []
        scans = []
        for scan in record.get("scans") or []:
            scan = dict(scan)
            if scan.get("state") in ("ready", "uploading"):
                found = [r for r in rows if scan.get("code") and r.get("code") == scan["code"]
                         and r.get("task_id") and r.get("model") == record.get("model")]
                if found:
                    scan.update(state="queued", task_id=found[-1]["task_id"],
                                percent_uploaded=None)
                else:
                    scan.update(state="interrupted", percent_uploaded=None)
            scans.append(scan)
        record.update(scans=scans, state="interrupted", reason=None, hint=None)
        return record

    def shutdown(self, timeout: float = 4.0) -> List[str]:
        """Stop every run at its next pause, wait up to ``timeout`` seconds,
        remove the temporary folders of runs still going, and return the ids
        of runs whose thread is still alive."""
        active = [r for r in list(self.runs.values()) if not r.finished.is_set()]
        for run in active:
            if run.client is not None:
                run.client._transport.stop.set()
        end = time.monotonic() + timeout
        for run in active:
            if run.thread is not None:
                run.thread.join(max(0.0, end - time.monotonic()))
        alive = []
        for run in active:
            if run.thread is not None and run.thread.is_alive():
                alive.append(run.id)
                for folder in run.workdirs():
                    shutil.rmtree(folder, ignore_errors=True)
                run._end("interrupted")
        return alive


def _text(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _alive(pid: Any) -> bool:
    """Whether another process with this id still runs (POSIX). On Windows
    ``os.kill`` would end the process, so a run there counts as stopped."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or pid == os.getpid() \
            or os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True                     # it exists, owned by someone else
    return True


def _charged(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    lab = raw.get("lab_id")
    return {"source": str(raw.get("source") or ""),
            "lab_id": int(lab) if isinstance(lab, int) and not isinstance(lab, bool) else None,
            "credits": _round(raw.get("credits"))}


def _failure_detail(failure: Any) -> Optional[str]:
    """Why a scan was not queued: the platform's reason for a rejected scan,
    else the package's message."""
    error = getattr(failure, "error", None)
    if isinstance(error, NalveraError):
        reason = error.extra.get("reason")
        if isinstance(reason, str) and reason:
            return reason[:1000]
    message = getattr(failure, "message", None)
    return str(message)[:1000] if message else None


__all__ = ["Run", "RunClient", "RunManager", "RunStore", "RUN_ID_RX"]
