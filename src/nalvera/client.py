"""The connection to Nalvera.AI: ``Client``.

``nalvera.login`` makes one and the module functions use it; a script with
several accounts makes its own::

    import nalvera

    client = nalvera.Client(email="you@university.org", api_key=key)
    tasks = client.run("scan.nii.gz", model="totalsegmentator_ct", task="total")
    client.wait(tasks)                   # reports queue position, time waited, time running
    paths = client.download(tasks, out="results")

What ``run`` does, in order, before any byte leaves this computer: checks
the key, the model and the budget; anonymizes every scan into a private
temporary folder; finds scans already queued or done with the same model and
task (the record on this computer remembers them); asks the platform what
the run costs from each scan's voxel grid; decides who pays (free tasks
first) and, when needed, asks once. Then, two scans at a time: records the
scan's code, encrypts it when asked, uploads it, removes the temporary copy
and queues the task. Results are saved under the original name:
``<original name>__<model>__<task>.<ext>``.

Every message goes to the client's reporter (see ``nalvera.report``);
nothing else prints or reads a terminal.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import importlib
import math
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple, Union

import requests

from . import auth, display, messages, settings
from . import budget as budget_mod
from . import encrypt as encrypt_mod
from . import generation as generate_mod
from . import report as report_mod
from ._http import DEFAULT_BASE_URL, Cancelled, Transport
from ._version import __version__
from .errors import (
    BudgetChoiceRequired,
    ClientTooOld,
    ConfigError,
    ConfirmationRequired,
    CreditCapExceeded,
    InsufficientCredits,
    NalveraError,
    NetworkError,
    NoFreeTasks,
    NotFound,
    PermissionDenied,
    PreparationError,
    RateLimited,
    RunInterrupted,
    ScanRejected,
    WaitInterrupted,
    bounded,
)
from .report import NullReporter, Reporter, default_reporter, printable
from .upload import Uploader

PathLike = Union[str, "os.PathLike[str]"]
TaskRef = Union["Task", str]

SCOPE_READ, SCOPE_RESULTS, SCOPE_WRITE = "tasks:read", "results:read", "tasks:write"
FINISHED_STATES = ("completed", "failed", "cancelled")
ACTIVE_STATES = ("queued", "processing")
#: States of an earlier task that make a re-run of the same scan a repeat.
_HELD_STATES = ("queued", "processing", "completed")
_LABEL_RX = re.compile(r"[A-Za-z0-9 ._-]{1,60}")
_UNSAFE_NAME_RX = re.compile(r"[\x00-\x1f\x7f/\\:*?\"<>|]+")
_NIFTI_SUFFIXES = (".nii", ".nii.gz")
_MAX_POLL_SECONDS = 30.0
# A ready upload is kept by the server for 6 hours; stop waiting for room in
# the queue well before it expires.
_ACTIVE_CAP_MAX_WAIT = 5 * 3600.0
_LIVE_TABLE_MAX_ROWS = 30
WORKDIR_PREFIX = "nalvera-run-"
#: The anonymizer's ``inspect(convert=True)`` folder (``anonymize.INSPECT_PREFIX``,
#: written out here so the clean-up does not load the anonymizer).
INSPECT_PREFIX = "nalvera-inspect-"
#: The DICOM conversion's private folder (``anonymize.DICOM_PREFIX``).
DICOM_PREFIX = "nalvera-dicom-"
_STALE_PREFIXES = (WORKDIR_PREFIX, INSPECT_PREFIX, DICOM_PREFIX)
_STALE_WORKDIR_SECONDS = 24 * 3600
KEY_WARN_DAYS = 14
# The largest numbers taken from an answer; past them a value counts as the
# cap (a price past it is more than any max_credits).
_MAX_QUEUE_POSITION = 1e9
_MAX_SECONDS = 10 * 366 * 86400.0
_MAX_CREDITS = 1e9
_MAX_POLL_FLOOR = 300.0
#: The most failed or cancelled earlier tasks one labelled generation steps
#: past in one call.
_MAX_RETIRED = 8


# ── What the server returns ───────────────────────────────────────────────────
@dataclass(repr=False)
class Task:
    """A task on Nalvera.AI.

    ``state`` is ``queued``, ``processing``, ``completed``, ``failed`` or
    ``cancelled``. ``queue_position`` is set while queued (1 = next).
    ``waited_seconds`` and ``running_seconds`` are the time spent waiting
    and running, as the platform measures them. ``cancel_requested`` is True
    once a cancel was asked: a running task stays ``processing`` until the
    worker stops it. ``name`` is what the platform shows (the label, or the
    scan's neutral code); ``original_name`` is the scan's name on this
    computer, which never leaves it. ``replayed`` is True when the platform
    answered with a task queued by an earlier request (nothing was charged
    again). ``earlier`` is True for a task an earlier call queued, returned
    by ``run`` or ``generate`` instead of queueing the same work twice.
    ``charged_now`` is True when ``charged`` counts toward this call: the
    platform queued the task for a request this call sent (also when the
    first answer was lost and the request was sent again).
    """
    id: str
    state: str
    name: Optional[str] = None
    model: Optional[str] = None
    model_version: Optional[str] = None
    task: Optional[str] = None
    task_label: Optional[str] = None
    queue_position: Optional[int] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    waited_seconds: Optional[float] = None
    running_seconds: Optional[float] = None
    error: Optional[str] = None
    output_ready: bool = False
    output_kind: Optional[str] = None
    parts: Optional[List[str]] = None
    source: Optional[str] = None
    cancel_requested: bool = False
    original_name: Optional[str] = None
    code: Optional[str] = None
    label: Optional[str] = None
    charged: Optional[Dict[str, Any]] = None
    replayed: bool = False
    earlier: bool = False
    charged_now: bool = False
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def _table(self):
        # A task this process queued (a generation) also shows what it charged.
        if not (self.charged_now or self.earlier or self.replayed):
            return display.TASK_COLUMNS, [display.task_row(self)], []
        fresh = self.charged_now and not self.earlier
        row = display.task_row(self) + [_charge_text(self.charged) if fresh
                                        else messages.text("table.earlier")]
        notes = [] if fresh else [messages.text("table.force_note")]
        return display.TASK_COLUMNS + ("charge",), [row], notes

    def __repr__(self) -> str:
        return display.text_table(*self._table())

    def _repr_html_(self) -> str:
        return display.html_table(*self._table())

    @classmethod
    def from_api(cls, data: Dict[str, Any], **local: Any) -> Task:
        """A Task from the server's task object, plus local fields
        (``original_name``, ``code``, ``label``, ``charged``). Fields this
        version does not know are kept in ``raw`` and otherwise ignored."""
        known = {k: data.get(k) for k in (
            "name", "model", "model_version", "task", "task_label",
            "created_at", "started_at", "completed_at",
            "error", "output_kind", "parts", "source")}
        # Numbers from the answer stay within sense: no infinity, no NaN.
        position = bounded(data.get("queue_position"), 0, _MAX_QUEUE_POSITION)
        known["queue_position"] = int(position) if position is not None else None
        for name in ("waited_seconds", "running_seconds"):
            known[name] = bounded(data.get(name), 0.0, _MAX_SECONDS)
        return cls(id=str(data["id"]), state=str(data.get("state") or ""),
                   output_ready=bool(data.get("output_ready")),
                   cancel_requested=bool(data.get("cancel_requested")), raw=dict(data),
                   **known, **local)

    @property
    def finished(self) -> bool:
        """True once the task is completed, failed or cancelled."""
        return self.state in FINISHED_STATES

    @property
    def cancelling(self) -> bool:
        """True while a cancel is under way (still ``processing``)."""
        return self.cancel_requested and not self.finished

    @property
    def display_name(self) -> str:
        """The original name when known on this computer, else the platform's."""
        return self.original_name or self.name or self.id


@dataclass
class ScanFailure:
    """A scan of a run that was not queued: ``code`` is the reason's code
    (an error code such as ``scan_rejected``, or ``no_free_tasks`` /
    ``max_credits_exceeded`` for a scan left out on purpose), ``message``
    says why, ``error`` is the exception when there was one."""
    original_name: str
    original_path: str
    code: str
    message: str
    error: Optional[BaseException] = field(default=None, repr=False)


@dataclass
class SkippedScan:
    """A scan a run left out because it is already queued, running or done
    with the same model and task (``reason`` is ``already_queued``,
    ``already_processing``, ``already_completed`` or ``duplicate_in_run``).
    ``task_id`` is the earlier task, when there is one. Pass ``force=True``
    to run it again."""
    original_name: str
    original_path: str
    reason: str
    task_id: Optional[str] = None


class RunResult(display._Table, list):
    """The tasks of a ``Client.run`` (a list of ``Task``, in the
    order of the scans): those it queued, and for a scan already sent with
    the same model and task, the earlier task (``Task.earlier`` True). And:

    ``failed``
        Scans that could not be queued (unreadable, rejected).
    ``skipped``
        Scans left out as repeats (``SkippedScan``); an earlier task
        named here is also in the list.
    ``not_queued``
        Scans left out on purpose: the free tasks ran out with
        ``budget="free"``, or the next task would pass ``max_credits``.
    ``charged_credits``
        Credits charged by this call.

    A notebook cell that ends in it shows a table of its tasks with what
    each one charged. A scan that was not queued, and why, is said when the
    run happens (``failed``, ``skipped`` and ``not_queued`` keep them).
    """

    def __init__(self, tasks: Iterable[Task] = (), failed: Iterable[ScanFailure] = (),
                 skipped: Iterable[SkippedScan] = (), not_queued: Iterable[ScanFailure] = (),
                 charged_credits: float = 0.0):
        super().__init__(tasks)
        self.failed: List[ScanFailure] = list(failed)
        self.skipped: List[SkippedScan] = list(skipped)
        self.not_queued: List[ScanFailure] = list(not_queued)
        self.charged_credits: float = round(float(charged_credits), 2)

    def _table(self):
        rows: display.Rows = []
        charged = earlier = 0
        for item in self:
            if item.charged_now and not item.earlier:
                charge = _charge_text(item.charged)
                charged += 1
            else:
                charge = messages.text("table.earlier")
                earlier += 1
            rows.append([item.display_name, item.id,
                         display.state_text(item.state, item.cancelling), charge])
        notes = []
        if charged > 1:                       # one task: its row says what it charged
            notes.append(messages.text("table.charged", credits=f"{self.charged_credits:.2f}"))
        if earlier:
            notes.append(messages.text("table.force_note"))
        return ("scan", "task_id", "state", "charge"), rows, notes


class Downloads(display._Table, list):
    """The files a ``Client.download`` saved or found saved (a list of
    ``Path``), and ``skipped``: ``(task id, state)`` of tasks that had no
    result. A notebook cell that ends in it shows a table."""

    def __init__(self, paths: Iterable[Path] = (), skipped: Iterable[Tuple[str, str]] = ()):
        super().__init__(paths)
        self.skipped: List[Tuple[str, str]] = list(skipped)

    def _table(self):
        rows: display.Rows = [[display.path_text(p)] for p in self]
        notes = [messages.text("table.no_result", task_id=printable(task_id),
                               state=display.state_text(state)) for task_id, state in self.skipped]
        return ("file",), rows, notes


@dataclass
class SavedCopy:
    """One anonymized copy written by ``save_anonymized``: the scan's
    ``original_name`` and ``original_path``, the copy's ``path`` (named by
    its ``code``), its ``shape`` (the voxel grid) and ``removed`` (what the
    copy no longer holds, in words). ``series`` is the DICOM series, when
    the scan has one."""
    original_name: str
    original_path: str
    path: Path
    code: str
    shape: Tuple[int, ...]
    removed: List[str]
    series: Optional[str] = None


class SavedCopies(display._Table, list):
    """The copies ``save_anonymized`` wrote (a list of
    ``SavedCopy``), ``failed`` (scans that could not be prepared, as
    ``ScanFailure``) and ``record`` (the file on this computer that
    links each code to its original). A notebook cell that ends in it
    shows a table."""

    def __init__(self, copies: Iterable[SavedCopy] = (), failed: Iterable[ScanFailure] = (),
                 record: Optional[Path] = None):
        super().__init__(copies)
        self.failed: List[ScanFailure] = list(failed)
        self.record = record

    def _table(self):
        rows: display.Rows = [[c.original_name if not c.series else messages.text(
            "run.series_name", name=c.original_name, series=c.series),
            display.path_text(c.path), c.shape, c.removed, None] for c in self]
        rows += [[f.original_name, None, None, None, f.message] for f in self.failed]
        columns: Tuple[str, ...] = ("scan", "copy", "shape", "removed", "note")
        if not self.failed:
            columns, rows = columns[:-1], [row[:-1] for row in rows]
        notes = [messages.text("table.record", path=display.path_text(self.record, cut=False))] \
            if self.record else []
        return columns, rows, notes


@dataclass
class ScanEstimate:
    """One scan of ``Client.estimate_run``: its original name and path,
    ``shape`` (the voxel grid) and ``credits`` (what its task costs before
    free tasks). ``series`` is the DICOM series, when it has one."""
    original_name: str
    original_path: str
    shape: Tuple[int, ...]
    credits: float
    series: Optional[str] = None


@dataclass
class RunEstimate:
    """What a run would cost, found without uploading anything.

    ``scans`` would be queued; ``skipped`` are repeats; ``failed`` could not
    be read. ``credits`` is the cost of ``scans`` before free tasks;
    ``free_count`` how many of them the free tasks pay for;
    ``paid_credits`` the cost of the rest. ``budgets`` lists each budget
    with its balance (``{"budget", "name", "available_credits",
    "sufficient"}``) and ``suggested`` the one the website would pick."""
    model: str
    task: str
    scans: List[ScanEstimate]
    skipped: List[SkippedScan]
    failed: List[ScanFailure]
    credits: float
    free_count: int
    paid_credits: float
    free_period: bool
    free_tasks_left: int
    budgets: List[Dict[str, Any]]
    suggested: str

    def _table(self):
        rows: display.Rows = [[s.original_name if not s.series else messages.text(
            "run.series_name", name=s.original_name, series=s.series), s.shape, s.credits, None]
            for s in self.scans]
        # Numbers alone in the credits column: why a scan is left out is a note.
        rows += [[s.original_name, None, None, messages.text(f"skipped.{s.reason}")
                  if messages.has(f"skipped.{s.reason}") else s.reason] for s in self.skipped]
        rows += [[f.original_name, None, None, f.message] for f in self.failed]
        columns: Tuple[str, ...] = ("scan", "shape", "credits", "note")
        if not (self.skipped or self.failed):
            columns, rows = columns[:-1], [row[:-1] for row in rows]
        notes = []
        if len(self.scans) > 1:               # one scan: its row says what it costs
            notes.append(messages.text("estimate.total", count=len(self.scans),
                                       credits=f"{self.credits:.2f}"))
        if self.free_count:
            notes.append(messages.text("estimate.free", free=self.free_count,
                                       paid=f"{self.paid_credits:.2f}"))
        options = [o for o in self.budgets if isinstance(o, dict)]
        # Without a lab there is no choice to make: personal pays.
        if not any(str(o.get("budget", "")).startswith("lab:") for o in options):
            options = []
        for option in options:
            mark = messages.text("budget.mark_suggested") \
                if option.get("budget") == self.suggested else ""
            notes.append(messages.text(
                "estimate.budget", budget=printable(option.get("budget")),
                name=printable(option.get("name")),
                available=display.cell(display._number(option.get("available_credits"))),
                mark=mark).strip())
        return columns, rows, notes

    def __repr__(self) -> str:
        return display.text_table(*self._table())

    def _repr_html_(self) -> str:
        return display.html_table(*self._table())


@dataclass
class _Entry:
    """One scan going through a run."""
    order: Tuple[int, int]
    name: str
    path: str
    source_kind: str = "file"
    prepared: Any = None
    credits: float = 0.0
    budget: Optional[str] = None
    idem_code: Optional[str] = None
    idem_sha: Optional[str] = None
    recovering: bool = False
    upload_path: Optional[str] = None
    upload_filename: Optional[str] = None
    upload_size: int = 0
    upload_id: Optional[str] = None
    reserved: float = 0.0
    holds_free: bool = False
    task: Optional[Task] = None
    earlier: Optional[Task] = None
    failure: Optional[ScanFailure] = None
    told: bool = False
    skipped: Optional[SkippedScan] = None
    not_queued: Optional[ScanFailure] = None

    @property
    def ready(self) -> bool:
        return (self.prepared is not None and self.failure is None and self.skipped is None
                and self.not_queued is None)


@dataclass
class _Source:
    """One path of a run after a folder of NIfTI files was opened."""
    path: str
    kind: str                      # "file", "nifti_in_folder" or "dicom_folder"
    index: int


# ── The client ────────────────────────────────────────────────────────────────
class Client:
    """A connection to Nalvera.AI with one account. ``nalvera.login`` makes
    one and the module functions (``nalvera.run``...) use it; make a
    ``Client`` yourself for a script that uses several accounts or needs the
    options below. Every argument is optional; making a client sends no
    request.

    ``email``, ``api_key``
        Default None: ``NALVERA_EMAIL`` and ``NALVERA_API_KEY``, then what
        ``nalvera.login(..., remember=True)`` stored (keychain or
        credentials file). With none found, ``NoCredentials`` is
        raised; its message lists the steps to get a key.
    ``base_url``
        Default None: ``NALVERA_BASE_URL``, else ``https://nalvera.ai``.
    ``mapping``
        The record on this computer that links each scan's neutral code to
        its original name, its tasks and their result files. Default None:
        ``NALVERA_MAPPING``, else one file per account and platform in the
        ``records`` folder of the configuration folder (``config_dir``),
        outside the working folder, which is often shared or synchronized.
    ``budget``
        Who pays beyond the free tasks: ``"personal"``, ``"lab:<id>"``,
        ``"free"`` or ``"ask"``. Default None: ``NALVERA_BUDGET``, else the
        rule of ``nalvera.budget`` (free tasks first, then personal when
        there is no lab, else a question or ``BudgetChoiceRequired``).
    ``max_credits``
        Default None: ``NALVERA_MAX_CREDITS``, else no cap. A run whose
        estimate passes it does not start, and a task is queued only while
        the credits the run has charged plus the task's estimate stay within
        it.
    ``encrypt``
        Encrypt uploads (NVLT, a second layer over HTTPS). Default None:
        ``NALVERA_ENCRYPT``, else off unless the account requires it.
        ``False`` is refused when the account requires encryption.
    ``config_dir``
        Where a login with ``remember=True`` stores credentials, and where
        the record lives. Default None: ``NALVERA_CONFIG_DIR``, else
        ``~/.config/nalvera`` (``%APPDATA%\\nalvera`` on Windows).
    ``reporter``
        Where every message goes (``nalvera.report``). Default None: in
        a Jupyter kernel a ``NotebookReporter``; elsewhere a
        ``StreamReporter`` on ``output`` (default ``sys.stdout``) and
        ``input`` (default ``sys.stdin``); a ``NullReporter`` with
        ``quiet=True`` (default False).
    ``interactive``
        Whether questions may be asked. Default None: whenever the reporter
        can ask (a notebook that accepts input, or a terminal). False never
        asks.
    ``allowed_folders``
        Default None (any path). A list of folders: every scan path and
        result folder must resolve inside one of them, checked before
        anything is read.
    ``clock``, ``sleep``
        For tests: replace ``time.monotonic`` and ``time.sleep``.
    """

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, mapping: Optional[PathLike] = None, *,
                 budget: Optional[str] = None, config_dir: Optional[PathLike] = None,
                 output: Optional[TextIO] = None, quiet: bool = False,
                 input: Optional[TextIO] = None, interactive: Optional[bool] = None,
                 reporter: Optional[Reporter] = None, encrypt: Optional[bool] = None,
                 max_credits: Optional[float] = None,
                 allowed_folders: Optional[Sequence[PathLike]] = None,
                 clock: Optional[Any] = None, sleep: Optional[Any] = None):
        # No local holds the environment or the raw key once it is read: a
        # traceback that prints locals then shows neither.
        address = base_url or settings.get("NALVERA_BASE_URL") or DEFAULT_BASE_URL
        try:
            # A stored key goes only to the platform it was stored for.
            credentials = auth.resolve(email, api_key,
                                       directory=Path(config_dir) if config_dir else None,
                                       base_url=address)
        finally:
            api_key = None                     # the key lives on only inside an ApiKey
        self._transport = Transport(address, credentials, clock=clock, sleep=sleep,
                                    notify=self._on_wait)
        self._config_dir = Path(config_dir) if config_dir else None
        chosen = mapping or settings.get("NALVERA_MAPPING")
        self._default_record = chosen is None
        self.mapping_path = Path(chosen) if chosen else auth.record_path(
            credentials.email, self._transport.base_url, directory=self._config_dir)
        self.budget = budget_mod.parse_budget(budget if budget is not None
                                              else settings.get("NALVERA_BUDGET"))
        self.max_credits = _parse_max_credits(max_credits if max_credits is not None
                                              else settings.get("NALVERA_MAX_CREDITS"))
        self.encrypt = _parse_flag(encrypt, "NALVERA_ENCRYPT")
        if reporter is None:
            reporter = NullReporter() if quiet else default_reporter(
                output=output, input=input, interactive=interactive)
        self.reporter: Reporter = reporter
        self._ask = interactive
        self._allowed = [os.path.realpath(os.fspath(p)) for p in allowed_folders] \
            if allowed_folders is not None else None
        self._mapping_lock = threading.Lock()
        self._pay_lock = threading.Lock()
        self._me: Optional[Dict[str, Any]] = None
        self._models: Optional[List[Dict[str, Any]]] = None
        self._labels: Dict[str, str] = {}
        self._mapping_cache: Tuple[Any, Dict[str, Optional[Dict[str, Any]]]] = (None, {})
        # Downloads still streaming (their .part files), and how often close()
        # was called: a download going on in another thread then stops.
        self._partials: set = set()
        self._partials_lock = threading.Lock()
        self._closings = 0
        # Where the record is, said once: at the first call that writes to it.
        self._record_told = False
        self._told_lock = threading.Lock()
        # The notebook cell whose sign-in line already shows this client.
        self._signed_in_cell: Any = None

    def __repr__(self) -> str:
        transport = getattr(self, "_transport", None)
        if transport is None:                  # a traceback of __init__ shows it unfinished
            return "Client(…)"
        return (f"Client(base_url={transport.base_url!r}, "
                f"email={transport.credentials.email!r}, "
                f"key={transport.credentials.api_key.prefix}…)")

    def _table(self):
        transport = getattr(self, "_transport", None)
        if transport is None:
            return ("account", "platform", "key"), [], []
        return ("account", "platform", "key"), [[transport.credentials.email,
                                                 transport.base_url,
                                                 transport.credentials.api_key.prefix]], []

    def _repr_html_(self) -> str:
        return display.html_table(*self._table())

    def _ipython_display_(self) -> None:
        """A notebook shows the client as a table, except in the cell whose
        ``nalvera.login(...)`` line just said who is signed in."""
        if self._signed_in_cell is not None and self._signed_in_cell == report_mod.cell_token():
            return
        shown = report_mod.display_module()
        if shown is not None:
            shown.display({"text/plain": display.text_table(*self._table()),
                           "text/html": display.html_table(*self._table())}, raw=True)

    def _call(self, kind: str):
        """The scope of one call of the package (``run``, ``wait``...): a
        notebook shows its progress as one display and keeps only what its
        returned table does not show (see ``nalvera.report``)."""
        return report_mod.call_scope(self.reporter, kind)

    def _tell_record(self) -> None:
        """Say where the record is, once per client, before its first write."""
        with self._told_lock:
            if self._record_told:
                return
            self._record_told = True
        self.reporter.say("run.record", path=display.path_text(self.mapping_path, cut=False))

    @property
    def _interactive(self) -> bool:
        """Whether a question can be asked now. Read at each question: a
        notebook accepts input in one cell run and not in another."""
        return self._ask is not False and self.reporter.can_ask()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the connections and sign out: the key is dropped from this
        client, which sends no request afterwards (``signed_out``). A
        download still going in another thread stops, and its partly written
        file is removed. ``nalvera.logout()`` closes the client of
        ``nalvera.login``."""
        with self._partials_lock:
            self._closings += 1
            partials, self._partials = list(self._partials), set()
        self._transport.close()
        for path in partials:
            _remove(path)

    @property
    def base_url(self) -> str:
        """The server's address."""
        return self._transport.base_url

    @property
    def email(self) -> str:
        """The account's email."""
        return self._transport.credentials.email

    @property
    def key_prefix(self) -> str:
        """The first 12 characters of the key (not secret)."""
        return self._transport.credentials.api_key.prefix

    # ── account ───────────────────────────────────────────────────────────────
    def me(self, refresh: bool = False) -> Dict[str, Any]:
        """The account, the key (prefix, name, permissions, end date), the
        limits to respect and the encryption the platform offers
        (``GET /api/v1/me``), as a dict that a notebook shows as a table.
        Read once and kept; ``refresh=True`` (default False) reads it
        again."""
        if self._me is None or refresh:
            answer = self._transport.json("GET", "/api/v1/me")
            self._me = display.Account(answer if isinstance(answer, dict) else {})
            self._transport.set_limits(self._me.get("limits") or {})
        return self._me

    def limits(self) -> Dict[str, Any]:
        """The limits that apply to this key: requests and task submits per
        minute, tasks waiting or running, uploads in progress, file size,
        part size and the shortest interval between status checks. A dict
        that a notebook shows as a table."""
        return display.Limits(self.me().get("limits") or {})

    def key_expiry(self) -> Tuple[Optional[str], Optional[int]]:
        """The key's end date (ISO text from the platform) and the whole
        days left, measured against the platform's own clock (the ``Date``
        of its answer). ``(None, None)`` when the platform names no end."""
        expires = (self.me().get("key") or {}).get("expires_at")
        end = _parse_time(expires)
        if end is None:
            return expires, None
        now = self._transport.server_date or datetime.now(timezone.utc)
        return expires, int((end - now).total_seconds() // 86400)

    def encryption(self) -> Dict[str, Any]:
        """Whether uploads are encrypted for this account: ``available``
        (the platform has a key), ``required`` (the account's switch),
        ``default`` (what a run does when not told: True when required or
        when ``encrypt``/``NALVERA_ENCRYPT`` asks for it)."""
        block = self.me().get("encryption") or {}
        required = bool(block.get("required"))
        return {"available": bool(block.get("available")), "required": required,
                "default": required or bool(self.encrypt)}

    def models(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """The models and tasks this account may run, with prices, and for a
        generation the settings each task accepts (``GET /api/v1/models``):
        a list of dicts that a notebook shows as a table, one row per task.
        Read once and kept; ``refresh=True`` (default False) reads it again."""
        if self._models is None or refresh:
            self.me()
            self._models = display.ModelList(
                self._transport.json("GET", "/api/v1/models").get("models") or [])
        return self._models

    def budgets(self) -> Dict[str, Any]:
        """Free tasks left, personal credits and each lab's balance, with the
        suggested budget (``GET /api/v1/budgets``): a dict that a notebook
        shows as a table."""
        self.me()
        return display.Budgets(budget_mod.checked_budgets(
            self._transport.json("GET", "/api/v1/budgets")))

    def estimate(self, model: str, task: Optional[str] = None, *, upload_id: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None,
                 shape: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """What one task would cost in credits, whether a free task covers
        it, and each budget's balance (``POST /api/v1/estimate``). Give
        ``shape`` (a scan's voxel grid, ``[X, Y, Z]``: no upload needed), an
        ``upload_id``, or ``params`` for a generation. ``task`` default None:
        the model's default task. An unknown model or task is refused here
        (``ConfigError``), before any request that could cost."""
        self.me()
        model_info, task_info = self._resolve(model, task, scan=params is None)
        model, task = model_info["id"], task_info["id"]
        body: Dict[str, Any] = {"model": model, "task": task}
        if shape is not None:
            body["shape"] = [int(n) for n in list(shape)[:3]]
        if upload_id is not None:
            body["upload_id"] = upload_id
        if params is not None:
            body["params"] = params
        return budget_mod.checked_estimate(
            self._transport.json("POST", "/api/v1/estimate", json=body, idempotent=True))

    # ── tasks ─────────────────────────────────────────────────────────────────
    def tasks(self, state: str = "all", limit: int = 50, offset: int = 0,
              name: Optional[str] = None) -> List[Task]:
        """The account's tasks, newest first: those queued from Python and on
        the website alike, as a list of ``Task`` that a notebook shows
        as a table. ``state``: ``"active"``, ``"finished"`` or ``"all"``
        (default). ``limit``: 1 to 100 (default 50). ``offset``: how many to
        skip (default 0). ``name`` (default None): only tasks with exactly
        this platform name, such as ``"scan-7F3K2Q9A.nii.gz"``."""
        return display.TaskList(self._local(t, None)
                                for t in self._list(state, limit, offset, name)[0])

    def task(self, task_id: TaskRef) -> Task:
        """One task, with its state, queue position and times. ``task_id``
        is a ``Task`` or its id."""
        self.me()
        tid = _task_id(task_id)
        return self._local(Task.from_api(self._transport.json("GET", f"/api/v1/tasks/{tid}")),
                           task_id)

    def cancel(self, tasks: Union[TaskRef, Iterable[TaskRef]]) -> Union[Task, List[Task]]:
        """Cancel tasks that are waiting or running. ``tasks`` is a
        ``Task`` or its id (returns the ``Task``), or a list of
        them (returns a list). A running task stops at the worker's next
        check: until then it stays ``processing`` with ``cancel_requested``.
        Asking twice is harmless. Credits are refunded as on the website.
        Raises ``NotCancellable`` for a finished task."""
        self.me()
        single = isinstance(tasks, (str, Task))
        done = display.TaskList()
        for ref in _as_refs(tasks):
            tid = _task_id(ref)
            data = self._transport.json("POST", f"/api/v1/tasks/{tid}/cancel", idempotent=True)
            done.append(self._local(Task.from_api(data), ref))
        return done[0] if single else done

    # ── estimate a run ────────────────────────────────────────────────────────
    def estimate_run(self, paths: Union[PathLike, Sequence[PathLike]], model: str,
                     task: Optional[str] = None, *, force: bool = False) -> RunEstimate:
        """What ``run()`` would cost, without uploading anything: the scans
        are anonymized into a temporary folder (removed afterwards), each
        one's voxel grid is priced by the platform, and repeats are left out
        as ``run()`` would (``force=True``, default False, keeps them).
        ``paths``, ``model`` and ``task`` (default None: the model's default
        task) as for ``run()``. Needs only a key that may see tasks.
        Returns a ``RunEstimate``."""
        with self._call("estimate"):
            return self._estimate_run(paths, model, task, force=force)

    def _estimate_run(self, paths: Any, model: str, task: Optional[str], *,
                      force: bool) -> RunEstimate:
        sources = _as_paths(paths)
        if not sources:
            raise ConfigError(messages.text("error.no_paths"), code="no_paths")
        me = self.me()
        _require_scope(me, SCOPE_READ)
        model_info, task_info = self._resolve(model, task, scan=True)
        job = self._new_job(model_info["id"], task_info["id"], label=None, text=None,
                            force=force, max_credits=None, key=None)
        try:
            self._survey(job, sources)
            estimate = job.estimate or {}
        finally:
            shutil.rmtree(job.workdir, ignore_errors=True)
            self._tell_failures(job)
        ready = [e for e in job.entries if e.ready]
        free = _free_state(self.budgets(), estimate)
        free_count = budget_mod.free_cover(free, len(ready))
        costs = [e.credits for e in ready]
        return RunEstimate(
            model=job.model, task=job.task,
            scans=[ScanEstimate(original_name=e.name, original_path=e.path,
                                shape=tuple(e.prepared.shape), credits=e.credits,
                                series=getattr(e.prepared, "series", None)) for e in ready],
            skipped=[e.skipped for e in job.entries if e.skipped],
            failed=[e.failure for e in job.entries if e.failure],
            credits=round(sum(costs), 2), free_count=free_count,
            paid_credits=round(sum(costs[free_count:]), 2),
            free_period=bool(free.get("period_active")),
            free_tasks_left=int(free.get("free_tasks_left") or 0),
            budgets=list(estimate.get("budgets") or []),
            suggested=str(estimate.get("suggested") or budget_mod.PERSONAL))

    # ── run ───────────────────────────────────────────────────────────────────
    def run(self, paths: Union[PathLike, Sequence[PathLike]], model: str,
            task: Optional[str] = None, *, budget: Optional[str] = None,
            label: Optional[str] = None, text: Optional[str] = None,
            yes: bool = False, encrypt: Optional[bool] = None,
            max_credits: Optional[float] = None, force: bool = False) -> RunResult:
        """Anonymize scans on this computer, upload them and queue one task
        per scan. Does not wait: use ``wait()``, then ``download()``.

        Example::

            tasks = nalvera.run("ct_scan.nii.gz", model="totalsegmentator_ct",
                                budget="personal")

        Running the same call again queues nothing twice: a scan already
        sent with this model and task (waiting, running or completed) comes
        back as its earlier task, marked ``earlier``, and costs nothing.

        ``paths``
            Required. A ``.nii``/``.nii.gz`` file, a folder of NIfTI files
            (each file at its top level is a scan), a folder of DICOM files
            (one task per series), or a list of these.
        ``model``
            Required. A model id from ``models()``, such as
            ``"totalsegmentator_ct"``.
        ``task``
            Default None: the model's default task. Options: the task ids
            ``models()`` lists for the model.
        ``budget``
            Who pays beyond the free tasks. Options: ``"personal"``,
            ``"lab:<id>"``, ``"free"`` (free tasks only: stop when they run
            out, charge nothing) or ``"ask"``. Default None: the client's
            ``budget`` (``NALVERA_BUDGET``); without one, free tasks first,
            then personal when the account has no active lab; with a lab, one
            question when the notebook or terminal can ask, the suggested
            budget with ``yes=True``, else ``BudgetChoiceRequired``
            before anything is uploaded.
        ``label``
            Default None. Shown on the platform instead of the neutral code
            (1-60 letters, digits, spaces, dots, dashes, underscores). It
            leaves this computer, so it must not identify a patient.
        ``text``
            Default None. The text query a task needs (``needs_text`` in
            ``models()``).
        ``yes``
            Default False. True accepts the suggested budget and goes ahead
            when a DICOM folder holds several scans, without asking.
        ``encrypt``
            Default None: the client's ``encrypt`` (``NALVERA_ENCRYPT``),
            else on only when the account requires it. True encrypts every
            upload or stops before uploading; False is refused when the
            account requires encryption.
        ``max_credits``
            Default None: the client's ``max_credits``
            (``NALVERA_MAX_CREDITS``), else no cap. The run does not start
            when its estimate passes the cap (``CreditCapExceeded``), and
            queues a task only while the credits it has charged plus the
            task's estimate stay within the cap.
        ``force``
            Default False. True queues scans again that are already queued,
            running or done with the same model and task.

        Returns a ``RunResult``: the tasks, in the order of the scans,
        with ``failed``, ``skipped`` and ``not_queued``; a notebook shows it
        as a table. A scan that cannot be read, or that the server rejects,
        is listed in ``failed`` and the others go on; if no scan could be
        queued, the first one's error is raised. An error that concerns the
        whole run (no credits, a permission, the network) stops it at once;
        the tasks queued before it are in the error's ``tasks``. Stopping the
        cell (or Ctrl-C) raises ``RunInterrupted`` (with ``tasks``):
        what was queued keeps running and is in the record.

        Voxel values and geometry of a NIfTI scan are uploaded unchanged. A
        DICOM series is uploaded as the converted volume: where the
        converter corrected a gantry tilt or uneven slice spacing, it
        resampled the series, and the voxel values and geometry are those of
        the resampled volume (the run says so for each such scan, and the
        record keeps it).
        """
        return self._run(paths, model, task, budget=budget, label=label, text=text, yes=yes,
                         encrypt=encrypt, max_credits=max_credits, force=force,
                         interactive=self._interactive)

    def run_in_background(self, paths: Union[PathLike, Sequence[PathLike]], model: str,
                          task: Optional[str] = None, **options: Any) -> BackgroundRun:
        """Start ``run()`` in a thread and return at once with a
        ``BackgroundRun`` to poll (``done()``), stop (``cancel()``) or
        collect (``result()``). ``paths``, ``model``, ``task`` and the
        keyword ``options`` (``budget``, ``label``, ``text``, ``yes``,
        ``encrypt``, ``max_credits``, ``force``) are those of ``run()``.
        Nothing is asked: a run that needs a budget choice fails with
        ``BudgetChoiceRequired``, so give ``budget`` (and ``yes`` for a
        DICOM folder with several scans). One run per client at a time."""
        return BackgroundRun(self, paths, model, task, options)

    def _run(self, paths: Any, model: str, task: Optional[str], **options: Any) -> RunResult:
        with self._call("run"):
            return self._run_scans(paths, model, task, **options)

    def _run_scans(self, paths: Any, model: str, task: Optional[str], *,
                   budget: Optional[str], label: Optional[str], text: Optional[str], yes: bool,
                   encrypt: Optional[bool], max_credits: Optional[float], force: bool,
                   interactive: bool) -> RunResult:
        sources = _as_paths(paths)
        if not sources:
            raise ConfigError(messages.text("error.no_paths"), code="no_paths")
        label = _clean_label(label)
        requested = budget_mod.parse_budget(budget if budget is not None else self.budget)
        cap = _parse_max_credits(max_credits) if max_credits is not None else self.max_credits
        asked_encrypt = encrypt if encrypt is not None else self.encrypt
        me = self.me()
        _check_version(me)
        _require_scope(me, SCOPE_WRITE)
        model_info, task_info = self._resolve(model, task, scan=True)
        if task_info.get("needs_text") and not (text and text.strip()):
            raise ConfigError(messages.text("error.text_required", model=model,
                                            task=task_info["id"]), code="text_required")
        key = encrypt_mod.decide(asked_encrypt, me.get("encryption"), self.base_url)
        budgets = self.budgets()
        # Validates a named lab, and stops before any work when the choice
        # can only end in BudgetChoiceRequired (a lab, no free task that
        # could make the question unnecessary, nobody to ask).
        budget_mod.plan(budgets, requested=requested,
                        interactive=interactive or _free_possible(budgets), assume_yes=yes)
        if key is not None:
            self.reporter.say("run.encrypting")
        job = self._new_job(model_info["id"], task_info["id"], label=label, text=text,
                            force=force, max_credits=cap, key=key)
        try:
            self._survey(job, sources)
            self._pay(job, budgets, requested, yes=yes, interactive=interactive)
            self._upload_all(job, interactive=interactive)
        except BaseException as error:
            self._transport.stop.set()
            # The account may hold only a few unfinished uploads: free the
            # slots of scans this run will not queue.
            self._release_uploads(job.entries)
            self._tell_failures(job)
            queued = _queued(job.entries)
            told = self._tell_stopped(queued, job.charged)
            if isinstance(error, NalveraError):
                error.tasks = queued                         # type: ignore[attr-defined]
                error.failed = _failures(job.entries)        # type: ignore[attr-defined]
            if isinstance(error, (Cancelled, KeyboardInterrupt)):
                raise RunInterrupted(queued, told=told) from None
            raise
        finally:
            shutil.rmtree(job.workdir, ignore_errors=True)
            self._transport.stop.clear()
        self._release_uploads(job.entries)

        ordered = sorted(job.entries, key=lambda e: e.order)
        result = RunResult([e.task or e.earlier for e in ordered if e.task or e.earlier],
                           _failures(job.entries),
                           [e.skipped for e in ordered if e.skipped],
                           [e.not_queued for e in sorted(job.entries, key=lambda e: e.order)
                            if e.not_queued],
                           charged_credits=job.charged)
        if not result and result.failed and not result.skipped and not result.not_queued:
            failure = result.failed[0]
            # Raised for the whole call: its text says what the scan's line would.
            self._tell_failures(job, skip=failure)
            first = failure.error
            if isinstance(first, NalveraError):
                first.failed = result.failed                 # type: ignore[attr-defined]
                # The text names the scan it is about.
                first.message = messages.text("error.scan_named",
                                              name=printable(failure.original_name),
                                              reason=first.message)
                first.args = (first.message,)
                raise first
            raise NalveraError(messages.text("error.nothing_queued"), code="nothing_queued")
        self._tell_failures(job)
        if len(job.entries) > 1:              # one scan: its own line says how it went
            counts = (("queued", sum(1 for t in result if not t.earlier)),
                      ("failed", len(result.failed)), ("skipped", len(result.skipped)),
                      ("not_queued", len(result.not_queued)))
            self._say_counts("run.summary", counts, "list.sentence")
        return result

    def _say_counts(self, key: str, counts: Sequence[Tuple[str, int]], joiner: str) -> None:
        """A summary of the counts that are not zero (``<key>.<name>`` each)."""
        items = [messages.text(f"{key}.{name}", count=n) for name, n in counts if n]
        if items:
            self.reporter.say(key, items=messages.text(joiner).join(items), **dict(counts))

    def _tell_stopped(self, tasks: Sequence[Task], credits: float) -> bool:
        """A call that raised or was stopped after it queued ``tasks``. In a
        notebook its display drops the progress lines, each "Queued ..."
        among them: one line stays that names the tasks, what they charged
        and the call that follows them. Plain output kept those lines, so
        nothing is added there. True when the line was said."""
        if not tasks or not getattr(self.reporter, "hides_progress", False):
            return False
        self.reporter.say("run.stopped_with_tasks", count=len(tasks),
                          ids=messages.text("list.separator").join(t.id for t in tasks),
                          credits=f"{credits:.2f}")
        return True

    def _tell_failures(self, job: _Job, skip: Optional[ScanFailure] = None) -> None:
        """Say, once and in the scans' order, each scan that failed and why;
        ``skip`` is a failure the call raises, whose text says it already."""
        for entry in sorted(list(job.entries), key=lambda e: e.order):
            if entry.failure is None or entry.told or entry.failure is skip:
                continue
            entry.told = True
            self.reporter.say("run.scan_failed", name=entry.name, reason=entry.failure.message)

    def _new_job(self, model: str, task: str, *, label: Optional[str], text: Optional[str],
                 force: bool, max_credits: Optional[float], key: Optional[bytes]) -> _Job:
        _remove_stale_workdirs()
        anonymize = _anonymizer()
        limits = self.me().get("limits") or {}
        mapping = anonymize.Mapping(str(self.mapping_path))
        uploader = Uploader(self._transport, min_poll_seconds=_poll_floor(limits),
                            max_file_bytes=_max_bytes(limits), say=self.reporter.say)
        try:
            workdir = tempfile.mkdtemp(prefix=WORKDIR_PREFIX)          # mode 0700
        except OSError:
            raise ConfigError(messages.text("error.temp_not_writable",
                                            path=tempfile.gettempdir()),
                              code="temp_not_writable") from None
        return _Job(anonymize=anonymize, mapping=mapping, uploader=uploader, workdir=workdir,
                    model=model, task=task, label=label, text=text, force=force,
                    max_credits=max_credits, key=key, max_bytes=_max_bytes(limits),
                    text_digest=_text_digest(text))

    # ── the survey: everything that happens before an upload ──────────────────
    def _survey(self, job: _Job, sources: List[PathLike]) -> None:
        """Open folders, check what can be checked, anonymize every scan,
        leave out repeats, and price each scan by its voxel grid."""
        expanded = self._expand(job, sources)
        self._preflight(job, expanded)
        self._prepare_all(job, expanded)
        self._leave_out_repeats(job)
        self._price(job)

    def _expand(self, job: _Job, sources: List[PathLike]) -> List[_Source]:
        """Each path as one or more scans to prepare (see ``_expand_paths``).
        A label that repeats a scan's file or folder name is refused before
        anything is read: the label is sent to the platform."""
        expanded = _expand_paths(sources, job.anonymize, self._check_allowed, self.reporter.say)
        if job.label:
            label = _name_key(job.label)
            for source in expanded:
                if _name_key(_strip_scan_suffix(os.path.basename(
                        source.path.rstrip(os.sep)))) == label:
                    raise ConfigError(messages.text("error.label_is_file_name"),
                                      code="label_is_file_name")
        return expanded

    def _allowed_check(self) -> Dict[str, Any]:
        """With allowed folders, the anonymizer's ``check`` argument: they are
        checked again where each file is read or linked, since a file may
        have become a link after it was listed. Without them, nothing."""
        return {"check": self._check_allowed} if self._allowed is not None else {}

    def _check_allowed(self, path: str) -> None:
        """Refuse a path outside the allowed folders, before reading it."""
        if self._allowed is None:
            return
        real = os.path.realpath(path)
        for folder in self._allowed:
            if real == folder or real.startswith(folder.rstrip(os.sep) + os.sep):
                return
        raise ConfigError(messages.text("error.path_not_allowed", path=printable(path)),
                          code="path_not_allowed")

    def _preflight(self, job: _Job, sources: List[_Source]) -> None:
        """What can fail later is checked now, before any upload: the
        mapping file can be read and its folder written, and the temporary
        folder has room for the anonymized copies."""
        folder = self.mapping_path.resolve().parent
        try:
            self._prepare_record_folder()
            folder.mkdir(parents=True, exist_ok=True)
            probe = tempfile.NamedTemporaryFile(dir=str(folder), prefix=".nalvera-probe-")
            probe.close()
        except OSError:
            raise ConfigError(messages.text("error.mapping_not_writable",
                                            path=printable(str(self.mapping_path))),
                              code="mapping_not_writable") from None
        self._mapping_rows(job)
        sizes = [_size_of(s.path) for s in sources]
        need = sum(sizes) + (max(sizes) if sizes and job.key is not None else 0)
        try:
            free = shutil.disk_usage(job.workdir).free
        except OSError:
            free = None
        if free is not None and need > free:
            raise ConfigError(messages.text("error.temp_space_low", path=job.workdir,
                                            need_mb=need // (1024 * 1024) + 1,
                                            free_mb=free // (1024 * 1024)),
                              code="temp_space_low")

    def _mapping_rows(self, job: _Job) -> List[Dict[str, Any]]:
        try:
            with self._mapping_lock:
                return list(job.mapping.rows())
        except job.anonymize.AnonymizeError as error:
            raise _mapping_error(error, self.mapping_path) from None

    def _prepare_all(self, job: _Job, sources: List[_Source]) -> None:
        """Anonymize every scan into the run's private folder, one path at a
        time. A scan that cannot be prepared is listed and the others go on."""
        noticed = False
        # A code the record already holds is drawn again: two scans never
        # share one, whatever the odds.
        taken = {str(r.get("code")) for r in self._mapping_rows(job) if r.get("code")}
        for number, source in enumerate(sources):
            if self._transport.stop.is_set():
                raise Cancelled()
            name = os.path.basename(source.path.rstrip(os.sep)) or source.path
            self.reporter.say("run.preparing", name=name)
            try:
                prepared = job.anonymize.prepare(source.path, workdir=job.workdir,
                                                 label=job.label, max_bytes=job.max_bytes,
                                                 avoid=taken, **self._allowed_check())
            except job.anonymize.AnonymizeError as error:
                code = getattr(error, "code", None) or "unreadable"
                if code == "mapping_unreadable":
                    raise _mapping_error(error, self.mapping_path) from None
                key = f"anonymize.{code}" if messages.has(f"anonymize.{code}") \
                    else "anonymize.other"
                failure = PreparationError(messages.text(key, code=code), code=code)
                self._fail(_Entry(order=(number, 0), name=name, path=source.path,
                                  source_kind=source.kind), failure, job)
                continue
            if not noticed:
                self.reporter.say("run.pixel_notice")
                noticed = True
            if source.kind == "dicom_folder":
                self.reporter.say("run.dicom_found", name=name, count=len(prepared))
            self._say_left_out(name, prepared)
            for k, scan in enumerate(prepared):
                taken.add(scan.code)
                entry = _Entry(order=(number, k), name=_display_name(scan), path=source.path,
                               source_kind=source.kind, prepared=scan)
                job.entries.append(entry)
                self.reporter.say("run.prepared", name=entry.name, code=scan.code)
                _say_corrected(self.reporter.say, entry.name, scan)

    def _say_left_out(self, name: str, prepared: Any) -> None:
        """What a DICOM conversion did not turn into scans: each
        volume dropped and why, the files in no scan, a partial conversion."""
        for volume in getattr(prepared, "dropped", None) or []:
            reason = getattr(volume, "reason", "")
            self.reporter.say("run.dicom_dropped", name=name,
                              series=getattr(volume, "series", "") or "-",
                              kept=getattr(volume, "kept", "") or "-",
                              reason=messages.text(f"dropped.{reason}")
                              if messages.has(f"dropped.{reason}") else reason)
        left_out = getattr(prepared, "left_out", None)
        if left_out:
            self.reporter.say("run.dicom_left_out", name=name, count=left_out)
        if getattr(prepared, "partial", False):
            self.reporter.say("run.dicom_partial", name=name, count=len(prepared))

    def _leave_out_repeats(self, job: _Job) -> None:
        """B1 and B2: a scan whose content was already queued, running or
        done with this model and task is left out (unless ``force``). A scan
        recorded by an earlier run that stopped before it learnt its task id
        is looked up by the name it was sent under; its Idempotency-Key is
        reused, so a request that did land is answered with its task."""
        ready = [e for e in job.entries if e.ready]
        if not ready:
            return
        seen: Dict[str, _Entry] = {}
        for entry in ready:
            keys = _content_keys(entry.prepared)
            if not job.force and any(k in seen for k in keys):
                entry.skipped = SkippedScan(entry.name, entry.path, "duplicate_in_run")
                self.reporter.say("run.duplicate_in_run", name=entry.name)
                self._remove_prepared(entry)
                continue
            for k in keys:
                seen.setdefault(k, entry)
        rows = self._mapping_rows(job)
        if not rows:
            return
        earlier: Dict[int, List[Dict[str, Any]]] = {}
        for entry in [e for e in ready if e.ready]:
            keys = set(_content_keys(entry.prepared))
            matches = [r for r in rows if keys & set(_row_content_keys(r))]
            if matches:
                earlier[id(entry)] = matches
        if not earlier:
            return
        # B2: codes recorded with no task: look the task up by its name.
        for entry in [e for e in ready if id(e) in earlier]:
            matches = earlier[id(entry)]
            codes_with_task = {r.get("code") for r in matches if r.get("task_id")}
            orphans = [r for r in matches if r.get("code") and r.get("code") not in codes_with_task]
            if not orphans or job.force:
                continue
            # The scan's first code: every later request of it reused that
            # code's key, so a request that landed was made with it.
            first = orphans[0]
            entry.idem_code, entry.recovering = first["code"], True
            entry.idem_sha = first.get("sha256") or None
            for row in orphans:
                if row.get("label"):
                    continue                     # sent under its label: the key finds it
                found = [t for t in self.tasks(state="all", limit=100,
                                               name=f"{row['code']}.nii.gz")
                         if t.model == job.model and (t.task or job.task) == job.task]
                for item in found:
                    self._record(job, "record_task", row["code"], item.id, job.model, job.task,
                                 job.text_digest)
                    matches.append({**row, "task_id": item.id, "model": job.model,
                                    "task": job.task, "text_sha256": job.text_digest})
                    self.reporter.say("run.recovered", name=entry.name, task_id=item.id)
        if job.force:
            return
        wanted = {r["task_id"] for e in ready if id(e) in earlier for r in earlier[id(e)]
                  if r.get("task_id") and _same_work(r, job)}
        states = self._lookup(sorted(wanted))
        for entry in [e for e in ready if id(e) in earlier and e.ready]:
            held = [states[r["task_id"]] for r in earlier[id(entry)]
                    if r.get("task_id") in states and states[r["task_id"]] is not None
                    and _same_work(r, job)]
            held = [t for t in held if t.state in _HELD_STATES]
            if held:
                item = self._local(held[-1], None)
                item.original_name = item.original_name or entry.name
                item.earlier = True
                entry.earlier = item
                entry.skipped = SkippedScan(entry.name, entry.path, f"already_{item.state}",
                                            task_id=item.id)
                self.reporter.say("run.already_done", name=entry.name, task_id=item.id,
                                  state=_state_text(item))
                self._remove_prepared(entry)

    def _price(self, job: _Job) -> None:
        """Ask the platform what each scan costs from its voxel grid, once
        per distinct grid (the cost depends on nothing else)."""
        prices: Dict[Tuple[int, ...], float] = {}
        for entry in [e for e in job.entries if e.ready]:
            shape = tuple(int(n) for n in list(entry.prepared.shape)[:3])
            if shape not in prices:
                answer = self.estimate(job.model, job.task, shape=shape)
                prices[shape] = _credits_of(answer.get("credits"))
                job.estimate = answer
            entry.credits = prices[shape]

    # ── paying ────────────────────────────────────────────────────────────────
    def _pay(self, job: _Job, budgets: Dict[str, Any], requested: Optional[str], *, yes: bool,
             interactive: bool) -> None:
        """Decide what each scan sends as its budget (the rules of budget.py),
        show the cost, and stop before uploading when the run cannot be
        paid, would pass ``max_credits``, or needs a confirmation."""
        ready = [e for e in job.entries if e.ready]
        if not ready:
            return
        state = dict(budgets)
        state["free"] = _free_state(budgets, job.estimate)
        free = state["free"]
        count = len(ready)
        expected_free = budget_mod.free_cover(free, count)
        plan = budget_mod.plan(state, requested=requested, interactive=interactive,
                               assume_yes=yes, count=count)
        job.plan, job.yes = plan, yes
        costs = [e.credits for e in ready]
        total = round(sum(costs), 2)
        self.reporter.say("run.plan", count=count, credits=f"{total:.2f}")
        if expected_free:
            self.reporter.say("run.plan_free", free=expected_free, count=count)
        if plan.budget == budget_mod.FREE and requested == budget_mod.FREE:
            if expected_free == 0:
                raise NoFreeTasks(extra={"budget": budget_mod.FREE})
            for entry in ready[expected_free:]:
                entry.not_queued = ScanFailure(entry.name, entry.path, "no_free_tasks",
                                               messages.text("run.left_out.no_free_tasks"))
                self._remove_prepared(entry)
            if expected_free < count:
                self.reporter.say("run.free_only_partial", free=expected_free,
                                  left=count - expected_free)
            for entry in ready[:expected_free]:
                entry.budget = budget_mod.FREE
            self._confirm_dicom(job, ready[:expected_free], 0.0, yes, interactive, False)
            return
        paid = ready[expected_free:]
        paid_credits = round(sum(e.credits for e in paid), 2)
        if job.max_credits is not None and paid_credits > job.max_credits + 1e-9:
            raise CreditCapExceeded(extra={"estimate": paid_credits,
                                           "max_credits": job.max_credits})
        asked = False
        paid_budget = plan.budget
        if plan.budget == budget_mod.FREE:
            paid_budget = None                         # the free tasks pay for everything
        elif plan.ask:
            paid_budget = budget_mod.ask(plan, credits=paid_credits, count=len(paid),
                                         reporter=self.reporter, free_count=expected_free,
                                         estimate_budgets=(job.estimate or {}).get("budgets"))
            asked = True
        job.paid_budget = paid_budget
        # A named budget is sent as given, and the server pays the first tasks
        # with free tasks. Under a spending cap the tasks the free tasks should
        # pay are sent as "free" instead: when those free tasks are gone by
        # then (spent elsewhere), each is held against the cap before it is
        # sent again with the named budget (rule 4), so the cap holds.
        free_first = plan.free_count if requested is None else \
            expected_free if job.max_credits is not None else 0
        for number, entry in enumerate(ready):
            if number < free_first:
                entry.budget = budget_mod.FREE
            else:
                entry.budget = paid_budget or budget_mod.FREE
        if paid_budget and paid:
            self.reporter.say("run.budget", budget=paid_budget)
            available = _available(job.estimate, budgets, paid_budget)
            if available is not None:
                covers = budget_mod.covered([e.credits for e in paid], available)
                if covers == 0:
                    raise InsufficientCredits(extra={"budget": paid_budget})
                if covers < len(paid):
                    self.reporter.say("run.plan_covers", budget=paid_budget, covers=covers,
                                      paid=len(paid))
        self._confirm_dicom(job, ready, paid_credits, yes or job.max_credits is not None,
                            interactive, asked)

    def _confirm_dicom(self, job: _Job, ready: List[_Entry], credits: float, yes: bool,
                       interactive: bool, asked: bool) -> None:
        """A DICOM folder that holds several scans is a cost surprise:
        go ahead only with ``yes``, after the budget question, or after a
        yes in a terminal."""
        per_folder: Dict[str, int] = {}
        for entry in ready:
            if entry.source_kind == "dicom_folder":
                per_folder[entry.path] = per_folder.get(entry.path, 0) + 1
        if not any(n > 1 for n in per_folder.values()) or yes or asked:
            return
        count = len(ready)
        if interactive:
            if budget_mod.confirm(self.reporter, "run.confirm_dicom", count=count,
                                  credits=f"{credits:.2f}"):
                return
            raise ConfirmationRequired(messages.text("error.confirmation_declined"),
                                       code="confirmation_declined")
        raise ConfirmationRequired(extra={"count": count, "credits": f"{credits:.2f}"})

    def _fallback_budget(self, job: _Job, interactive: bool) -> str:
        """Rule 4: a task sent as ``free`` was refused because the free task
        was spent elsewhere. The remaining tasks follow rule 3 (asked once)."""
        with self._pay_lock:
            if job.paid_budget:
                return job.paid_budget
            plan = job.plan
            remaining = [e for e in job.entries if e.ready and e.task is None
                         and e.budget == budget_mod.FREE]
            if plan is None or not plan.has_lab:
                job.paid_budget = budget_mod.PERSONAL
            elif job.yes:
                job.paid_budget = plan.suggested
            elif interactive:
                job.paid_budget = budget_mod.ask(
                    plan, credits=round(sum(e.credits for e in remaining), 2),
                    count=len(remaining), reporter=self.reporter,
                    estimate_budgets=(job.estimate or {}).get("budgets"))
            else:
                raise BudgetChoiceRequired(plan.options, suggested=plan.suggested)
            self.reporter.say("run.free_ran_out", budget=job.paid_budget)
            return job.paid_budget

    # ── uploading and queueing ────────────────────────────────────────────────
    def _upload_all(self, job: _Job, *, interactive: bool) -> None:
        """Upload and queue every scan, two at a time. The first error that
        concerns the whole run stops the other worker at its next pause."""
        todo = [e for e in sorted(job.entries, key=lambda e: e.order) if e.ready]
        if not todo:
            return
        job.interactive = interactive
        if any(e.budget != budget_mod.FREE for e in todo):
            # The tasks sent as "free" are submitted before the paid ones:
            # the server pays the first task it receives with a free task,
            # whatever budget that task names.
            for entry in todo:
                entry.holds_free = entry.budget == budget_mod.FREE
            job.free_inflight = sum(1 for e in todo if e.holds_free)
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nalvera-run")
        # Each worker runs in a copy of the caller's context: in a Jupyter
        # kernel its lines are then shown under the cell that started the run.
        futures = [pool.submit(contextvars.copy_context().run, self._one, job, entry)
                   for entry in todo]
        try:
            wait_futures(futures, return_when=FIRST_EXCEPTION)
        except BaseException:
            self._transport.stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        if any(f.done() and not f.cancelled() and f.exception() is not None for f in futures):
            self._transport.stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        failed = [f.exception() for f in futures if f.done() and not f.cancelled()
                  and f.exception() is not None]
        if failed:
            # The cause, not a worker that stopped because of it.
            raise next((e for e in failed if not isinstance(e, Cancelled)), failed[0])

    def _one(self, job: _Job, entry: _Entry) -> None:
        """One scan (in a worker thread): encrypt, record, upload, queue."""
        try:
            if self._transport.stop.is_set():
                raise Cancelled()
            if job.halted:
                self._leave_out(entry, job.halted)
                return
            scan = entry.prepared
            try:
                self._make_upload_file(job, entry)
                if not entry.recovering:
                    # Write-ahead (B2). A recovered scan keeps its first code
                    # and row: no second code is ever recorded for it.
                    self._record(job, "record_scan", scan)
                uploaded = job.uploader.upload(entry.upload_path,
                                               filename=entry.upload_filename,
                                               size=entry.upload_size, display_name=entry.name)
                entry.upload_id = uploaded.upload_id
            except (ScanRejected, NotFound) as error:
                self._fail(entry, error, job)
                return
            except NalveraError as error:
                if error.code != "upload_failed":
                    raise
                self._fail(entry, error, job)
                return
            finally:
                self._remove_prepared(entry)
            self._submit_scan(job, entry)
        finally:
            self._free_settled(job, entry)

    def _free_settled(self, job: _Job, entry: _Entry) -> None:
        """A task sent as ``free`` has its answer (or will never be sent)."""
        with job.free_done:
            if entry.holds_free:
                entry.holds_free = False
                job.free_inflight -= 1
                job.free_done.notify_all()

    def _after_free_tasks(self, job: _Job, entry: _Entry) -> None:
        """Hold a paid submit until every other task sent as ``free`` has
        its answer."""
        with job.free_done:
            while job.free_inflight > (1 if entry.holds_free else 0):
                if self._transport.stop.is_set():
                    raise Cancelled()
                job.free_done.wait(0.2)

    def _make_upload_file(self, job: _Job, entry: _Entry) -> None:
        """The file to upload: the anonymized scan, or its NVLT container.
        The plaintext copy is removed as soon as the container is written,
        and a resume re-sends the same container (encryption is randomized)."""
        scan = entry.prepared
        if job.key is None:
            entry.upload_path, entry.upload_filename = str(scan.path), _sent_name(entry)
            entry.upload_size = int(scan.size)
            return
        target = os.path.join(job.workdir, scan.filename + encrypt_mod.SUFFIX)
        try:
            size = encrypt_mod.encrypt_file(str(scan.path), target, job.key)
        finally:
            _remove(scan.path)
        entry.upload_path, entry.upload_filename = target, _sent_name(entry) + encrypt_mod.SUFFIX
        entry.upload_size = int(size)

    def _submit_scan(self, job: _Job, entry: _Entry) -> None:
        """Queue the task of one uploaded scan, with its budget; follows
        rule 4 when a ``free`` task is refused, and the spending cap."""
        scan = entry.prepared
        # A recovered scan is sent again under its first code, with the key
        # the earlier run used: a request of it that landed is answered with
        # its task, never queued twice.
        code = entry.idem_code or scan.code
        key = _scan_idempotency_key(code, entry.idem_sha or scan.sha256, job.model, job.task,
                                    job.text_digest)
        while True:
            if entry.budget != budget_mod.FREE:
                self._after_free_tasks(job, entry)
            with self._submit_turn(job, entry):
                if self._transport.stop.is_set():
                    raise Cancelled()
                if job.halted:
                    self._leave_out(entry, job.halted)
                    return
                if not self._reserve(job, entry):
                    return
                body = {"upload_id": entry.upload_id, "model": job.model, "task": job.task,
                        "budget": entry.budget or budget_mod.PERSONAL,
                        "name": _sent_name(entry), "label": job.label, "text": job.text}
                try:
                    answer, attempts, ran = self._submit(body, key)
                except NoFreeTasks:
                    self._unreserve(job, entry)
                    self._free_settled(job, entry)
                    if job.plan is not None and job.plan.requested == budget_mod.FREE:
                        job.halted = "no_free_tasks"
                        self._leave_out(entry, "no_free_tasks")
                        return
                    fallback = self._fallback_budget(job, job.interactive)
                    for other in job.entries:
                        if other.task is None and other.budget == budget_mod.FREE:
                            other.budget = fallback
                            # No longer sent as free: nothing waits for its answer.
                            self._free_settled(job, other)
                    continue                  # the same key: the refused request made nothing
                except (ScanRejected, NotFound) as error:
                    self._unreserve(job, entry)
                    self._fail(entry, error, job)
                    return
                except BaseException as error:
                    # A submit that may have queued and charged the task keeps
                    # its reservation, and no other task is sent: the run
                    # stops (a run again finds the task by its key).
                    if not getattr(error, "may_have_run", False):
                        self._unreserve(job, entry)
                    self._transport.stop.set()
                    raise
                replayed = bool(answer.get("idempotent_replay"))
                task = Task.from_api(answer["task"], original_name=entry.name, code=code,
                                     label=job.label, charged=answer.get("charged"),
                                     replayed=replayed, charged_now=_counts(replayed, ran))
                entry.task = task
                self._settle(job, entry, task)
            break
        self._free_settled(job, entry)
        self._record(job, "record_task", code, task.id, job.model, job.task, job.text_digest)
        self._say_queued(task, answer, expected=entry.recovering or attempts > 1)

    @staticmethod
    def _submit_turn(job: _Job, entry: _Entry) -> Any:
        """Under a spending cap, paid tasks are submitted and settled one at
        a time (uploads still run two at a time): each is weighed against
        what the run has charged so far, so a charge above its estimate is
        counted before the next task is weighed."""
        if job.max_credits is None or entry.budget == budget_mod.FREE:
            return contextlib.nullcontext()
        return job.submit_lock

    def _reserve(self, job: _Job, entry: _Entry) -> bool:
        """Hold this task's estimate against ``max_credits`` (B4). False
        (and the scan left out) when it would pass the cap."""
        if job.max_credits is None or entry.budget == budget_mod.FREE:
            return True
        with self._pay_lock:
            if job.charged + job.reserved + entry.credits > job.max_credits + 1e-9:
                job.halted = "max_credits_exceeded"
                self._leave_out(entry, "max_credits_exceeded")
                return False
            job.reserved += entry.credits
            entry.reserved = entry.credits
        return True

    def _unreserve(self, job: _Job, entry: _Entry) -> None:
        with self._pay_lock:
            job.reserved -= entry.reserved
            entry.reserved = 0.0

    def _settle(self, job: _Job, entry: _Entry, task: Task) -> None:
        # A replay of an earlier run's request charges nothing now; a replay
        # that answers this call's own request (its first answer lost) does.
        charged = _credits_of((task.charged or {}).get("credits")) if task.charged_now else 0.0
        with self._pay_lock:
            job.reserved -= entry.reserved
            entry.reserved = 0.0
            job.charged = round(job.charged + charged, 2)

    def _leave_out(self, entry: _Entry, code: str) -> None:
        if entry.not_queued is None:
            entry.not_queued = ScanFailure(entry.name, entry.path, code,
                                           messages.text(f"run.left_out.{code}"))
            self.reporter.say("run.left_out", name=entry.name,
                              reason=messages.text(f"run.left_out.{code}"))

    def _submit(self, body: Dict[str, Any], key: str) -> Tuple[Dict[str, Any], int, bool]:
        """POST /api/v1/tasks with ``key``, waiting while the account has the
        maximum number of API tasks waiting or running. Returns the answer,
        how many attempts the last request took, and whether an attempt of
        this submit may have run on the server without its answer arriving
        (then a replayed answer is this submit's own task)."""
        waited, ran = 0.0, False
        try:
            while True:
                try:
                    answer = self._transport.json(
                        "POST", "/api/v1/tasks", json=body, headers={"Idempotency-Key": key},
                        idempotent=True,
                        retry_codes=("rate_limited", "queue_full", "storage_full",
                                     "temporarily_unavailable"))
                    ran = ran or self._transport.last_may_have_run
                    return answer, self._transport.last_attempts, ran
                except RateLimited as full:
                    ran = ran or self._transport.last_may_have_run
                    if full.code != "too_many_active_tasks":
                        raise
                    minimum = _poll_floor(self.me().get("limits") or {})
                    seconds = max(minimum, full.retry_after or 60.0)
                    if waited + seconds > _ACTIVE_CAP_MAX_WAIT:
                        raise
                    self.reporter.say("run.active_cap_wait", limit=full.extra.get("limit", "?"),
                                      seconds=int(seconds))
                    self._transport.pause(seconds)
                    waited += seconds
        except BaseException as error:
            # Whether the platform may have queued (and charged) the task
            # although no answer says so: the caller keeps the charge held.
            error.may_have_run = ran or self._transport.last_may_have_run  # type: ignore
            raise

    def _release_uploads(self, entries: List[_Entry]) -> None:
        """Best effort, after a run stopped: delete the uploads the server
        stored for scans that were never queued, instead of leaving them
        until they expire."""
        for entry in list(entries):
            if entry.upload_id and entry.task is None:
                try:
                    self._transport.request("DELETE", f"/api/v1/uploads/{entry.upload_id}",
                                            max_total_wait=0, timeout=(5.0, 10.0))
                except Exception:
                    pass

    def _fail(self, entry: _Entry, error: NalveraError, job: _Job) -> None:
        """A scan that will not be queued. Its line is said when the run
        ends (``_tell_failures``): when no scan was queued, the call raises
        its error instead, which says the same."""
        entry.failure = ScanFailure(original_name=entry.name, original_path=entry.path,
                                    code=error.code, message=str(error), error=error)
        with job.lock:
            if entry not in job.entries:
                job.entries.append(entry)

    def _record(self, job: _Job, method: str, *args: Any) -> None:
        self._tell_record()
        try:
            with self._mapping_lock:
                getattr(job.mapping, method)(*args)
        except job.anonymize.AnonymizeError as error:
            raise _mapping_error(error, self.mapping_path) from None

    @staticmethod
    def _remove_prepared(entry: _Entry) -> None:
        if entry.prepared is not None:
            _remove(entry.prepared.path)
        if entry.upload_path:
            _remove(entry.upload_path)

    # ── generate ──────────────────────────────────────────────────────────────
    def generate(self, model: str, task: str, params: Optional[Dict[str, Any]] = None, *,
                 budget: Optional[str] = None, label: Optional[str] = None,
                 yes: bool = False, max_credits: Optional[float] = None,
                 force: bool = False) -> Task:
        """Queue a generation: a task that makes new images from settings,
        with no input scan. Does not wait: use ``wait()``, then
        ``download()``.

        Example::

            task = nalvera.generate("nv_generate_ctmr", "rflow-ct",
                                    {"mode": "image-only", "output_size": [256, 256, 128],
                                     "spacing": [1.5, 1.5, 2.0]},
                                    budget="personal", label="phantom-01")

        ``model``
            Required. A generative model from ``models()``:
            ``"nv_generate_ctmr"``.
        ``task``
            Required. Options: ``"rflow-ct"`` or ``"rflow-mr-brain"``.
        ``params``
            Default None (no settings). The settings, such as
            ``{"mode": "image-only", "output_size": [256, 256, 128],
            "spacing": [1.5, 1.5, 2.0], "seed": 7}``. ``models()`` lists
            what each task accepts (its ``generation``); a setting the
            platform refuses raises ``InvalidRequest`` with its
            explanation.
        ``budget``, ``yes``, ``max_credits``
            As for ``run()``: a free task pays first.
        ``label``
            Default None. Shown on the platform; the result's file name
            starts with it. With a label, calling again with the same model,
            task and settings returns the earlier task (``earlier`` True)
            and queues nothing, also after a lost answer or a stopped call;
            one that failed or was cancelled is queued anew. Without a
            label, every call queues a task.
        ``force``
            Default False. True queues it again although a task with the
            same label, model, task and settings exists.

        Returns the ``Task``."""
        with self._call("generate"):
            return self._generate(model, task, params, budget=budget, label=label, yes=yes,
                                  max_credits=max_credits, force=force)

    def _generate(self, model: str, task: str, params: Optional[Dict[str, Any]], *,
                  budget: Optional[str], label: Optional[str], yes: bool,
                  max_credits: Optional[float], force: bool) -> Task:
        label = _clean_label(label)
        requested = budget_mod.parse_budget(budget if budget is not None else self.budget)
        cap = _parse_max_credits(max_credits) if max_credits is not None else self.max_credits
        me = self.me()
        _check_version(me)
        _require_scope(me, SCOPE_WRITE)
        model_info, task_info = self._resolve(model, task, scan=False)
        model_id, task_id = model_info["id"], task_info["id"]
        params = dict(params or {})
        digest = _params_digest(params)
        # With a label, every call with the same account, label, model, task
        # and settings sends the same Idempotency-Key: a call again after a
        # lost answer, a stop or failed retries gets the first task back and
        # charges nothing. An earlier task of these that failed or was
        # cancelled is "retired": it changes the key, so a call again queues
        # anew. Without a label (or with force) every call has a new key.
        stable = bool(label) and not force
        retired: List[str] = []
        if stable:
            earlier, retired = self._earlier_generation(model_id, task_id, label, digest)
            if earlier is not None:
                self.reporter.say("generate.earlier", label=label, task_id=earlier.id,
                                  state=_state_text(earlier))
                return earlier
        budgets = self.budgets()
        estimate = self.estimate(model_id, task_id, params=params)
        state = dict(budgets)
        state["free"] = _free_state(budgets, estimate)
        plan = budget_mod.plan(state, requested=requested, interactive=self._interactive,
                               assume_yes=yes, count=1)
        credits = _credits_of(estimate.get("credits"))
        is_free = plan.budget == budget_mod.FREE
        if not is_free and cap is not None and credits > cap + 1e-9:
            raise CreditCapExceeded(extra={"estimate": credits, "max_credits": cap})
        chosen = plan.budget
        if plan.free_count:
            self.reporter.say("run.plan_free", free=1, count=1)
        if chosen is None:
            chosen = budget_mod.ask(plan, credits=credits, count=1, reporter=self.reporter,
                                    estimate_budgets=estimate.get("budgets"))
        self.reporter.say("run.budget", budget=chosen)
        body = {"model": model_id, "task": task_id, "budget": chosen, "label": label,
                "params": params}

        def new_key() -> str:
            return _generation_key(self.email, self.base_url, label, model_id, task_id, digest,
                                   retired) if stable else generate_mod.new_idempotency_key()

        key = new_key()
        try:
            while True:
                try:
                    answer, attempts, ran = self._submit(body, key)
                except NoFreeTasks:
                    if requested == budget_mod.FREE or body["budget"] != budget_mod.FREE:
                        raise
                    if not plan.has_lab:
                        chosen = budget_mod.PERSONAL
                    elif yes:
                        chosen = plan.suggested
                    elif self._interactive:
                        chosen = budget_mod.ask(plan, credits=credits, count=1,
                                                reporter=self.reporter,
                                                estimate_budgets=estimate.get("budgets"))
                    else:
                        raise BudgetChoiceRequired(plan.options, suggested=plan.suggested) \
                            from None
                    if cap is not None and credits > cap + 1e-9:
                        raise CreditCapExceeded(extra={"estimate": credits, "max_credits": cap}) \
                            from None
                    self.reporter.say("run.free_ran_out", budget=chosen)
                    body = {**body, "budget": chosen}
                    continue                 # the same key: the refused request made nothing
                replayed = bool(answer.get("idempotent_replay"))
                result = Task.from_api(answer["task"], label=label, charged=answer.get("charged"),
                                       replayed=replayed, charged_now=_counts(replayed, ran))
                if stable and replayed and not result.charged_now \
                        and result.state not in _HELD_STATES and len(retired) < _MAX_RETIRED:
                    # An earlier call's task with this key that failed or was
                    # cancelled, which the record of this computer did not hold.
                    self._record_generation(result.id, model_id, task_id, label, digest)
                    retired.append(result.id)
                    key = new_key()
                    continue
                break
        except BaseException as error:
            if not stable and getattr(error, "may_have_run", False):
                # A call again would queue a second generation: say that this
                # one may be queued (a labelled call again is safe).
                if isinstance(error, NalveraError):
                    error.message = f"{error.message} {messages.text('generate.may_be_queued')}"
                    error.args = (error.message,)
                else:
                    self.reporter.say("generate.may_be_queued")
            raise
        if stable and result.replayed and not result.charged_now:
            result.earlier = True                    # queued by an earlier call
        try:
            if label:
                self._labels[result.id] = label
            self._record_generation(result.id, model_id, task_id, label, digest)
            if result.earlier:
                self.reporter.say("generate.earlier", label=label, task_id=result.id,
                                  state=_state_text(result))
            else:
                self._say_queued(result, answer, expected=attempts > 1)
        except BaseException:
            charge = _credits_of((result.charged or {}).get("credits")) \
                if result.charged_now else 0.0
            self._tell_stopped([result], charge)
            raise
        return result

    def _earlier_generation(self, model: str, task: str, label: str,
                            digest: str) -> Tuple[Optional[Task], List[str]]:
        """The task of a generation with this label, model, task and
        settings that the record holds and that is still waiting, running or
        completed (None when there is none), and the ids of the record's
        other tasks of it (failed, cancelled or deleted), which change the
        Idempotency-Key of a new one."""
        rows = [r for r in self._record_rows() if r.get("kind") == "generation"
                and r.get("model") == model and r.get("task") == task
                and r.get("label") == label and r.get("params_sha256") == digest
                and r.get("task_id")]
        if not rows:
            return None, []
        ids = list(dict.fromkeys(str(r["task_id"]) for r in rows))
        found = self._lookup(ids)
        held = [found[t] for t in ids if found.get(t) is not None
                and found[t].state in _HELD_STATES]
        if not held:
            return None, ids
        item = self._local(held[-1], None)
        item.label = item.label or label
        item.earlier = True
        return item, []

    def _record_rows(self) -> List[Dict[str, Any]]:
        """Every row of the record; none when it is missing or unreadable."""
        if not self.mapping_path.exists():
            return []
        try:
            with self._mapping_lock:
                return list(_anonymizer().Mapping(str(self.mapping_path)).rows())
        except Exception:
            return []

    def _record_generation(self, task_id: str, model: str, task: str, label: Optional[str],
                           digest: str) -> None:
        """Record a queued generation, so that a call again finds it and a
        download names it by its label after a restart."""
        self._tell_record()
        try:
            self._prepare_record_folder()
            with self._mapping_lock:
                _anonymizer().Mapping(str(self.mapping_path)).record_generation(
                    task_id, model, task, label, digest)
        except Exception:
            self.reporter.say("download.not_recorded", path=str(self.mapping_path))

    def _prepare_record_folder(self) -> None:
        """The default record's folder, private to the user (mode 0700)."""
        if self._default_record:
            auth.private_dir(self.mapping_path.parent, levels=2)   # records, config folder

    # ── wait ──────────────────────────────────────────────────────────────────
    def wait(self, tasks: Union[TaskRef, Iterable[TaskRef], None] = None, *,
             timeout: Optional[float] = None,
             poll_seconds: Optional[float] = None) -> List[Task]:
        """Wait until every task has finished (completed, failed or
        cancelled) and return them, updated, as a list of ``Task`` that
        a notebook shows as a table.

        Example::

            finished = nalvera.wait(tasks)

        ``tasks``
            Default None: every task of the account that is waiting or
            running (after a restart, this finds them again). Options: a
            ``Task``, a task id, or a list of them (such as what
            ``run()`` returned).
        ``timeout``
            Default None (no limit). Seconds; past them
            ``NalveraError`` (``wait_timeout``) is raised and the tasks
            keep running.
        ``poll_seconds``
            Default None: the key's ``min_poll_seconds``. A slower start
            for the checks.

        Shows, per task, the original name, the state (``cancelling`` while
        a cancel is under way), the queue position, the time waited and the
        time running (in a notebook as one table redrawn in place), then a
        summary. Checks no more often than the key's ``min_poll_seconds``,
        slowing to every 30 seconds while nothing changes. Finished tasks
        return at once. Stopping the cell raises ``WaitInterrupted``;
        the tasks keep running, and ``nalvera.wait()`` picks them up
        again."""
        with self._call("wait"):
            return self._wait(tasks, timeout=timeout, poll_seconds=poll_seconds)

    def _wait(self, tasks: Any, *, timeout: Optional[float],
              poll_seconds: Optional[float]) -> List[Task]:
        refs = self._active_refs() if tasks is None else _as_refs(tasks)
        if not refs:
            self.reporter.say("wait.nothing")
            return display.TaskList()
        ids = list(dict.fromkeys(_task_id(r) for r in refs))
        known: Dict[str, Task] = {_task_id(r): r for r in refs if isinstance(r, Task)}
        shown: Dict[str, Tuple[str, Optional[int], bool]] = {}
        live = self.reporter.supports_live and len(ids) <= _LIVE_TABLE_MAX_ROWS
        clock = self._transport.clock
        start = clock()
        minimum = interval = None
        current: Dict[str, Task] = {}
        try:
            while True:
                current = self._fetch(ids, known)
                changed = False
                lines = []
                for tid in ids:
                    item = current[tid]
                    mark = (item.state, item.queue_position, item.cancel_requested)
                    fields = self._wait_fields(item)
                    lines.append(messages.text("wait.line", **fields))
                    if shown.get(tid) != mark:
                        changed = True
                        shown[tid] = mark
                        if not live:
                            self.reporter.say("wait.line", **fields)
                if live:
                    self.reporter.live(lines)
                if all(current[tid].finished for tid in ids):
                    break
                if minimum is None:
                    minimum = _poll_floor(self.me().get("limits") or {})
                    interval = max(minimum, float(poll_seconds or 0))
                if timeout is not None and clock() - start + interval > timeout:
                    raise NalveraError(messages.text("error.wait_timeout",
                                                     seconds=int(timeout)),
                                       code="wait_timeout")
                self._transport.pause(interval)
                interval = max(minimum, float(poll_seconds or 0)) if changed \
                    else min(_MAX_POLL_SECONDS, interval * 1.5)
                known.update(current)
        except KeyboardInterrupt:
            raise WaitInterrupted([current.get(t) or known.get(t) or t for t in ids]) from None
        except NalveraError as error:
            if error.code != "wait_timeout":           # its text says the tasks keep running
                going = [t for t in ids if not getattr(current.get(t) or known.get(t),
                                                       "finished", False)]
                if going and getattr(self.reporter, "hides_progress", False):
                    # The live table is emptied when the call ends: one line stays.
                    self.reporter.say("wait.stopped", count=len(going),
                                      ids=messages.text("list.separator").join(going))
            raise
        finally:
            if live:
                self.reporter.live_done()
        done = display.TaskList(current[tid] for tid in ids)
        if len(done) > 1:                     # one task: its own line says how it ended
            self._say_counts("wait.summary", [(s, sum(1 for t in done if t.state == s))
                                              for s in FINISHED_STATES], "list.separator")
        for item in done:
            if item.state == "failed" and item.error:
                self.reporter.say("wait.failed_detail", name=item.display_name, error=item.error)
        return done

    def _active_refs(self) -> List[Task]:
        """Every task of the account that is waiting or running (one request
        per 100 tasks)."""
        found: List[Task] = []
        offset = 0
        for _page in range(20):
            page, total = self._list("active", 100, offset)
            found += [self._local(t, None) for t in page]
            offset += len(page)
            if not page or offset >= total:
                break
        return found

    def _fetch(self, ids: List[str], known: Dict[str, Any]) -> Dict[str, Task]:
        """The tasks' current state. A finished task is final: what is known
        of it needs no request. The others: a few one by one, many through
        the list (one request per 100 tasks)."""
        found: Dict[str, Task] = {tid: known[tid] for tid in ids
                                  if isinstance(known.get(tid), Task) and known[tid].finished}
        pending = [tid for tid in ids if tid not in found]
        if len(pending) > 3:
            found.update(self._from_list(pending))
        for tid in pending:
            if tid not in found:
                self.me()
                found[tid] = Task.from_api(self._transport.json("GET", f"/api/v1/tasks/{tid}"))
        return {tid: self._local(found[tid], known.get(tid)) for tid in ids}

    def _from_list(self, ids: Sequence[str]) -> Dict[str, Task]:
        found: Dict[str, Task] = {}
        wanted, offset, pages = set(ids), 0, 0
        while wanted - set(found) and pages < 20:
            page, total = self._list("all", 100, offset)
            for item in page:
                if item.id in wanted:
                    found[item.id] = item
            offset += len(page)
            pages += 1
            if not page or offset >= total:
                break
        return found

    def _lookup(self, ids: Sequence[str]) -> Dict[str, Optional[Task]]:
        """Tasks by id; None for a task that no longer exists (deleted on
        the website)."""
        if not ids:
            return {}
        found: Dict[str, Optional[Task]] = dict(self._from_list(ids)) if len(ids) > 3 else {}
        for tid in ids:
            if tid not in found:
                try:
                    found[tid] = Task.from_api(self._transport.json("GET",
                                                                    f"/api/v1/tasks/{tid}"))
                except NotFound:
                    found[tid] = None
        return found

    def _wait_fields(self, item: Task) -> Dict[str, Any]:
        position = ""
        if item.state == "queued" and item.queue_position is not None:
            position = messages.text("wait.position", position=item.queue_position)
        return {"name": printable(item.display_name), "state": _state_text(item),
                "position": position, "waited": _duration(item.waited_seconds),
                "running": _duration(item.running_seconds)}

    def _wait_line(self, item: Task) -> str:
        return messages.text("wait.line", **self._wait_fields(item))

    # ── download ──────────────────────────────────────────────────────────────
    def download(self, tasks: Union[TaskRef, Iterable[TaskRef], None] = None,
                 out: PathLike = "results", *, part: Optional[str] = None,
                 overwrite: bool = False) -> Downloads:
        """Save the results of completed tasks in a folder and return their
        paths, as a ``Downloads`` list that a notebook shows as a table
        (``skipped`` lists tasks without a result).

        Example::

            paths = nalvera.download(finished, out="results")

        ``tasks``
            Default None: every completed task of this computer's record
            whose result is not saved yet (after a restart, this finds
            them again). Options: a ``Task``, a task id, or a list of
            them.
        ``out``
            Default ``"results"``. The folder; created when missing.
        ``part``
            Default None: the whole result (a generation with several files
            comes as one ZIP). Options: one of the task's ``parts``, such as
            ``"image"`` or ``"mask"``.
        ``overwrite``
            Default False. True downloads every result again and replaces
            the file of the same name.

        A scan's result is named after the original scan:
        ``<original name>__<model>__<task>.<ext>``; where two results of one
        download would get the same name, the parent folder's name is added
        in front. A generation's result is
        ``<label or task id>__<model>__<task>.<ext>``. The extension comes
        from the platform's file name. A result an earlier call saved in
        ``out``, still there with its recorded size, is returned without
        downloading it again; a missing one is downloaded again. An existing
        file is never replaced unless ``overwrite=True``: when the name is
        taken (by another file, or by a saved result that changed since),
        the new one gets a number."""
        with self._call("download"):
            return self._download(tasks, out, part=part, overwrite=overwrite)

    def _download(self, tasks: Any, out: PathLike, *, part: Optional[str],
                  overwrite: bool) -> Downloads:
        if tasks is None:
            return self._download_pending(Path(os.path.abspath(os.path.expanduser(
                os.fspath(out)))), part)
        refs = _as_refs(tasks)
        me = self.me()
        _require_scope(me, SCOPE_RESULTS)
        folder = Path(os.path.abspath(os.path.expanduser(os.fspath(out))))
        self._check_allowed(str(folder))
        items, skipped = [], []
        for ref in refs:
            item = ref if isinstance(ref, Task) and ref.state == "completed" else self.task(ref)
            item = self._local(item, ref)
            if item.state != "completed":
                skipped.append((item.id, item.state))
                self.reporter.say("download.skipped", name=item.display_name,
                                  state=_state_text(item))
                continue
            items.append(item)
        kept: Dict[int, Path] = {}
        fetch: List[Task] = []
        for number, item in enumerate(items):
            saved = None if overwrite else self._saved_result(item, folder, part)
            if saved is not None:
                kept[number] = saved
                self.reporter.say("download.already_saved", name=item.display_name,
                                  path=display.path_text(saved, cut=False))
                continue
            fetch.append(item)
        fetched: List[Path] = []
        if fetch:
            _writable_folder(folder)
            fetched = self._save_results(fetch, folder, part, overwrite,
                                         kept=[items[n] for n in sorted(kept)])
        paths, queue = [], iter(fetched)
        for number in range(len(items)):
            paths.append(kept[number] if number in kept else next(queue))
        return Downloads(paths, skipped)

    def _download_pending(self, folder: Path, part: Optional[str]) -> Downloads:
        """``download()`` with no tasks: the completed tasks of the record
        whose result is not saved yet."""
        if not self.mapping_path.exists():
            raise ConfigError(messages.text("error.no_local_record",
                                            path=printable(str(self.mapping_path))),
                              code="no_local_record")
        pending = _not_downloaded(self._record_rows(), part)
        if not pending:
            self.reporter.say("download.nothing_new", path=str(self.mapping_path))
            return Downloads()
        _require_scope(self.me(), SCOPE_RESULTS)
        found = self._lookup(pending)
        ready = [found[t] for t in pending if found.get(t) and found[t].state == "completed"]
        running = sum(1 for t in pending if found.get(t) and not found[t].finished)
        saved = self.download(ready, out=folder, part=part) if ready else Downloads()
        if not ready:
            self.reporter.say("download.nothing_new", path=str(self.mapping_path))
        if running:
            self.reporter.say("download.still_running", count=running)
        return saved

    def _saved_result(self, item: Task, folder: Path, part: Optional[str]) -> Optional[Path]:
        """The result (or ``part``) an earlier call saved in ``folder``, when
        it is still there with the size the record holds; None otherwise. A
        file that changed since is left as it is, and the result is
        downloaded again under a new name."""
        for row in reversed(self._record_rows()):
            if row.get("task_id") != item.id or not row.get("result_path") \
                    or (row.get("result_part") or "") != (part or ""):
                continue
            path = Path(row["result_path"])
            if path.parent != folder:
                continue
            try:
                if path.is_file() and path.stat().st_size == int(row.get("result_size") or -1):
                    return path
            except (OSError, ValueError):
                continue
        return None

    def _save_results(self, items: List[Task], folder: Path, part: Optional[str],
                      overwrite: bool, kept: Sequence[Task] = ()) -> List[Path]:
        """Download and save each result. A file name is made of the
        stem, model, task, part and extension; the parent folder's name goes
        in front of the stem only where two results of this download would
        otherwise share one (``kept``: results of this download found saved,
        which count for that). The extension is known once a result arrives,
        so results whose names agree without it are held until all of them
        have arrived; the others are saved at once."""
        stems = [self._result_stem(item) for item in items]
        bases = [_result_base(stem, item, part) for stem, item in zip(stems, items)]
        counts: Dict[str, int] = {}
        for base in bases:
            counts[base] = counts.get(base, 0) + 1
        everywhere = dict(counts)
        for item in kept:
            base = _result_base(self._result_stem(item), item, part)
            everywhere[base] = everywhere.get(base, 0) + 1
        saved: Dict[int, Path] = {}
        held: Dict[str, List[Tuple[int, Path, str]]] = {}
        try:
            for number, item in enumerate(items):
                temp, extension = self._fetch_result(item, folder, bases[number], part)
                if everywhere[bases[number]] == 1:
                    saved[number] = self._place_result(item, temp, folder, bases[number],
                                                       extension, overwrite, part=part)
                    continue
                group = held.setdefault(bases[number], [])
                group.append((number, temp, extension))
                if len(group) == counts[bases[number]]:
                    del held[bases[number]]
                    # A name shared with a saved result: the folder names, as
                    # that result has.
                    self._place_group(items, stems, group, folder, part, overwrite, saved,
                                      complete=everywhere[bases[number]] == len(group))
        except BaseException:
            # What arrived is kept. A group cut short gets the folder names,
            # as it would have had with its other members.
            for group in held.values():
                try:
                    self._place_group(items, stems, group, folder, part, overwrite, saved,
                                      complete=False)
                except Exception:
                    for _number, temp, _extension in group:
                        self._drop_partial(temp)
            raise
        return [saved[number] for number in range(len(items))]

    def _place_group(self, items: List[Task], stems: List[str],
                     group: List[Tuple[int, Path, str]], folder: Path, part: Optional[str],
                     overwrite: bool, saved: Dict[int, Path], complete: bool = True) -> None:
        """Save results whose names agree up to the extension: those that
        share their extension too get their parent folder's name in front."""
        extensions: Dict[str, int] = {}
        for _number, _temp, extension in group:
            extensions[extension] = extensions.get(extension, 0) + 1
        for number, temp, extension in group:
            item, stem = items[number], stems[number]
            if extensions[extension] > 1 or not complete:
                row = self._mapping_row(item.id)
                parent = Path(str(row.get("original_path") or "")).parent.name if row else ""
                if parent:
                    stem = f"{parent}_{stem}"
            saved[number] = self._place_result(item, temp, folder,
                                               _result_base(stem, item, part), extension,
                                               overwrite, part=part)

    def _fetch_result(self, item: Task, folder: Path, base: str,
                      part: Optional[str]) -> Tuple[Path, str]:
        """Download one result into a hidden ``.part`` file in ``folder``;
        returns that file and the result's extension. A file that arrives
        short or a broken connection starts over, three times at most."""
        attempts, started = 0, self._closings
        while True:
            attempts += 1
            temp = folder / f".{base}.{uuid.uuid4().hex[:8]}.part"
            with self._partials_lock:
                if self._closings != started:
                    raise Cancelled()                    # close() from another thread
                self._partials.add(str(temp))
            try:
                response = self._transport.request(
                    "GET", f"/api/v1/tasks/{item.id}/result",
                    params={"part": part} if part else None, stream=True, timeout=(10.0, 300.0))
                try:
                    extension = _extension(response.headers)
                    expected = _expected_length(response.headers)
                    received = 0
                    with open(temp, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if self._closings != started:
                                raise Cancelled()
                            if chunk:
                                fh.write(chunk)
                                received += len(chunk)
                    if expected is not None and received != expected:
                        raise _ShortDownload()
                finally:
                    response.close()
                return temp, extension
            except requests.RequestException:
                # The file arrived short or the connection broke: start over.
                self._drop_partial(temp)
                if attempts >= 3:
                    raise NetworkError(messages.text(
                        "error.network_error", reason=messages.text("net.failed")),
                        extra={"reason": "failed"}) from None
                self._transport.pause(min(30.0, 2.0 ** attempts))
            except BaseException:
                self._drop_partial(temp)
                raise

    def _place_result(self, item: Task, temp: Path, folder: Path, base: str, extension: str,
                      overwrite: bool, *, part: Optional[str] = None) -> Path:
        """Move a downloaded result into place and record its path, part and
        size."""
        try:
            target, existing = _place(temp, folder, base, extension, overwrite)
        except BaseException:
            self._drop_partial(temp)
            raise
        with self._partials_lock:
            self._partials.discard(str(temp))
        self._record_result(item.id, target, part)
        if existing is not None:
            self.reporter.say("download.renamed", name=item.display_name,
                              existing=existing.name, path=display.path_text(target, cut=False))
        else:
            self.reporter.say("download.saved", name=item.display_name,
                              path=display.path_text(target, cut=False))
        return target

    def _drop_partial(self, temp: Any) -> None:
        _remove(temp)
        with self._partials_lock:
            self._partials.discard(str(temp))

    def _result_stem(self, item: Task) -> str:
        row = self._mapping_row(item.id)
        if row:
            stem = _strip_scan_suffix(row.get("original_name") or "")
            series = row.get("series")
            if series:
                stem = f"{stem}_{series}"
            if stem:
                return stem
        if self._is_generative(item.model):
            # Another client or process knows the label from the task's name.
            label = item.label or self._labels.get(item.id) or _label_from_name(item.name)
            return (label or item.id).replace(" ", "_")
        return _strip_scan_suffix(item.name or "") or item.id

    # ── the record on this computer ───────────────────────────────────────────
    def scans(self) -> List[Dict[str, Any]]:
        """The record on this computer: for each scan its neutral code,
        original name, label, tasks and result files, and each generation;
        one dict per row (a list that a notebook shows as a table). Its
        ``path`` is the record's file. Sends no request and needs no key:
        without a sign-in it reads the record of the account named by
        ``NALVERA_EMAIL`` or a stored login.

        Example::

            nalvera.scans()
        """
        return read_record(self.mapping_path)

    def save_anonymized(self, paths: Union[PathLike, Sequence[PathLike]],
                        out: PathLike) -> SavedCopies:
        """Write the anonymized copies of scans into the folder ``out``,
        exactly as ``run`` would upload them. Sends nothing; needs no key.

        Example::

            copies = nalvera.save_anonymized("ct_scan.nii.gz", out="anonymized")

        ``paths``
            Required. A ``.nii``/``.nii.gz`` file, a folder of NIfTI files,
            a folder of DICOM files, or a list of these.
        ``out``
            Required. The folder; created when missing. It holds the copies
            and nothing else, each named by its new code, so it can be
            shared. The record that links each code to its original name
            stays on this computer, in the account's record (``scans()``
            shows it).

        Returns the copies (a list that a notebook shows as a table): per
        scan the copy's path, its shape and what was removed; ``failed``
        lists scans that could not be prepared."""
        return save_anonymized(paths, out, mapping=self.mapping_path, reporter=self.reporter,
                               check=self._check_allowed)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _resolve(self, model: str, task: Optional[str], *, scan: bool):
        listed = self.models()
        info = next((m for m in listed if m.get("id") == model), None)
        if info is None:
            raise ConfigError(messages.text("error.unknown_model", model=model),
                              code="unknown_model")
        if scan and info.get("generative"):
            raise ConfigError(messages.text("error.model_is_generative", model=model),
                              code="model_is_generative")
        if not scan and not info.get("generative"):
            raise ConfigError(messages.text("error.model_needs_scan", model=model),
                              code="model_needs_scan")
        tasks = [t for t in info.get("tasks") or [] if isinstance(t, dict)]
        if task is None:
            chosen = next((t for t in tasks if t.get("default")), tasks[0] if tasks else None)
        else:
            chosen = next((t for t in tasks if t.get("id") == task), None)
        if chosen is None:
            raise ConfigError(messages.text(
                "error.unknown_task", model=model, task=task,
                tasks=messages.text("list.separator").join(str(t.get("id")) for t in tasks)),
                code="unknown_task")
        return info, chosen

    def _is_generative(self, model: Optional[str]) -> bool:
        try:
            return any(m.get("id") == model and m.get("generative") for m in self.models())
        except NalveraError:
            return False

    def _list(self, state: str, limit: int, offset: int,
              name: Optional[str] = None) -> Tuple[List[Task], int]:
        self.me()
        params: Dict[str, Any] = {"state": state, "limit": int(limit), "offset": int(offset)}
        if name is not None:
            params["name"] = name
        data = self._transport.json("GET", "/api/v1/tasks", params=params)
        items = [Task.from_api(t) for t in data.get("tasks") or []]
        return items, int(data.get("total") or 0)

    def _local(self, item: Task, ref: Any) -> Task:
        """Fill the fields only this computer knows: the original name (from
        an earlier Task object, or the mapping file), the code, the label."""
        if isinstance(ref, Task):
            item.original_name = item.original_name or ref.original_name
            item.code = item.code or ref.code
            item.label = item.label or ref.label
            item.charged = item.charged or ref.charged
        if item.original_name is None:
            row = self._mapping_row(item.id)
            if row:
                item.original_name = _display_name_row(row)
                item.code = item.code or row.get("code")
                item.label = item.label or row.get("label") or None
        if item.label is None and item.id in self._labels:
            item.label = self._labels[item.id]
        return item

    def _mapping_row(self, task_id: str) -> Optional[Dict[str, Any]]:
        """The mapping's row for a task, or None. Never creates the file and
        never fails: without a mapping, tasks show their platform names.
        Answers are kept until the file changes."""
        try:
            st = self.mapping_path.stat()
        except OSError:
            return None
        stamp = (st.st_mtime_ns, st.st_size)
        if self._mapping_cache[0] != stamp:
            self._mapping_cache = (stamp, {})
        cache = self._mapping_cache[1]
        if task_id not in cache:
            try:
                mapping = _anonymizer().Mapping(str(self.mapping_path))
                with self._mapping_lock:
                    rows = mapping.find(task_id=task_id)
            except Exception:
                rows = []
            cache[task_id] = rows[0] if rows else None
        return cache[task_id]

    def _record_result(self, task_id: str, path: Path, part: Optional[str] = None) -> None:
        """Record a saved result with its part and size, so that a download
        again finds it complete and fetches nothing."""
        self._tell_record()
        try:
            size = os.path.getsize(path)
            self._prepare_record_folder()
            mapping = _anonymizer().Mapping(str(self.mapping_path))
            with self._mapping_lock:
                mapping.record_result(task_id, str(path), part=part, size=size)
        except Exception:
            self.reporter.say("download.not_recorded", path=str(self.mapping_path))

    def _say_queued(self, item: Task, answer: Dict[str, Any], *, expected: bool) -> None:
        if answer.get("idempotent_replay") and not item.charged_now:
            # B6: the platform answered with an earlier request's task. For a
            # recovered run that is expected; otherwise say so. (A replay of
            # this call's own lost answer is this call's task: said as queued.)
            key = "run.already_queued" if expected else "run.unexpected_replay"
            self.reporter.say(key, name=item.display_name, task_id=item.id,
                              state=_state_text(item))
            return
        self.reporter.say("run.queued", name=item.display_name, task_id=item.id,
                          charge=_charge_text(answer.get("charged")))

    def _on_wait(self, seconds: float, error: NalveraError) -> None:
        if error.code == "too_many_active_tasks":
            return                                   # _submit announces it
        self.reporter.say("run.retry_wait", seconds=int(round(seconds)),
                          reason=_short_reason(error))


@dataclass
class _Job:
    """The state of one run."""
    anonymize: Any
    mapping: Any
    uploader: Uploader
    workdir: str
    model: str
    task: str
    label: Optional[str]
    text: Optional[str]
    force: bool
    max_credits: Optional[float]
    key: Optional[bytes]
    max_bytes: Optional[int]
    text_digest: str = ""
    entries: List[_Entry] = field(default_factory=list)
    lock: Any = field(default_factory=threading.Lock)
    estimate: Optional[Dict[str, Any]] = None
    plan: Optional[budget_mod.BudgetPlan] = None
    paid_budget: Optional[str] = None
    yes: bool = False
    interactive: bool = False
    charged: float = 0.0
    reserved: float = 0.0
    free_inflight: int = 0
    free_done: Any = field(default_factory=threading.Condition)
    halted: Optional[str] = None
    # Under a spending cap: one paid submit and its settling at a time.
    submit_lock: Any = field(default_factory=threading.Lock)


class BackgroundRun:
    """A ``Client.run`` started with ``Client.run_in_background``.

    Progress goes to the client's reporter (a ``ListReporter`` keeps
    it for polling). ``done()`` says whether the run ended; ``result()``
    waits for it and returns its ``RunResult`` or raises its error;
    ``cancel()`` stops it at the next pause (the tasks already queued keep
    running; ``result()`` then raises ``RunInterrupted``)."""

    def __init__(self, client: Client, paths: Any, model: str, task: Optional[str],
                 options: Dict[str, Any]):
        unknown = set(options) - {"budget", "label", "text", "yes", "encrypt", "max_credits",
                                  "force"}
        if unknown:
            raise TypeError(f"unexpected arguments: {', '.join(sorted(unknown))}")
        self._client = client
        self._result: Optional[RunResult] = None
        self._error: Optional[BaseException] = None
        self._finished = threading.Event()
        kwargs = {"budget": None, "label": None, "text": None, "yes": False, "encrypt": None,
                  "max_credits": None, "force": False, **options}

        def work() -> None:
            try:
                self._result = client._run(paths, model, task, interactive=False, **kwargs)
            except BaseException as error:             # handed to result()
                self._error = error
            finally:
                self._finished.set()

        # The thread runs in a copy of the caller's context: in a Jupyter
        # kernel that context names the cell that started the run, so the
        # run's lines are shown under that cell.
        context = contextvars.copy_context()
        self._thread = threading.Thread(target=context.run, args=(work,),
                                        name="nalvera-background-run", daemon=True)
        self._thread.start()

    def done(self) -> bool:
        """True once the run has ended (queued, failed or cancelled)."""
        return self._finished.is_set()

    def cancel(self) -> None:
        """Stop the run at its next pause. Tasks already queued keep running."""
        self._client._transport.stop.set()

    def result(self, timeout: Optional[float] = None) -> RunResult:
        """Wait (at most ``timeout`` seconds; default None: no limit) and
        return the ``RunResult``, or raise the run's error.
        ``TimeoutError`` when the run is still going."""
        if not self._finished.wait(timeout):
            raise TimeoutError("the run is still going")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    @property
    def error(self) -> Optional[BaseException]:
        """The error that ended the run, once it ended; None otherwise."""
        return self._error if self._finished.is_set() else None


class _ShortDownload(requests.RequestException):
    """A result that arrived with fewer or more bytes than announced."""


# ── module helpers ────────────────────────────────────────────────────────────
def _anonymizer():
    """The anonymizer module (``nalvera.anonymize``), imported when first
    needed so that ``--help`` and the account commands never load it."""
    return importlib.import_module("nalvera.anonymize")


def _mapping_error(error: Any, path: Path) -> ConfigError:
    return ConfigError(messages.text("error.mapping_unreadable", path=printable(str(path))),
                       code="mapping_unreadable")


def _parse_flag(value: Optional[bool], setting: str) -> Optional[bool]:
    """A yes/no choice: the value given, else the setting (``1``, ``true``,
    ``yes``, ``on`` or ``0``, ``false``, ``no``, ``off``), else None."""
    if value is not None:
        return bool(value)
    raw = settings.get(setting)
    if raw is None:
        return None
    word = raw.strip().lower()
    if word in ("1", "true", "yes", "on"):
        return True
    if word in ("0", "false", "no", "off"):
        return False
    raise ConfigError(messages.text("error.invalid_flag", name=setting, value=printable(raw)),
                      code="invalid_setting")


def _parse_max_credits(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        number = -1.0
    if not number >= 0:                                    # also refuses NaN
        raise ConfigError(messages.text("error.invalid_max_credits", value=printable(value)),
                          code="invalid_max_credits")
    return number


def _version_tuple(text: Any) -> Tuple[int, ...]:
    match = re.match(r"\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(text or ""))
    if not match:
        return (0,)
    return tuple(int(g or 0) for g in match.groups())


def _check_version(me: Dict[str, Any]) -> None:
    """A9: stop before any work when the platform needs a newer package."""
    minimum = me.get("minimum_client_version")
    if minimum and _version_tuple(__version__) < _version_tuple(minimum):
        raise ClientTooOld(code="client_too_old", extra={"minimum_version": str(minimum)})


def _free_state(budgets: Dict[str, Any], estimate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The free tasks as ``{"period_active", "free_tasks_left"}``: the
    estimate's figures are the newest (its ``source`` is ``global`` during a
    free period), the budgets' fill what it leaves out."""
    known = dict((budgets or {}).get("free") or {})
    latest = (estimate or {}).get("free") or {}
    period = bool(known.get("period_active")) or latest.get("source") == "global" \
        or bool(latest.get("period_active"))
    left = latest.get("free_tasks_left", known.get("free_tasks_left"))
    try:
        left = int(left or 0)
    except (TypeError, ValueError):
        left = 0
    return {"period_active": period, "free_tasks_left": left}


def _free_possible(budgets: Dict[str, Any]) -> bool:
    free = budgets.get("free") or {}
    return bool(free.get("period_active")) or int(free.get("free_tasks_left") or 0) > 0


def _available(estimate: Optional[Dict[str, Any]], budgets: Dict[str, Any],
               budget: str) -> Optional[float]:
    """What ``budget`` can spend, from the estimate's own figures."""
    for option in (estimate or {}).get("budgets") or []:
        if isinstance(option, dict) and option.get("budget") == budget:
            return _finite(option.get("available_credits"))
    for option in budget_mod.options_from_budgets(budgets):
        if option["budget"] == budget:
            return _finite(option["available_credits"])
    return None


def _finite(value: Any) -> Optional[float]:
    """A balance from an answer; None when it is no finite number."""
    number = bounded(value, 0.0, _MAX_CREDITS)
    return None if number is None or not math.isfinite(float(value)) else number


def _credits_of(value: Any) -> float:
    """Credits from an answer (a price, a charge): 0 or more. A negative,
    infinite, NaN or absurd value counts as more than any spending cap."""
    if value is None or value == "":
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return _MAX_CREDITS
    if not math.isfinite(number) or number < 0 or number > _MAX_CREDITS:
        return _MAX_CREDITS
    return number


def _poll_floor(limits: Dict[str, Any]) -> float:
    """The key's shortest interval between status checks (seconds), within
    sense: 5 when the answer names none."""
    return bounded(limits.get("min_poll_seconds"), 0.0, _MAX_POLL_FLOOR, 5.0) or 5.0


def _max_bytes(limits: Dict[str, Any]) -> Optional[int]:
    """The key's largest upload, or None when the answer names none."""
    number = bounded(limits.get("max_file_bytes"), 1, 2 ** 50)
    return int(number) if number else None


def _short_reason(error: NalveraError) -> str:
    reason = error.extra.get("reason") if error.code == "network_error" else None
    if reason and messages.has(f"net.{reason}"):
        return messages.text(f"net.{reason}")
    if messages.has(f"reason.{error.code}"):
        return messages.text(f"reason.{error.code}")
    return error.code


def _state_text(item: Task) -> str:
    return display.state_text(item.state, item.cancelling)


def _counts(replayed: bool, ran: bool) -> bool:
    """Whether a submit's charge counts toward this call: a new task, or a
    replay that may answer an attempt of this call (its answer lost). Where
    that replay could also be an earlier run's request, the charge counts:
    the spending cap errs on the safe side."""
    return not replayed or ran


def _charge_text(charged: Optional[Dict[str, Any]]) -> str:
    """What a submit charged, in words (``run.charge_*``)."""
    charged = charged if isinstance(charged, dict) else {}
    try:
        credits = f"{float(charged.get('credits') or 0.0):.2f}"
    except (TypeError, ValueError):
        credits = "-"
    source = charged.get("source")
    if source == "free":
        return messages.text("run.charge_free")
    if source == "lab":
        return messages.text("run.charge_lab", credits=credits,
                             lab_id=printable(charged.get("lab_id")))
    return messages.text("run.charge_personal", credits=credits)


def _generation_key(email: str, base_url: str, label: str, model: str, task: str,
                    digest: str, retired: Sequence[str] = ()) -> str:
    """The Idempotency-Key of a labelled generation: the same account,
    platform, label, model, task and settings (``digest``) give the same
    key. ``retired`` are the ids of earlier such tasks that failed or were
    cancelled; each makes another key."""
    identity = "\n".join([str(email).strip().lower(), str(base_url), label, model, task,
                          digest, *sorted(str(t) for t in retired)])
    return "gen-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:48]


def _params_digest(params: Dict[str, Any]) -> str:
    """A generation's settings as one SHA-256: the same settings in any
    order give the same digest."""
    import json
    text = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _not_downloaded(rows: List[Dict[str, Any]], part: Optional[str] = None) -> List[str]:
    """The task ids of the record, in its order, with no saved result (of
    ``part``) that is still there."""
    order: Dict[str, None] = {}
    saved = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        order.setdefault(task_id, None)
        path = row.get("result_path")
        if path and (row.get("result_part") or "") == (part or "") and os.path.exists(path):
            saved.add(task_id)
    return [t for t in order if t not in saved]


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _remove_stale_workdirs(base: Optional[str] = None,
                           max_age: float = _STALE_WORKDIR_SECONDS) -> int:
    """Remove this package's own temporary folders (``nalvera-run-*``,
    and ``nalvera-inspect-*`` of a DICOM estimate) left by a process that
    crashed or was killed, once older than a day. Only folders owned by this
    user. Returns how many were removed."""
    base = base or tempfile.gettempdir()
    removed = 0
    now = time.time()
    try:
        names = os.listdir(base)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(_STALE_PREFIXES):
            continue
        path = os.path.join(base, name)
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode) or now - st.st_mtime < max_age:
            continue
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


def _expand_paths(sources: Sequence[PathLike], anonymize: Any, check: Any,
                  say: Any) -> List[_Source]:
    """Each path as one or more scans to prepare. A folder of NIfTI
    files gives each file at its top level; a folder of DICOM files is
    converted as a whole; a folder with both, or with neither, stops naming
    the folder. ``check(path)`` refuses a path before it is read; ``say``
    reports a folder of NIfTI files."""
    out: List[_Source] = []
    finder = getattr(getattr(anonymize, "dicom", None), "list_dicom_files", None)
    for index, raw in enumerate(sources):
        path = os.path.abspath(os.path.expanduser(os.fspath(raw)))
        check(path)
        if not os.path.isdir(path):
            out.append(_Source(path=path, kind="file", index=index))
            continue
        nifti = _nifti_files(path)
        nifti_real = {os.path.realpath(p) for p in nifti}
        try:
            dicom = [f for f in (finder(path) if finder else _dicom_like_files(path))
                     if os.path.realpath(f) not in nifti_real]
        except Exception:
            # The finder could not tell (too many files, unreadable ones): a
            # folder without NIfTI files goes to the converter, which reports
            # what it finds.
            dicom = [] if nifti else [path]
        if nifti and dicom:
            raise ConfigError(messages.text("error.mixed_folder", path=printable(path)),
                              code="mixed_folder")
        if nifti:
            folder_real = os.path.realpath(path)
            for item in nifti:
                real = os.path.realpath(item)
                if os.path.dirname(real) != folder_real:
                    raise ConfigError(messages.text("error.path_outside_folder",
                                                    path=printable(item)),
                                      code="path_outside_folder")
                check(real)
                out.append(_Source(path=item, kind="nifti_in_folder", index=index))
            say("run.folder_nifti", path=path, count=len(nifti))
            continue
        if dicom:
            for item in dicom:
                check(os.path.realpath(item))   # a link out of the allowed folders is refused
            out.append(_Source(path=path, kind="dicom_folder", index=index))
            continue
        raise ConfigError(messages.text("error.no_scans_in_folder", path=printable(path)),
                          code="no_scans_in_folder")
    return out


def _nifti_files(folder: str) -> List[str]:
    """The NIfTI files at the top level of ``folder`` (by name), sorted."""
    out = []
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    for name in names:
        if name.startswith("."):
            continue
        path = os.path.join(folder, name)
        if name.lower().endswith(_NIFTI_SUFFIXES) and os.path.isfile(path):
            out.append(path)
    return out


def _dicom_like_files(folder: str, limit: int = 100000) -> List[str]:
    """Files under ``folder`` that look like DICOM ("DICM" at byte 128, or a
    .dcm name); used only when the anonymizer offers no finder."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "rb") as fh:
                    head = fh.read(132)
            except OSError:
                continue
            if head[128:132] == b"DICM" or name.lower().endswith(".dcm"):
                found.append(path)
                if len(found) >= limit:
                    return found
    return found


def _size_of(path: str) -> int:
    try:
        if os.path.isdir(path):
            total = 0
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
            return total
        return os.path.getsize(path)
    except OSError:
        return 0


def _writable_folder(folder: Path) -> None:
    """The result folder exists (or is created) and accepts files."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = tempfile.NamedTemporaryFile(dir=str(folder), prefix=".nalvera-probe-")
        probe.close()
    except OSError:
        raise ConfigError(messages.text("error.output_not_writable", path=printable(str(folder))),
                          code="output_not_writable") from None


def check_output_folder(folder: PathLike) -> None:
    """Raise ``ConfigError`` (``output_not_writable``) when results
    cannot be saved in ``folder``; creates it when missing."""
    _writable_folder(Path(os.path.abspath(os.path.expanduser(os.fspath(folder)))))


def _content_keys(scan: Any) -> List[str]:
    keys = []
    if getattr(scan, "sha256", None):
        keys.append("a:" + str(scan.sha256))
    if getattr(scan, "sha256_original", None):
        keys.append("o:" + str(scan.sha256_original))
    return keys


def _row_content_keys(row: Dict[str, Any]) -> List[str]:
    keys = []
    if row.get("sha256"):
        keys.append("a:" + str(row["sha256"]))
    if row.get("sha256_original"):
        keys.append("o:" + str(row["sha256_original"]))
    return keys


def _as_paths(paths: Any) -> List[PathLike]:
    if isinstance(paths, (str, os.PathLike)):
        return [paths]
    return list(paths or [])


def _as_refs(tasks: Any) -> List[TaskRef]:
    if isinstance(tasks, (str, Task)):
        return [tasks]
    return list(tasks or [])


def _task_id(ref: TaskRef) -> str:
    return ref.id if isinstance(ref, Task) else str(ref).strip()


def _clean_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    cleaned = str(label).strip()
    if not cleaned:
        return None
    if not _LABEL_RX.fullmatch(cleaned):
        raise ConfigError(messages.text("error.invalid_label"), code="invalid_label")
    return cleaned


def _require_scope(me: Dict[str, Any], scope: str) -> None:
    """Refuse on this computer what the server would refuse after the work
    of anonymizing: a key without ``scope``, or Terms awaiting acceptance."""
    scopes = (me.get("key") or {}).get("scopes") or []
    if scope not in scopes:
        raise PermissionDenied(code="scope_missing", extra={"required_scope": scope})
    if scope == SCOPE_WRITE and me.get("terms_reaccept_required"):
        raise PermissionDenied(code="terms_reaccept_required")


def _scan_idempotency_key(code: Optional[str], sha256: Optional[str], model: str, task: str,
                          text_digest: str = "") -> str:
    """The key of one logical submit: the scan's code (recorded in the
    mapping before the upload), the SHA-256 of its anonymized file, the
    model, the task and the digest of its text query. A retry of the same
    submit sends the same key, so it never queues or charges twice; another
    scan (even one that drew the same code), model, task or text never
    shares it. A run that recovers an earlier run's scan reuses that run's
    code and digest, so a request that did land is answered with its task."""
    identity = f"{code}\n{sha256 or ''}\n{model}\n{task}\n{text_digest}"
    return "scan-" + hashlib.sha256(identity.encode()).hexdigest()[:48]


def _text_digest(text: Optional[str]) -> str:
    """The SHA-256 of a task's text query (empty without one): part of what
    makes a task the same work as an earlier one."""
    cleaned = (text or "").strip()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest() if cleaned else ""


def _same_work(row: Dict[str, Any], job: _Job) -> bool:
    """Whether a record row's task is the work this run would queue: the
    same model, task and text query."""
    return (row.get("model") == job.model and (row.get("task") or job.task) == job.task
            and (row.get("text_sha256") or "") == job.text_digest)


def _sent_name(entry: _Entry) -> str:
    """The neutral file name a scan is sent under: its code's, or for a
    recovered scan the first code's."""
    scan = entry.prepared
    if entry.idem_code and str(scan.filename).startswith(str(scan.code)):
        return entry.idem_code + str(scan.filename)[len(str(scan.code)):]
    return str(scan.filename)


def _name_key(name: str) -> str:
    """A name as the platform would show it, for comparing a label with a
    file name: case and the space or underscore between words ignored."""
    return str(name).strip().replace(" ", "_").lower()


def _say_corrected(say: Any, name: str, scan: Any) -> None:
    """Say that the converter resampled a DICOM series (a gantry tilt or
    uneven slice spacing corrected): its voxels are not the original ones."""
    corrections = [c for c in getattr(scan, "corrections", None) or () if c]
    if corrections:
        say("run.dicom_corrected", name=name, corrections=messages.text("list.separator").join(
            messages.text(f"corrected.{c}") if messages.has(f"corrected.{c}") else c
            for c in corrections))


def _display_name(scan: Any) -> str:
    name = str(getattr(scan, "original_name", None) or getattr(scan, "code", "scan"))
    series = getattr(scan, "series", None)
    return messages.text("run.series_name", name=name, series=series) if series else name


def _display_name_row(row: Dict[str, Any]) -> Optional[str]:
    name = row.get("original_name")
    if not name:
        return None
    series = row.get("series")
    return messages.text("run.series_name", name=name, series=series) if series else name


def _queued(entries: List[_Entry]) -> List[Task]:
    return [e.task for e in sorted(entries, key=lambda e: e.order) if e.task is not None]


def _failures(entries: List[_Entry]) -> List[ScanFailure]:
    return [e.failure for e in sorted(entries, key=lambda e: e.order) if e.failure is not None]


def _remove(path: Any) -> None:
    try:
        os.remove(str(path))
    except OSError:
        pass


def _strip_scan_suffix(name: str) -> str:
    base = Path(str(name)).name
    for suffix in (".nii.gz", ".nii", ".gz", ".zip"):
        if base.lower().endswith(suffix):
            return base[: -len(suffix)]
    return base


def _safe(text: str) -> str:
    """One part of a file name: no folder separator, no character a file
    system refuses, no control or format character, no leading dot, and no
    trailing dot or space (Windows drops them)."""
    cleaned = _UNSAFE_NAME_RX.sub("_", printable(text)).strip().lstrip(".").rstrip(". ")
    return cleaned[:120].rstrip(". ") or "result"


#: Device names Windows reserves, with or without an extension.
_RESERVED_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
                             *(f"COM{n}" for n in "0123456789¹²³"),
                             *(f"LPT{n}" for n in "0123456789¹²³")})
#: The longest name (bytes) a result's base may have: with ``.`` + 8 + ``.part``
#: around it, or ``-<n>`` and the longest extension after it, the file name
#: stays within the 255 bytes file systems allow.
_MAX_BASE_BYTES = 240


def _result_base(stem: str, item: Task, part: Optional[str]) -> str:
    """A result's file name without its extension:
    ``<stem>__<model>__<task>[__<part>]``, each made safe, never a
    reserved device name, at most ``_MAX_BASE_BYTES`` bytes."""
    base = "__".join(_safe(x) for x in (stem, item.model or "model", item.task or "default")
                     + ((part,) if part else ()))
    if base.split(".")[0].rstrip(" ").upper() in _RESERVED_NAMES:
        base = "_" + base
    encoded = base.encode("utf-8")
    if len(encoded) > _MAX_BASE_BYTES:
        base = encoded[:_MAX_BASE_BYTES].decode("utf-8", errors="ignore").rstrip(". ")
    return base


def _label_from_name(name: Optional[str]) -> Optional[str]:
    """A generation's label, read from its platform name. The platform names
    a generation by its label as given, else by a name of its own with
    parentheses (``NV-Generate rflow-ct (image-only)``), which no label can
    hold."""
    cleaned = str(name or "").strip()
    return cleaned if _LABEL_RX.fullmatch(cleaned) else None


_TYPE_EXTENSIONS = {"application/zip": ".zip", "application/json": ".json",
                    "text/plain": ".txt", "text/csv": ".csv", "application/gzip": ".gz"}
#: The result types the platform sends, compound ones first. ``.npy`` is a
#: vector result (Merlin's embedding). Any other extension becomes ``.bin``,
#: so a server cannot make this computer save a program or a web page.
RESULT_EXTENSIONS = (".nii.gz", ".tar.gz", ".zip", ".json", ".txt", ".csv", ".npy", ".gz")


def _extension(headers: Any) -> str:
    """The result's file extension: one of ``RESULT_EXTENSIONS``, from
    the base name of the Content-Disposition file name (untrusted: no folder
    part is ever used), else from its type; ``.bin`` for anything else."""
    filename = _disposition_filename(headers.get("Content-Disposition") or "")
    if filename:
        lower = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
        for extension in RESULT_EXTENSIONS:
            if lower.endswith(extension):
                return extension
    media = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    return _TYPE_EXTENSIONS.get(media, ".bin")


def _expected_length(headers: Any) -> Optional[int]:
    """The announced size, when the body is not re-encoded on the way."""
    if headers.get("Content-Encoding"):
        return None
    try:
        return int(headers.get("Content-Length"))
    except (TypeError, ValueError):
        return None


def _disposition_filename(value: str) -> Optional[str]:
    from urllib.parse import unquote
    star = re.search(r"filename\*\s*=\s*([^']*)'[^']*'([^;]+)", value, re.IGNORECASE)
    if star:
        return unquote(star.group(2).strip().strip('"'))
    plain = re.search(r'filename\s*=\s*"([^"]*)"', value, re.IGNORECASE) or \
        re.search(r"filename\s*=\s*([^;]+)", value, re.IGNORECASE)
    return plain.group(1).strip() if plain else None


def _place(temp: Path, folder: Path, base: str, extension: str,
           overwrite: bool) -> Tuple[Path, Optional[Path]]:
    """Move the downloaded file to ``base + extension``; when that name is
    taken (and ``overwrite`` is False), to ``base-2``, ``base-3``…. Returns
    (the path, the existing file that was kept or None)."""
    first = folder / f"{base}{extension}"
    if overwrite:
        os.replace(temp, first)
        return first, None
    number = 1
    while True:
        target = first if number == 1 else folder / f"{base}-{number}{extension}"
        try:
            os.link(temp, target)          # fails when the name is taken: never replaces
            os.remove(temp)
        except FileExistsError:
            number += 1
            continue
        except OSError:
            if target.exists():
                number += 1
                continue
            os.replace(temp, target)       # a file system without hard links
        return target, (first if number > 1 else None)


_duration = display.duration


# ── The record and anonymized copies without a client ─────────────────────────
def read_record(path: PathLike) -> display.ScanRecord:
    """The rows of the record at ``path`` (none when it is missing), as a
    table. A CSV this package did not write raises ``ConfigError``
    (``mapping_unreadable``) and is left unchanged."""
    anonymize = _anonymizer()
    try:
        rows = list(anonymize.Mapping(str(path)).rows())
    except anonymize.AnonymizeError as error:
        raise _mapping_error(error, Path(path)) from None
    return display.ScanRecord(rows, path=str(path))


def default_record_path(config_dir: Optional[PathLike] = None) -> Path:
    """The record a call without a client uses: ``NALVERA_MAPPING``, else
    the file of the account named by ``NALVERA_EMAIL`` or a stored login
    (none: a file for no account) on the platform of ``NALVERA_BASE_URL``,
    in the configuration folder. Reads no key and sends no request."""
    chosen = settings.get("NALVERA_MAPPING")
    if chosen:
        return Path(chosen)
    directory = Path(config_dir) if config_dir else None
    return auth.record_path(auth.known_email(directory=directory),
                            settings.get("NALVERA_BASE_URL") or DEFAULT_BASE_URL,
                            directory=directory)


def save_anonymized(paths: Union[PathLike, Sequence[PathLike]], out: PathLike, *,
                    mapping: Optional[PathLike] = None, reporter: Optional[Reporter] = None,
                    check: Optional[Any] = None) -> SavedCopies:
    """Write the anonymized copies of scans into the folder ``out``, exactly
    the files a run would upload. Sends nothing and needs no key.

    ``paths`` are as for ``run()`` (files, folders of NIfTI files,
    DICOM folders). ``out`` is created when missing and afterwards holds the
    copies and nothing else, each named by its new code, so no file in it is
    ever replaced. The record that links each code to its original name is
    the key that re-identifies the scans: it goes to ``mapping`` (default:
    ``default_record_path()``), never into ``out``. The DICOM conversion
    works in a private folder of the system's temporary folder.
    ``reporter`` (default: a ``NullReporter``) gets the progress lines;
    ``check(path)`` may refuse a path before it is read.

    Returns ``SavedCopies``: per scan the copy's path, its shape and
    what was removed; ``failed`` lists scans that could not be prepared."""
    reporter = reporter or NullReporter()
    with report_mod.call_scope(reporter, "anonymize"):
        return _save_anonymized(paths, out, mapping=mapping, reporter=reporter, check=check)


def _save_anonymized(paths: Union[PathLike, Sequence[PathLike]], out: PathLike, *,
                     mapping: Optional[PathLike], reporter: Reporter,
                     check: Optional[Any]) -> SavedCopies:
    sources = _as_paths(paths)
    if not sources:
        raise ConfigError(messages.text("error.no_paths"), code="no_paths")
    say = reporter.say
    anonymize = _anonymizer()
    record = Path(mapping) if mapping else default_record_path()
    folder = Path(os.path.abspath(os.path.expanduser(os.fspath(out))))
    if check is not None:
        check(str(folder))
    _writable_folder(folder)
    if not mapping and not settings.get("NALVERA_MAPPING"):
        auth.private_dir(record.parent, levels=2)               # records, config folder
    links = anonymize.Mapping(str(record))
    try:
        rows = links.rows()                               # a file not ours stops here
    except anonymize.AnonymizeError as error:
        raise _mapping_error(error, record) from None
    taken = {str(r.get("code")) for r in rows if r.get("code")}
    expanded = _expand_paths(sources, anonymize, check or (lambda path: None),
                             lambda key, **fields: None)
    _remove_stale_workdirs()
    try:
        workdir = tempfile.mkdtemp(prefix=WORKDIR_PREFIX)          # mode 0700
    except OSError:
        raise ConfigError(messages.text("error.temp_not_writable", path=tempfile.gettempdir()),
                          code="temp_not_writable") from None
    copies = SavedCopies(record=record)
    noticed = False
    try:
        for source in expanded:
            name = os.path.basename(source.path.rstrip(os.sep)) or source.path
            try:
                prepared = anonymize.prepare(source.path, workdir=workdir, avoid=taken,
                                             **({"check": check} if check else {}))
            except anonymize.AnonymizeError as error:
                code = getattr(error, "code", None) or "unreadable"
                if code == "mapping_unreadable":
                    raise _mapping_error(error, record) from None
                key = f"anonymize.{code}" if messages.has(f"anonymize.{code}") \
                    else "anonymize.other"
                reason = messages.text(key, code=code)
                copies.failed.append(ScanFailure(name, source.path, code, reason))
                say("run.scan_failed", name=name, reason=reason)
                continue
            if not noticed:
                say("run.pixel_notice")
                noticed = True
            for scan in prepared:
                taken.add(scan.code)
                target = folder / scan.filename
                _move_new(Path(scan.path), target)
                try:
                    links.record_scan(scan)
                except anonymize.AnonymizeError as error:
                    raise _mapping_error(error, record) from None
                removed = _removed_from(scan, source.path)
                copies.append(SavedCopy(original_name=str(scan.original_name),
                                        original_path=str(scan.original_path), path=target,
                                        code=scan.code, shape=tuple(scan.shape), removed=removed,
                                        series=getattr(scan, "series", None)))
                say("anonymized.copy", name=_display_name(scan),
                    path=display.path_text(target, cut=False),
                    shape=" x ".join(str(n) for n in scan.shape))
                _say_corrected(say, _display_name(scan), scan)
                say("anonymized.removed",
                    removed=messages.text("list.separator").join(removed))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    say("anonymized.done", count=len(copies), path=display.path_text(folder, cut=False),
        mapping=display.path_text(record, cut=False))
    return copies


def _move_new(source: Path, target: Path) -> None:
    """Move ``source`` to ``target``, which must not exist: nothing in the
    folder is ever replaced."""
    try:
        os.link(source, target)
    except FileExistsError:
        raise ConfigError(messages.text("error.output_not_writable", path=printable(str(target))),
                          code="output_not_writable") from None
    except OSError:
        if target.exists():
            raise ConfigError(messages.text("error.output_not_writable",
                                            path=printable(str(target))),
                              code="output_not_writable") from None
        shutil.copyfile(source, target)               # another file system, or no hard links
    _remove(source)


def _removed_from(scan: Any, original: str) -> List[str]:
    """What the anonymized copy of ``original`` no longer holds, in words:
    the file name always; for DICOM every tag; for NIfTI the text fields
    that held something and the header extensions."""
    removed = [messages.text("anonymized.removed_name")]
    if getattr(scan, "source", "") == "dicom":
        return removed + [messages.text("anonymized.removed_dicom")]
    try:
        import nibabel
        header = nibabel.load(original).header
    except Exception:
        return removed                     # the copy was checked; the original is unreadable
    for name in ("descrip", "aux_file", "intent_name", "db_name"):
        try:
            value = bytes(header[name].item()).strip(b"\x00 ")
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        if value:
            removed.append(messages.text("anonymized.removed_field", field=name))
    if len(header.extensions):
        removed.append(messages.text("anonymized.removed_extensions",
                                     count=len(header.extensions)))
    return removed


__all__ = ["Client", "Task", "RunResult", "Downloads", "ScanFailure", "SkippedScan",
           "RunEstimate", "ScanEstimate", "BackgroundRun", "SavedCopy", "SavedCopies",
           "check_output_folder", "save_anonymized", "default_record_path"]
