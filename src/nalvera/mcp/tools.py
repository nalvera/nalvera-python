"""What each tool of the local server does: arguments in, data out.

The manifest (``mcp_tools.json``) is the one source of every tool's name,
description, schemas and annotations, and of the fixed texts (error
instructions, reasons, notes). The tools both servers offer answer here as
the hosted server answers (``mcp_server/tools.py`` in the Platform), through
the package's ``nalvera.Client`` instead of raw HTTP; the local-only
tools (``run_scan``, ``get_run``, ``download_result``, and ``estimate_cost``
with a ``path``) work on this computer.

Rules every tool follows:
  * the key is never an argument and never part of an answer: the client
    finds it where ``nalvera.login(..., remember=True)`` stored it, or in the environment;
  * every path is resolved, symbolic links included, and must lie inside an
    allowed folder (``NALVERA_MCP_ALLOWED_DIRS``) before anything is read;
  * spending needs ``budget`` and ``max_credits``; above the cap nothing is
    charged;
  * a call never blocks for long: platform requests give up after a few
    seconds of retries, waits end within 25 seconds, scan runs and large
    downloads go on in the background;
  * what comes back is data: names, labels, errors and result texts sit in
    named fields;
  * the assistant's provider sees no more than the assistant needs: a scan
    is named by its code and its position in the run, paths are relative to
    the allowed folder, and a saved result is reported as its folder and a
    count. ``NALVERA_MCP_SHOW_NAMES=1`` returns the original names and the
    result's own path, for a person who accepts that the provider sees them;
  * every pattern of an argument is checked against the whole value
    (``fullmatch``): a trailing newline never passes.

This module imports no MCP SDK (``server.py`` does).
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import auth, display, messages, settings
from .. import client as client_mod
from .._http import RetryPolicy
from ..errors import (
    EncryptionError,
    NalveraError,
    NetworkError,
    NoCredentials,
    NoFreeTasks,
    UnexpectedResponse,
)
from ..report import NullReporter
from . import LOCAL, load_manifest
from .runs import RunClient, RunManager, RunStore

ALLOWED_DIRS_SETTING = "NALVERA_MCP_ALLOWED_DIRS"
SHOW_NAMES_SETTING = "NALVERA_MCP_SHOW_NAMES"
FINAL_STATES = frozenset({"completed", "failed", "cancelled"})
TASK_ID_RX = re.compile(r"[A-Za-z0-9]{1,32}")
_STRING_MAX = 4000
_TREE_MAX_ENTRIES = 200_000
_RETRY_CODES = ("rate_limited", "queue_full", "storage_full", "temporarily_unavailable")
# Package codes for an argument the platform would refuse as well.
_AS_INVALID_REQUEST = frozenset({
    "unknown_model", "unknown_task", "text_required", "model_is_generative", "model_needs_scan",
    "invalid_label", "invalid_max_credits", "invalid_budget", "invalid_param", "no_paths",
    "label_is_file_name"})
# Fields of a refusal passed on beside the code (the platform's own, or the
# package's when it refused first).
_EXTRA_FIELDS = ("required_scope", "state", "budget", "limit", "max_bytes", "minimum_version",
                 "reason")
#: The status-poll floor until the key's own value (GET /me) is known.
DEFAULT_POLL_FLOOR = 5.0


# ── Allowed folders ───────────────────────────────────────────────────────────
def allowed_folders(value: Optional[str], start: str) -> List[str]:
    """The folders the server may read and write, resolved: those of
    ``value`` (``NALVERA_MCP_ALLOWED_DIRS``, separated by ``os.pathsep``;
    a folder that does not exist is left out), else ``start`` (the folder
    the server was started in) unless that is the root of the file system or
    the home folder, which an assistant's app often starts servers in: then
    none until the setting names some."""
    if value is not None:
        found: List[str] = []
        for part in value.split(os.pathsep):
            part = part.strip().strip('"')
            if not part:
                continue
            real = os.path.realpath(os.path.expanduser(part))
            if os.path.isdir(real) and real not in found:
                found.append(real)
        return found
    real = os.path.realpath(start)
    if os.path.dirname(real) == real or real == os.path.realpath(os.path.expanduser("~")):
        return []
    return [real]


def inside(real: str, folders: Sequence[str]) -> bool:
    """Whether the resolved path ``real`` lies in one of ``folders``."""
    return any(real == f or real.startswith(f.rstrip(os.sep) + os.sep) for f in folders)


def show_names(value: Optional[str]) -> bool:
    """Whether ``NALVERA_MCP_SHOW_NAMES`` asks for original names and paths
    (``1``, ``true``, ``yes``, ``on``); off by default."""
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def clean_stale() -> int:
    """Remove the package's temporary folders a killed process left (older
    than a day, owned by this user): done when the server starts."""
    return client_mod._remove_stale_workdirs()


def patterns(entry: Dict[str, Any]) -> Dict[str, re.Pattern[str]]:
    """The ``pattern`` of each string argument of a manifest entry,
    compiled, local-only inputs included."""
    properties = dict(entry["inputSchema"].get("properties") or {})
    properties.update(entry.get("localInputs") or {})
    return {name: re.compile(spec["pattern"]) for name, spec in properties.items()
            if isinstance(spec, dict) and isinstance(spec.get("pattern"), str)}


# ── Shaping what the platform returned (as mcp_server/tools.py does) ─────────
def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text[:_STRING_MAX]


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def task_out(raw: Dict[str, Any]) -> Dict[str, Any]:
    """The platform's task object with exactly the fields the output schema
    names: a field a newer platform adds is not passed on unannounced."""
    parts = raw.get("parts")
    return {
        "id": str(raw.get("id") or ""),
        "name": _text(raw.get("name")),
        "model": _text(raw.get("model")),
        "model_version": _text(raw.get("model_version")),
        "task": _text(raw.get("task")),
        "task_label": _text(raw.get("task_label")),
        "state": str(raw.get("state") or "unknown"),
        "queue_position": _integer(raw.get("queue_position")),
        "created_at": _text(raw.get("created_at")),
        "started_at": _text(raw.get("started_at")),
        "completed_at": _text(raw.get("completed_at")),
        "waited_seconds": _number(raw.get("waited_seconds")),
        "running_seconds": _number(raw.get("running_seconds")),
        "error": _text(raw.get("error")),
        "output_ready": bool(raw.get("output_ready")),
        "output_kind": _text(raw.get("output_kind")),
        "parts": [str(p) for p in parts] if isinstance(parts, list) else None,
        "source": _text(raw.get("source")),
        "cancel_requested": bool(raw.get("cancel_requested")),
    }


def models_out(listed: Sequence[Dict[str, Any]], only: Optional[str]) -> Dict[str, Any]:
    models = []
    for m in listed or []:
        if only is not None and m.get("id") != only:
            continue
        tasks = [{
            "id": str(t.get("id") or ""), "name": _text(t.get("name")),
            "default": bool(t.get("default")),
            # The task's own modality, else the model's.
            "modality": display.task_modality(m, t),
            "price_eur_per_mvoxel": _number(t.get("price_eur_per_mvoxel")),
            "needs_text": bool(t.get("needs_text")),
            "generation": t.get("generation") if isinstance(t.get("generation"), dict) else None,
        } for t in m.get("tasks") or []]
        models.append({
            "id": str(m.get("id") or ""), "name": _text(m.get("name")),
            "version": _text(m.get("version")),
            "modality": [str(x) for x in m.get("modality") or []],
            "type": _text(m.get("type")), "generative": bool(m.get("generative")),
            "needs_scan": bool(m.get("needs_scan")),
            "price_eur_per_mvoxel": _number(m.get("price_eur_per_mvoxel")),
            "tasks": tasks,
        })
    return {"models": models}


def budgets_out(raw: Dict[str, Any]) -> Dict[str, Any]:
    free = raw.get("free") or {}
    personal = raw.get("personal") or {}
    return {
        "free": {"period_active": bool(free.get("period_active")),
                 "free_tasks_left": _integer(free.get("free_tasks_left")) or 0},
        "personal": {"monthly_left": _number(personal.get("monthly_left")),
                     "topup": _number(personal.get("topup")),
                     "available_credits": _number(personal.get("available_credits")),
                     "period_end": _text(personal.get("period_end"))},
        "labs": [{"id": _integer(lab.get("id")) or 0, "name": _text(lab.get("name")),
                  "balance": _number(lab.get("balance")),
                  "member_status": _text(lab.get("member_status")),
                  "spending_limit": _number(lab.get("spending_limit")),
                  "spent": _number(lab.get("spent"))}
                 for lab in raw.get("labs") or []],
        "suggested": _text(raw.get("suggested")),
    }


def estimate_out(raw: Dict[str, Any]) -> Dict[str, Any]:
    free = raw.get("free") or {}
    return {
        "credits": _number(raw.get("credits")) or 0.0,
        "free": {"available": bool(free.get("available")), "source": _text(free.get("source")),
                 "free_tasks_left": _integer(free.get("free_tasks_left")) or 0},
        "budgets": [{"budget": str(b.get("budget") or ""), "name": _text(b.get("name")),
                     "available_credits": _number(b.get("available_credits")),
                     "sufficient": bool(b.get("sufficient"))}
                    for b in raw.get("budgets") or []],
        "suggested": _text(raw.get("suggested")),
    }


def charged_out(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    return {"source": str(raw.get("source") or ""), "lab_id": _integer(raw.get("lab_id")),
            "credits": _number(raw.get("credits")) or 0.0}


class _TtlCache:
    """Answers kept for a few seconds, so a status asked again sooner than
    the platform's poll floor costs the platform nothing."""

    def __init__(self, max_entries: int = 2_000):
        self._data: OrderedDict = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def get(self, key: Any, ttl: float) -> Any:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            stamp, value = hit
            if time.monotonic() - stamp > ttl:
                self._data.pop(key, None)
                return None
            return value

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class ToolFailure(Exception):
    """A refusal decided on this computer, before or without the platform."""

    def __init__(self, code: str, detail: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None):
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.extra = dict(extra or {})


class _Download:
    """One download_result going on (or done) in a thread."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.path: Optional[str] = None
        self.size = 0
        self.error: Optional[BaseException] = None
        self.thread: Optional[threading.Thread] = None


class LocalTools:
    """Every tool of the local server. Each method takes the tool's
    arguments (already checked against its input schema) and returns its
    structured result, or raises."""

    def __init__(self, manifest: Optional[Dict[str, Any]] = None, *,
                 allowed: Optional[Sequence[str]] = None, start: Optional[str] = None,
                 mapping: Optional[str] = None, config_dir: Optional[str] = None,
                 max_wait: float = 25.0, text_cap: int = 65536, decision_wait: float = 20.0,
                 download_wait: float = 25.0, min_poll_floor: float = 1.0,
                 api_timeout: float = 15.0, api_retry_wait: float = 8.0,
                 sleep: Optional[Callable[[float], None]] = None):
        self.manifest = manifest or load_manifest()
        self.entries = {t["name"]: t for t in self.manifest["tools"] if LOCAL in t["servers"]}
        self.allowed: List[str] = [os.path.realpath(os.fspath(f)) for f in allowed] \
            if allowed is not None else allowed_folders(settings.get(ALLOWED_DIRS_SETTING),
                                                        start or os.getcwd())
        self.show_names = show_names(settings.get(SHOW_NAMES_SETTING))
        # Checked with fullmatch: a JSON Schema pattern is a search, and "$"
        # matches before a trailing newline.
        self._patterns = {name: patterns(entry) for name, entry in self.entries.items()}
        # The record re-identifies every scan: by default it lives in the
        # configuration folder (one file per account and platform), never in
        # the project folder an assistant works in.
        mapping = mapping or settings.get("NALVERA_MAPPING")
        self.mapping = Path(os.path.abspath(os.path.expanduser(mapping))) if mapping \
            else client_mod.default_record_path(config_dir)
        folder = Path(config_dir) if config_dir else auth.config_dir()
        self.max_wait = max_wait
        self.text_cap = text_cap
        self.download_wait = download_wait
        self.min_poll_floor = min_poll_floor
        self.api_timeout = api_timeout
        self.api_retry_wait = api_retry_wait
        self._sleep = sleep or time.sleep
        self._client: Optional[client_mod.Client] = None
        self._client_lock = threading.Lock()
        self._status = _TtlCache()
        self._downloads: Dict[Tuple[str, str, str], _Download] = {}
        self._downloads_lock = threading.Lock()
        # The temporary folders of estimates converting DICOM right now:
        # shutdown removes those a call is still inside.
        self._inspecting: set = set()
        self._inspecting_lock = threading.Lock()
        self.runs = RunManager(store=RunStore(folder / "mcp-runs"), mapping=self.mapping,
                               texts=self.manifest, make_client=self._run_client,
                               error_fields=self._run_error, has_dicom=self._has_dicom,
                               decision_wait=decision_wait, scrub=self.scrub,
                               show_names=self.show_names, relative=self.relative)
        self._impl: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "list_models": self._list_models, "get_budgets": self._get_budgets,
            "estimate_cost": self._estimate_cost, "generate": self._generate,
            "list_tasks": self._list_tasks, "get_task": self._get_task,
            "wait_for_task": self._wait_for_task, "cancel_task": self._cancel_task,
            "get_text_result": self._get_text_result, "run_scan": self._run_scan,
            "get_run": self._get_run, "download_result": self._download_result,
        }
        differ = set(self.entries) ^ set(self._impl)
        if differ:
            raise RuntimeError(f"manifest and implementation disagree on: {sorted(differ)}")

    # ── the one entry point ───────────────────────────────────────────────────
    def call(self, name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Run tool ``name``: ``(False, result)``, or ``(True, {"error":
        {...}})`` for a refusal (the API's ``code``, the manifest's
        instruction, ``detail`` and ``retry_after`` when there are any)."""
        try:
            for field, pattern in self._patterns.get(name, {}).items():
                value = arguments.get(field)
                if isinstance(value, str) and pattern.fullmatch(value) is None:
                    raise ToolFailure("invalid_arguments", field)
            return False, self._impl[name](dict(arguments))
        except Exception as error:      # never a stack trace or an argument in the answer
            return True, {"error": self.error_fields(error)}

    def error_fields(self, error: BaseException) -> Dict[str, Any]:
        code, detail, extra, retry_after = "internal_error", None, {}, None
        if isinstance(error, ToolFailure):
            code, detail, extra = error.code, error.detail, error.extra
        elif isinstance(error, NalveraError):
            code, detail, extra = self._package_error(error)
            retry_after = error.retry_after
        texts = self.manifest["errors"]
        fields: Dict[str, Any] = {"code": code,
                                  "message": texts.get(code, texts["invalid_request"])}
        if detail:
            fields["detail"] = self.scrub(str(detail))[:_STRING_MAX]
        if retry_after is not None:
            fields["retry_after"] = int(round(retry_after))
        for key, value in extra.items():
            if isinstance(value, (str, int, float, bool)):
                fields.setdefault(key, value)
        return fields

    def _package_error(self, error: NalveraError) -> Tuple[str, Optional[str], Dict[str, Any]]:
        code = error.code
        known = self.manifest["errors"]
        extra = {k: error.extra[k] for k in _EXTRA_FIELDS if k in error.extra}
        if isinstance(error, NoCredentials):
            return "no_credentials", None, {}
        if isinstance(error, NetworkError):
            return "api_unreachable", None, {}
        if isinstance(error, UnexpectedResponse) or code.startswith("unexpected_"):
            return "unexpected_answer", None, {}
        if isinstance(error, EncryptionError):
            return "encryption_error", error.message, {}
        if code == "invalid_key_format":
            return "invalid_credentials", error.message, {}
        if code in known:
            # The platform's refusal (its own text in detail), or the package
            # refusing first what the platform would refuse.
            return code, error.detail, extra
        if error.status is not None:
            return code, error.detail, extra    # a code a newer platform added
        if code in _AS_INVALID_REQUEST:
            return "invalid_request", error.message, {}
        return "local_error", error.message, {}

    def scrub(self, text: str) -> str:
        """``text`` without the location of local folders: a path inside an
        allowed folder becomes relative to it, the folder itself its name,
        the temporary and home folders their name. A message of the package
        may name such a path; the assistant gets base names only."""
        places = []
        for folder in [*self.allowed, tempfile.gettempdir(), os.path.expanduser("~")]:
            name = os.path.basename(folder.rstrip(os.sep)) or folder
            relative = os.path.realpath(folder) in self.allowed
            for form in {folder, os.path.realpath(folder), os.path.abspath(folder)}:
                form = form.rstrip(os.sep)
                if form:
                    places.append((form, name, relative))
        for form, name, relative in sorted(places, key=lambda p: -len(p[0])):  # inner first
            if relative:
                text = text.replace(form + os.sep, "")
            text = text.replace(form, name)
        return text

    def relative(self, real: str) -> str:
        """``real`` (resolved, inside an allowed folder) as the assistant may
        see it: relative to that folder (``.`` for the folder itself)."""
        for folder in self.allowed:
            if inside(real, [folder]):
                return os.path.relpath(real, folder)
        return os.path.basename(real.rstrip(os.sep)) or "."

    def _run_error(self, error: BaseException) -> Dict[str, Any]:
        fields = self.error_fields(error)
        return {"code": fields["code"], "message": fields["message"],
                "detail": fields.get("detail")}

    def close(self, timeout: float = 4.0) -> List[str]:
        """Stop the runs (see ``RunManager.shutdown``, waiting up to
        ``timeout`` seconds) and close the connections. A download still
        streaming stops and its partly written file is removed (the client's
        ``close``); so is the temporary folder of an estimate still
        converting DICOM. Returns the ids of runs whose thread did not stop."""
        alive = self.runs.shutdown(timeout)
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None
        with self._inspecting_lock:
            folders, self._inspecting = list(self._inspecting), set()
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)
        return alive

    # ── clients ───────────────────────────────────────────────────────────────
    def client(self) -> client_mod.Client:
        """The client of the quick tools, made on first use: retries end
        after a few seconds, so a call stays short."""
        with self._client_lock:
            if self._client is None:
                made = client_mod.Client(mapping=str(self.mapping), reporter=NullReporter(),
                                         interactive=False, allowed_folders=self.allowed)
                made._transport.timeout = (5.0, self.api_timeout)
                made._transport.retry = RetryPolicy(max_attempts=4,
                                                    max_total_wait=self.api_retry_wait)
                self._client = made
            return self._client

    def _run_client(self, run: Any) -> RunClient:
        return RunClient(mapping=str(self.mapping), reporter=run.progress, interactive=False,
                         allowed_folders=self.allowed, run=run)

    # ── paths ─────────────────────────────────────────────────────────────────
    def resolve(self, path: str, *, must_exist: bool = True) -> str:
        """``path`` resolved (a relative one against the first allowed
        folder), checked to lie inside an allowed folder with every symbolic
        link under it, before anything in it is read."""
        expanded = os.path.expanduser(str(path))
        if not os.path.isabs(expanded):
            expanded = os.path.join(self.allowed[0] if self.allowed else os.getcwd(), expanded)
        real = os.path.realpath(expanded)
        if not self.allowed or not inside(real, self.allowed):
            raise ToolFailure("path_not_allowed")
        if must_exist and not os.path.exists(real):
            raise ToolFailure("path_not_found")
        if os.path.isdir(real):
            self._check_tree(real)
        return real

    def _check_tree(self, folder: str) -> None:
        """A link inside an allowed folder that points out of the allowed
        folders makes the whole folder refused. So does a folder too large to
        check: it is refused at the entry limit, never passed unchecked."""
        seen = 0
        for root, dirs, files in os.walk(folder, followlinks=False):
            for name in dirs + files:
                seen += 1
                if seen > _TREE_MAX_ENTRIES:
                    raise ToolFailure("local_error", messages.text(
                        "error.folder_too_large", path=os.path.basename(folder),
                        count=_TREE_MAX_ENTRIES))
                candidate = os.path.join(root, name)
                if os.path.islink(candidate) and \
                        not inside(os.path.realpath(candidate), self.allowed):
                    raise ToolFailure("path_not_allowed")

    def _has_dicom(self, paths: Sequence[str]) -> bool:
        anonymize = client_mod._anonymizer()
        for path in paths:
            try:
                if anonymize.inspect(path).kind in ("dicom", "mixed"):
                    return True
            except Exception:
                continue                  # the run reports what it cannot read
        return False

    # ── tools ─────────────────────────────────────────────────────────────────
    def _list_models(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return models_out(self.client().models(refresh=True), args.get("model"))

    def _get_budgets(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return budgets_out(self.client().budgets())

    def _estimate_cost(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "path" in args:
            if "shape" in args or "params" in args:
                raise ToolFailure("invalid_arguments", messages.text("mcp.path_alone"))
            return self._estimate_path(args)
        raw = self.client().estimate(args["model"], args.get("task"), shape=args.get("shape"),
                                     params=args.get("params"))
        return estimate_out(raw)

    def _estimate_path(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """The cost of every scan at a path: each one's voxel grid read on
        this computer (DICOM converted in a temporary folder), priced by the
        platform. Nothing is uploaded."""
        real = self.resolve(args["path"])
        anonymize = client_mod._anonymizer()
        # inspect makes its own folder inside this one: known here, it can be
        # removed at shutdown while a conversion is still going.
        parent = tempfile.mkdtemp(prefix=client_mod.INSPECT_PREFIX)
        with self._inspecting_lock:
            self._inspecting.add(parent)
        try:
            found = anonymize.inspect(real, convert=True, workdir=parent)
        except anonymize.AnonymizeError as error:
            code = getattr(error, "code", None) or "unreadable"
            if code == "not_found":
                raise ToolFailure("path_not_found") from None
            raise ToolFailure("local_error", _anonymize_text(code)) from None
        finally:
            with self._inspecting_lock:
                self._inspecting.discard(parent)
            shutil.rmtree(parent, ignore_errors=True)
        if found.kind == "mixed":
            raise ToolFailure("local_error", messages.text(
                "error.mixed_folder", path=os.path.basename(real)))
        shapes = [tuple(int(n) for n in list(s.shape)[:3]) for s in found.scans
                  if s.shape and not s.error]
        if not shapes:
            errors = [s.error for s in found.scans if s.error]
            raise ToolFailure("local_error", _anonymize_text(errors[0] if errors
                                                              else "not_a_scan"))
        client = self.client()
        prices: Dict[Tuple[int, ...], float] = {}
        last: Dict[str, Any] = {}
        for shape in shapes:
            if shape not in prices:
                last = client.estimate(args["model"], args.get("task"), shape=shape)
                prices[shape] = _number(last.get("credits")) or 0.0
        total = round(sum(prices[s] for s in shapes), 2)
        out = estimate_out(last)
        out["credits"] = total
        for option in out["budgets"]:
            available = option["available_credits"]
            option["sufficient"] = available is not None and available + 1e-9 >= total
        return out

    def _generate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate, then queue. Never charges more than ``max_credits``:
        above it the request goes out with budget ``free``, which the
        platform queues only when a free task pays (or answers with the task
        an earlier call with this request_id queued), charging nothing
        otherwise. The Idempotency-Key is the hosted server's, so one
        request_id is one task through either server."""
        client = self.client()
        model, task, params = args["model"], args["task"], args["params"]
        budget, cap = args["budget"], float(args["max_credits"])
        estimated = client.estimate(model, task, params=params)
        credits = _number(estimated.get("credits")) or 0.0
        free = estimated.get("free") or {}
        estimate = {"credits": credits, "free_available": bool(free.get("available")),
                    "free_tasks_left": _integer(free.get("free_tasks_left")) or 0}
        send = budget if budget == "free" or credits <= cap + 1e-9 else "free"
        body: Dict[str, Any] = {"model": model, "task": task, "budget": send, "params": params}
        if args.get("label"):
            body["label"] = args["label"]
        key = f"{self.manifest['idempotency_key_prefix']}{args['request_id']}"
        try:
            answer = client._transport.json("POST", "/api/v1/tasks", json=body,
                                            headers={"Idempotency-Key": key}, idempotent=True,
                                            retry_codes=_RETRY_CODES)
        except NoFreeTasks:
            if send != budget:
                return {"queued": False, "reason": "over_max_credits",
                        "hint": self.manifest["reasons"]["over_max_credits"],
                        "estimate": estimate, "budget_sent": send, "task": None,
                        "charged": None, "idempotent_replay": False}
            raise
        return {"queued": True, "reason": None, "hint": None, "estimate": estimate,
                "budget_sent": send, "task": task_out(answer.get("task") or {}),
                "charged": charged_out(answer.get("charged")),
                "idempotent_replay": bool(answer.get("idempotent_replay"))}

    def _floor(self) -> float:
        """The key's status-poll floor (GET /me ``min_poll_seconds``)."""
        limits = self.client().me().get("limits") or {}
        return max(self.min_poll_floor,
                   _number(limits.get("min_poll_seconds")) or DEFAULT_POLL_FLOOR)

    def _list_tasks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state, limit = args.get("state", "all"), int(args.get("limit", 20))
        offset, name = int(args.get("offset", 0)), args.get("name")
        key = ("list", state, limit, offset, name)
        hit = self._status.get(key, self._floor())
        if hit is not None:
            return hit
        items, total = self.client()._list(state, limit, offset, name)
        listed = {"tasks": [task_out(t.raw) for t in items], "total": int(total)}
        self._status.put(key, listed)
        return listed

    def _fetch_task(self, task_id: str, *, use_cache: bool) -> Dict[str, Any]:
        if not TASK_ID_RX.fullmatch(str(task_id)):
            raise ToolFailure("invalid_arguments", "task_id")
        key = ("task", task_id)
        if use_cache:
            hit = self._status.get(key, self._floor())
            if hit is not None:
                return hit
        task = task_out(self.client().task(task_id).raw)
        self._status.put(key, task)
        return task

    def _get_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._fetch_task(args["task_id"], use_cache=True)

    def _wait_for_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Poll no faster than the key's floor; return on a change of state
        (or of cancel_requested / output_ready), or when the wait is over."""
        started = time.monotonic()
        limit = min(float(args.get("timeout_seconds", self.max_wait)), self.max_wait)
        since = args.get("since_state")
        floor = self._floor()
        task = self._fetch_task(args["task_id"], use_cache=True)

        def answer(current: Dict[str, Any], changed: bool) -> Dict[str, Any]:
            return {"task": current, "changed": changed,
                    "waited_seconds": round(time.monotonic() - started, 1)}

        if since is not None and task["state"] != since:
            return answer(task, True)
        if task["state"] in FINAL_STATES:
            return answer(task, False)
        seen = (task["state"], task["cancel_requested"], task["output_ready"])
        while limit - (time.monotonic() - started) >= floor:
            self._sleep(floor)
            task = self._fetch_task(args["task_id"], use_cache=False)
            if (task["state"], task["cancel_requested"], task["output_ready"]) != seen:
                return answer(task, True)
        return answer(task, False)

    def _cancel_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        task_id = args["task_id"]
        if not TASK_ID_RX.fullmatch(str(task_id)):
            raise ToolFailure("invalid_arguments", "task_id")
        task = task_out(self.client().cancel(task_id).raw)
        # The state just changed: no cached answer may hide that.
        self._status.clear()
        self._status.put(("task", task_id), task)
        return task

    def _get_text_result(self, args: Dict[str, Any]) -> Dict[str, Any]:
        task_id = args["task_id"]
        task = self._fetch_task(task_id, use_cache=False)
        if task["state"] != "completed":
            raise ToolFailure("not_ready", extra={"state": task["state"]})
        if task["output_kind"] != "text":
            raise ToolFailure("not_text")
        client = self.client()
        response = client._transport.request(
            "GET", f"/api/v1/tasks/{task_id}/result",
            params={"part": args["part"]} if args.get("part") else None, stream=True,
            timeout=(5.0, self.api_timeout))
        try:
            chunks, size = [], 0
            for chunk in response.iter_content(chunk_size=8192):
                chunks.append(chunk)
                size += len(chunk)
                if size > self.text_cap:
                    break
            given = client_mod._disposition_filename(
                response.headers.get("Content-Disposition") or "")
        finally:
            response.close()
        data = b"".join(chunks)
        truncated = len(data) > self.text_cap
        data = data[:self.text_cap]
        filename = os.path.basename(given.replace("\\", "/"))[:255] if given else None
        return {"task_id": task_id, "filename": filename or None,
                # A cut inside a multi-byte character becomes one U+FFFD.
                "text": data.decode("utf-8", errors="replace"),
                "truncated": truncated, "bytes": len(data), "source": "ai_model",
                "notice": self.manifest["notes"]["ai_output"]}

    def _run_scan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        resolved: List[str] = []
        shown: Dict[str, str] = {}
        for given in args["paths"]:
            real = self.resolve(given)
            if real not in shown:
                resolved.append(real)
                shown[real] = given
        series = args.get("series")
        request = {"paths": resolved, "model": args["model"], "task": args.get("task"),
                   "budget": args["budget"], "max_credits": float(args["max_credits"]),
                   "label": args.get("label"), "text": args.get("text"),
                   "series": sorted({str(s) for s in series}) if series else None,
                   "force": bool(args.get("force", False))}
        return self.runs.start(request, resolved, shown)

    def _get_run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        found = self.runs.get(str(args["run_id"]))
        if found is None:
            raise ToolFailure("run_not_found")
        return found

    def _download_result(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Save the result in an allowed folder; the same call repeated
        waits for (or returns) the same file instead of downloading again."""
        task_id, part = args["task_id"], args.get("part")
        if not TASK_ID_RX.fullmatch(str(task_id)):
            raise ToolFailure("invalid_arguments", "task_id")
        if args.get("folder") is None and not self.allowed:
            raise ToolFailure("path_not_allowed")
        folder = self.resolve(args.get("folder") or os.path.join(self.allowed[0], "results"),
                              must_exist=False)
        key = (task_id, part or "", folder)
        with self._downloads_lock:
            known = self._downloads.get(key)
            stale = known is not None and known.done.is_set() and (
                known.error is not None or not (known.path and os.path.exists(known.path)))
            if known is None or stale:
                known = self._downloads[key] = self._start_download(task_id, part, folder)
        known.done.wait(self.download_wait)
        if not known.done.is_set():
            raise ToolFailure("download_in_progress")
        if known.error is not None:
            raise known.error
        # The saved file's name starts with the scan's original name: by
        # default the assistant gets the folder and a count.
        shown = self.relative(str(known.path)) if self.show_names else self.relative(folder)
        return {"task_id": task_id, "path": shown, "files": 1, "bytes": int(known.size)}

    def _start_download(self, task_id: str, part: Optional[str], folder: str) -> _Download:
        download = _Download()
        client = self.client()

        def work() -> None:
            try:
                saved = client.download(task_id, out=folder, part=part)
                if saved.skipped:
                    raise ToolFailure("not_ready", extra={"state": saved.skipped[0][1]})
                download.path = str(saved[0])
                download.size = os.path.getsize(download.path)
            except BaseException as error:
                download.error = error
            finally:
                download.done.set()

        download.thread = threading.Thread(target=work, name="nalvera-mcp-download",
                                           daemon=True)
        download.thread.start()
        return download


def _anonymize_text(code: str) -> str:
    key = f"anonymize.{code}"
    return messages.text(key, code=code) if messages.has(key) else \
        messages.text("anonymize.other", code=code)


__all__ = ["LocalTools", "ToolFailure", "allowed_folders", "inside", "task_out",
           "ALLOWED_DIRS_SETTING"]
