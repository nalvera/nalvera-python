"""What a notebook cell shows for the values the package returns.

The functions return plain data for code (lists, dicts, dataclasses). The
types here are list and dict subclasses that add a readable ``__repr__`` (a
text table) and ``_repr_html_`` (the same content as an HTML table), so a
cell that ends in ``nalvera.models()`` or ``nalvera.tasks()`` shows a table.

Every value in a table comes from a server or a file name and is untrusted:
anything shaped like an API key is replaced by its first 12 characters, it
is made printable (control and format characters become ``?``), cut to
``CELL_MAX`` characters and, in HTML, escaped with ``html.escape``. No
attribute, style, class or link is ever built from data.
"""
from __future__ import annotations

import html
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import messages
from .errors import redact
from .report import printable

#: The longest cell shown, in characters; a longer value ends in "…".
CELL_MAX = 200
#: The longest path shown in a table; a longer one is cut in the middle of its
#: folder part, and its file name stays whole.
PATH_MAX = 60

Rows = List[List[Any]]


def price_text(value: Any) -> Optional[str]:
    """A price with the digits it has (``0.006``, ``0.01``, ``0.02``) and at
    least two decimals; None when it is no finite number. Credits keep two
    decimals: this is for prices only."""
    number = _number(value)
    if number is None or not math.isfinite(number):
        return None
    whole, _, decimals = f"{number:.6f}".partition(".")
    decimals = decimals.rstrip("0").ljust(2, "0")
    return f"{whole}.{decimals}"


def path_text(path: Any, *, cut: bool = True) -> str:
    """A file's path as a person reads it: relative to the working folder
    when it is inside it, else with ``~`` for the home folder. With ``cut``
    (default True, for a table cell) a path longer than ``PATH_MAX`` is
    cut in the middle of its folder part, keeping the file name whole."""
    if path is None or path == "":
        return ""
    text = os.fspath(path)
    try:
        absolute = os.path.abspath(text)
        here = os.getcwd()
        home = os.path.expanduser("~")
        if absolute.startswith(here.rstrip(os.sep) + os.sep):
            text = os.path.relpath(absolute, here)
        elif home not in ("", "~") and absolute.startswith(home.rstrip(os.sep) + os.sep):
            text = "~" + absolute[len(home.rstrip(os.sep)):]
        else:
            text = absolute
    except (OSError, ValueError):
        pass
    text = printable(text)
    if not cut or len(text) <= PATH_MAX:
        return text
    folder, name = os.path.split(text)
    room = PATH_MAX - len(name) - len(os.sep) - 1          # the "…" takes one
    if room < 10:
        return "…" + os.sep + name
    left = room // 2
    return folder[:left] + "…" + folder[len(folder) - (room - left):] + os.sep + name


def cell(value: Any) -> str:
    """``value`` as one safe line for a table cell: None is ``-``, a
    key-shaped value shows its prefix only, control and format characters
    become ``?``, a long value is cut."""
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        text = f"{value:.2f}"
    elif isinstance(value, (list, tuple)):
        text = " x ".join(str(v) for v in value) if all(isinstance(v, int) for v in value) \
            else messages.text("list.separator").join(str(v) for v in value)
    else:
        text = str(value)
    text = printable(redact(text))
    return text if len(text) <= CELL_MAX else text[:CELL_MAX - 1] + "…"


def note(text: str) -> str:
    """A line under a table: a key-shaped value shows its prefix only, it is
    printable, and cut at twice ``CELL_MAX`` (a note may quote a value
    from a server)."""
    text = printable(redact(text))
    return text if len(text) <= 2 * CELL_MAX else text[:2 * CELL_MAX - 1] + "…"


def header(name: str) -> str:
    """A column's title (``table.<name>`` in ``nalvera.messages``)."""
    return messages.text(f"table.{name}")


def text_table(columns: Sequence[str], rows: Rows, notes: Iterable[str] = ()) -> str:
    """The table as aligned text lines, then each note on a line."""
    titles = [header(c) for c in columns]
    body = [[cell(v) for v in row] for row in rows]
    widths = [max([len(t)] + [len(r[i]) for r in body]) for i, t in enumerate(titles)]
    lines = ["  ".join(t.ljust(w) for t, w in zip(titles, widths)).rstrip()]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip() for row in body]
    if not body:
        lines.append(messages.text("table.empty"))
    lines += [note(n) for n in notes]
    return "\n".join(lines)


def html_table(columns: Sequence[str], rows: Rows, notes: Iterable[str] = ()) -> str:
    """The same content as ``text_table``, as an HTML table without any
    attribute; every value escaped."""
    def esc(value: str) -> str:
        return html.escape(value, quote=True)
    out = ["<table>", "<thead><tr>"]
    out += [f"<th>{esc(header(c))}</th>" for c in columns]
    out.append("</tr></thead>")
    out.append("<tbody>")
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{esc(cell(v))}</td>" for v in row) + "</tr>")
    if not rows:
        out.append(f"<tr><td>{esc(messages.text('table.empty'))}</td></tr>")
    out.append("</tbody></table>")
    out += [f"<p>{esc(note(n))}</p>" for n in notes]
    return "".join(out)


class _Table:
    """A value that shows itself as a table: subclasses give ``_table()``,
    which returns (columns, rows, notes)."""

    def _table(self) -> Tuple[Sequence[str], Rows, List[str]]:   # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return text_table(*self._table())

    def _repr_html_(self) -> str:
        return html_table(*self._table())


# ── Small helpers shared with the client ──────────────────────────────────────
def duration(seconds: Optional[float]) -> str:
    """A time span in words (``time.*``); ``-`` when not known."""
    if seconds is None:
        return messages.text("wait.none")
    total = int(round(float(seconds)))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    if h:
        return messages.text("time.hours", h=h, m=m)
    if m:
        return messages.text("time.minutes", m=m, s=s)
    return messages.text("time.seconds", s=s)


def state_text(state: Optional[str], cancelling: bool = False) -> str:
    """A task's state in words (``wait.state.*``)."""
    shown = "cancelling" if cancelling else (state or "")
    key = f"wait.state.{shown}"
    return messages.text(key) if messages.has(key) else printable(shown)


TASK_COLUMNS = ("task_id", "name", "model", "task", "state", "queue", "waited", "running")


def task_row(task: Any) -> List[Any]:
    """One task (a ``nalvera.Task``) as a row of ``TASK_COLUMNS``."""
    position = task.queue_position if task.state == "queued" else None
    return [task.id, task.display_name, task.model, task.task,
            state_text(task.state, task.cancelling), position,
            duration(task.waited_seconds), duration(task.running_seconds)]


# ── The values the functions return ───────────────────────────────────────────
class TaskList(_Table, list):
    """Tasks (a list of ``nalvera.Task``), shown as a table."""

    def _table(self):
        return TASK_COLUMNS, [task_row(t) for t in self], []


class ModelList(_Table, list):
    """The models of ``models()`` (a list of dicts as the platform sends
    them), shown as a table with a row per task."""

    def _table(self):
        rows = []
        for model in self:
            if not isinstance(model, dict):
                continue
            for task in model.get("tasks") or [{}]:
                if not isinstance(task, dict):
                    continue
                rows.append([model.get("id"), task.get("id"),
                             messages.text("table.yes") if task.get("default") else "",
                             model.get("name"), task.get("name"), task_modality(model, task),
                             price_text(task.get("price_eur_per_mvoxel"))])
        return ("model", "task", "default", "model_name", "task_name", "modality",
                "price"), rows, []


class Budgets(_Table, dict):
    """``budgets()``: free tasks, personal credits and each lab, as the
    platform sends them, shown as a table."""

    def _table(self):
        rows = []
        free = self.get("free") or {}
        if isinstance(free, dict):
            rows.append(["free", messages.text("table.free_tasks"),
                         messages.text("table.free_period") if free.get("period_active")
                         else free.get("free_tasks_left", 0)])
        personal = self.get("personal") or {}
        if isinstance(personal, dict):
            rows.append(["personal", messages.text("budget.personal_name"),
                         _number(personal.get("available_credits"))])
        for lab in self.get("labs") or []:
            if isinstance(lab, dict):
                rows.append([f"lab:{lab.get('id')}", lab.get("name"),
                             _number(lab.get("balance"))])
        notes = []
        # Without a lab there is no choice to make: personal pays.
        if self.get("suggested") and self.get("labs"):
            notes.append(messages.text("table.suggested", budget=printable(self["suggested"])))
        return ("budget", "budget_name", "available"), rows, notes


class Limits(_Table, dict):
    """``limits()``: the numbers of the key's limits, shown as a table."""

    def _table(self):
        rows = [[messages.text(f"table.limit.{k}") if messages.has(f"table.limit.{k}") else k,
                 v] for k, v in self.items()]
        return ("limit", "value"), rows, []


class Account(_Table, dict):
    """``me()``: the account, the key (its prefix, never the key), its
    permissions and end date, as the platform sends them; shown as a table
    of one row."""

    def _table(self):
        key = self.get("key") if isinstance(self.get("key"), dict) else {}
        scopes = [messages.text(f"scope.{s}") for s in key.get("scopes") or []
                  if isinstance(s, str) and messages.has(f"scope.{s}")]
        row = [self.get("email"), self.get("plan_label") or self.get("plan"), key.get("prefix"),
               key.get("name"), scopes, str(key.get("expires_at") or "")[:10]]
        notes = [messages.text("whoami.terms")] if self.get("terms_reaccept_required") else []
        return ("account", "plan", "key", "key_name", "permissions", "key_ends"), [row], notes


class ScanRecord(_Table, list):
    """``scans()``: the record on this computer, one dict per row (code,
    original name, label, task, model, result file...), shown as a table.
    ``path`` is the record's file (the first call that writes to it says
    where it is)."""

    def __init__(self, rows: Iterable[Dict[str, Any]] = (), path: Optional[str] = None):
        super().__init__(rows)
        self.path = path

    def _table(self):
        rows = [[r.get("code"), _series_name(r), r.get("label"), r.get("task_id"),
                 r.get("model"), r.get("task"), path_text(r.get("result_path"))] for r in self]
        return ("code", "original_name", "label", "task_id", "model", "task", "result"), \
            rows, []


def task_modality(model: Dict[str, Any], task: Dict[str, Any]) -> List[str]:
    """A task's modality: its own when the platform names one, else its
    model's (a model such as MRSegmentator has a CT task beside its MRI
    ones)."""
    own = [str(m) for m in task.get("modality") or [] if m]
    return own or [str(m) for m in model.get("modality") or [] if m]


def _series_name(row: Dict[str, Any]) -> Any:
    name = row.get("original_name")
    if name and row.get("series"):
        return messages.text("run.series_name", name=name, series=row.get("series"))
    return name


def _number(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["TaskList", "ModelList", "Budgets", "Limits", "Account", "ScanRecord", "cell",
           "text_table", "html_table", "duration", "state_text", "price_text", "path_text",
           "task_modality", "CELL_MAX", "PATH_MAX"]
