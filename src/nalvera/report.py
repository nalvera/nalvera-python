"""Where the package's messages go: one injectable reporter.

Nothing in the library prints or reads a terminal by itself. Every progress
line, notice and question of a ``nalvera.Client`` goes to its
reporter, and only a reporter calls ``input`` or ``getpass``:

``NotebookReporter``
    Chosen by default inside a Jupyter kernel. Each call of the package
    (``run``, ``wait``, ``download``...) shows its progress in one display
    redrawn in place; when the call returns, the progress is gone and the
    lines a person must read stay, then the cell shows the returned table.
    Questions are asked with ``input`` and ``getpass`` when the notebook
    accepts input.
``StreamReporter``
    Chosen by default elsewhere. Writes each message as a line to a stream
    (default ``sys.stdout``) and, when both its streams are terminals, can
    ask a question. The ``nalvera`` command uses one.
``NullReporter``
    Discards everything and never asks (``Client(quiet=True)``).
``ListReporter``
    Keeps every message as an ``Event``, for a program that shows
    progress its own way (a widget, a server speaking a protocol on
    standard output). It never asks.

A reporter that cannot ask makes the package take the non-interactive path:
a run that needs a budget choice raises ``nalvera.BudgetChoiceRequired``
instead of prompting.
"""
from __future__ import annotations

import contextvars
import getpass
import html
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, TextIO

from . import messages
from .errors import printable, redact

# Fields whose value is a block of lines the package built itself.
_MULTILINE_FIELDS = frozenset({"options"})

TERMINAL, KERNEL, OTHER = "terminal", "kernel", "other"

#: Lines that say how far a call has come. In a notebook they are shown in
#: the call's one display while it runs and are gone when it ends.
PROGRESS_KEYS: FrozenSet[str] = frozenset({
    "login.checking", "run.preparing", "run.prepared", "run.folder_nifti", "run.dicom_found",
    "run.checking", "run.plan", "run.plan_free", "run.budget", "run.queued",
    "run.already_queued", "run.recovered", "run.active_cap_wait", "run.retry_wait",
    "upload.progress", "upload.done", "upload.resume"})

#: For each kind of call, the lines whose facts the table it returns shows.
#: In a notebook they are left out when the call returns (the table says
#: it); when the call raises instead, they are shown.
SHOWN_BY_TABLE: Dict[str, FrozenSet[str]] = {
    "run": frozenset({"run.already_done", "run.summary"}),
    "estimate": frozenset({"run.scan_failed", "run.duplicate_in_run", "run.already_done"}),
    "generate": frozenset({"generate.earlier"}),
    "wait": frozenset({"wait.summary", "wait.nothing"}),
    "download": frozenset({"download.saved", "download.already_saved", "download.skipped"}),
    "anonymize": frozenset({"run.scan_failed", "anonymized.copy", "anonymized.removed",
                            "anonymized.done"}),
}


def clean_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """The fields with every text value made printable (see ``printable``)."""
    return {k: (printable(v) if isinstance(v, str) and k not in _MULTILINE_FIELDS else v)
            for k, v in fields.items()}


@dataclass
class Event:
    """One message: ``key`` (a stable key of ``nalvera.messages``), ``text``
    (the filled text) and ``fields`` (the values that filled it)."""
    key: str
    text: str
    fields: Dict[str, Any] = field(default_factory=dict)


class Reporter:
    """The interface. Subclass it and override what you need; the defaults
    discard everything and never ask."""

    #: True when ``live`` redraws a block of lines in place.
    supports_live = False
    #: True when a call's progress lines (``run.queued`` among them) are gone
    #: once it ends: a call that raises after it queued tasks then says them
    #: in one line (``run.stopped_with_tasks``).
    hides_progress = False

    def message(self, event: Event) -> None:
        """A progress line or notice."""

    def error(self, text: str) -> None:
        """An error that ends a command (used by the command line)."""

    def live(self, lines: List[str]) -> None:
        """Redraw the waiting table (only called when ``supports_live``)."""

    def live_done(self) -> None:
        """The waiting table is final: the next ``live`` starts a new one."""

    def begin_call(self, kind: str) -> None:
        """A call of the package starts (``kind``: a key of
        ``SHOWN_BY_TABLE``, such as ``"run"``). Calls may nest."""

    def end_call(self, error: Optional[BaseException] = None) -> None:
        """The call started last ends: it returned (``error`` None), or it
        raised ``error``."""

    def can_ask(self) -> bool:
        """Whether ``ask`` reaches a person who can answer."""
        return False

    def ask(self, prompt: str, *, hidden: bool = False) -> Optional[str]:
        """Show ``prompt`` and return the answer line, or None when there is
        nobody to answer. ``hidden`` (default False): the answer is not
        shown while typed (an API key)."""
        return None

    # ── helpers for the package ───────────────────────────────────────────────
    def say(self, key: str, /, **fields: Any) -> Event:
        """Fill the text for ``key`` and pass it to ``message``."""
        shown = clean_fields(fields)
        event = Event(key=key, text=redact(messages.text(key, **shown)), fields=shown)
        self.message(event)
        return event


@contextmanager
def call_scope(reporter: Any, kind: str) -> Iterator[None]:
    """One call of the package, for ``reporter``: ``begin_call(kind)``, then
    ``end_call`` with the error the call raised, if any. A reporter without
    these methods is left as it is."""
    begin = getattr(reporter, "begin_call", None)
    end = getattr(reporter, "end_call", None)
    if begin is None or end is None:
        yield
        return
    begin(kind)
    try:
        yield
    except BaseException as error:
        end(error)
        raise
    end(None)


class NullReporter(Reporter):
    """Shows nothing and never asks."""


class ListReporter(Reporter):
    """Keeps every message in ``events`` (thread-safe) and never asks."""

    def __init__(self) -> None:
        self.events: List[Event] = []
        self.errors: List[str] = []
        self._lock = threading.Lock()

    def message(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)

    def error(self, text: str) -> None:
        with self._lock:
            self.errors.append(redact(text))

    def keys(self) -> List[str]:
        """The keys of the messages so far, in order."""
        with self._lock:
            return [e.key for e in self.events]


def _isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _kernel_shell() -> Any:
    """The IPython shell of a Jupyter kernel running this code, or None.
    IPython is never imported here: only a program that loaded it can be
    running in its kernel."""
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return None
    try:
        shell = ipython.get_ipython()
    except Exception:
        return None
    if shell is None or getattr(shell, "kernel", None) is None:
        return None
    return shell


def session_kind() -> str:
    """Where this code runs: ``"kernel"`` (a Jupyter kernel), ``"terminal"``
    (standard input and output are both terminals) or ``"other"`` (a file,
    a pipe, a scheduled task)."""
    if _kernel_shell() is not None:
        return KERNEL
    if _isatty(sys.stdin) and _isatty(sys.stdout):
        return TERMINAL
    return OTHER


def cell_token() -> Any:
    """What tells the notebook cell (or IPython input) running now from any
    other: the id of the kernel's current request, else the execution
    count. None outside IPython. IPython is never imported here."""
    ipython = sys.modules.get("IPython")
    try:
        shell = ipython.get_ipython() if ipython is not None else None
    except Exception:
        shell = None
    if shell is None:
        return None
    try:
        header = (getattr(shell, "parent_header", None) or {}).get("header") or {}
        if header.get("msg_id"):
            return header["msg_id"]
    except Exception:
        pass
    return ("execution", getattr(shell, "execution_count", None))


def shows_beside_results(reporter: Any) -> bool:
    """Whether ``reporter`` writes where IPython shows a cell's result: a
    notebook's cell, or the standard output of a terminal IPython."""
    return isinstance(reporter, NotebookReporter) or (
        isinstance(reporter, StreamReporter) and reporter.output is sys.stdout)


def safe_write(stream: TextIO, text: str) -> None:
    """Write ``text``; a character the stream's encoding cannot show (such as
    "⋮" on an old Windows console) is replaced instead of raising."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    try:
        stream.flush()
    except Exception:
        pass


class StreamReporter(Reporter):
    """Writes messages as lines to ``output`` (default ``sys.stdout``) and
    errors to ``errors`` (default ``sys.stderr``). ``interactive`` (default:
    both ``input`` and ``output`` are terminals) decides whether questions
    are asked on ``input`` (default ``sys.stdin``). A hidden question reads
    with ``secret_input`` (default: ``getpass.getpass`` on a terminal, else
    a line of ``input``).

    The waiting table is redrawn in place only on a terminal; on a file or
    a pipe a line is written when a task's state or queue position
    changes, no more often than the package checks."""

    def __init__(self, output: Optional[TextIO] = None, input: Optional[TextIO] = None,
                 errors: Optional[TextIO] = None, interactive: Optional[bool] = None,
                 secret_input: Optional[Callable[[str], str]] = None):
        self.output = output if output is not None else sys.stdout
        self.input = input if input is not None else sys.stdin
        self.errors = errors if errors is not None else sys.stderr
        self.interactive = (_isatty(self.input) and _isatty(self.output)) \
            if interactive is None else bool(interactive)
        self.supports_live = _isatty(self.output)
        self.secret_input = secret_input
        self._lock = threading.Lock()
        self._drawn = 0

    def message(self, event: Event) -> None:
        with self._lock:
            self._drawn = 0
            safe_write(self.output, event.text + "\n")

    def error(self, text: str) -> None:
        with self._lock:
            safe_write(self.errors, redact(text) + "\n")

    def live(self, lines: List[str]) -> None:
        with self._lock:
            out = []
            if self._drawn:
                out.append(f"\x1b[{self._drawn}F")
            for line in lines:
                out.append("\x1b[2K" + redact(line) + "\n")
            safe_write(self.output, "".join(out))
            self._drawn = len(lines)

    def live_done(self) -> None:
        with self._lock:
            self._drawn = 0

    def can_ask(self) -> bool:
        return self.interactive

    def ask(self, prompt: str, *, hidden: bool = False) -> Optional[str]:
        if not self.interactive:
            return None
        with self._lock:
            self._drawn = 0
        if hidden:
            reader = self.secret_input
            if reader is None and self.input is sys.stdin and _isatty(self.input):
                reader = getpass.getpass
            if reader is not None:
                try:
                    return reader(prompt)
                except EOFError:
                    return None
        with self._lock:
            safe_write(self.output, prompt)
        line = self.input.readline()
        return line if line else None


class _Call:
    """What one call has shown so far in a notebook: ``lines`` stay,
    ``progress`` is gone when the call ends, ``held`` (facts the returned
    table shows) is shown only when the call raises."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.depth = 1
        self.handle: Any = None
        self.lines: List[str] = []
        self.progress: List[str] = []
        self.held: List[str] = []
        self.token: Any = None


class NotebookReporter(Reporter):
    """The reporter of a Jupyter kernel, chosen by default there.

    * A call of the package (``run``, ``estimate_run``, ``generate``,
      ``wait``, ``download``, ``save_anonymized``) shows its progress as one
      display, redrawn in place through the notebook's own display
      mechanism (plain text and an escaped HTML ``<pre>``; never a terminal
      escape code). When the call returns, the progress is gone; what a
      person must read stays (that nothing is removed from the image, a resampled
      series, a scan left out and why...); a line whose fact the returned
      table shows is left out. When the call raises, those lines stay too.
    * Outside a call (``login``), messages are plain lines on ``output``
      (default: the kernel's ``sys.stdout``, shown under the cell).
    * A question is asked with ``input``, a hidden one (the API key) with
      ``getpass.getpass``, only while the notebook accepts input. A notebook
      run by ``nbconvert``, papermill or a scheduler does not: the package
      then takes the non-interactive path (an exception that names the
      argument to pass).
    * Safe from a background thread: the lines of ``run_in_background`` are
      shown under the cell that started the run.
    """

    supports_live = True

    def __init__(self, output: Optional[TextIO] = None):
        self._output = output
        self._lock = threading.RLock()
        self._handle: Any = None
        # The call running in this context (a copy of it, in the threads of a
        # run, refers to the same call).
        self._call: contextvars.ContextVar = contextvars.ContextVar(
            f"nalvera_call_{id(self)}", default=None)

    @property
    def output(self) -> TextIO:
        """Where lines go (the kernel's current ``sys.stdout`` by default)."""
        return self._output if self._output is not None else sys.stdout

    @property
    def hides_progress(self) -> bool:                  # type: ignore[override]
        """True while the notebook's display mechanism is there (a call's
        progress is then one display, emptied when the call ends)."""
        return display_module() is not None

    def _scope(self) -> Optional[_Call]:
        state = self._call.get()
        return state if state is not None and display_module() is not None else None

    def begin_call(self, kind: str) -> None:
        state = self._call.get()
        if state is not None:
            with self._lock:
                state.depth += 1
            return
        state = _Call(kind)
        state.token = self._call.set(state)

    def end_call(self, error: Optional[BaseException] = None) -> None:
        state = self._call.get()
        if state is None:
            return
        with self._lock:
            state.depth -= 1
            if state.depth:
                return
        try:
            self._call.reset(state.token)
        except ValueError:                      # ended in another context
            self._call.set(None)
        with self._lock:
            state.progress = []
            if error is not None:
                state.lines += state.held
            if state.handle is not None or state.lines:
                self._draw(state)

    def message(self, event: Event) -> None:
        state = self._scope()
        with self._lock:
            if state is None:
                safe_write(self.output, event.text + "\n")
                return
            if event.key in PROGRESS_KEYS:
                state.progress = [event.text]
            elif event.key in SHOWN_BY_TABLE.get(state.kind, ()):
                state.held.append(event.text)
                return
            else:
                state.lines.append(event.text)
            self._draw(state)

    def error(self, text: str) -> None:
        with self._lock:
            safe_write(self.output, redact(text) + "\n")

    def live(self, lines: List[str]) -> None:
        shown = [redact(line) for line in lines]
        state = self._scope()
        with self._lock:
            if state is not None:
                state.progress = shown
                self._draw(state)
                return
            display = display_module()
            body = "\n".join(shown)
            if display is None:
                safe_write(self.output, body + "\n")
            elif self._handle is None:
                self._handle = display.display(_bundle(body), raw=True, display_id=True)
            else:
                self._handle.update(_bundle(body), raw=True)

    def live_done(self) -> None:
        state = self._scope()
        with self._lock:
            if state is not None:
                state.progress = []
            else:
                self._handle = None

    def _draw(self, state: _Call) -> None:
        """Show the call's lines and progress in its one display: made at
        its first line, redrawn after; emptied when nothing stays."""
        body = "\n".join(state.lines + state.progress)
        if state.handle is None:
            display = display_module()
            if display is not None and body:
                state.handle = display.display(_bundle(body), raw=True, display_id=True)
        else:
            state.handle.update(_bundle(body), raw=True)

    def can_ask(self) -> bool:
        shell = _kernel_shell()
        if shell is None:
            return False
        return bool(getattr(shell.kernel, "_allow_stdin", False))

    def ask(self, prompt: str, *, hidden: bool = False) -> Optional[str]:
        if not self.can_ask():
            return None
        try:
            return getpass.getpass(prompt) if hidden else input(prompt)
        except EOFError:
            return None
        except Exception as error:
            # IPython's StdinNotImplementedError: the front end takes no input.
            if type(error).__name__ == "StdinNotImplementedError":
                return None
            raise


def _bundle(body: str) -> Dict[str, str]:
    """A display's content: plain text and the same in an escaped ``<pre>``
    (an empty display stays empty)."""
    return {"text/plain": body,
            "text/html": "<pre>" + html.escape(body, quote=True) + "</pre>" if body else ""}


def display_module() -> Any:
    """``IPython.display`` when IPython is loaded (it always is in a
    kernel), else None. Never imports IPython into a program without it."""
    if "IPython" not in sys.modules:
        return None
    try:
        import importlib
        return importlib.import_module("IPython.display")
    except Exception:
        return None


_display_module = display_module


def default_reporter(output: Optional[TextIO] = None, input: Optional[TextIO] = None,
                     interactive: Optional[bool] = None) -> Reporter:
    """The reporter a client uses when none is given: a
    ``NotebookReporter`` in a Jupyter kernel (unless ``output`` or
    ``input`` is given), else a ``StreamReporter`` on ``output`` and
    ``input``."""
    if output is None and input is None and session_kind() == KERNEL:
        return NotebookReporter()
    return StreamReporter(output=output, input=input, interactive=interactive)


__all__ = ["Reporter", "StreamReporter", "NotebookReporter", "NullReporter", "ListReporter",
           "Event", "printable", "safe_write", "session_kind", "default_reporter", "call_scope",
           "cell_token", "PROGRESS_KEYS", "SHOWN_BY_TABLE"]
