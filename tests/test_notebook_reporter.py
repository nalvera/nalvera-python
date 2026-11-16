"""Notebooks: telling a kernel from a terminal, the
NotebookReporter (one display per call that ends with only what its table
does not show, questions only while the notebook accepts input), the lines
of plain output said once, and stopping a cell. A fake IPython stands in for
the kernel here; tests/test_notebooks.py runs a real one."""
from __future__ import annotations

import contextvars
import io
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import nalvera
from nalvera import display, messages, report
from nalvera.errors import BudgetChoiceRequired, RunInterrupted, WaitInterrupted

MODEL = "totalsegmentator_ct"
SRC = Path(__file__).resolve().parent.parent / "src"


class FakeKernel:
    def __init__(self, allow_stdin=False):
        self._allow_stdin = allow_stdin


class FakeHandle:
    def __init__(self, shown):
        self.shown = shown
        self.updates = []

    def update(self, bundle, raw=False):
        assert raw is True
        self.updates.append(bundle)


@pytest.fixture
def kernel(monkeypatch):
    """A fake Jupyter kernel: IPython in sys.modules, a shell with a kernel,
    and IPython.display recording what is shown."""
    state = types.SimpleNamespace(kernel=FakeKernel(), displays=[], handles=[], plain=[])
    shell = types.SimpleNamespace(kernel=state.kernel, parent_header={}, execution_count=1)
    state.shell = shell
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    shown = types.ModuleType("IPython.display")

    def display(bundle, raw=False, display_id=None):
        assert raw is True
        if display_id is None:                   # shown once, never updated
            state.plain.append(bundle)
            return None
        assert display_id is True
        state.displays.append(bundle)
        handle = FakeHandle(bundle)
        state.handles.append(handle)
        return handle

    shown.display = display
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", shown)
    return state


class _TTY(io.StringIO):
    def isatty(self):
        return True


# ── which session ─────────────────────────────────────────────────────────────
def test_a_kernel_is_told_from_a_terminal_and_from_neither(monkeypatch, kernel):
    assert report.session_kind() == report.KERNEL
    monkeypatch.delitem(sys.modules, "IPython")
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert report.session_kind() == report.OTHER
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", _TTY())
    assert report.session_kind() == report.TERMINAL


def test_terminal_ipython_is_not_a_kernel(monkeypatch):
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: types.SimpleNamespace()        # no kernel
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    assert report.session_kind() != report.KERNEL


def test_ipython_is_never_imported_by_the_package(tmp_path):
    code = ("import sys, nalvera\nfrom nalvera import report\n"
            "report.session_kind(); report.default_reporter()\n"
            "r = report.NotebookReporter(); r.live(['a']); r.can_ask()\n"
            "assert 'IPython' not in sys.modules, 'imported'\nprint('ok')\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(SRC), "PATH": ""}, cwd=tmp_path, timeout=60)
    assert done.returncode == 0 and "ok" in done.stdout, done.stderr


def test_in_a_kernel_the_default_reporter_is_the_notebook_reporter(kernel, env, server):
    assert isinstance(report.default_reporter(), report.NotebookReporter)
    client = nalvera.Client()
    assert isinstance(client.reporter, report.NotebookReporter)
    client.close()
    assert isinstance(report.default_reporter(output=io.StringIO()), report.StreamReporter)


# ── the notebook reporter ─────────────────────────────────────────────────────
def test_the_waiting_table_is_one_display_redrawn_in_place(kernel, capsys):
    reporter = report.NotebookReporter()
    reporter.live(["<script>a</script>: waiting", "b: running"])
    reporter.live(["a: completed", "b: running"])
    assert len(kernel.displays) == 1 and len(kernel.handles[0].updates) == 1
    first = kernel.displays[0]
    assert set(first) == {"text/plain", "text/html"}
    assert "<script>" not in first["text/html"] and "&lt;script&gt;" in first["text/html"]
    assert first["text/html"].startswith("<pre>")
    assert kernel.handles[0].updates[0]["text/plain"] == "a: completed\nb: running"
    reporter.live_done()
    reporter.live(["c"])
    assert len(kernel.displays) == 2
    assert "\x1b" not in capsys.readouterr().out


def test_messages_are_plain_lines_without_escape_codes(kernel):
    out = io.StringIO()
    reporter = report.NotebookReporter(output=out)
    reporter.say("run.preparing", name="a\x1b[2Kb.nii.gz")
    assert "\x1b" not in out.getvalue() and out.getvalue().endswith("\n")


def test_questions_are_asked_only_while_the_notebook_accepts_input(kernel, monkeypatch):
    reporter = report.NotebookReporter()
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "2")
    monkeypatch.setattr(report.getpass, "getpass",
                        lambda prompt="": asked.append(("hidden", prompt)) or "secret")
    assert not reporter.can_ask() and reporter.ask("Number: ") is None and asked == []
    kernel.kernel._allow_stdin = True
    assert reporter.can_ask()
    assert reporter.ask("Number: ") == "2"
    assert reporter.ask("Key: ", hidden=True) == "secret"
    assert asked == ["Number: ", ("hidden", "Key: ")]


def test_a_front_end_without_input_counts_as_nobody_to_ask(kernel, monkeypatch):
    kernel.kernel._allow_stdin = True

    class StdinNotImplementedError(RuntimeError):
        pass

    def refuse(prompt=""):
        raise StdinNotImplementedError("no input")

    monkeypatch.setattr("builtins.input", refuse)
    assert report.NotebookReporter().ask("Number: ") is None


def test_wait_in_a_kernel_redraws_one_table_and_prints_no_escape_code(kernel, make_client,
                                                                     server, scans, capsys):
    server.script = [("queued", 2), ("processing", None), ("completed", None)]
    client = make_client(reporter=report.NotebookReporter(), output=None)
    tasks = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL, budget="personal")
    shown = len(kernel.displays)
    done = client.wait(tasks)
    assert {t.state for t in done} == {"completed"}
    assert len(kernel.displays) == shown + 1                    # one display for the wait
    handle = kernel.handles[-1]
    drawn = [kernel.displays[-1]["text/plain"]] + [u["text/plain"] for u in handle.updates]
    assert any(d.count(messages.text("wait.state.completed")) == 2 for d in drawn)
    # The returned table shows the states and times: the display ends empty.
    assert handle.updates[-1] == {"text/plain": "", "text/html": ""}
    assert "\x1b" not in capsys.readouterr().out


# ── one display per call, each idea once ──────────────────────────────────────
def _final(kernel, index):
    """What display ``index`` shows once its call has ended."""
    handle = kernel.handles[index]
    return (handle.updates[-1] if handle.updates else kernel.displays[index])["text/plain"]


def test_a_run_shows_one_display_that_keeps_only_what_its_table_does_not(kernel, make_client,
                                                                        server, scans, env):
    client = make_client(reporter=report.NotebookReporter(), output=None)
    result = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL,
                        budget="personal")
    assert len(kernel.displays) == 1
    final = _final(kernel, 0).splitlines()
    record = display.path_text(env / "nalvera_scans.csv", cut=False)
    assert final == [messages.text("run.pixel_notice"),
                     messages.text("run.record", path=record)]
    seen = "\n".join([kernel.displays[0]["text/plain"]]
                     + [u["text/plain"] for u in kernel.handles[0].updates])
    assert messages.text("run.preparing", name="a.nii.gz") in seen      # progress, then gone
    table = repr(result)
    for task in result:
        assert task.id in table
    assert len(result) == 2


def test_the_record_is_said_once_per_sign_in_at_the_first_call_that_writes(make_client,
                                                                           server, scans):
    reporter = nalvera.ListReporter()
    client = make_client(reporter=reporter, output=None)
    client.estimate_run(scans("a.nii.gz"), model=MODEL)
    assert "run.record" not in reporter.keys()                   # an estimate writes nothing
    client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    client.run(scans("b.nii.gz"), model=MODEL, budget="personal")
    assert reporter.keys().count("run.record") == 1


def test_a_download_shows_no_line_that_its_table_shows(kernel, make_client, server, scans,
                                                       env):
    client = make_client(reporter=report.NotebookReporter(), output=None)
    done = client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    shown = len(kernel.displays)
    finals = [_final(kernel, i) for i in range(shown)]
    saved = client.download(done, out=env / "results")
    assert saved and all(_final(kernel, i) == finals[i] for i in range(shown))
    assert len(kernel.displays) == shown                         # nothing but the table


def test_a_call_that_raises_keeps_the_other_lines_and_never_repeats_its_error(
        kernel, make_client, server, scans):
    client = make_client(reporter=report.NotebookReporter(), output=None)
    with pytest.raises(nalvera.PreparationError) as caught:
        client.run([scans("bad.nii.gz", b"NOTASCAN"), scans("worse.nii.gz", b"NOTASCAN 2")],
                   model=MODEL, budget="personal")
    final = _final(kernel, 0)
    reason = messages.text("anonymize.not_a_scan")
    assert messages.text("run.scan_failed", name="worse.nii.gz", reason=reason) in final
    assert "bad.nii.gz" not in final and "bad.nii.gz" in str(caught.value)
    assert messages.text("run.summary.failed", count=2) not in final


def test_an_estimate_that_raises_shows_what_its_table_would_have(kernel, make_client, server,
                                                                 scans, monkeypatch):
    client = make_client(reporter=report.NotebookReporter(), output=None)

    def broken(*args, **kwargs):
        raise nalvera.NalveraError("the platform is away", code="server_error")

    monkeypatch.setattr(client, "estimate", broken)
    with pytest.raises(nalvera.NalveraError):
        client.estimate_run([scans("a.nii.gz"), scans("bad.nii.gz", b"NOTASCAN")], model=MODEL)
    reason = messages.text("anonymize.not_a_scan")
    assert messages.text("run.scan_failed", name="bad.nii.gz", reason=reason) in \
        _final(kernel, 0)


def test_in_terminal_ipython_the_client_after_login_shows_nothing_either(monkeypatch, env,
                                                                         server):
    shell = types.SimpleNamespace(execution_count=4)            # no kernel: a terminal
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    shown_module = types.ModuleType("IPython.display")
    shown = []
    shown_module.display = lambda bundle, raw=False: shown.append(bundle)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", shown_module)
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    client = nalvera.login(server.email, server.key)
    client._ipython_display_()
    assert shown == [] and len(out.getvalue().splitlines()) == 1
    shell.execution_count = 5
    client._ipython_display_()
    assert len(shown) == 1 and server.email in shown[0]["text/plain"]


def test_a_quiet_login_leaves_the_client_its_table(kernel, env, server):
    client = nalvera.login(server.email, server.key, reporter=nalvera.NullReporter())
    client._ipython_display_()
    assert len(kernel.plain) == 1


def test_right_after_login_the_client_shows_nothing_and_later_its_table(kernel, env, server):
    kernel.shell.parent_header = {"header": {"msg_id": "cell-1"}}
    reporter = report.NotebookReporter(output=io.StringIO())
    client = nalvera.login(server.email, server.key, reporter=reporter)
    client._ipython_display_()
    assert kernel.plain == []                                    # the sign-in line says it
    kernel.shell.parent_header = {"header": {"msg_id": "cell-2"}}
    client._ipython_display_()
    [table] = kernel.plain
    assert server.email in table["text/plain"] and "<table>" in table["text/html"]
    assert server.key[12:] not in table["text/plain"] + table["text/html"]
    lines = reporter.output.getvalue().splitlines()
    assert len(lines) == 1 and server.key[:12] in lines[0]


def test_a_budget_question_in_a_kernel_without_input_names_the_argument(
        kernel, make_client, server, scans):
    server.add_lab(12, "Imaging Lab", 100.0)
    client = make_client(reporter=report.NotebookReporter(output=io.StringIO()), output=None,
                         interactive=None)
    with pytest.raises(BudgetChoiceRequired) as caught:
        client.run(scans("a.nii.gz"), model=MODEL)
    assert messages.fragment("budget", messages.PYTHON) in str(caught.value)
    assert not server.calls("POST", "/api/v1/uploads")


def test_a_budget_question_in_a_kernel_with_input_is_asked_once(kernel, make_client, server,
                                                                scans, monkeypatch):
    server.add_lab(12, "Imaging Lab", 100.0)
    kernel.kernel._allow_stdin = True
    answers = []
    monkeypatch.setattr("builtins.input", lambda prompt="": answers.append(prompt) or "1")
    client = make_client(reporter=report.NotebookReporter(output=io.StringIO()), output=None,
                         interactive=None)
    [task] = client.run(scans("a.nii.gz"), model=MODEL)
    assert len(answers) == 1 and server.tasks[task.id]["budget"] == "personal"


# ── outside a terminal and a kernel ───────────────────────────────────────────
def test_on_a_file_a_line_is_written_only_when_something_changes(make_client, server, scans):
    out = io.StringIO()
    server.script = [("queued", 1), ("queued", 1), ("queued", 1), ("completed", None)]
    client = make_client(output=out)
    client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    waiting = [line for line in out.getvalue().splitlines()
               if messages.text("wait.state.queued") in line]
    assert len(waiting) == 1 and "\x1b" not in out.getvalue()


def test_on_a_file_a_failed_scan_is_said_once_and_no_count_is_zero(make_client, server, scans):
    out = io.StringIO()
    client = make_client(output=out)
    result = client.run([scans("a.nii.gz"), scans("bad.nii.gz", b"NOTASCAN")], model=MODEL,
                        budget="personal")
    shown = out.getvalue()
    reason = messages.text("anonymize.not_a_scan")
    assert shown.count(messages.text("run.scan_failed", name="bad.nii.gz", reason=reason)) == 1
    summary = messages.text("run.summary", items=messages.text("list.sentence").join(
        [messages.text("run.summary.queued", count=1),
         messages.text("run.summary.failed", count=1)]))
    assert summary in shown.splitlines()
    assert messages.text("run.summary.skipped", count=0) not in shown
    assert len(result) == 1 and len(result.failed) == 1


def test_on_a_file_one_scan_and_one_task_have_no_summary_line(make_client, server, scans):
    out = io.StringIO()
    client = make_client(output=out)
    client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    shown = out.getvalue()
    assert messages.text("run.summary.queued", count=1) not in shown
    assert messages.text("wait.summary.completed", count=1) not in shown


def test_on_a_file_a_run_that_queues_nothing_says_its_reason_once(make_client, server, scans):
    out = io.StringIO()
    with pytest.raises(nalvera.PreparationError) as caught:
        make_client(output=out).run(scans("bad.nii.gz", b"NOTASCAN"), model=MODEL,
                                    budget="personal")
    reason = messages.text("anonymize.not_a_scan")
    assert reason in str(caught.value) and reason not in out.getvalue()


def test_on_a_file_several_tasks_end_with_the_counts_that_are_not_zero(make_client, server,
                                                                       scans):
    out = io.StringIO()
    client = make_client(output=out)
    client.wait(client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL,
                           budget="personal"))
    assert messages.text("wait.summary", items=messages.text(
        "wait.summary.completed", count=2)) in out.getvalue().splitlines()


# ── stopping a cell ───────────────────────────────────────────────────────────
def test_stopping_wait_names_the_function_that_waits_again(make_client, server, scans,
                                                           monkeypatch):
    server.script = [("queued", 1)]
    client = make_client(quiet=True)
    tasks = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")

    def stop(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(client._transport, "pause", stop)
    with pytest.raises(WaitInterrupted) as caught:
        client.wait(tasks)
    assert isinstance(caught.value, KeyboardInterrupt)
    text = str(caught.value)
    assert messages.fragment("wait", messages.PYTHON) in text and len(text.splitlines()) <= 2
    assert [t.id for t in caught.value.tasks] == [tasks[0].id]
    assert len(server.tasks) == 1                              # it keeps running


def test_stopping_a_run_keeps_what_was_queued_and_says_how_to_follow_it(make_client, server,
                                                                        scans, env):
    server.patch_delay = 0.05
    client = make_client(quiet=True, output=None)
    from conftest import scan_bytes
    background = client.run_in_background([scans(f"s{i}.nii.gz", scan_bytes(b"x", 9000))
                                           for i in range(4)], model=MODEL, budget="personal")
    end = time.monotonic() + 20
    while not server.calls("PATCH") and time.monotonic() < end:
        time.sleep(0.01)
    background.cancel()
    with pytest.raises(RunInterrupted) as caught:
        background.result(timeout=20)
    text = str(caught.value)
    assert messages.fragment("follow", messages.PYTHON) in text and len(text.splitlines()) <= 2
    rows = Path(env / "nalvera_scans.csv").read_text()
    for task in caught.value.tasks:
        assert task.id in rows                                 # what was queued is recorded


def test_a_background_run_reports_in_the_context_that_started_it(make_client, server, scans):
    marker = contextvars.ContextVar("cell", default=None)
    marker.set("cell 7")
    seen = []

    class Recording(nalvera.ListReporter):
        def message(self, event):
            seen.append(marker.get())
            super().message(event)

    client = make_client(reporter=Recording(), output=None)
    background = client.run_in_background(scans("a.nii.gz"), model=MODEL, budget="personal")
    background.result(timeout=20)
    assert seen and set(seen) == {"cell 7"}
