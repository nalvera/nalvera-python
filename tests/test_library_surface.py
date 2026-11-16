"""The library stays silent, returns data, can run in the background and
keeps to allowed folders; and the public surface: the settings list, the
exit statuses in the help, docstrings."""
from __future__ import annotations

import inspect
import io
import re
import sys
import time
from pathlib import Path

import pytest

import nalvera
import stub_anonymizer
from conftest import scan_bytes
from nalvera import cli, messages, settings
from nalvera.errors import BudgetChoiceRequired, ConfigError, RunInterrupted

MODEL, TASK = "totalsegmentator_ct", "total"
SRC = Path(__file__).resolve().parent.parent / "src" / "nalvera"


class _NoInput(io.StringIO):
    def readline(self, *args):
        raise AssertionError("the library read standard input")

    read = readline


# ── nothing in the library prints or reads a terminal ─────────────────────────
def test_a_whole_run_with_a_reporter_writes_nothing_to_the_terminal(make_client, server, scans,
                                                                    env, capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", _NoInput())
    reporter = nalvera.ListReporter()
    server.add_lab(12, "Imaging Lab", 100.0)
    client = make_client(reporter=reporter, output=None)
    found = client.estimate_run(scans("a.nii.gz"), model=MODEL)
    tasks = client.run([scans("a.nii.gz"), scans("b.nii.gz", b"NOTASCAN")], model=MODEL,
                       budget="lab:12")
    done = client.wait(tasks)
    client.download(done, out=env / "results")
    client.generate("nv_generate_ctmr", "rflow-ct", {"mode": "image-only",
                                                     "output_size": [256, 256, 128]},
                    budget="personal")
    with pytest.raises(BudgetChoiceRequired):                 # needs a choice: no prompt
        client.run(scans("c.nii.gz"), model=MODEL)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    keys = reporter.keys()
    for key in ("run.preparing", "run.queued", "run.scan_failed", "wait.line",
                "download.saved"):
        assert key in keys
    assert found.scans and all(isinstance(e.text, str) for e in reporter.events)


def test_the_library_source_never_prints_or_reads_a_terminal():
    """Code, not docstring examples: calls of print/input/getpass and any use
    of sys.stdout, sys.stderr or sys.stdin, outside the command line and the
    reporter."""
    import ast
    allowed = {"cli.py", "report.py", "__main__.py"}
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # input and getpass: only a reporter asks a person.
            if path.name != "report.py":
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "input":
                    found.append((path.name, node.lineno, "input"))
                if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                        a.name == "getpass" for a in node.names) or (
                        isinstance(node, ast.ImportFrom) and node.module == "getpass"):
                    found.append((path.name, node.lineno, "getpass"))
            if path.name in allowed:
                continue
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "print":
                found.append((path.name, node.lineno, "print"))
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "sys" and node.attr in ("stdout", "stderr", "stdin"):
                found.append((path.name, node.lineno, node.attr))
    assert found == []


def test_every_command_calls_public_documented_functions():
    import argparse
    commands = next(a for a in cli.build_parser()._actions
                    if isinstance(a, argparse._SubParsersAction)).choices
    assert set(cli.COMMAND_FUNCTIONS) == set(commands)
    for command, names in cli.COMMAND_FUNCTIONS.items():
        for name in names:
            if name == "nalvera.mcp":
                import nalvera.mcp.__main__ as start
                assert start.main.__doc__
                continue
            assert name in nalvera.__all__, (command, name)
            assert inspect.getdoc(getattr(nalvera, name)), (command, name)


def test_a_quiet_client_shows_nothing(make_client, server, scans, capsys):
    client = make_client(quiet=True, output=None)
    client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    assert capsys.readouterr().out == ""


# ── operations return data ────────────────────────────────────────────────────
def test_estimate_run_returns_data_without_uploading(make_client, server, scans):
    server.free_tasks_left = 1
    server.add_lab(12, "Imaging Lab", 50.0)
    found = make_client(quiet=True).estimate_run([scans("a.nii.gz"), scans("b.nii.gz")],
                                                 model=MODEL)
    assert isinstance(found, nalvera.RunEstimate)
    assert (found.model, found.task, len(found.scans)) == (MODEL, TASK, 2)
    assert found.credits == 2.5 and found.free_count == 1 and found.paid_credits == 1.25
    assert {b["budget"] for b in found.budgets} == {"personal", "lab:12"}
    assert found.suggested == "lab:12"
    assert server.calls("POST", "/api/v1/uploads") == [] and server.tasks == {}


def test_a_failed_scan_carries_its_code_and_message(client, server, scans):
    result = client.run([scans("a.nii.gz"), scans("bad.nii.gz", b"NOTASCAN")], model=MODEL,
                        budget="personal")
    (failure,) = result.failed
    assert failure.code == "not_a_scan" and failure.message and failure.original_name


def test_the_estimate_command(env, server, stub_anon, no_sleep):
    from test_cli import run_cli
    scan = env / "a.nii.gz"
    scan.write_bytes(scan_bytes(b"a"))
    status, out, _err = run_cli("estimate", str(scan), "--model", MODEL)
    assert status == 0
    assert messages.text("estimate.scan", name="a.nii.gz", credits="1.25",
                         shape="4 x 4 x 4") in out
    # One scan: no total line that repeats its own.
    assert messages.text("estimate.total", count=1, credits="1.25") not in out
    assert server.calls("POST", "/api/v1/uploads") == []


# ── long work in the background ───────────────────────────────────────────────
def _until(predicate, seconds=20.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_run_in_the_background_can_be_polled_and_collected(make_client, server, scans):
    reporter = nalvera.ListReporter()
    client = make_client(reporter=reporter, output=None)
    background = client.run_in_background([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL,
                                          budget="personal")
    assert _until(background.done)
    result = background.result(timeout=1)
    assert isinstance(result, nalvera.RunResult) and len(result) == 2
    assert "run.queued" in reporter.keys() and background.error is None


def test_a_background_run_never_asks(make_client, server, scans):
    server.add_lab(12, "Imaging Lab", 100.0)
    client = make_client(interactive=True, input=_NoInput(), output=io.StringIO())
    background = client.run_in_background(scans("a.nii.gz"), model=MODEL)
    assert _until(background.done)
    with pytest.raises(BudgetChoiceRequired):
        background.result()


def test_a_background_run_can_be_cancelled(make_client, server, scans):
    server.patch_delay = 0.05                                  # each part takes real time
    client = make_client(quiet=True, output=None)
    background = client.run_in_background([scans(f"s{i}.nii.gz", scan_bytes(b"x", 9000))
                                           for i in range(6)], model=MODEL, budget="personal")
    assert _until(lambda: server.calls("PATCH"))
    background.cancel()
    assert _until(background.done)
    with pytest.raises(RunInterrupted) as caught:
        background.result()
    assert len(caught.value.tasks) < 6
    with pytest.raises(TypeError):
        client.run_in_background(scans("x.nii.gz"), model=MODEL, unknown_option=1)


# ── allowed folders ───────────────────────────────────────────────────────────
def test_paths_outside_the_allowed_folders_are_refused_before_reading(make_client, server,
                                                                      scans, env):
    inside = env / "input"
    client = make_client(allowed_folders=[inside])
    allowed = scans("a.nii.gz")                               # written under input/
    assert client.run(allowed, model=MODEL, budget="personal")
    outside = env / "private" / "b.nii.gz"
    outside.parent.mkdir()
    outside.write_bytes(scan_bytes(b"b"))
    before = len(stub_anonymizer.prepared)
    with pytest.raises(ConfigError) as caught:
        client.run(outside, model=MODEL, budget="personal")
    assert caught.value.code == "path_not_allowed"
    assert len(stub_anonymizer.prepared) == before
    with pytest.raises(ConfigError):
        client.download(next(iter(server.tasks)), out=env / "elsewhere")


@pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="symlinks")
def test_a_link_inside_an_allowed_folder_that_points_out_is_refused(make_client, server, env):
    import os
    secret = env / "private.nii.gz"
    secret.write_bytes(scan_bytes(b"s"))
    inside = env / "input"
    inside.mkdir()
    os.symlink(secret, inside / "innocent.nii.gz")
    with pytest.raises(ConfigError) as caught:
        make_client(allowed_folders=[inside]).run(inside / "innocent.nii.gz", model=MODEL,
                                                  budget="personal")
    assert caught.value.code == "path_not_allowed"


# ── the settings list ─────────────────────────────────────────────────────────
def test_settings_are_importable_data_with_a_meaning_and_a_default():
    names = [s.name for s in nalvera.SETTINGS]
    for name in ("NALVERA_EMAIL", "NALVERA_API_KEY", "NALVERA_BUDGET", "NALVERA_ENCRYPT",
                 "NALVERA_MAPPING", "NALVERA_BASE_URL", "NALVERA_CONFIG_DIR",
                 "NALVERA_MAX_CREDITS"):
        assert name in names
    for setting in nalvera.SETTINGS:
        assert messages.has(f"setting.{setting.name}")
        assert messages.has(f"setting.{setting.name}.default")
    with pytest.raises(KeyError):
        settings.get("NALVERA_NOT_A_SETTING")


_READ = re.compile(r"""(?:environ(?:\.get)?|getenv|\benv\.get|settings\.get|\bget)\s*[\[(]\s*"""
                   r"""["']([A-Z][A-Z0-9_]+)["']""")


def test_every_setting_the_code_reads_is_listed():
    names = {s.name for s in nalvera.SETTINGS}
    found = set()
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        found |= set(_READ.findall(source))
        found |= set(re.findall(r"\bNALVERA_[A-Z0-9_]+", source))
    assert found, "the scan found nothing"
    assert sorted(found - names) == []


def test_help_shows_every_setting_and_every_exit_status(capsys):
    assert cli.main(["--help"]) == 0
    shown = " ".join(capsys.readouterr().out.split())          # lines are wrapped
    for setting in nalvera.SETTINGS:
        assert setting.name in shown
        assert " ".join(setting.meaning.split()) in shown
    for status, key in cli.EXIT_STATUSES.items():
        line = messages.text("help.exit_line", status=status, meaning=messages.text(key))
        assert " ".join(line.split()) in shown
    assert set(cli.EXIT_STATUSES) == {0, 1, 2, 3, 4, 5, 130}


# ── docstrings of the public surface ──────────────────────────────────────────
def test_every_public_method_says_what_its_arguments_do():
    for cls in (nalvera.Client, nalvera.BackgroundRun):
        for name, member in inspect.getmembers(cls, inspect.isfunction):
            if name.startswith("_"):
                continue
            doc = inspect.getdoc(member) or ""
            assert doc, f"{cls.__name__}.{name} has no docstring"
            for parameter in list(inspect.signature(member).parameters)[1:]:
                if parameter in ("options", "exc"):
                    continue
                assert parameter in doc or parameter in (inspect.getdoc(cls) or ""), \
                    f"{cls.__name__}.{name}: {parameter} is not described"
