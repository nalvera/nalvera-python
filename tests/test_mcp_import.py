"""``nalvera mcp`` where it cannot run (no ``mcp`` extra): one line on
standard error and status 2. Importing ``nalvera`` never imports the MCP
SDK. Runs without the extra."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from nalvera import cli
from nalvera.mcp import unavailable_reason

SRC = str(Path(__file__).resolve().parent.parent / "src")


def _python(code: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": SRC, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True,
                          timeout=120, env=env, stdin=subprocess.DEVNULL)


def test_importing_the_package_and_the_server_logic_loads_no_sdk():
    done = _python("import sys, nalvera, nalvera.cli, nalvera.mcp, nalvera.mcp.tools, "
                   "nalvera.mcp.runs; nalvera.Client; "
                   "loaded = [m for m in ('mcp', 'mcp_types', 'anyio', 'jsonschema') "
                   "if m in sys.modules]; print(loaded)")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_without_the_extra_the_command_says_so_in_one_line_and_exits_2():
    # sys.modules["mcp"] = None makes the SDK impossible to import here.
    done = _python("import sys; sys.modules['mcp'] = None; from nalvera.cli import main; "
                   "raise SystemExit(main(['mcp']))")
    assert done.returncode == cli.EXIT_USAGE
    assert done.stdout == ""
    lines = [line for line in done.stderr.splitlines() if line.strip()]
    assert len(lines) == 1


def test_the_command_has_help(capsys):
    assert cli.main(["mcp", "--help"]) == 0
    assert capsys.readouterr().out.startswith("usage: nalvera mcp")


def test_no_refusal_by_interpreter_version_is_left():
    import nalvera.mcp as mcp_pkg
    from nalvera import messages
    assert not hasattr(mcp_pkg, "MIN_PYTHON")
    assert not messages.has("error.mcp_python")


def test_the_reason_names_what_is_missing(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "mcp" else real(name, *a))
    assert unavailable_reason() == "sdk"
