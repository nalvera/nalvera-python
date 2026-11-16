"""Importing the package and asking for help make no network request."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

GUARD = r"""
import socket, sys
attempts = []
def refuse(*args, **kwargs):
    attempts.append(args)
    raise OSError("network use during a no-network test")
socket.socket.connect = refuse
socket.socket.connect_ex = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse

import nalvera
from nalvera import cli
codes = []
for argv in [["--help"], ["--version"]] + [[c, "--help"] for c in (
        "login", "logout", "whoami", "limits", "models", "run", "generate", "status", "wait",
        "download", "cancel", "scans")]:
    try:
        codes.append(cli.main(argv))
    except SystemExit as stop:
        codes.append(stop.code)
assert all(c == 0 for c in codes), codes
assert "nalvera.anonymize" not in sys.modules
print("ATTEMPTS", len(attempts))
"""


def test_import_and_help_make_no_request(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NALVERA_")}
    env["PYTHONPATH"] = str(SRC)
    env["NALVERA_CONFIG_DIR"] = str(tmp_path)
    result = subprocess.run([sys.executable, "-c", GUARD], capture_output=True, text=True,
                            env=env, cwd=tmp_path, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "ATTEMPTS 0" in result.stdout


def test_logout_and_scans_make_no_request(tmp_path):
    script = GUARD.split("import nalvera")[0] + (
        "from nalvera import cli\n"
        "assert cli.main(['logout']) == 0\n"
        "assert cli.main(['scans']) == 0\n"
        "print('ATTEMPTS', len(attempts))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("NALVERA_")}
    env.update(PYTHONPATH=str(SRC), NALVERA_CONFIG_DIR=str(tmp_path),
               NALVERA_MAPPING=str(tmp_path / "none.csv"))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            env=env, cwd=tmp_path, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "ATTEMPTS 0" in result.stdout
