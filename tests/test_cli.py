"""The nalvera command: help, login, each command, exit statuses, no key in output."""
from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import scan_bytes
from fake_server import make_key
from nalvera import cli, messages

pytestmark = pytest.mark.usefixtures("cli_dialect")

COMMANDS = ["login", "logout", "whoami", "limits", "models", "anonymize", "estimate", "run",
            "generate", "status", "wait", "download", "cancel", "scans"]


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    return slept


def run_cli(*argv, stdin="", secret=None):
    out, err = io.StringIO(), io.StringIO()
    status = cli.main(list(argv), stdin=io.StringIO(stdin), stdout=out, stderr=err,
                      secret_input=(lambda prompt: secret) if secret is not None else None)
    return status, out.getvalue(), err.getvalue()


# ── help ──────────────────────────────────────────────────────────────────────
def test_help_lists_every_command(capsys):
    assert cli.main(["--help"]) == 0
    shown = capsys.readouterr().out
    for name in COMMANDS:
        assert name in shown


@pytest.mark.parametrize("name", COMMANDS)
def test_each_command_has_help(capsys, name):
    assert cli.main([name, "--help"]) == 0
    shown = capsys.readouterr().out
    assert shown.startswith("usage: nalvera " + name)
    assert messages.text(f"help.{name}") in " ".join(shown.split())


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip().startswith("nalvera ")


def test_no_command_prints_help_and_a_usage_status():
    status, out, _err = run_cli()
    assert status == cli.EXIT_USAGE and "usage" in out


def test_a_usage_error_is_status_2(capsys):
    assert cli.main(["run"]) == 2


def test_python_dash_m_works(env):
    src = str(Path(__file__).resolve().parent.parent / "src")
    environment = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([sys.executable, "-m", "nalvera", "--help"], capture_output=True,
                            text=True, timeout=60, env=environment)
    assert result.returncode == 0 and "login" in result.stdout


# ── without credentials ───────────────────────────────────────────────────────
@pytest.mark.parametrize("argv", [["whoami"], ["limits"], ["models"], ["status"], ["wait"],
                                  ["download", "abc"], ["cancel", "abc"],
                                  ["run", "x.nii.gz", "--model", "m"],
                                  ["generate", "--model", "m", "--task", "t"]])
def test_without_credentials_every_command_prints_the_steps(env, monkeypatch, argv):
    monkeypatch.delenv("NALVERA_API_KEY")
    monkeypatch.delenv("NALVERA_EMAIL")
    status, _out, err = run_cli(*argv)
    assert status == cli.EXIT_AUTH
    assert messages.text("error.no_credentials") in err


# ── login / logout ────────────────────────────────────────────────────────────
def test_login_checks_the_key_then_stores_it_privately(env, server, monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    monkeypatch.delenv("NALVERA_EMAIL")
    status, out, err = run_cli("login", "--email", server.email, secret=server.key)
    assert status == 0, err
    assert server.calls("GET", "/api/v1/me")
    stored = env / "config" / "credentials.json"
    assert stored.exists()
    if os.name == "posix":
        assert stat.S_IMODE(stored.stat().st_mode) == 0o600
    assert server.key not in out and server.key not in err
    status, out, _err = run_cli("whoami")
    assert status == 0 and server.key[:12] in out


def test_login_asks_for_the_email_when_not_given(env, server, monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    monkeypatch.delenv("NALVERA_EMAIL")
    status, _out, err = run_cli("login", stdin=server.email + "\n", secret=server.key)
    assert status == 0, err


def test_a_typo_in_the_key_is_caught_before_any_request(env, server, monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    typo = server.key[:-1] + ("A" if server.key[-1] != "A" else "B")
    status, out, err = run_cli("login", "--email", server.email, secret=typo)
    assert status == cli.EXIT_AUTH and server.requests == []
    assert typo not in out + err
    assert not (env / "config" / "credentials.json").exists()


def test_a_key_the_server_refuses_is_not_stored(env, server, monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    other = make_key()
    status, out, err = run_cli("login", "--email", server.email, secret=other)
    assert status == cli.EXIT_AUTH and other not in out + err
    assert messages.text("error.invalid_credentials") in err
    assert not (env / "config" / "credentials.json").exists()


def test_logout_removes_the_stored_key(env, server, monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    run_cli("login", "--email", server.email, secret=server.key)
    status, out, _err = run_cli("logout")
    assert status == 0 and out.strip() == messages.text("logout.done")
    assert not (env / "config" / "credentials.json").exists()
    assert run_cli("logout")[1].strip() == messages.text("logout.nothing")


# ── account commands ──────────────────────────────────────────────────────────
def test_whoami_shows_account_key_scopes_free_tasks_and_budgets(env, server):
    server.free_tasks_left = 2
    server.add_lab(12, "Imaging Lab", 100.0)
    status, out, _err = run_cli("whoami")
    assert status == 0
    for expected in (server.email.lower(), server.key[:12], "2027-11-15", "Imaging Lab",
                     "lab:12", messages.text("scope.tasks:write")):
        assert expected in out
    assert server.key not in out


def test_limits_shows_the_numbers(env, server):
    status, out, _err = run_cli("limits")
    assert status == 0 and "120" in out and "1024" in out


def test_models_lists_models_and_tasks(env, server):
    status, out, _err = run_cli("models")
    assert status == 0
    assert "totalsegmentator_ct" in out and "--task total" in out and "nv_generate_ctmr" in out


def test_models_for_a_generation_shows_its_settings(env, server):
    status, out, _err = run_cli("models", "--model", "nv_generate_ctmr")
    assert status == 0
    for key in ("xy_sizes", "spacing_mm", "modes", "contrasts"):
        assert key in out
    assert run_cli("models", "--model", "nope")[0] == cli.EXIT_USAGE


# ── run, generate, follow, download ───────────────────────────────────────────
def test_run_waits_and_saves_under_the_original_name(env, server, stub_anon, no_sleep):
    scan = env / "Doe_John.nii.gz"
    scan.write_bytes(scan_bytes(b"x"))
    status, out, err = run_cli("run", str(scan), "--model", "totalsegmentator_ct",
                               "--budget", "personal", "--out", str(env / "res"))
    assert status == 0, err
    assert (env / "res" / "Doe_John__totalsegmentator_ct__total.nii.gz").exists()
    assert server.key not in out + err


def test_run_with_a_failed_scan_ends_with_status_1(env, server, stub_anon, no_sleep):
    good, bad = env / "a.nii.gz", env / "b.nii.gz"
    good.write_bytes(scan_bytes(b"a"))
    bad.write_bytes(b"NOTASCAN")
    status, _out, _err = run_cli("run", str(good), str(bad), "--model", "totalsegmentator_ct",
                                 "--budget", "personal", "--no-wait")
    assert status == cli.EXIT_FAILED


def test_run_no_wait_prints_how_to_follow(env, server, stub_anon, no_sleep):
    scan = env / "a.nii.gz"
    scan.write_bytes(scan_bytes(b"a"))
    status, out, _err = run_cli("run", str(scan), "--model", "totalsegmentator_ct",
                                "--budget", "personal", "--no-wait")
    task_id = next(iter(server.tasks))
    assert status == 0 and messages.text("run.no_wait_hint", ids=task_id) in out


def test_run_with_a_lab_and_no_terminal_needs_a_budget(env, server, stub_anon, no_sleep):
    server.add_lab(12, "Imaging Lab", 100.0)
    scan = env / "a.nii.gz"
    scan.write_bytes(scan_bytes(b"a"))
    status, _out, err = run_cli("run", str(scan), "--model", "totalsegmentator_ct")
    assert status == cli.EXIT_CREDITS and "lab:12" in err and "personal" in err
    assert server.calls("POST", "/api/v1/uploads") == []


def test_no_credits_is_status_4(env, server, stub_anon, no_sleep):
    server.personal_credits = 1.3
    files = []
    for name in ("a", "b"):
        path = env / f"{name}.nii.gz"
        path.write_bytes(scan_bytes(name.encode()))
        files.append(str(path))
    status, out, err = run_cli("run", files[0], "--model", "totalsegmentator_ct", "--budget",
                               "personal", "--no-wait")
    assert status == 0
    status, out, err = run_cli("run", files[1], "--model", "totalsegmentator_ct", "--budget",
                               "personal", "--no-wait")
    assert status == cli.EXIT_CREDITS
    assert messages.text("error.insufficient_credits", budget="personal") in err


def test_generate_with_params(env, server, no_sleep):
    status, out, err = run_cli("generate", "--model", "nv_generate_ctmr", "--task", "rflow-ct",
                               "--param", "mode=image-only", "--param", "output_size=[256,256,128]",
                               "--param", "spacing=[1.5,1.5,2.0]", "--param", "seed=7",
                               "--budget", "personal", "--out", str(env / "gen"))
    assert status == 0, err
    task = next(iter(server.tasks.values()))
    assert task["body"]["params"] == {"mode": "image-only", "output_size": [256, 256, 128],
                                      "spacing": [1.5, 1.5, 2.0], "seed": 7}
    assert len(list((env / "gen").iterdir())) == 1


def test_a_bad_param_is_a_usage_error(env, server):
    status, _out, _err = run_cli("generate", "--model", "nv_generate_ctmr", "--task",
                                 "rflow-ct", "--param", "seed")
    assert status == cli.EXIT_USAGE and server.calls("POST", "/api/v1/tasks") == []


def test_status_wait_download_cancel(env, server, no_sleep):
    done = server.add_task("completed")
    waiting = server.add_task("queued")
    status, out, _err = run_cli("status")
    assert status == 0 and "website-scan.nii.gz" in out
    status, out, _err = run_cli("status", done)
    assert status == 0
    status, _out, _err = run_cli("cancel", waiting)
    assert status == 0 and server._state(server.tasks[waiting]) == "cancelled"
    status, _out, err = run_cli("cancel", done)
    assert status == cli.EXIT_FAILED and messages.text("error.not_cancellable") in err
    status, _out, _err = run_cli("wait", done)
    assert status == 0
    status, _out, _err = run_cli("download", done, "--out", str(env / "dl"))
    assert status == 0 and len(list((env / "dl").iterdir())) == 1
    status, _out, _err = run_cli("download", waiting, "--out", str(env / "dl"))
    assert status == cli.EXIT_FAILED


def test_status_without_active_tasks(env, server):
    status, out, _err = run_cli("status")
    assert status == 0 and out.strip() == messages.text("status.none")


def test_scans_lists_the_local_mapping(env, server, stub_anon, no_sleep):
    assert run_cli("scans")[1].strip() == messages.text(
        "scans.none", path=os.environ["NALVERA_MAPPING"])
    scan = env / "Doe_John.nii.gz"
    scan.write_bytes(scan_bytes(b"x"))
    run_cli("run", str(scan), "--model", "totalsegmentator_ct", "--budget", "personal",
            "--out", str(env / "res"))
    status, out, _err = run_cli("scans")
    task_id = next(iter(server.tasks))
    assert status == 0 and "Doe_John.nii.gz" in out and task_id in out
    assert "Doe_John__totalsegmentator_ct__total.nii.gz" in out
    assert server.requests and all(r["path"] != "/scans" for r in server.requests)


def test_ctrl_c_is_status_130(env, server, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_whoami", interrupted)
    status, _out, err = run_cli("whoami")
    assert status == cli.EXIT_INTERRUPTED
    assert messages.text("cli.interrupted") in err


def test_the_key_never_appears_in_any_output(env, server, stub_anon, no_sleep):
    scan = env / "a.nii.gz"
    scan.write_bytes(scan_bytes(b"a"))
    runs = [("whoami",), ("limits",), ("models",), ("status",),
            ("run", str(scan), "--model", "totalsegmentator_ct", "--budget", "personal",
             "--out", str(env / "r")),
            ("download", "ffffffffffff"), ("cancel", "ffffffffffff")]
    for argv in runs:
        _status, out, err = run_cli(*argv)
        assert server.key not in out + err and server.key[12:] not in out + err, argv
