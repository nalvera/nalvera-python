"""run_scan in the background: its state survives a restart of the server
(the run file and the mapping file), and shutdown stops uploads cleanly and
removes temporary files. Calls the tools directly (no MCP SDK needed)."""
from __future__ import annotations

import json
import os
import time

import pytest

from test_mcp_support import make_tools, nifti, prepare_env, task_count, until

MODEL = "totalsegmentator_ct"


@pytest.fixture
def mcp_env(server, tmp_path, monkeypatch):
    return prepare_env(server, tmp_path, monkeypatch)


def _ok(tools, name, arguments):
    failed, data = tools.call(name, arguments)
    assert not failed, data
    return data


def _finished(tools, run_id, seconds=60.0):
    def done():
        data = _ok(tools, "get_run", {"run_id": run_id})
        return data if data["state"] not in ("preparing", "waiting", "uploading") else None
    return until(done, seconds, step=0.1)


def _run_files(env):
    folder = env.config / "mcp-runs"
    return sorted(folder.glob("run-*.json")) if folder.exists() else []


def test_a_finished_run_is_found_after_a_restart(mcp_env):
    nifti(mcp_env, "ct_a.nii.gz", seed=1)
    first = make_tools(mcp_env)
    run_id = _ok(first, "run_scan", {"paths": ["ct_a.nii.gz"], "model": MODEL,
                                     "budget": "personal", "max_credits": 5})["run_id"]
    before = _finished(first, run_id)
    first.close(1.0)
    second = make_tools(mcp_env)
    try:
        after = _ok(second, "get_run", {"run_id": run_id})
        assert after == before and after["state"] == "done"
    finally:
        second.close(1.0)
    [path] = _run_files(mcp_env)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    record = path.read_text(encoding="utf-8")
    assert mcp_env.server.key not in record and str(mcp_env.work) not in record


def test_a_run_stopped_by_a_restart_shows_what_was_queued(mcp_env):
    """The server stopped between two scans: the mapping file names the task
    of the scan that was queued; the other scan was not queued."""
    server = mcp_env.server
    nifti(mcp_env, "ct_a.nii.gz", seed=1)
    nifti(mcp_env, "ct_b.nii.gz", seed=2)
    first = make_tools(mcp_env)
    run_id = _ok(first, "run_scan", {"paths": ["ct_a.nii.gz", "ct_b.nii.gz"], "model": MODEL,
                                     "budget": "personal", "max_credits": 5})["run_id"]
    _finished(first, run_id)
    first.close(1.0)
    # Turn the files into what a server killed between the two scans leaves:
    # the run file says "uploading" (written by another process), the first
    # scan's task is in the mapping only, the second scan never reached the
    # platform.
    [path] = _run_files(mcp_env)
    record = json.loads(path.read_text(encoding="utf-8"))
    queued, lost = (dict(scan) for scan in record["scans"])
    record.update(state="uploading", boot="another-process")
    record["scans"][0].update(state="uploading", task_id=None, percent_uploaded=100)
    record["scans"][1].update(state="ready", task_id=None)
    path.write_text(json.dumps(record), encoding="utf-8")
    [rows_path] = list((mcp_env.config / "records").glob("*.csv"))     # the record
    assert not list(mcp_env.work.glob("*.csv"))                        # not in the project
    rows = rows_path.read_text(encoding="utf-8-sig").splitlines()
    rows_path.write_text("\n".join(r for r in rows if lost["code"] not in r) + "\n",
                         encoding="utf-8")
    with server.lock:
        server.tasks.pop(lost["task_id"])
        server.order.remove(lost["task_id"])
        server.personal_credits += 1.25
    second = make_tools(mcp_env)
    try:
        data = _ok(second, "get_run", {"run_id": run_id})
        assert data["state"] == "interrupted"
        assert data["scans"][0]["state"] == "queued"
        assert data["scans"][0]["task_id"] == queued["task_id"]
        assert data["scans"][1]["state"] == "interrupted"
        # Running the same scans again queues the lost one and charges once.
        again = _finished(second, _ok(second, "run_scan", {
            "paths": ["ct_a.nii.gz", "ct_b.nii.gz"], "model": MODEL, "budget": "personal",
            "max_credits": 5})["run_id"])
        assert sorted(s["state"] for s in again["scans"]) == ["queued", "skipped"]
        assert task_count(server) == 2 and server.personal_credits == 97.5
    finally:
        second.close(1.0)


def test_shutdown_stops_uploads_and_removes_temporary_files(mcp_env):
    server = mcp_env.server
    server.patch_delay = 0.3
    for number in range(3):
        nifti(mcp_env, f"ct_{number}.nii.gz", seed=number + 1, shape=(40, 40, 30))
    tools = make_tools(mcp_env)
    run_id = _ok(tools, "run_scan", {"paths": [f"ct_{n}.nii.gz" for n in range(3)],
                                     "model": MODEL, "budget": "personal",
                                     "max_credits": 10})["run_id"]
    until(lambda: server.calls("PATCH"), 20)
    started = time.monotonic()
    alive = tools.close(4.0)
    assert time.monotonic() - started < 5 and alive == []
    data = json.loads(_run_files(mcp_env)[0].read_text(encoding="utf-8"))
    assert data["run_id"] == run_id and data["state"] == "interrupted"
    assert task_count(server) < 3
    assert not [p for p in os.listdir(mcp_env.temp) if p.startswith("nalvera-run-")]
    # The scans that were not queued run on a later call, each charged once.
    later = make_tools(mcp_env)
    try:
        server.patch_delay = 0.0
        again = _finished(later, _ok(later, "run_scan", {
            "paths": [f"ct_{n}.nii.gz" for n in range(3)], "model": MODEL,
            "budget": "personal", "max_credits": 10})["run_id"])
        assert again["state"] == "done" and task_count(server) == 3
        assert server.personal_credits == 100.0 - 3 * 1.25
    finally:
        later.close(1.0)


def test_a_run_that_fails_names_no_local_folder(mcp_env):
    nifti(mcp_env, "study/ct.nii.gz", seed=1)
    (mcp_env.work / "study" / "IM0001").write_bytes(b"\0" * 128 + b"DICM" + b"\0" * 64)
    tools = make_tools(mcp_env)
    try:
        data = _finished(tools, _ok(tools, "run_scan", {
            "paths": ["study"], "model": MODEL, "budget": "personal",
            "max_credits": 5})["run_id"])
        assert data["state"] == "failed" and data["error"]["code"] == "local_error"
        assert "study" in data["error"]["detail"]
        assert str(mcp_env.work) not in json.dumps(data)
        assert mcp_env.server.calls("POST", "/api/v1/uploads") == []
    finally:
        tools.close(1.0)


def test_a_run_of_a_server_still_running_elsewhere_is_shown_as_it_is(mcp_env):
    import subprocess
    import sys
    folder = mcp_env.config / "mcp-runs"
    folder.mkdir(parents=True)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        record = {"run_id": "run-00000000abcd", "state": "uploading", "boot": "elsewhere",
                  "pid": other.pid, "reason": "in_progress", "hint": None, "error": None,
                  "estimate": None, "series_to_confirm": None, "charged_credits": 0.0,
                  "scans": [{"name": "ct.nii.gz", "code": "scan-AAAAAAAA", "series": None,
                             "state": "uploading", "percent_uploaded": 50, "task_id": None,
                             "reason": None, "detail": None, "charged": None}]}
        path = folder / "run-00000000abcd.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        tools = make_tools(mcp_env)
        try:
            data = _ok(tools, "get_run", {"run_id": "run-00000000abcd"})
            assert data["state"] == "uploading" and data["scans"][0]["percent_uploaded"] == 50
            other.kill()
            other.wait(10)
            data = _ok(tools, "get_run", {"run_id": "run-00000000abcd"})
            assert data["state"] == "interrupted"
            assert json.loads(path.read_text())["state"] == "uploading"    # not its file
        finally:
            tools.close(1.0)
    finally:
        other.kill()


def test_old_run_files_are_removed_when_the_server_starts(mcp_env):
    folder = mcp_env.config / "mcp-runs"
    folder.mkdir(parents=True)
    old = folder / "run-0123456789ab.json"
    old.write_text("{}", encoding="utf-8")
    past = time.time() - 40 * 24 * 3600
    os.utime(old, (past, past))
    make_tools(mcp_env).close(1.0)
    assert not old.exists()
