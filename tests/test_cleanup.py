"""Temporary files the package leaves nowhere: stale folders of a killed
process are removed at the start of a run, and the local MCP server's
shutdown removes the partly written file of a download and the folder of
an estimate it cut off. The MCP tools are called directly (no SDK needed)."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

import nalvera.anonymize as anonymize
from nalvera import client as client_module
from test_mcp_support import make_tools, nifti, prepare_env, until

MODEL = "totalsegmentator_ct"


@pytest.fixture
def mcp_env(server, tmp_path, monkeypatch):
    return prepare_env(server, tmp_path, monkeypatch)


def _hidden_parts(folder: Path):
    return sorted(p.name for p in folder.rglob(".*.part")) if folder.exists() else []


# ── stale folders of a process that was killed ───────────────────────────────
def test_stale_inspect_folders_are_removed_at_the_start(client, server, scans):
    base = Path(tempfile.gettempdir())
    old, fresh = base / "nalvera-inspect-old", base / "nalvera-inspect-fresh"
    stray = base / "nalvera-inspect-file"
    for folder in (old, fresh):
        folder.mkdir()
        (folder / "volume.nii").write_bytes(b"x")
    stray.write_bytes(b"not a folder")
    two_days_ago = time.time() - 2 * 86400
    for path in (old, stray):
        os.utime(path, (two_days_ago, two_days_ago))
    client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert not old.exists() and fresh.exists() and stray.exists()


def test_the_clean_up_knows_the_anonymizers_inspect_prefix():
    assert anonymize.INSPECT_PREFIX in client_module._STALE_PREFIXES
    assert client_module.WORKDIR_PREFIX in client_module._STALE_PREFIXES


# ── the MCP server's shutdown ────────────────────────────────────────────────
class _SlowResult:
    """A result that sends its first bytes, then nothing until ``gate``."""

    def __init__(self, gate: threading.Event):
        self.gate = gate
        self.headers = {"Content-Disposition": 'attachment; filename="r.nii.gz"',
                        "Content-Type": "application/gzip"}

    def iter_content(self, chunk_size: int = 0):
        yield b"\x1f\x8b" + b"x" * 100
        self.gate.wait(10)
        yield b"y" * 100

    def close(self) -> None:
        pass


def test_shutdown_removes_the_part_file_of_a_download_still_streaming(mcp_env):
    server = mcp_env.server
    tools = make_tools(mcp_env, download_wait=0.05)
    task_id = server.add_task()
    gate = threading.Event()
    client = tools.client()
    real_request = client._transport.request

    def request(method, path, **kwargs):
        if path.endswith("/result"):
            return _SlowResult(gate)
        return real_request(method, path, **kwargs)

    client._transport.request = request
    failed, data = tools.call("download_result", {"task_id": task_id})
    assert failed and data["error"]["code"] == "download_in_progress"
    results = mcp_env.work / "results"
    until(lambda: _hidden_parts(results), 10)
    [download] = tools._downloads.values()
    tools.close(1.0)
    try:
        assert _hidden_parts(results) == []
    finally:
        gate.set()
        download.thread.join(10)
    assert not download.thread.is_alive()
    assert _hidden_parts(results) == [] and not any(results.iterdir())


def test_shutdown_removes_the_folder_of_an_estimate_cut_off(mcp_env, monkeypatch):
    scan = nifti(mcp_env, "scan.nii.gz")
    gate, inside = threading.Event(), threading.Event()

    def slow_inspect(path, *, convert=False, workdir=None):    # a long DICOM conversion
        private = tempfile.mkdtemp(prefix="nalvera-inspect-", dir=workdir)
        (Path(private) / "volume.nii").write_bytes(b"converted, still with DICOM details")
        try:
            inside.set()
            gate.wait(10)
            raise anonymize.AnonymizeError("conversion_failed")
        finally:
            shutil.rmtree(private, ignore_errors=True)

    monkeypatch.setattr(anonymize, "inspect", slow_inspect)
    tools = make_tools(mcp_env)
    worker = threading.Thread(target=tools.call,
                              args=("estimate_cost", {"model": MODEL, "path": str(scan)}))
    worker.start()
    try:
        assert inside.wait(10)
        tools.close(1.0)
        left = [p for p in os.listdir(mcp_env.temp) if p.startswith("nalvera-inspect-")]
        assert left == []
    finally:
        gate.set()
        worker.join(10)
