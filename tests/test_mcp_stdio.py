"""``nalvera mcp`` as a separate process, over its real standard input and
output: driven by the MCP SDK's stdio client (or by hand, line by line),
against the fake server, with the real anonymizer. Needs the ``mcp`` extra."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import examples_helpers as helpers
from test_mcp_support import SRC, dicom_ready, fast, sdk_ready, until

pytestmark = pytest.mark.skipif(not sdk_ready(), reason="needs the mcp extra")

MODEL = "totalsegmentator_ct"
# Copies the server's standard output, line by line, to a log file while
# passing it on, so the test sees every byte the client was sent.
TEE = r"""
import os, subprocess, sys, threading
log_path, command = sys.argv[1], sys.argv[2:]
child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def forward():
    while True:
        chunk = os.read(0, 65536)
        if not chunk:
            break
        child.stdin.write(chunk)
        child.stdin.flush()
    child.stdin.close()
threading.Thread(target=forward, daemon=True).start()
with open(log_path, "wb") as log:
    for line in iter(child.stdout.readline, b""):
        log.write(line)
        log.flush()
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
sys.exit(child.wait())
"""


class Process:
    """The working folder and environment of a server process."""

    def __init__(self, server, root: Path):
        fast(server)
        self.server = server
        self.root = root
        self.work = (root / "work")
        self.temp, self.home, self.config = root / "tmp", root / "home", root / "config"
        for folder in (self.work, self.temp, self.home):
            folder.mkdir(exist_ok=True)
        self.work = self.work.resolve()
        self.env = {
            "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1",
            "NALVERA_EMAIL": server.email, "NALVERA_API_KEY": server.key,
            "NALVERA_BASE_URL": server.url, "NALVERA_CONFIG_DIR": str(self.config),
            "TMPDIR": str(self.temp), "HOME": str(self.home),
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            "PATH": os.environ.get("PATH", ""),
        }
        self.stdout_log = root / "stdout.log"
        self.stderr_log = root / "stderr.log"

    def command(self):
        return [sys.executable, "-c", TEE, str(self.stdout_log), sys.executable, "-m",
                "nalvera", "mcp"]

    def sdk(self, scenario, mode: str = "auto"):
        """Run ``scenario(client)`` with the SDK's stdio client."""
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters, stdio_client
        command = self.command()
        params = StdioServerParameters(command=command[0], args=command[1:], env=self.env,
                                       cwd=str(self.work))

        async def main():
            with open(self.stderr_log, "a") as errlog:
                async with Client(stdio_client(params, errlog=errlog), mode=mode) as client:
                    return await scenario(client)
        return asyncio.run(main())

    def popen(self):
        return subprocess.Popen([sys.executable, "-m", "nalvera", "mcp"], cwd=str(self.work),
                                env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=open(self.stderr_log, "a"))

    def leftovers(self):
        return [p for p in os.listdir(self.temp) if p.startswith("nalvera-")]

    def run_file(self, run_id: str):
        return json.loads((self.config / "mcp-runs" / f"{run_id}.json").read_text())


@pytest.fixture
def proc(server, tmp_path):
    return Process(server, tmp_path)


async def _follow(client, run_id: str, seconds: float = 90.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        result = await client.call_tool("get_run", {"run_id": run_id})
        data = result.structured_content
        if data["state"] not in ("preparing", "waiting", "uploading"):
            return data
        await asyncio.sleep(0.1)
    raise AssertionError("the run did not end")


def _lines(path: Path):
    return [line for line in path.read_bytes().splitlines() if line.strip()]


def test_standard_output_carries_only_protocol_messages_during_a_whole_run(proc):
    helpers.write_nifti(proc.work / "study" / "ct_a.nii.gz", seed=1)
    helpers.write_nifti(proc.work / "study" / "ct_b.nii.gz", seed=2)
    paths = ["study"]
    if dicom_ready():                                  # dcm2niix's own output must stay off
        helpers.write_dicom(proc.work / "dicom_one", (3,), seed=20)
        paths.append("dicom_one")

    async def scenario(client):
        listed = await client.list_tools()
        started = await client.call_tool("run_scan", {"paths": paths, "model": MODEL,
                                                      "budget": "personal", "max_credits": 10})
        assert not started.is_error, started.content[0].text
        done = await _follow(client, started.structured_content["run_id"])
        task_id = done["scans"][0]["task_id"]
        for _ in range(20):
            waited = await client.call_tool("wait_for_task", {"task_id": task_id,
                                                              "timeout_seconds": 5})
            if waited.structured_content["task"]["state"] == "completed":
                break
        saved = await client.call_tool("download_result", {"task_id": task_id})
        assert not saved.is_error, saved.content[0].text
        return listed, done, saved

    listed, done, saved = proc.sdk(scenario)
    assert len(listed.tools) == 12
    assert done["state"] == "done" and {s["state"] for s in done["scans"]} == {"queued"}
    assert len(done["scans"]) == len(paths) + 1
    assert (saved.structured_content["path"], saved.structured_content["files"]) == \
        ("results", 1)
    lines = _lines(proc.stdout_log)
    assert len(lines) >= 4
    for line in lines:
        message = json.loads(line)                     # every line is a protocol message
        assert message.get("jsonrpc") == "2.0", line[:200]
    logged = proc.stderr_log.read_text(errors="replace")
    assert proc.server.key not in logged and proc.server.key.encode() not in \
        proc.stdout_log.read_bytes()
    assert not proc.leftovers()


def test_the_state_of_a_run_survives_a_restart_of_the_server(proc):
    helpers.write_nifti(proc.work / "ct_a.nii.gz", seed=1)

    async def first(client):
        started = await client.call_tool("run_scan", {"paths": ["ct_a.nii.gz"], "model": MODEL,
                                                      "budget": "personal", "max_credits": 5})
        return await _follow(client, started.structured_content["run_id"])

    async def second(client):
        return (await client.call_tool("get_run", {"run_id": before["run_id"]})
                ).structured_content

    before = proc.sdk(first)
    after = proc.sdk(second, mode="legacy")            # the handshake revisions, too
    assert after == before and after["state"] == "done"


def _send(process, message):
    process.stdin.write((json.dumps(message) + "\n").encode())
    process.stdin.flush()


def _answer(process, request_id, seconds=60.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        line = process.stdout.readline()
        if not line:
            break
        message = json.loads(line)
        if message.get("id") == request_id:
            return message
    raise AssertionError("no answer")


def _start_slow_run(proc):
    proc.server.patch_delay = 0.3
    for number in range(3):
        helpers.write_nifti(proc.work / f"ct_{number}.nii.gz", seed=number + 1,
                            shape=(40, 40, 30))
    process = proc.popen()
    _send(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                               "clientInfo": {"name": "test", "version": "1"}}})
    _answer(process, 1)
    _send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    _send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "run_scan", "arguments": {"paths": [f"ct_{n}.nii.gz" for n in range(3)],
                                          "model": MODEL, "budget": "personal",
                                          "max_credits": 10}}})
    answer = _answer(process, 2)
    run_id = answer["result"]["structuredContent"]["run_id"]
    until(lambda: proc.server.calls("PATCH"), 30)
    return process, run_id


def test_closing_standard_input_stops_the_server_within_seconds(proc):
    process, run_id = _start_slow_run(proc)
    started = time.monotonic()
    process.stdin.close()
    assert process.wait(timeout=15) == 0
    assert time.monotonic() - started < 8
    assert not proc.leftovers()
    assert proc.run_file(run_id)["state"] == "interrupted"


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="POSIX signals")
def test_sigterm_stops_uploads_and_removes_temporary_files(proc):
    process, run_id = _start_slow_run(proc)
    started = time.monotonic()
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=15)
    assert time.monotonic() - started < 8
    assert not proc.leftovers()
    assert proc.run_file(run_id)["state"] == "interrupted"
    queued = sum(1 for r in proc.server.tasks.values() if r["source"] == "api")
    assert queued < 3
