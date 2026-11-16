"""The local MCP server, tool by tool, through the MCP SDK's own client
(in process) against the fake server. Needs the ``mcp`` extra; skipped
without it."""
from __future__ import annotations

import json
import os
import threading
import time

import pytest

from test_mcp_support import (
    INJECTION,
    Session,
    make_tools,
    needs_dicom,
    nifti,
    prepare_env,
    run_to_end,
    sdk_ready,
    task_count,
)

pytestmark = pytest.mark.skipif(not sdk_ready(), reason="needs the mcp extra")


@pytest.fixture
def mcp_env(server, tmp_path, monkeypatch):
    return prepare_env(server, tmp_path, monkeypatch)


@pytest.fixture
def local_tools(mcp_env):
    tools = make_tools(mcp_env)
    yield tools
    tools.close(2.0)

MODEL, TASK = "totalsegmentator_ct", "total"
GEN = {"model": "nv_generate_ctmr", "task": "rflow-ct",
       "params": {"mode": "image-only", "output_size": [256, 256, 128],
                  "spacing": [1.5, 1.5, 2.0]}}


@pytest.fixture
def session(local_tools):
    return Session(local_tools)


def _posts(server, path="/api/v1/tasks"):
    return server.calls("POST", path)


# ── read-only tools ───────────────────────────────────────────────────────────
def test_the_tool_list_is_the_manifests_local_list(session):
    from nalvera.mcp.server import local_entries
    listed = session.list_tools().tools
    entries = local_entries(session.tools.manifest)
    assert [t.name for t in listed] == [e["name"] for e in entries]
    for tool, entry in zip(listed, entries):
        assert (tool.title, tool.description) == (entry["title"], entry["description"])
        assert tool.input_schema == entry["inputSchema"]
        assert tool.output_schema == entry["outputSchema"]
        hints = tool.annotations
        assert (hints.read_only_hint, hints.destructive_hint, hints.idempotent_hint,
                hints.open_world_hint) == tuple(entry["annotations"][k] for k in (
                    "readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"))
    assert "path" in next(t for t in listed if t.name == "estimate_cost").input_schema[
        "properties"]


def test_list_models_and_one_model(session):
    data = session.ok("list_models", {})
    assert {m["id"] for m in data["models"]} >= {MODEL, "nv_generate_ctmr"}
    only = session.ok("list_models", {"model": "nv_generate_ctmr"})
    assert [m["id"] for m in only["models"]] == ["nv_generate_ctmr"]


def test_get_budgets_names_each_lab_and_the_suggestion(session, mcp_env):
    mcp_env.server.add_lab(12, "Imaging Lab", 80.0)
    mcp_env.server.free_tasks_left = 2
    data = session.ok("get_budgets", {})
    assert data["free"] == {"period_active": False, "free_tasks_left": 2}
    assert [lab["id"] for lab in data["labs"]] == [12] and data["suggested"] == "lab:12"


def test_estimate_by_shape_by_params_and_by_path(session, mcp_env):
    server = mcp_env.server
    server.credits_by_shape[(8, 8, 6)] = 2.0
    server.personal_credits = 3.0
    assert session.ok("estimate_cost", {"model": MODEL, "shape": [100, 100, 50]})[
        "credits"] == 1.25
    assert session.ok("estimate_cost", {k: GEN[k] for k in ("model", "task", "params")})[
        "credits"] == 1.25
    one = nifti(mcp_env, "scans/ct_a.nii.gz", seed=1)
    nifti(mcp_env, "scans/ct_b.nii.gz", seed=2)
    assert session.ok("estimate_cost", {"model": MODEL, "path": str(one)})["credits"] == 2.0
    folder = session.ok("estimate_cost", {"model": MODEL, "path": "scans"})   # relative
    assert folder["credits"] == 4.0
    personal = next(b for b in folder["budgets"] if b["budget"] == "personal")
    assert personal["sufficient"] is False                 # 3 credits for a total of 4
    assert _posts(server, "/api/v1/uploads") == [] and server.tasks == {}
    assert session.err("estimate_cost", {"model": MODEL, "path": str(one),
                                         "shape": [8, 8, 6]})["code"] == "invalid_arguments"


# ── generate: the spend rules ─────────────────────────────────────────────────
def test_generate_needs_a_budget_a_cap_and_a_request_id(session, mcp_env):
    for missing in ("budget", "max_credits", "request_id"):
        arguments = {**GEN, "budget": "personal", "max_credits": 5, "request_id": "req-0001"}
        del arguments[missing]
        assert session.err("generate", arguments)["code"] == "invalid_arguments"
    assert _posts(mcp_env.server) == [] and mcp_env.server.calls("POST", "/estimate") == []


def test_generate_over_the_cap_charges_nothing_and_returns_the_estimate(session, mcp_env):
    server = mcp_env.server
    data = session.ok("generate", {**GEN, "budget": "personal", "max_credits": 1.0,
                                   "request_id": "req-over-cap"})
    assert data["queued"] is False and data["reason"] == "over_max_credits"
    assert data["hint"] == session.tools.manifest["reasons"]["over_max_credits"]
    assert data["estimate"]["credits"] == 1.25 and data["task"] is None
    assert server.tasks == {} and server.personal_credits == 100.0
    [sent] = _posts(server)                                  # sent as free: refused, no charge
    assert sent["headers"]["Idempotency-Key"] == "mcp-gen-req-over-cap"


def test_generate_over_the_cap_still_runs_when_a_free_task_pays(session, mcp_env):
    mcp_env.server.free_tasks_left = 1
    data = session.ok("generate", {**GEN, "budget": "personal", "max_credits": 0,
                                   "request_id": "req-free-0001"})
    assert data["queued"] is True and data["budget_sent"] == "free"
    assert data["charged"]["source"] == "free" and mcp_env.server.personal_credits == 100.0


def test_a_retried_generate_returns_the_first_task_and_charges_once(session, mcp_env):
    server = mcp_env.server
    arguments = {**GEN, "budget": "personal", "max_credits": 5, "request_id": "req-retry-01",
                 "label": "phantom 01"}
    first = session.ok("generate", arguments)
    again = session.ok("generate", arguments)
    assert first["queued"] and first["idempotent_replay"] is False
    assert again["task"]["id"] == first["task"]["id"] and again["idempotent_replay"] is True
    assert task_count(server) == 1 and server.personal_credits == 98.75
    assert {c["headers"]["Idempotency-Key"] for c in _posts(server)} == {"mcp-gen-req-retry-01"}


def test_generate_with_budget_free_and_none_left(session, mcp_env):
    error = session.err("generate", {**GEN, "budget": "free", "max_credits": 0,
                                     "request_id": "req-free-none"})
    assert error["code"] == "no_free_tasks" and mcp_env.server.tasks == {}


def test_a_read_only_key_cannot_generate_or_cancel(session, mcp_env):
    server = mcp_env.server
    task_id = server.add_task(state="processing")
    server.scopes = ["tasks:read"]
    error = session.err("generate", {**GEN, "budget": "personal", "max_credits": 5,
                                     "request_id": "req-readonly"})
    assert (error["code"], error["required_scope"]) == ("scope_missing", "tasks:write")
    error = session.err("cancel_task", {"task_id": task_id})
    assert (error["code"], error["required_scope"]) == ("scope_missing", "tasks:write")


# ── tasks ─────────────────────────────────────────────────────────────────────
def test_list_tasks_and_get_task_answer_from_a_cache_within_the_poll_floor(session, mcp_env):
    server = mcp_env.server
    ids = [server.add_task() for _ in range(3)]
    listed = session.ok("list_tasks", {})
    assert listed["total"] == 3 and [t["id"] for t in listed["tasks"]] == ids[::-1]
    assert "limit=20" in server.calls("GET", "/api/v1/tasks")[0]["query"]
    server.limits["min_poll_seconds"] = 30
    session.tools.client().me(refresh=True)
    session.ok("get_task", {"task_id": ids[0]})
    before = len(server.calls("GET", f"/api/v1/tasks/{ids[0]}"))
    assert session.ok("get_task", {"task_id": ids[0]})["id"] == ids[0]
    assert len(server.calls("GET", f"/api/v1/tasks/{ids[0]}")) == before
    assert session.err("get_task", {"task_id": "ffffffffffff"})["code"] == "not_found"
    assert session.err("get_task", {"task_id": "../me"})["code"] == "invalid_arguments"


def test_wait_for_task_returns_when_the_state_changes(session, mcp_env):
    server = mcp_env.server
    task_id = server.add_task(state="queued")
    server.tasks[task_id]["script"] = [("queued", 2), ("processing", None), ("completed", None)]
    data = session.ok("wait_for_task", {"task_id": task_id, "since_state": "queued",
                                        "timeout_seconds": 10})
    assert data["changed"] is True and data["task"]["state"] != "queued"
    done = session.ok("wait_for_task", {"task_id": task_id, "timeout_seconds": 10})
    assert done["task"]["state"] == "completed"
    started = time.monotonic()
    still = session.ok("wait_for_task", {"task_id": task_id, "since_state": "completed",
                                         "timeout_seconds": 1})
    assert still["changed"] is False and time.monotonic() - started < 5


def test_cancel_a_running_task_then_a_finished_one(session, mcp_env):
    server = mcp_env.server
    running = server.add_task(state="processing")
    data = session.ok("cancel_task", {"task_id": running})
    assert data["cancel_requested"] is True and data["state"] == "processing"
    finished = server.add_task()
    assert session.err("cancel_task", {"task_id": finished})["code"] == "not_cancellable"


def test_a_text_result_is_capped_and_marked_as_model_output(mcp_env):
    server = mcp_env.server
    tools = make_tools(mcp_env, text_cap=64)
    try:
        session = Session(tools)
        report = ("Findings: " + "no acute abnormality. " * 20).encode()
        server.result_bytes, server.result_filename = report, "../../report.txt"
        task_id = server.add_task(output_kind="text")
        data = session.ok("get_text_result", {"task_id": task_id})
        assert data["truncated"] is True and data["bytes"] == 64 and len(data["text"]) == 64
        assert data["source"] == "ai_model" and data["notice"] == tools.manifest["notes"][
            "ai_output"]
        assert data["filename"] == "report.txt"                # base name only
        volume = server.add_task()
        assert session.err("get_text_result", {"task_id": volume})["code"] == "not_text"
        waiting = server.add_task(state="queued")
        error = session.err("get_text_result", {"task_id": waiting})
        assert (error["code"], error["state"]) == ("not_ready", "queued")
    finally:
        tools.close(1.0)


def test_text_written_by_people_or_models_arrives_only_in_data_fields(session, mcp_env):
    server = mcp_env.server
    server.result_bytes = INJECTION.encode()
    failed = server.add_task(state="failed", error=INJECTION)
    text_task = server.add_task(output_kind="text")
    for task_id in (failed, text_task):
        server.tasks[task_id]["name"] = INJECTION                # a label typed by someone
    listed = session.ok("list_tasks", {"state": "all"})
    for task in listed["tasks"]:
        found = [k for k, v in task.items() if isinstance(v, str) and "IGNORE ALL" in v]
        assert set(found) <= {"name", "error"}
    assert next(t for t in listed["tasks"] if t["id"] == failed)["error"] == INJECTION
    report = session.ok("get_text_result", {"task_id": text_task})
    assert report["text"] == INJECTION and report["source"] == "ai_model"
    assert set(report) == {"task_id", "filename", "text", "truncated", "bytes", "source",
                           "notice"}
    server.inject("GET", "/api/v1/budgets", status=403,
                  body={"code": "lab_not_allowed", "detail": INJECTION})
    error = session.err("get_budgets", {})
    assert error["detail"] == INJECTION and error["message"] != INJECTION


# ── run_scan and get_run ──────────────────────────────────────────────────────
def test_run_scan_queues_a_folder_and_reports_each_scan(session, mcp_env):
    server = mcp_env.server
    nifti(mcp_env, "study/ct_a.nii.gz", seed=1)
    nifti(mcp_env, "study/ct_b.nii.gz", seed=2)
    started = session.ok("run_scan", {"paths": [str(mcp_env.work / "study")], "model": MODEL,
                                      "budget": "personal", "max_credits": 10})
    assert started["run_id"].startswith("run-") and started["estimate"] == {
        "scans": 2, "credits": 2.5, "free_tasks": 0, "paid_credits": 2.5,
        "budget": "personal", "max_credits": 10.0}
    data = run_to_end(session, started["run_id"])
    assert data["state"] == "done" and data["charged_credits"] == 2.5
    # The assistant gets each scan's code and position, never its file name.
    assert [s["position"] for s in data["scans"]] == [1, 2]
    assert all(s["name"] == s["code"] for s in data["scans"])
    assert "ct_a" not in json.dumps(data) and "ct_b" not in json.dumps(data)
    for scan in data["scans"]:
        assert scan["state"] == "queued" and scan["task_id"] in server.tasks
        assert scan["code"].startswith("scan-") and scan["charged"]["credits"] == 1.25
    assert all(u["filename"].startswith("scan-") for u in server.uploads.values())
    assert str(mcp_env.work) not in json.dumps(data)          # base names, never folders


def test_a_run_over_the_cap_uploads_nothing_and_returns_the_estimate(session, mcp_env):
    server = mcp_env.server
    nifti(mcp_env, "ct_a.nii.gz", seed=1)
    nifti(mcp_env, "ct_b.nii.gz", seed=2)
    data = session.ok("run_scan", {"paths": ["ct_a.nii.gz", "ct_b.nii.gz"], "model": MODEL,
                                   "budget": "personal", "max_credits": 2})
    data = run_to_end(session, data["run_id"])
    assert data["state"] == "over_max_credits" and data["reason"] == "over_max_credits"
    assert data["estimate"]["paid_credits"] == 2.5 and data["charged_credits"] == 0
    assert {s["state"] for s in data["scans"]} == {"not_queued"}
    assert _posts(server, "/api/v1/uploads") == [] and server.tasks == {}


def test_budget_free_with_a_cap_of_zero_queues_what_the_free_tasks_pay(session, mcp_env):
    server = mcp_env.server
    server.free_tasks_left = 1
    nifti(mcp_env, "ct_a.nii.gz", seed=1)
    nifti(mcp_env, "ct_b.nii.gz", seed=2)
    data = session.ok("run_scan", {"paths": ["ct_a.nii.gz", "ct_b.nii.gz"], "model": MODEL,
                                   "budget": "free", "max_credits": 0})
    data = run_to_end(session, data["run_id"])
    assert sorted(s["state"] for s in data["scans"]) == ["not_queued", "queued"]
    assert next(s for s in data["scans"] if s["state"] == "not_queued")["reason"] == \
        "no_free_tasks"
    assert server.personal_credits == 100.0 and data["charged_credits"] == 0


def test_a_retried_run_scan_charges_nothing_twice(session, mcp_env):
    server = mcp_env.server
    nifti(mcp_env, "ct_a.nii.gz", seed=1)
    arguments = {"paths": ["ct_a.nii.gz"], "model": MODEL, "budget": "personal",
                 "max_credits": 5}
    first = run_to_end(session, session.ok("run_scan", arguments)["run_id"])
    [queued] = first["scans"]
    again = run_to_end(session, session.ok("run_scan", arguments)["run_id"])
    [skipped] = again["scans"]
    assert skipped["state"] == "skipped" and skipped["reason"] == "already_queued"
    assert skipped["task_id"] == queued["task_id"]
    assert task_count(server) == 1 and server.personal_credits == 98.75
    assert again["charged_credits"] == 0


def test_the_same_run_scan_while_it_is_going_returns_the_same_run(session, mcp_env):
    server = mcp_env.server
    server.patch_delay, server.chunk_bytes = 0.2, 20000
    nifti(mcp_env, "ct_a.nii.gz", seed=1, shape=(40, 40, 30))
    arguments = {"paths": ["ct_a.nii.gz"], "model": MODEL, "budget": "personal",
                 "max_credits": 5}
    first = session.ok("run_scan", arguments)
    second = session.ok("run_scan", arguments)
    assert second["run_id"] == first["run_id"]
    run_to_end(session, first["run_id"])
    assert task_count(server) == 1


def test_get_run_reports_upload_progress(session, mcp_env):
    server = mcp_env.server
    server.patch_delay, server.chunk_bytes = 0.15, 15000
    nifti(mcp_env, "ct_a.nii.gz", seed=1, shape=(40, 40, 30))
    run_id = session.ok("run_scan", {"paths": ["ct_a.nii.gz"], "model": MODEL,
                                     "budget": "personal", "max_credits": 5})["run_id"]
    seen = set()
    for _ in range(400):
        data = session.ok("get_run", {"run_id": run_id})
        for scan in data["scans"]:
            if scan["state"] == "uploading":
                seen.add(scan["percent_uploaded"])
        if data["state"] == "done":
            break
        time.sleep(0.02)
    assert data["state"] == "done" and seen and all(0 <= p <= 100 for p in seen)


def test_a_rejected_scan_reports_the_platforms_reason(session, mcp_env):
    server = mcp_env.server
    nifti(mcp_env, "ct_a.nii.gz", seed=1)
    original = server._process

    def reject(up):
        original(up)
        up["state"], up["error"] = "rejected", "This file holds a label mask, not a scan."
    server._process = reject
    data = session.ok("run_scan", {"paths": ["ct_a.nii.gz"], "model": MODEL,
                                   "budget": "personal", "max_credits": 5})
    data = run_to_end(session, data["run_id"])
    [scan] = data["scans"]
    assert scan["state"] == "failed" and scan["reason"] == "scan_rejected"
    assert scan["detail"] == "This file holds a label mask, not a scan."
    assert server.tasks == {}


def test_an_unknown_run_id(session):
    assert session.err("get_run", {"run_id": "run-000000000000"})["code"] == "run_not_found"
    assert session.err("get_run", {"run_id": "../../etc"})["code"] == "invalid_arguments"


@needs_dicom
def test_several_dicom_series_are_listed_and_queue_only_when_confirmed(session, mcp_env):
    import examples_helpers as helpers
    server = mcp_env.server
    helpers.write_dicom(mcp_env.work / "study", (3, 5), seed=40)
    arguments = {"paths": ["study"], "model": MODEL, "budget": "personal", "max_credits": 10}
    data = session.ok("run_scan", arguments)
    data = run_to_end(session, data["run_id"])
    assert data["state"] == "needs_confirmation" and data["reason"] == "series_to_confirm"
    listed = data["series_to_confirm"]
    assert sorted(s["series"] for s in listed) == ["3", "5"]
    assert all(s["path"] == "study" and s["credits"] == 1.25 and s["shape"] for s in listed)
    assert data["estimate"]["scans"] == 2
    assert _posts(server, "/api/v1/uploads") == []
    wrong = run_to_end(session, session.ok("run_scan", {**arguments, "series": ["9"]})[
        "run_id"])
    assert wrong["state"] == "failed" and wrong["error"]["code"] == "series_not_found"
    assert _posts(server, "/api/v1/uploads") == []
    done = run_to_end(session, session.ok("run_scan", {**arguments, "series": ["5"]})[
        "run_id"])
    [scan] = done["scans"]
    assert done["state"] == "done" and scan["series"] == "5" and scan["state"] == "queued"
    assert task_count(server) == 1 and len(_posts(server, "/api/v1/uploads")) == 1


def test_run_scan_without_credentials_starts_nothing(session, mcp_env, monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    nifti(mcp_env, "ct_a.nii.gz", seed=1)
    assert session.err("list_models", {})["code"] == "no_credentials"
    error = session.err("run_scan", {"paths": ["ct_a.nii.gz"], "model": MODEL,
                                     "budget": "personal", "max_credits": 5})
    assert error["code"] == "no_credentials"
    assert session.tools.runs.runs == {} and mcp_env.server.requests == []


# ── download_result ───────────────────────────────────────────────────────────
def test_download_result_never_overwrites_and_a_repeat_returns_the_same_file(session, mcp_env):
    server = mcp_env.server
    task_id = server.add_task()
    first = session.ok("download_result", {"task_id": task_id})
    # The folder, relative to the allowed folder, and a count: the file's name
    # starts with the scan's original name.
    assert (first["path"], first["files"]) == ("results", 1)
    assert first["bytes"] == len(server.result_bytes)
    [saved] = list((mcp_env.work / "results").iterdir())
    again = session.ok("download_result", {"task_id": task_id})
    assert again == first and len(server.calls("GET", "/result")) == 1
    other = mcp_env.work / "elsewhere"
    other.mkdir()
    taken = other / saved.name
    taken.write_bytes(b"keep me")
    moved = session.ok("download_result", {"task_id": task_id, "folder": "elsewhere"})
    assert moved["path"] == "elsewhere" and taken.read_bytes() == b"keep me"
    assert len(list(other.iterdir())) == 2
    waiting = server.add_task(state="queued")
    error = session.err("download_result", {"task_id": waiting})
    assert (error["code"], error["state"]) == ("not_ready", "queued")


def test_a_slow_download_goes_on_and_is_collected_by_the_same_call(mcp_env):
    server = mcp_env.server
    tools = make_tools(mcp_env, download_wait=0.05)
    try:
        session = Session(tools)
        task_id = server.add_task()
        gate = threading.Event()
        client = tools.client()
        real_download = client.download

        def held_download(*args, **kwargs):          # a large file still arriving
            gate.wait(5)
            return real_download(*args, **kwargs)
        client.download = held_download
        error = session.err("download_result", {"task_id": task_id})
        assert error["code"] == "download_in_progress"
        gate.set()
        tools.download_wait = 10
        data = session.ok("download_result", {"task_id": task_id})
        assert os.path.exists(data["path"]) and len(server.calls("GET", "/result")) == 1
    finally:
        tools.close(1.0)


# ── errors, timeouts, protocol revisions ──────────────────────────────────────
def test_a_limit_comes_back_with_retry_after(session, mcp_env):
    mcp_env.server.inject("GET", "/api/v1/budgets", status=429, times=5,
                          body={"code": "rate_limited", "detail": "Slow down.",
                                "retry_after": 120}, headers={"Retry-After": "120"})
    error = session.err("get_budgets", {})
    assert (error["code"], error["retry_after"]) == ("rate_limited", 120)


def test_a_platform_that_does_not_answer(mcp_env, monkeypatch):
    monkeypatch.setenv("NALVERA_BASE_URL", "http://127.0.0.1:9")
    tools = make_tools(mcp_env, api_retry_wait=0.5)
    try:
        assert Session(tools).err("get_budgets", {})["code"] == "api_unreachable"
    finally:
        tools.close(1.0)


def test_no_tool_call_runs_past_its_time_limit(local_tools):
    local_tools._impl["get_budgets"] = lambda args: time.sleep(3) or {}
    session = Session(local_tools, tool_timeout=0.3)
    started = time.monotonic()
    assert session.err("get_budgets", {})["code"] == "timeout"
    assert time.monotonic() - started < 2.5


def test_an_unknown_tool_is_a_protocol_error(session):
    from mcp.shared.exceptions import MCPError
    with pytest.raises(Exception) as caught:                 # the client's task group wraps it
        session.call("delete_everything", {})
    found, todo = [], [caught.value]
    while todo:
        error = todo.pop()
        found.append(error)
        todo += list(getattr(error, "exceptions", ()))
    assert any(isinstance(e, MCPError) for e in found)


def test_the_handshake_revisions_are_served_too(local_tools):
    legacy = Session(local_tools, mode="legacy")
    assert len(legacy.list_tools().tools) == 12
    assert legacy.ok("get_budgets", {})["suggested"] == "personal"
