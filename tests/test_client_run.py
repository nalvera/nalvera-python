"""Client.run, wait and download end to end: fake server, stand-in anonymizer."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest

import nalvera
import stub_anonymizer
from conftest import TTYStringIO, scan_bytes
from fake_server import error_body
from nalvera.errors import (
    BudgetChoiceRequired,
    ConfigError,
    InsufficientCredits,
    NalveraError,
    PermissionDenied,
    PreparationError,
    ScanRejected,
)

MODEL, TASK = "totalsegmentator_ct", "total"


def _expected_key(code: str, sha256: str, model: str = MODEL, task: str = TASK) -> str:
    # The scan's code is recorded in the mapping before the upload, so the
    # key of its submit can be found again by a later run (80/B2); the
    # anonymized file's digest keeps two scans with one code apart, and the
    # text query (none here) is part of it too.
    identity = f"{code}\n{sha256}\n{model}\n{task}\n"
    return "scan-" + hashlib.sha256(identity.encode()).hexdigest()[:48]


def _no_temp_files_left():
    return all(not os.path.exists(s.path) for s in stub_anonymizer.prepared)


# ── run ───────────────────────────────────────────────────────────────────────
def test_one_scan_is_anonymized_uploaded_and_queued(client, server, scans):
    path = scans("Doe_John_CT.nii.gz")
    tasks = client.run(path, model=MODEL, task=TASK, budget="personal")
    assert isinstance(tasks, nalvera.RunResult) and len(tasks) == 1 and tasks.failed == []
    task = tasks[0]
    assert task.state == "queued" and task.original_name == "Doe_John_CT.nii.gz"
    assert task.code.startswith("scan-") and task.charged["source"] == "personal"
    submit = server.calls("POST", "/api/v1/tasks")[0]
    body = server.tasks[task.id]["body"]
    assert body["name"] == f"{task.code}.nii.gz" and body["budget"] == "personal"
    assert submit["headers"]["Idempotency-Key"] == _expected_key(
        task.code, hashlib.sha256(path.read_bytes()).hexdigest())
    # The original name never left this computer.
    created = server.calls("POST", "/api/v1/uploads")[0]
    assert created["body_length"] < 200
    assert all("Doe_John" not in str(r) for r in server.requests)
    assert _no_temp_files_left()


def test_the_mapping_links_code_task_and_original(client, server, scans, env):
    path = scans("P012_baseline.nii.gz")
    task = client.run(path, model=MODEL, budget="personal")[0]
    rows = stub_anonymizer.Mapping(str(env / "nalvera_scans.csv")).find(task_id=task.id)
    assert rows and rows[0]["original_name"] == "P012_baseline.nii.gz"
    assert rows[0]["code"] == task.code and rows[0]["model"] == MODEL and rows[0]["task"] == TASK


def test_a_label_reaches_the_platform(client, server, scans):
    task = client.run(scans("a.nii.gz"), model=MODEL, label="P012 baseline",
                      budget="personal")[0]
    assert server.tasks[task.id]["body"]["label"] == "P012 baseline"
    assert task.name == "P012_baseline.nii.gz"


@pytest.mark.parametrize("label", ["<b>x</b>", "a/b", "x" * 61, "tab\tname"])
def test_a_label_that_could_carry_markup_or_a_path_is_refused_first(client, server, scans,
                                                                    label):
    with pytest.raises(ConfigError) as caught:
        client.run(scans("a.nii.gz"), model=MODEL, label=label, budget="personal")
    assert caught.value.code == "invalid_label"
    assert server.calls("POST", "/api/v1/uploads") == []


def test_a_dicom_folder_gives_one_task_per_series(client, server, env):
    folder = env / "input" / "Patient_DICOM"
    folder.mkdir(parents=True)
    for series in ("s1", "s2", "s3"):
        (folder / f"{series}.dcm").write_bytes(scan_bytes(series.encode()))
    tasks = client.run(folder, model=MODEL, budget="personal", yes=True)
    assert len(tasks) == 3 and len(server.tasks) == 3
    assert {t.original_name for t in tasks} == {
        nalvera.messages.text("run.series_name", name="Patient_DICOM", series=s)
        for s in ("s1", "s2", "s3")}


def test_several_paths_keep_their_order(client, server, scans):
    paths = [scans(f"scan_{i}.nii.gz") for i in range(5)]
    tasks = client.run(paths, model=MODEL, budget="personal")
    assert [t.original_name for t in tasks] == [p.name for p in paths]


def test_a_scan_that_cannot_be_read_is_skipped_and_the_rest_go_on(client, server, scans):
    good = scans("good.nii.gz")
    bad = scans("notes.nii.gz", b"NOTASCAN hello")
    rejected = scans("mask.nii.gz", scan_bytes(b"mask") + b"REJECT")
    tasks = client.run([good, bad, rejected], model=MODEL, budget="personal")
    assert [t.original_name for t in tasks] == ["good.nii.gz"]
    failed = {f.original_name: f.code for f in tasks.failed}
    assert failed == {"notes.nii.gz": "not_a_scan", "mask.nii.gz": "scan_rejected"}
    assert _no_temp_files_left()


def test_when_no_scan_could_be_queued_the_error_is_raised(client, server, scans):
    with pytest.raises(PreparationError) as caught:
        client.run(scans("notes.nii.gz", b"NOTASCAN"), model=MODEL, budget="personal")
    assert caught.value.code == "not_a_scan" and len(caught.value.failed) == 1
    with pytest.raises(ScanRejected):
        client.run(scans("m.nii.gz", scan_bytes(b"m") + b"REJECT"), model=MODEL,
                   budget="personal")


def test_a_budget_that_covers_no_task_stops_before_any_upload(client, server, scans):
    # 80/B3: the estimate shows the budget pays for none of the tasks.
    server.personal_credits = 0.5
    with pytest.raises(InsufficientCredits) as caught:
        client.run([scans(f"t{i}.nii.gz") for i in range(3)], model=MODEL, budget="personal")
    assert caught.value.code == "insufficient_credits"
    assert caught.value.extra["budget"] == "personal"
    assert server.calls("POST", "/api/v1/uploads") == []
    assert _no_temp_files_left()


def test_no_credits_mid_run_stops_at_once_and_reports_what_was_queued(client, server, scans,
                                                                      monkeypatch):
    # 80/B5: the estimate said the budget covers the run, then credits went
    # elsewhere: the first refusal stops the run, without a retry.
    paths = [scans(f"t{i}.nii.gz") for i in range(3)]
    real = client._upload_all

    def spend_elsewhere(job, **kwargs):
        server.personal_credits = 1.3                    # one more task at 1.25
        return real(job, **kwargs)

    monkeypatch.setattr(client, "_upload_all", spend_elsewhere)
    with pytest.raises(InsufficientCredits) as caught:
        client.run(paths, model=MODEL, budget="personal")
    assert caught.value.code == "insufficient_credits"
    assert len(caught.value.tasks) == 1 and len(server.tasks) == 1
    submits = server.calls("POST", "/api/v1/tasks")
    assert len(submits) <= 3 and len({s["headers"]["Idempotency-Key"] for s in submits}) \
        == len(submits)                                  # no submit was repeated
    assert _no_temp_files_left()
    # The refused scan's upload is deleted: the platform counts every unfinished
    # upload of the account against a small cap.
    assert not [u for u in server.uploads.values() if u["state"] in ("ready", "receiving")]


def test_too_many_active_tasks_waits_until_one_finishes(client, server, scans, fake_time, out):
    server.inject("POST", "/api/v1/tasks", status=429, times=2, headers={"Retry-After": "60"},
                  body=error_body("too_many_active_tasks", retry_after=60, limit=50))
    tasks = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert len(tasks) == 1 and fake_time.slept >= 120
    assert len(server.calls("POST", "/api/v1/tasks")) == 3


def test_a_lost_submit_answer_is_repeated_with_the_same_key_and_charged_once(client, server,
                                                                             scans):
    server.inject("POST", "/api/v1/tasks", drop=True, after=True)
    credits = server.personal_credits
    tasks = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    submits = server.calls("POST", "/api/v1/tasks")
    assert len(submits) == 2
    assert submits[0]["headers"]["Idempotency-Key"] == submits[1]["headers"]["Idempotency-Key"]
    assert len(server.tasks) == 1 and tasks[0].id in server.tasks
    assert server.personal_credits == pytest.approx(credits - server.credits_per_task)


def test_free_tasks_are_used_first_and_announced(client, server, scans, out):
    server.free_tasks_left = 1
    tasks = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL, budget="personal")
    assert sorted(t.charged["source"] for t in tasks) == ["free", "personal"]
    assert nalvera.messages.text("run.plan_free", free=1, count=2) in out.getvalue()


def test_a_lab_and_no_terminal_stop_before_any_upload(client, server, scans):
    server.add_lab(12, "Imaging Lab", 100.0)
    with pytest.raises(BudgetChoiceRequired) as caught:
        client.run(scans("a.nii.gz"), model=MODEL)
    assert {o["budget"] for o in caught.value.options} == {"personal", "lab:12"}
    assert server.calls("POST", "/api/v1/uploads") == [] and stub_anonymizer.prepared == []


def test_the_budget_setting_skips_the_question(make_client, server, scans, monkeypatch):
    server.add_lab(12, "Imaging Lab", 100.0)
    monkeypatch.setenv("NALVERA_BUDGET", "lab:12")
    tasks = make_client().run(scans("a.nii.gz"), model=MODEL)
    assert tasks[0].charged == {"source": "lab", "lab_id": 12, "credits": 1.25}


def test_yes_takes_the_suggested_lab(client, server, scans):
    server.add_lab(12, "Small Lab", 10.0)
    server.add_lab(14, "Big Lab", 500.0)
    tasks = client.run(scans("a.nii.gz"), model=MODEL, yes=True)
    assert tasks[0].charged["lab_id"] == 14


def test_in_a_terminal_the_question_comes_once_with_the_runs_cost(make_client, server, scans):
    server.add_lab(12, "Imaging Lab", 100.0)
    stdin, stdout = io.StringIO("1\n"), io.StringIO()
    client = make_client(interactive=True, input=stdin, output=stdout)
    tasks = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL)
    assert [t.charged["source"] for t in tasks] == ["personal", "personal"]
    # Priced by voxel grid before uploading: one estimate per distinct grid.
    estimates = server.calls("POST", "/api/v1/estimate")
    assert len(estimates) == 1
    first_upload = server.calls("POST", "/api/v1/uploads")[0]
    assert server.requests.index(estimates[0]) < server.requests.index(first_upload)
    shown = stdout.getvalue()
    assert shown.count(nalvera.messages.text("budget.prompt", default=2)) == 1
    assert "2.50" in shown                                       # 2 tasks x 1.25
    assert 'budget="personal"' in shown and "NALVERA_BUDGET=personal" in shown


def test_a_key_that_may_not_run_tasks_is_refused_before_anything_is_done(client, server,
                                                                         scans):
    server.scopes = ["tasks:read", "results:read"]
    with pytest.raises(PermissionDenied) as caught:
        client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert caught.value.code == "scope_missing"
    assert caught.value.required_scope == "tasks:write"
    assert stub_anonymizer.prepared == [] and server.calls("POST", "/api/v1/uploads") == []


def test_terms_awaiting_acceptance_stop_the_run_before_anonymizing(client, server, scans):
    server.terms_reaccept_required = True
    with pytest.raises(PermissionDenied) as caught:
        client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert caught.value.code == "terms_reaccept_required"
    assert stub_anonymizer.prepared == [] and server.calls("POST", "/api/v1/uploads") == []


@pytest.mark.parametrize("model,task,code", [
    ("no_such_model", None, "unknown_model"), (MODEL, "no_such_task", "unknown_task"),
    ("nv_generate_ctmr", None, "model_is_generative"), ("merlin", None, "text_required")])
def test_requests_that_cannot_work_are_refused_before_uploading(client, server, scans, model,
                                                                task, code):
    with pytest.raises(ConfigError) as caught:
        client.run(scans("a.nii.gz"), model=model, task=task, budget="personal")
    assert caught.value.code == code
    assert server.calls("POST", "/api/v1/uploads") == []


def test_a_text_query_is_sent_for_a_task_that_needs_one(client, server, scans):
    task = client.run(scans("a.nii.gz"), model="merlin", text="pleural effusion",
                      budget="personal")[0]
    assert server.tasks[task.id]["body"]["text"] == "pleural effusion"


def test_ctrl_c_removes_the_temporary_folder(client, server, scans, monkeypatch, stub_anon):
    seen = []
    real = stub_anon.prepare

    def interrupted(path, *, workdir, label=None, max_bytes=None, avoid=None):
        seen.append(workdir)
        if len(seen) == 2:
            raise KeyboardInterrupt
        return real(path, workdir=workdir, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(stub_anon, "prepare", interrupted)
    with pytest.raises(KeyboardInterrupt):
        client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL, budget="personal")
    assert seen and not os.path.exists(seen[0])


# ── wait ──────────────────────────────────────────────────────────────────────
def test_wait_follows_the_queue_and_ends_with_a_summary(client, server, scans, out, fake_time):
    tasks = client.run(scans("Doe_John.nii.gz"), model=MODEL, budget="personal")
    before = len(fake_time.sleeps)
    done = client.wait(tasks)
    assert [t.state for t in done] == ["completed"]
    lines = [line for line in out.getvalue().splitlines() if line.startswith("Doe_John.nii.gz")]
    # One line per change: positions 2 and 1, running, completed.
    for position in (2, 1):
        fragment = nalvera.messages.text("wait.position", position=position)
        assert sum(1 for line in lines if fragment in line) == 1
    for state in ("processing", "completed"):
        shown = nalvera.messages.text(f"wait.state.{state}")
        assert sum(1 for line in lines if shown in line) >= 1
    assert nalvera.messages.text("wait.summary", completed=1, failed=0, cancelled=0) \
        in out.getvalue()
    assert all(s <= 30 for s in fake_time.sleeps[before:])


def test_polls_respect_the_minimum_and_slow_down_while_nothing_changes(client, server, scans,
                                                                       fake_time, monkeypatch):
    server.script = [("queued", 1)] * 12 + [("completed", None)]
    tasks = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    gets = []
    real = client._transport.json

    def spy(method, path, **kwargs):
        if method == "GET" and path.startswith("/api/v1/tasks/"):
            gets.append(fake_time.now)
        return real(method, path, **kwargs)

    monkeypatch.setattr(client._transport, "json", spy)
    client.wait(tasks)
    gaps = [b - a for a, b in zip(gets, gets[1:])]
    assert gaps and min(gaps) >= 5
    assert max(gaps) <= 31 and gaps[-1] > gaps[0]


def test_many_tasks_are_followed_with_the_list(client, server, scans):
    tasks = client.run([scans(f"s{i}.nii.gz") for i in range(6)], model=MODEL,
                       budget="personal")
    before = len(server.calls("GET", "/api/v1/tasks/"))
    client.wait(tasks)
    assert len(server.calls("GET", "/api/v1/tasks/")) == before      # no task-by-task polls
    assert server.calls("GET", "/api/v1/tasks")


def test_a_failed_task_is_reported_with_its_error(client, server, scans, out):
    server.script = [("queued", 1), ("failed", None)]
    tasks = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    server.tasks[tasks[0].id]["error"] = "Out of memory."
    done = client.wait(tasks)
    assert done[0].state == "failed" and done[0].error == "Out of memory."
    assert "Out of memory." in out.getvalue()


def test_wait_redraws_in_place_on_a_terminal(make_client, server, scans):
    screen = TTYStringIO()
    client = make_client(output=screen)
    client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    assert "\x1b[1F" in screen.getvalue()


def test_wait_can_time_out(client, server, scans):
    server.script = [("queued", 1)]
    tasks = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    with pytest.raises(NalveraError) as caught:
        client.wait(tasks, timeout=60)
    assert caught.value.code == "wait_timeout"


# ── download ──────────────────────────────────────────────────────────────────
def test_results_are_saved_under_the_original_name(client, server, scans, env):
    tasks = client.run(scans("Doe_John_CT.nii.gz"), model=MODEL, budget="personal")
    paths = client.download(client.wait(tasks), out=env / "results")
    assert [p.name for p in paths] == [f"Doe_John_CT__{MODEL}__{TASK}.nii.gz"]
    assert paths[0].read_bytes() == server.result_bytes
    rows = stub_anonymizer.Mapping(str(env / "nalvera_scans.csv")).find(task_id=tasks[0].id)
    assert any(r["result_path"] == str(paths[0]) for r in rows)
    assert not [p for p in (env / "results").iterdir() if p.name.endswith(".part")]


def test_an_existing_file_is_never_replaced(client, server, scans, env):
    tasks = client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    first = client.download(tasks, out=env / "results")[0]
    first.write_bytes(b"my edits")
    second = client.download(tasks, out=env / "results")[0]
    assert first.read_bytes() == b"my edits"
    assert second != first and second.name == f"a__{MODEL}__{TASK}-2.nii.gz"


def test_a_task_without_a_result_is_skipped(client, server, scans, env):
    server.script = [("queued", 1)]
    tasks = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    saved = client.download(tasks, out=env / "results")
    assert list(saved) == [] and saved.skipped == [(tasks[0].id, "queued")]


def test_a_new_client_finds_the_original_name_in_the_mapping(make_client, server, scans, env):
    first = make_client()
    task_id = first.wait(first.run(scans("Smith_MR.nii.gz"), model=MODEL,
                                   budget="personal"))[0].id
    later = make_client()
    paths = later.download([task_id], out=env / "later")
    assert paths[0].name == f"Smith_MR__{MODEL}__{TASK}.nii.gz"


def test_a_website_task_is_named_after_the_platforms_name(client, server, env):
    task_id = server.add_task("completed")
    paths = client.download(task_id, out=env / "results")
    assert paths[0].name == f"website-scan__{MODEL}__{TASK}.nii.gz"


def test_download_needs_the_results_permission(client, server, env):
    server.scopes = ["tasks:read", "tasks:write"]
    task_id = server.add_task("completed")
    with pytest.raises(PermissionDenied) as caught:
        client.download(task_id, out=env / "results")
    assert caught.value.required_scope == "results:read"


def test_a_part_of_a_result_is_named_with_the_part(client, server, env):
    task_id = server.add_task("completed")
    path = client.download(task_id, out=env / "results", part="mask")[0]
    assert path.name == f"website-scan__{MODEL}__{TASK}__mask.nii.gz"
    assert server.calls("GET", "/result")[-1]["query"] == "part=mask"


# ── the key never shows ───────────────────────────────────────────────────────
def test_the_key_never_appears_in_output_errors_or_reprs(make_client, server, scans, env):
    out = io.StringIO()
    client = make_client(output=out)
    tasks = client.run([scans("a.nii.gz"), scans("bad.nii.gz", b"NOTASCAN")], model=MODEL,
                       budget="personal")
    client.download(client.wait(tasks), out=env / "results")
    texts = [out.getvalue(), repr(client), repr(tasks), repr(client._transport)]
    server.inject("GET", "/api/v1/budgets", status=400,
                  body={"code": "invalid_request", "detail": f"echo {server.key}"})
    with pytest.raises(NalveraError) as caught:
        client.budgets()
    texts += [str(caught.value), repr(caught.value), caught.value.detail]
    for text in texts:
        assert server.key not in text and server.key[12:] not in text
    mapping = Path(env / "nalvera_scans.csv").read_text()
    assert server.key not in mapping and server.key[:12] not in mapping
