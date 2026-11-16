"""Money. Re-runs never charge twice, the
mapping is written ahead, the cost is asked once, the spending cap holds."""
from __future__ import annotations

import io
import os

import pytest

import stub_anonymizer
from conftest import scan_bytes
from nalvera import cli, messages
from nalvera import client as client_module
from nalvera.errors import CreditCapExceeded, InsufficientCredits
from nalvera.upload import Uploader

MODEL, TASK = "totalsegmentator_ct", "total"


def _mapping(env):
    return stub_anonymizer.Mapping(str(env / "nalvera_scans.csv"))


# ── B1: never queue the same scan twice by accident ───────────────────────────
@pytest.mark.parametrize("state", ["queued", "processing", "completed"])
def test_a_rerun_skips_a_scan_already_sent(make_client, server, scans, state):
    path = scans("Doe_John.nii.gz")
    server.script = [(state, 1 if state == "queued" else None)]
    first = make_client().run(path, model=MODEL, budget="personal")[0]
    uploads, credits = len(server.uploads), server.personal_credits
    again = make_client().run(path, model=MODEL, budget="personal")
    assert [(t.id, t.earlier) for t in again] == [(first.id, True)] and again.failed == []
    assert again.charged_credits == 0
    assert [(s.reason, s.task_id) for s in again.skipped] == [(f"already_{state}", first.id)]
    assert len(server.uploads) == uploads and server.personal_credits == credits


def test_the_same_scan_under_another_name_is_recognised(make_client, server, scans, env):
    data = scan_bytes(b"same content")
    make_client().run(scans("a.nii.gz", data), model=MODEL, budget="personal")
    again = make_client().run(scans("copy_of_a.nii.gz", data), model=MODEL, budget="personal")
    assert len(again.skipped) == 1 and len(server.tasks) == 1


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_a_scan_whose_task_failed_runs_again(make_client, server, scans, state):
    path = scans("a.nii.gz")
    server.script = [(state, None)]
    make_client().run(path, model=MODEL, budget="personal")
    again = make_client().run(path, model=MODEL, budget="personal")
    assert len(again) == 1 and len(server.tasks) == 2


def test_a_scan_whose_task_was_deleted_on_the_website_runs_again(make_client, server, scans):
    path = scans("a.nii.gz")
    first = make_client().run(path, model=MODEL, budget="personal")[0]
    del server.tasks[first.id]
    server.order.remove(first.id)
    again = make_client().run(path, model=MODEL, budget="personal")
    assert len(again) == 1


def test_another_model_or_task_is_not_a_repeat(make_client, server, scans):
    path = scans("a.nii.gz")
    make_client().run(path, model=MODEL, budget="personal")
    again = make_client().run(path, model=MODEL, task="lung_vessels", budget="personal")
    assert len(again) == 1 and len(server.tasks) == 2


def test_force_runs_it_again(make_client, server, scans):
    path = scans("a.nii.gz")
    make_client().run(path, model=MODEL, budget="personal")
    again = make_client().run(path, model=MODEL, budget="personal", force=True)
    assert len(again) == 1 and len(server.tasks) == 2
    keys = {c["headers"]["Idempotency-Key"] for c in server.calls("POST", "/api/v1/tasks")}
    assert len(keys) == 2


def test_the_same_file_twice_in_one_run_is_queued_once(make_client, server, scans):
    data = scan_bytes(b"one")
    result = make_client().run([scans("a.nii.gz", data), scans("b.nii.gz", data)],
                               model=MODEL, budget="personal")
    assert len(result) == 1 and [s.reason for s in result.skipped] == ["duplicate_in_run"]


def test_a_dicom_folder_run_again_is_skipped(make_client, server, env):
    folder = env / "input" / "Patient_DICOM"
    folder.mkdir(parents=True)
    for series in ("s1", "s2"):
        (folder / f"{series}.dcm").write_bytes(scan_bytes(series.encode()))
    first = make_client().run(folder, model=MODEL, budget="personal", yes=True)
    again = make_client().run(folder, model=MODEL, budget="personal", yes=True)
    assert [t.id for t in again] == [t.id for t in first] and all(t.earlier for t in again)
    assert len(again.skipped) == 2 and len(server.tasks) == 2


def test_the_command_line_has_force(env, server, stub_anon, no_sleep):
    from test_cli import run_cli
    scan = env / "a.nii.gz"
    scan.write_bytes(scan_bytes(b"a"))
    argv = ("run", str(scan), "--model", MODEL, "--budget", "personal", "--no-wait")
    assert run_cli(*argv)[0] == 0 and run_cli(*argv)[0] == 0
    assert len(server.tasks) == 1
    assert run_cli(*argv, "--force")[0] == 0 and len(server.tasks) == 2


# ── B2: the mapping is written ahead ──────────────────────────────────────────
def test_the_scan_is_recorded_before_its_upload(make_client, server, scans, env, monkeypatch):
    seen = []
    real = Uploader.upload

    def spy(self, path, *, filename, size, display_name):
        code = filename.split(".")[0]
        seen.append([r["code"] for r in _mapping(env).rows()].count(code))
        return real(self, path, filename=filename, size=size, display_name=display_name)

    monkeypatch.setattr(Uploader, "upload", spy)
    make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert seen == [1]


def _die_after_the_submit(client, monkeypatch):
    """The process dies after the server queued the task, before the answer
    was recorded."""
    real = client._submit

    def submit_then_die(body, key):
        real(body, key)
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "_submit", submit_then_die)


def test_a_run_that_died_after_its_submit_is_found_by_name(make_client, server, scans, env,
                                                           monkeypatch):
    path = scans("a.nii.gz")
    first = make_client()
    _die_after_the_submit(first, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        first.run(path, model=MODEL, budget="personal")
    assert len(server.tasks) == 1 and not any(r["task_id"] for r in _mapping(env).rows())
    uploads, credits = len(server.uploads), server.personal_credits
    again = make_client().run(path, model=MODEL, budget="personal")
    task_id = next(iter(server.tasks))
    assert [(s.reason, s.task_id) for s in again.skipped] == [("already_queued", task_id)]
    assert len(server.uploads) == uploads and server.personal_credits == credits
    lookups = [c for c in server.calls("GET", "/api/v1/tasks") if "name=scan-" in c["query"]]
    assert len(lookups) == 1
    assert any(r["task_id"] == task_id for r in _mapping(env).rows())      # recorded now


def test_a_labelled_run_that_died_reuses_its_key_and_is_charged_once(make_client, server,
                                                                     scans, monkeypatch):
    path = scans("a.nii.gz")
    first = make_client()
    _die_after_the_submit(first, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        first.run(path, model=MODEL, budget="personal", label="P012 baseline")
    credits = server.personal_credits
    out = io.StringIO()
    again = make_client(output=out).run(path, model=MODEL, budget="personal",
                                        label="P012 baseline")
    assert len(server.tasks) == 1 and server.personal_credits == credits
    assert again[0].id == next(iter(server.tasks)) and again[0].replayed
    keys = [c["headers"]["Idempotency-Key"] for c in server.calls("POST", "/api/v1/tasks")]
    assert len(keys) == 2 and keys[0] == keys[1]
    assert messages.text("run.unexpected_replay", name="a.nii.gz", task_id=again[0].id,
                         state=messages.text("wait.state.queued")) not in out.getvalue()


# ── B3: the cost, once, before anything is uploaded ───────────────────────────
def test_the_plan_is_shown_before_any_upload_and_says_what_the_budget_covers(
        make_client, server, scans, monkeypatch):
    server.personal_credits = 2.6                    # 2 of 3 tasks at 1.25
    server.free_tasks_left = 1
    out = io.StringIO()
    client = make_client(output=out)
    with pytest.raises(InsufficientCredits):
        client.run([scans(f"s{i}.nii.gz") for i in range(4)], model=MODEL, budget="personal")
    shown = out.getvalue()
    for line in (messages.text("run.plan", count=4, credits="5.00"),
                 messages.text("run.plan_free", free=1, count=4),
                 messages.text("run.plan_covers", budget="personal", covers=2, paid=3)):
        assert line in shown
    plan_at = shown.index(messages.text("run.plan", count=4, credits="5.00"))
    assert plan_at < shown.index(messages.text("upload.done", name="s0.nii.gz"))
    assert len(server.tasks) == 3                     # one free, two paid, then it stopped


def test_scans_of_different_size_are_priced_each(make_client, server, scans):
    server.credits_by_shape = {(4, 4, 4): 1.0, (8, 8, 8): 3.0}
    found = make_client().estimate_run(
        [scans("small.nii.gz"), scans("big.nii.gz", scan_bytes(b"big") + b"SHAPE=8x8x8;")],
        model=MODEL)
    assert [s.credits for s in found.scans] == [1.0, 3.0] and found.credits == 4.0
    assert server.calls("POST", "/api/v1/uploads") == [] and server.tasks == {}
    assert all(not os.path.exists(s.path) for s in stub_anonymizer.prepared)


# ── B4: the spending cap ──────────────────────────────────────────────────────
def test_a_run_whose_estimate_passes_the_cap_does_not_start(make_client, server, scans):
    with pytest.raises(CreditCapExceeded) as caught:
        make_client().run([scans(f"s{i}.nii.gz") for i in range(3)], model=MODEL,
                          budget="personal", max_credits=3.0)
    assert caught.value.code == "max_credits_exceeded"
    assert caught.value.extra == {"estimate": 3.75, "max_credits": 3.0}
    assert server.calls("POST", "/api/v1/uploads") == []
    assert cli.exit_status(caught.value) == cli.EXIT_CREDITS


def test_free_tasks_do_not_count_against_the_cap(make_client, server, scans):
    server.free_tasks_left = 2
    result = make_client().run([scans(f"s{i}.nii.gz") for i in range(3)], model=MODEL,
                               budget="personal", max_credits=1.25)
    assert len(result) == 3 and result.charged_credits == 1.25


def test_queueing_stops_before_the_charged_total_would_pass_the_cap(make_client, server,
                                                                    scans, monkeypatch):
    client = make_client()
    real = client._upload_all

    def prices_rise(job, **kwargs):
        server.credits_per_task = 2.0                # charged more than estimated
        return real(job, **kwargs)

    monkeypatch.setattr(client, "_upload_all", prices_rise)
    result = client.run([scans(f"s{i}.nii.gz") for i in range(4)], model=MODEL,
                        budget="personal", max_credits=5.0)
    assert result.charged_credits <= 5.0 + 2.0 - 1.25
    assert [f.code for f in result.not_queued] and all(
        f.code == "max_credits_exceeded" for f in result.not_queued)
    assert len(result) + len(result.not_queued) == 4


def test_the_cap_from_the_environment_and_the_command_line(env, server, stub_anon, no_sleep,
                                                           monkeypatch):
    from test_cli import run_cli
    scan = env / "a.nii.gz"
    scan.write_bytes(scan_bytes(b"a"))
    status, _out, _err = run_cli("run", str(scan), "--model", MODEL, "--budget", "personal",
                                 "--max-credits", "1", "--no-wait")
    assert status == cli.EXIT_CREDITS and server.tasks == {}
    monkeypatch.setenv("NALVERA_MAX_CREDITS", "1")
    assert run_cli("run", str(scan), "--model", MODEL, "--budget", "personal",
                   "--no-wait")[0] == cli.EXIT_CREDITS
    assert run_cli("run", str(scan), "--model", MODEL, "--max-credits", "abc")[0] == 2


def test_a_generation_over_the_cap_is_refused(client, server):
    with pytest.raises(CreditCapExceeded):
        client.generate("nv_generate_ctmr", "rflow-ct", {"mode": "image-only"},
                        budget="personal", max_credits=1.0)
    assert server.calls("POST", "/api/v1/tasks") == []


# ── B6: an unexpected idempotent replay is reported ───────────────────────────
def test_an_unexpected_replay_is_reported_and_charges_nothing(make_client, server, scans,
                                                              monkeypatch):
    earlier = server.add_task("queued")
    server.idempotency["fixed-key"] = earlier
    monkeypatch.setattr(client_module, "_scan_idempotency_key", lambda *args: "fixed-key")
    out = io.StringIO()
    credits = server.personal_credits
    task = make_client(output=out).run(scans("a.nii.gz"), model=MODEL, budget="personal")[0]
    assert task.id == earlier and task.replayed
    assert server.personal_credits == credits
    assert messages.text("run.unexpected_replay", name="a.nii.gz", task_id=earlier,
                         state=messages.text("wait.state.queued")) in out.getvalue()


def test_a_replay_after_a_lost_answer_is_this_calls_own_task(make_client, server, scans):
    server.inject("POST", "/api/v1/tasks", drop=True, after=True)
    out = io.StringIO()
    task = make_client(output=out).run(scans("a.nii.gz"), model=MODEL, budget="personal")[0]
    assert task.replayed and task.charged_now
    assert messages.text("run.unexpected_replay", name="a.nii.gz", task_id=task.id,
                         state=messages.text("wait.state.queued")) not in out.getvalue()
    assert messages.text("run.queued", name="a.nii.gz", task_id=task.id,
                         charge=client_module._charge_text(task.charged)) in out.getvalue()


# ── B7: exit statuses ─────────────────────────────────────────────────────────
def test_exit_statuses_by_error():
    from nalvera import errors
    cases = [
        (errors.PreparationError(code="not_a_scan"), 1),
        (errors.ScanRejected(code="scan_rejected"), 1),
        (errors.ConfigError(code="invalid_label"), 2),
        (errors.InvalidRequest(code="invalid_request"), 2),
        (errors.NoCredentials(), 3),
        (errors.ConfigError(code="invalid_key_format"), 3),
        (errors.AuthenticationError(code="key_revoked"), 3),
        (errors.PermissionDenied(code="scope_missing"), 3),
        (errors.ApiDisabled(code="api_disabled"), 3),
        (errors.ClientTooOld(), 3),
        (errors.InsufficientCredits(), 4),
        (errors.NoFreeTasks(), 4),
        (errors.BudgetChoiceRequired([]), 4),
        (errors.CreditCapExceeded(), 4),
        (errors.ConfirmationRequired(), 4),
        (errors.PermissionDenied(code="lab_not_allowed"), 4),
        (errors.RateLimited(code="too_many_active_tasks"), 5),
        (errors.ServiceUnavailable(code="queue_full"), 5),
        (errors.NetworkError(), 5),
        (errors.ServerError(), 5),
        (errors.RunInterrupted(), 130),
    ]
    for error, status in cases:
        assert cli.exit_status(error) == status, error
