"""Free tasks first, then the paid budget; the budget
``free``; ``no_free_tasks`` in the middle of a run. Against the fake server,
which follows backend/routes/api_v1.py (tests/test_api_v1_budget_free.py)."""
from __future__ import annotations

import io

import pytest

import stub_anonymizer
from nalvera import cli, messages
from nalvera.errors import BudgetChoiceRequired, NoFreeTasks

MODEL = "totalsegmentator_ct"
CT = {"mode": "image-only", "output_size": [256, 256, 128], "spacing": [1.5, 1.5, 2.0]}


def _budgets_sent(server):
    return [server.tasks[t]["body"]["budget"] for t in server.order
            if server.tasks[t]["source"] == "api"]


def _terminal(make_client, answers=""):
    out = io.StringIO()
    client = make_client(interactive=True, input=io.StringIO(answers), output=out)
    return client, out


# ── rule 1: a budget given is sent as given ───────────────────────────────────
@pytest.mark.parametrize("budget", ["personal", "lab:12", "free"])
def test_a_budget_given_is_sent_as_given(make_client, server, scans, budget):
    server.add_lab(12, "Imaging Lab", 100.0)
    server.free_tasks_left = 1
    make_client().run(scans("a.nii.gz"), model=MODEL, budget=budget)
    assert _budgets_sent(server) == [budget]


def test_the_budget_setting_is_sent_as_given(make_client, server, scans, monkeypatch):
    monkeypatch.setenv("NALVERA_BUDGET", "free")
    server.free_tasks_left = 1
    make_client().run(scans("a.nii.gz"), model=MODEL)
    assert _budgets_sent(server) == ["free"]


# ── rule 2: free tasks cover the run: "free", no question ─────────────────────
@pytest.mark.parametrize("scene", ["free_tasks", "free_period"])
def test_free_tasks_covering_the_run_need_no_budget_and_no_question(make_client, server,
                                                                    scans, scene):
    server.add_lab(12, "Imaging Lab", 100.0)            # a lab, and no terminal
    if scene == "free_tasks":
        server.free_tasks_left = 3
    else:
        server.free_period = True
    tasks = make_client().run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL)
    assert _budgets_sent(server) == ["free", "free"]
    assert [t.charged["source"] for t in tasks] == ["free", "free"]
    assert server.labs[0]["balance"] == 100.0


def test_free_tasks_covering_the_run_ask_nothing_in_a_terminal(make_client, server, scans):
    server.add_lab(12, "Imaging Lab", 100.0)
    server.free_tasks_left = 2
    client, out = _terminal(make_client)                 # no answer would be read
    client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL)
    assert messages.text("budget.prompt", default=2) not in out.getvalue()
    assert _budgets_sent(server) == ["free", "free"]


# ── rule 3: free tasks cover part ─────────────────────────────────────────────
def test_without_a_lab_the_rest_is_personal(make_client, server, scans):
    server.free_tasks_left = 1
    make_client().run([scans("a.nii.gz"), scans("b.nii.gz"), scans("c.nii.gz")], model=MODEL)
    assert _budgets_sent(server) == ["free", "personal", "personal"]


def test_the_free_task_is_submitted_before_the_paid_ones(make_client, server, scans,
                                                         monkeypatch):
    # The server pays the first task it receives with the free task, whatever
    # budget it names: a paid submit that overtook the free one would turn
    # the free one into a refusal and a second request.
    import time

    from nalvera.upload import Uploader
    real = Uploader.upload

    def first_scan_is_slow(self, path, *, filename, size, display_name):
        if display_name == "a.nii.gz":
            time.sleep(0.3)
        return real(self, path, filename=filename, size=size, display_name=display_name)

    monkeypatch.setattr(Uploader, "upload", first_scan_is_slow)
    server.free_tasks_left = 1
    make_client().run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL)
    assert _budgets_sent(server) == ["free", "personal"]
    assert len(server.calls("POST", "/api/v1/tasks")) == 2


def test_with_a_lab_a_terminal_is_asked_once_for_the_rest(make_client, server, scans):
    server.add_lab(12, "Imaging Lab", 100.0)
    server.free_tasks_left = 1
    client, out = _terminal(make_client, "2\n")
    client.run([scans("a.nii.gz"), scans("b.nii.gz"), scans("c.nii.gz")], model=MODEL)
    assert _budgets_sent(server) == ["free", "lab:12", "lab:12"]
    shown = out.getvalue()
    assert shown.count(messages.text("budget.prompt", default=2)) == 1
    assert messages.text("budget.question_free", free=1, count=2, credits="2.50") in shown


def test_with_a_lab_and_no_terminal_the_choice_is_required_before_any_upload(make_client,
                                                                            server, scans):
    server.add_lab(12, "Imaging Lab", 100.0)
    server.free_tasks_left = 1
    with pytest.raises(BudgetChoiceRequired) as caught:
        make_client().run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL)
    assert {o["budget"] for o in caught.value.options} == {"personal", "lab:12"}
    assert server.calls("POST", "/api/v1/uploads") == []
    assert server.calls("POST", "/api/v1/estimate")          # priced first, by shape


def test_with_a_lab_and_yes_the_rest_goes_to_the_suggested_lab(make_client, server, scans):
    server.add_lab(12, "Small Lab", 10.0)
    server.add_lab(14, "Big Lab", 500.0)
    server.free_tasks_left = 1
    make_client().run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL, yes=True)
    assert _budgets_sent(server) == ["free", "lab:14"]


# ── rule 4: the free task was spent elsewhere ─────────────────────────────────
def _spend_free_tasks_during_upload(client, server, monkeypatch, left):
    real = client._upload_all

    def elsewhere(job, **kwargs):
        server.free_tasks_left = left                    # spent on the website meanwhile
        return real(job, **kwargs)

    monkeypatch.setattr(client, "_upload_all", elsewhere)


def test_no_free_tasks_mid_run_without_a_lab_pays_personal_with_the_same_key(
        make_client, server, scans, monkeypatch):
    server.free_tasks_left = 2
    client = make_client()
    _spend_free_tasks_during_upload(client, server, monkeypatch, left=1)
    tasks = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL)
    assert sorted(t.charged["source"] for t in tasks) == ["free", "personal"]
    submits = server.calls("POST", "/api/v1/tasks")
    keys = [s["headers"]["Idempotency-Key"] for s in submits]
    assert len(submits) == 3 and len(set(keys)) == 2      # the refused one was resent
    assert len(server.uploads) == 2                        # with the same upload


def test_no_free_tasks_mid_run_with_a_lab_and_no_terminal_stops(make_client, server, scans,
                                                                monkeypatch):
    server.add_lab(12, "Imaging Lab", 100.0)
    server.free_tasks_left = 2
    client = make_client()
    _spend_free_tasks_during_upload(client, server, monkeypatch, left=1)
    with pytest.raises(BudgetChoiceRequired) as caught:
        client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL)
    assert len(caught.value.tasks) == 1
    assert _budgets_sent(server) == ["free"]               # never resent as personal
    assert server.personal_credits == 100.0 and server.labs[0]["balance"] == 100.0


def test_no_free_tasks_mid_run_with_a_lab_asks_in_a_terminal(make_client, server, scans,
                                                             monkeypatch):
    server.add_lab(12, "Imaging Lab", 100.0)
    server.free_tasks_left = 2
    client, out = _terminal(make_client, "2\n")
    _spend_free_tasks_during_upload(client, server, monkeypatch, left=1)
    tasks = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL)
    assert sorted(t.charged["source"] for t in tasks) == ["free", "lab"]
    assert messages.text("run.free_ran_out", budget="lab:12") in out.getvalue()


# ── the budget "free" given by the user ───────────────────────────────────────
def test_budget_free_queues_what_the_free_tasks_pay_and_stops(make_client, server, scans):
    server.free_tasks_left = 1
    result = make_client().run([scans("a.nii.gz"), scans("b.nii.gz"), scans("c.nii.gz")],
                               model=MODEL, budget="free")
    assert [t.charged["source"] for t in result] == ["free"]
    assert [f.code for f in result.not_queued] == ["no_free_tasks", "no_free_tasks"]
    assert len(server.uploads) == 1 and server.personal_credits == 100.0


def test_budget_free_with_no_free_task_left_stops_before_uploading(make_client, server,
                                                                   scans):
    with pytest.raises(NoFreeTasks) as caught:
        make_client().run(scans("a.nii.gz"), model=MODEL, budget="free")
    assert caught.value.code == "no_free_tasks"
    assert server.calls("POST", "/api/v1/uploads") == []
    assert cli.exit_status(caught.value) == cli.EXIT_CREDITS


def test_budget_free_refused_mid_run_charges_nothing(make_client, server, scans, monkeypatch):
    server.free_tasks_left = 2
    client = make_client()
    _spend_free_tasks_during_upload(client, server, monkeypatch, left=1)
    result = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL, budget="free")
    assert len(result) == 1 and [f.code for f in result.not_queued] == ["no_free_tasks"]
    assert server.personal_credits == 100.0


def test_the_command_line_exits_4_when_free_tasks_ran_out(env, server, stub_anon, no_sleep):
    from test_cli import run_cli
    server.free_tasks_left = 1
    paths = []
    for name in ("a", "b"):
        path = env / f"{name}.nii.gz"
        from conftest import scan_bytes
        path.write_bytes(scan_bytes(name.encode()))
        paths.append(str(path))
    status, _out, _err = run_cli("run", *paths, "--model", MODEL, "--budget", "free",
                                 "--no-wait")
    assert status == cli.EXIT_CREDITS and len(server.tasks) == 1


# ── generation ────────────────────────────────────────────────────────────────
def test_a_generation_covered_by_a_free_task_sends_free(make_client, server):
    server.add_lab(12, "Imaging Lab", 100.0)
    server.free_tasks_left = 1
    task = make_client().generate("nv_generate_ctmr", "rflow-ct", CT)
    assert server.tasks[task.id]["body"]["budget"] == "free"


def test_a_generation_whose_free_task_was_spent_elsewhere(make_client, server, monkeypatch):
    server.free_tasks_left = 1
    client = make_client()
    real = client._submit
    calls = []

    def spend_first(body, key):
        if not calls:
            server.free_tasks_left = 0
        calls.append(body["budget"])
        return real(body, key)

    monkeypatch.setattr(client, "_submit", spend_first)
    task = client.generate("nv_generate_ctmr", "rflow-ct", CT)
    assert calls == ["free", "personal"] and task.charged["source"] == "personal"
    assert stub_anonymizer.prepared == []
