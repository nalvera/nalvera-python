"""Details of the messages (the pixel-data notice, the retry line, a wait
known from a header, a task without a price, the state in not_ready, the
mapping texts) and two server facts: cancel_requested, and finding a task by
name."""
from __future__ import annotations

import io
import re

from fake_server import error_body
from nalvera import errors, messages
from nalvera.report import ListReporter

MODEL = "totalsegmentator_ct"


# ── the pixel-data notice, once per run ───────────────────────────────────────
def test_the_pixel_notice_is_shown_once_per_run(make_client, server, scans):
    reporter = ListReporter()
    client = make_client(reporter=reporter, output=None)
    client.run([scans(f"s{i}.nii.gz") for i in range(3)], model=MODEL, budget="personal")
    assert reporter.keys().count("run.pixel_notice") == 1
    client.run(scans("t.nii.gz"), model=MODEL, budget="personal")
    assert reporter.keys().count("run.pixel_notice") == 2


# ── a short reason in the retry line ──────────────────────────────────────────
def test_the_retry_line_gives_a_short_reason(make_client, server):
    reporter = ListReporter()
    client = make_client(reporter=reporter, output=None)
    server.inject("GET", "/api/v1/budgets", status=429, headers={"Retry-After": "30"},
                  body=error_body("rate_limited", retry_after=30))
    client.budgets()
    (event,) = [e for e in reporter.events if e.key == "run.retry_wait"]
    assert event.fields["reason"] == messages.text("reason.rate_limited")
    assert event.fields["seconds"] >= 30


def test_a_network_retry_names_what_failed(make_client, server):
    reporter = ListReporter()
    client = make_client(reporter=reporter, output=None)
    server.inject("GET", "/api/v1/budgets", drop=True)
    client.budgets()
    (event,) = [e for e in reporter.events if e.key == "run.retry_wait"]
    assert event.fields["reason"] == messages.text("net.failed")


# ── Retry-After from the header into the message ──────────────────────────────
def test_a_wait_known_only_from_the_header_is_in_the_message(make_client, server):
    server.inject("GET", "/api/v1/budgets", status=503, times=100,
                  headers={"Retry-After": "3000"}, body=error_body("queue_full"))
    client = make_client()
    try:
        client.budgets()
    except errors.ServiceUnavailable as error:
        assert str(error) == messages.text("error.queue_full",
                                           wait=messages.text("time.wait_minutes", m=50))
    else:                                                    # pragma: no cover
        raise AssertionError("no error")


# ── a console that cannot encode a character: see test_edge_files ─────────────

# ── no "None EUR" ─────────────────────────────────────────────────────────────
def test_a_task_without_a_price_is_listed_without_one(cli_dialect, env, server, no_sleep):
    from test_cli import run_cli
    server.models[0]["tasks"][0]["price_eur_per_mvoxel"] = None
    status, out, _err = run_cli("models")
    assert status == 0 and "None" not in out
    assert messages.text("models.task_no_price", id="total", name="Total: full body",
                         mark=messages.text("models.default_mark")) in out


# ── the state label in not_ready ──────────────────────────────────────────────
def test_not_ready_shows_the_state_as_the_waiting_table_does():
    error = errors.from_response(409, {"code": "not_ready", "state": "processing",
                                       "detail": "x"})
    assert str(error) == messages.text("error.not_ready",
                                       state=messages.text("wait.state.processing"))
    assert error.extra["state"] == "processing"            # the code still sees the raw state


# ── the mapping texts the anonymizer reads ────────────────────────────────────
def test_the_mapping_texts_exist_for_the_anonymizer():
    assert messages.has("anonymize.mapping_unreadable")
    assert "{" not in messages.text("anonymize.mapping_unreadable")    # shown as it is
    assert re.findall(r"{(\w+)}", messages.MESSAGES["mapping.notice"]) == ["name"]
    assert messages.has("anonymize.nifti2_unsupported")


# ── server facts: cancel_requested, cancelling twice ──────────────────────────
def test_a_cancelled_running_task_shows_cancelling_until_it_stops(make_client, server, scans):
    server.script = [("processing", None)]
    out = io.StringIO()
    client = make_client(output=out)
    task = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")[0]
    client.task(task)                                       # now processing
    asked = client.cancel(task)
    assert asked.state == "processing" and asked.cancel_requested and asked.cancelling
    again = client.cancel(task)                             # harmless
    assert again.cancel_requested
    done = client.wait([task])
    assert done[0].state == "cancelled"
    lines = out.getvalue().splitlines()
    cancelling = messages.text("wait.state.cancelling")
    assert any(cancelling in line for line in lines)
    assert any(messages.text("wait.state.cancelled") in line for line in lines)


def test_tasks_can_be_found_by_their_platform_name(client, server, scans):
    task = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")[0]
    client.run(scans("b.nii.gz"), model=MODEL, budget="personal")
    found = client.tasks(name=f"{task.code}.nii.gz")
    assert [t.id for t in found] == [task.id]
    assert client.tasks(name="scan-NOTHERE.nii.gz") == []
