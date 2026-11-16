"""A lost answer's charge is counted, an upload the package
will not use is deleted, ``encryption_unavailable`` is waited out, and the
key pin accepts the one key the platform lists."""
from __future__ import annotations

import base64
import hashlib
import io

import pytest

import nvlt_reference
from fake_server import error_body
from nalvera import _http, auth, encrypt, messages
from nalvera import client as client_module
from nalvera.upload import Uploader

MODEL, TASK = "totalsegmentator_ct", "total"
GEN = {"mode": "image-only", "output_size": [256, 256, 128], "spacing": [1.5, 1.5, 2.0]}
PROXY_PAGE = b"<html><body><h1>502 Bad Gateway</h1></body></html>"


def _charged(server, before=100.0):
    return round(before - server.personal_credits, 2)


def _lose_the_next_submit_answer(server, how="drop"):
    """The platform queues and charges the task; its answer never arrives."""
    if how == "drop":
        server.inject("POST", "/api/v1/tasks", drop=True, after=True)
    else:
        server.inject("POST", "/api/v1/tasks", status=502, after=True, body=PROXY_PAGE,
                      headers={"Content-Type": "text/html"})


def _die_after_the_submit(client, monkeypatch):
    """The process dies after the server queued the task, before the answer
    was recorded (an earlier run, for the recovery cases)."""
    real = client._submit

    def submit_then_die(body, key):
        real(body, key)
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "_submit", submit_then_die)


# ── the transport says whether an attempt may have run ────────────────────────
@pytest.fixture
def transport(env, server, fake_time):
    creds = auth.Credentials(email=server.email, api_key=auth.ApiKey(server.key))
    t = _http.Transport(server.url, creds, clock=fake_time.clock, sleep=fake_time.sleep,
                        rand=lambda: 0.5)
    yield t
    t.close()


def test_a_clean_request_ran_nowhere_else(transport):
    transport.json("GET", "/api/v1/me")
    assert transport.last_may_have_run is False


@pytest.mark.parametrize("how", ["drop", "proxy"])
def test_a_lost_answer_may_have_run(transport, server, how):
    server.inject("GET", "/api/v1/me", drop=how == "drop", after=True,
                  **({} if how == "drop" else {"status": 502, "body": PROXY_PAGE}))
    transport.json("GET", "/api/v1/me")
    assert transport.last_may_have_run is True and transport.last_attempts == 2


def test_a_coded_refusal_did_not_run(transport, server):
    server.inject("GET", "/api/v1/me", status=503, headers={"Retry-After": "1"},
                  body=error_body("temporarily_unavailable", retry_after=1))
    transport.json("GET", "/api/v1/me")
    assert transport.last_may_have_run is False and transport.last_attempts == 2


def test_the_next_request_starts_clean(transport, server):
    server.inject("GET", "/api/v1/me", drop=True, after=True)
    transport.json("GET", "/api/v1/me")
    transport.json("GET", "/api/v1/me")
    assert transport.last_may_have_run is False


# ── run: the three cases ──────────────────────────────────────────────────────
@pytest.mark.parametrize("how", ["drop", "proxy"])
def test_run_counts_the_charge_of_its_own_lost_answer(make_client, server, scans, out, how):
    _lose_the_next_submit_answer(server, how)
    result = make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert len(server.tasks) == 1 and result[0].replayed
    assert result.charged_credits == _charged(server) == server.credits_per_task
    assert result[0].charged_now
    assert messages.text("run.unexpected_replay", name="a.nii.gz", task_id=result[0].id,
                         state=messages.text("wait.state.queued")) not in out.getvalue()


def test_run_recovering_an_earlier_runs_task_counts_nothing(make_client, server, scans,
                                                            monkeypatch):
    path = scans("a.nii.gz")
    first = make_client()
    _die_after_the_submit(first, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        first.run(path, model=MODEL, budget="personal", label="P012 baseline")
    credits = server.personal_credits
    again = make_client().run(path, model=MODEL, budget="personal", label="P012 baseline")
    assert again[0].replayed and not again[0].charged_now
    assert again.charged_credits == 0 and server.personal_credits == credits


def test_run_recovering_with_a_lost_answer_counts_the_charge(make_client, server, scans,
                                                             monkeypatch):
    path = scans("a.nii.gz")
    first = make_client()
    _die_after_the_submit(first, monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        first.run(path, model=MODEL, budget="personal", label="P012 baseline")
    _lose_the_next_submit_answer(server)
    again = make_client().run(path, model=MODEL, budget="personal", label="P012 baseline")
    assert len(server.tasks) == 1 and again[0].replayed
    # The cap errs on the safe side: this charge may be the earlier run's.
    assert again.charged_credits == server.credits_per_task and again[0].charged_now


def test_the_spending_cap_holds_after_a_lost_answer(make_client, server, scans, monkeypatch):
    """Estimated 1.0 per task, charged 1.5: the first answer is lost, and its
    charge, counted, leaves no room for the second task under a cap of 2."""
    server.credits_per_task = 1.0
    client = make_client()
    real = client._upload_all

    def charge_more(job, **kwargs):
        server.credits_per_task = 1.5
        return real(job, **kwargs)

    monkeypatch.setattr(client, "_upload_all", charge_more)
    # One upload at a time, so the first task is settled before the second asks.
    monkeypatch.setattr(client_module, "ThreadPoolExecutor",
                        lambda max_workers, **kw: _Pool(1, **kw))
    _lose_the_next_submit_answer(server)
    result = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL,
                        budget="personal", max_credits=2.0)
    assert len(result) == 1 and [n.code for n in result.not_queued] == ["max_credits_exceeded"]
    assert result.charged_credits == 1.5 and _charged(server) == 1.5


class _Pool(client_module.ThreadPoolExecutor):
    def __init__(self, max_workers, **kwargs):
        super().__init__(max_workers=max_workers, **kwargs)


# ── generate: the same rule ───────────────────────────────────────────────────
def test_generate_counts_the_charge_of_its_own_lost_answer(make_client, server, out):
    _lose_the_next_submit_answer(server)
    task = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal")
    assert len(server.tasks) == 1 and task.replayed and task.charged_now
    charge = client_module._charge_text(task.charged)
    assert messages.text("run.queued", name=task.display_name, task_id=task.id,
                         charge=charge) in out.getvalue()


def test_generate_answered_with_an_earlier_requests_task_counts_nothing(make_client, server,
                                                                        out, monkeypatch):
    earlier = server.add_task("queued")
    server.idempotency["fixed-key"] = earlier
    monkeypatch.setattr(client_module.generate_mod, "new_idempotency_key", lambda: "fixed-key")
    task = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal")
    assert task.id == earlier and task.replayed and not task.charged_now
    assert messages.text("run.unexpected_replay", name=task.display_name, task_id=earlier,
                         state=messages.text("wait.state.queued")) in out.getvalue()


def test_generate_with_both_counts_the_charge(make_client, server, monkeypatch):
    earlier = server.add_task("queued")
    server.idempotency["fixed-key"] = earlier
    monkeypatch.setattr(client_module.generate_mod, "new_idempotency_key", lambda: "fixed-key")
    _lose_the_next_submit_answer(server)
    task = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal")
    assert task.id == earlier and task.charged_now


# ── an upload the package will not use is deleted ─────────────────────────────
def test_an_upload_left_out_by_the_cap_is_deleted(make_client, server, scans, monkeypatch):
    server.credits_per_task = 1.0
    client = make_client()
    real = client._upload_all

    def charge_more(job, **kwargs):
        server.credits_per_task = 1.5
        return real(job, **kwargs)

    monkeypatch.setattr(client, "_upload_all", charge_more)
    monkeypatch.setattr(client_module, "ThreadPoolExecutor",
                        lambda max_workers, **kw: _Pool(1, **kw))
    result = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL,
                        budget="personal", max_credits=2.0)
    assert len(result.not_queued) == 1
    states = sorted(u["state"] for u in server.uploads.values())
    assert states in (["cancelled", "consumed"], ["consumed"])
    assert not [u for u in server.uploads.values() if u["state"] in ("ready", "receiving")]


def test_an_upload_refused_for_credits_is_deleted(make_client, server, scans, monkeypatch):
    client = make_client()
    real = client._upload_all

    def spend_elsewhere(job, **kwargs):
        server.personal_credits = 1.3                    # one more task at 1.25
        return real(job, **kwargs)

    monkeypatch.setattr(client, "_upload_all", spend_elsewhere)
    with pytest.raises(client_module.InsufficientCredits):
        client.run([scans(f"t{i}.nii.gz") for i in range(3)], model=MODEL, budget="personal")
    assert not [u for u in server.uploads.values() if u["state"] in ("ready", "receiving")]


def test_a_free_only_run_deletes_the_upload_it_could_not_queue(make_client, server, scans,
                                                               monkeypatch):
    server.free_tasks_left = 2
    client = make_client()
    real = client._upload_all

    def spend_a_free_task(job, **kwargs):
        server.free_tasks_left = 1                     # spent elsewhere meanwhile
        return real(job, **kwargs)

    monkeypatch.setattr(client, "_upload_all", spend_a_free_task)
    monkeypatch.setattr(client_module, "ThreadPoolExecutor",
                        lambda max_workers, **kw: _Pool(1, **kw))
    result = client.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL, budget="free")
    assert len(result) == 1 and len(result.not_queued) == 1
    assert not [u for u in server.uploads.values() if u["state"] in ("ready", "receiving")]


# ── encryption_unavailable: waited out, then the upload goes on ───────────────
def test_encryption_unavailable_is_waited_out_and_the_upload_resumes(make_client, server,
                                                                     scans, fake_time):
    server.enable_encryption(required=True)
    server.inject("PATCH", "/api/v1/uploads/", status=503, times=1,
                  headers={"Retry-After": "300"},
                  body=error_body("encryption_unavailable", retry_after=300))
    result = make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert len(result) == 1 and fake_time.slept >= 300
    (up,) = [u for u in server.uploads.values()]
    assert up["state"] == "consumed" and up["plaintext"]


def test_a_wait_for_encryption_says_why(make_client, server, scans):
    server.enable_encryption(required=True)
    server.inject("PATCH", "/api/v1/uploads/", status=503, times=1,
                  headers={"Retry-After": "300"},
                  body=error_body("encryption_unavailable", retry_after=300))
    out = io.StringIO()
    make_client(output=out).run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert messages.text("reason.encryption_unavailable") in out.getvalue()


def test_after_waiting_long_enough_the_error_names_the_wait_once(make_client, server, scans):
    server.enable_encryption(required=True)
    server.inject("PATCH", "/api/v1/uploads/", status=503, times=100,
                  headers={"Retry-After": "300"},
                  body=error_body("encryption_unavailable", retry_after=300))
    with pytest.raises(encrypt.EncryptionError) as caught:
        make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    text = str(caught.value)
    assert caught.value.code == "encryption_unavailable"
    assert text.count(messages.text("time.wait_minutes", m=5)) == 1
    assert messages.text("error.try_again_in", wait=messages.text("time.wait_minutes", m=5)) \
        not in text
    assert not [u for u in server.uploads.values() if u["state"] in ("ready", "receiving")]


def test_an_encryption_error_found_on_this_computer_reads_whole():
    with pytest.raises(encrypt.EncryptionError) as caught:
        encrypt.platform_key({"available": False}, "https://nalvera.ai")
    assert messages.text("time.wait_unknown") in str(caught.value)


def test_the_upload_retries_name_encryption_unavailable():
    from nalvera import errors
    assert "encryption_unavailable" in Uploader.RETRY_CODES
    assert "encryption_unavailable" not in errors.RETRYABLE_CODES


# ── the key pin and the one key /me lists ─────────────────────────────────────
def test_the_one_listed_key_is_used_when_it_is_trusted():
    _old_private, old = nvlt_reference.new_key_pair()
    _private, current = nvlt_reference.new_key_pair()
    block = {"available": True, "required": False, "format": "NVLT/2",
             "keys": [{"public_key": base64.b64encode(current).decode(),
                       "sha256": hashlib.sha256(current).hexdigest()}]}
    pins = (hashlib.sha256(current).hexdigest(), hashlib.sha256(old).hexdigest())
    assert encrypt.platform_key(block, "https://nalvera.ai", trusted=pins) == current
    with pytest.raises(encrypt.EncryptionError):
        encrypt.platform_key(block, "https://nalvera.ai",
                             trusted=(hashlib.sha256(old).hexdigest(),))
