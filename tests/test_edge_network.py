"""Network and server answers."""
from __future__ import annotations

import email.utils
import io

import pytest
import requests

import stub_anonymizer
from conftest import scan_bytes
from fake_server import error_body
from nalvera import _http, auth, cli, messages
from nalvera.errors import (
    AuthenticationError,
    ClientTooOld,
    NetworkError,
    ServiceUnavailable,
    TooLarge,
    UnexpectedResponse,
)
from nalvera.upload import Uploader

MODEL = "totalsegmentator_ct"


@pytest.fixture
def transport(env, server, fake_time):
    creds = auth.Credentials(email=server.email, api_key=auth.ApiKey(server.key))
    t = _http.Transport(server.url, creds, clock=fake_time.clock, sleep=fake_time.sleep,
                        rand=lambda: 0.5)
    yield t
    t.close()


# ── A1: answers that are not JSON ─────────────────────────────────────────────
@pytest.mark.parametrize("status", [502, 503, 504])
def test_a_proxy_page_during_a_restart_is_retried(transport, server, status):
    server.inject("GET", "/api/v1/budgets", status=status, times=2,
                  body=b"<html><body><h1>Bad Gateway</h1></body></html>",
                  headers={"Content-Type": "text/html"})
    assert "free" in transport.json("GET", "/api/v1/budgets")
    assert len(server.calls("GET", "/api/v1/budgets")) == 3


def test_an_html_413_is_too_large(transport, server):
    server.inject("POST", "/api/v1/uploads", status=413,
                  body=b"<html><head><title>413 Request Entity Too Large</title></head></html>",
                  headers={"Content-Type": "text/html"})
    with pytest.raises(TooLarge) as caught:
        transport.json("POST", "/api/v1/uploads", json={"filename": "a.nii.gz", "size": 5},
                       idempotent=False)
    assert caught.value.code == "too_large"
    assert str(caught.value) == messages.text("error.too_large_no_limit")


def test_an_unknown_page_shows_its_status_and_first_line_without_a_key(transport, server):
    server.inject("GET", "/api/v1/me", status=418,
                  body=f"\n<html>\n<p>teapot echo {server.key}</p>\nsecond line\n".encode(),
                  headers={"Content-Type": "text/html"})
    with pytest.raises(UnexpectedResponse) as caught:
        transport.json("GET", "/api/v1/me")
    error = caught.value
    assert error.code == "unexpected_status" and error.status == 418
    assert "418" in str(error) and "teapot" in str(error) and "second line" not in str(error)
    assert server.key not in str(error) and server.key[12:] not in str(error)


def test_a_proxy_page_in_the_middle_of_an_upload_resumes_from_head(transport, server, tmp_path):
    data = scan_bytes(b"patient", 2500)
    path = tmp_path / "scan-AAAAAAAA.nii.gz"
    path.write_bytes(data)
    server.inject("PATCH", "/api/v1/uploads/", status=502, body=b"<html>Bad Gateway</html>")
    Uploader(transport).upload(str(path), filename=path.name, size=len(data), display_name="x")
    up = next(iter(server.uploads.values()))
    assert bytes(up["data"]) == data and server.calls("HEAD")


# ── A2: a connection that drops in the middle of a part ───────────────────────
def test_a_dropped_part_continues_from_the_stored_offset_never_from_zero(transport, server,
                                                                         tmp_path):
    data = scan_bytes(b"patient", 3500)
    path = tmp_path / "scan-BBBBBBBB.nii.gz"
    path.write_bytes(data)
    # The server stores the first part (bytes 0-999), then the answer is lost.
    server.inject("PATCH", "/api/v1/uploads/", drop=True, after=True, times=1)
    Uploader(transport).upload(str(path), filename=path.name, size=len(data), display_name="x")
    offsets = [int(p["headers"]["Upload-Offset"]) for p in server.calls("PATCH")]
    # HEAD said 1000: the upload goes on from there; byte 0 is never sent again.
    assert offsets == [0, 1000, 2000, 3000]
    assert len(server.calls("HEAD")) == 1
    up = next(iter(server.uploads.values()))
    assert bytes(up["data"]) == data


# ── A3: Retry-After ───────────────────────────────────────────────────────────
def test_retry_after_as_an_http_date_is_measured_by_the_servers_clock(transport, server,
                                                                      fake_time):
    server_now = 1_800_000_000                     # unrelated to this computer's clock
    server.inject("GET", "/api/v1/me", status=503, body=error_body("temporarily_unavailable"),
                  headers={"Date": email.utils.formatdate(server_now, usegmt=True),
                           "Retry-After": email.utils.formatdate(server_now + 20,
                                                                 usegmt=True)})
    transport.json("GET", "/api/v1/me")
    assert 20 <= fake_time.slept <= 22


def test_a_missing_retry_after_falls_back_to_backoff(transport, server, fake_time):
    server.inject("GET", "/api/v1/me", status=503, body=error_body("queue_full"))
    transport.json("GET", "/api/v1/me")
    assert 0 < fake_time.slept <= transport.retry.base_delay * 2


def test_a_huge_retry_after_stops_at_once_and_says_when(transport, server, fake_time):
    server.inject("GET", "/api/v1/me", status=503, headers={"Retry-After": "7200"},
                  body=error_body("temporarily_unavailable"))
    with pytest.raises(ServiceUnavailable) as caught:
        transport.json("GET", "/api/v1/me")
    assert fake_time.sleeps == [] and caught.value.retry_after == 7200
    assert _http.describe_wait(7200) in str(caught.value)
    assert len(server.calls("GET", "/api/v1/me")) == 1


def test_the_header_wait_reaches_the_message(transport, server):
    server.inject("GET", "/api/v1/me", status=429, times=100, headers={"Retry-After": "1200"},
                  body={"detail": "slow down"})                        # no code, no retry_after
    with pytest.raises(Exception) as caught:
        transport.json("GET", "/api/v1/me")
    assert caught.value.code == "rate_limited"
    assert str(caught.value) == messages.text("error.rate_limited", wait="20 min")


# ── A5: proxies and certificate bundles from the environment ──────────────────
def test_the_proxy_and_the_ca_bundle_from_the_environment_apply(env, server, monkeypatch,
                                                                tmp_path):
    bundle = tmp_path / "hospital-ca.pem"
    bundle.write_text("not used: the connection is intercepted below")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.hospital.example:3128")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    seen = {}

    def send(self, request, **kwargs):
        seen.update(kwargs)
        seen["authorization"] = request.headers.get("Authorization")
        response = requests.Response()
        response.status_code, response._content = 200, b'{"email": "x"}'
        response.request = request
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    creds = auth.Credentials(email=server.email, api_key=auth.ApiKey(server.key))
    t = _http.Transport("https://nalvera.example", creds)
    t.json("GET", "/api/v1/me")
    assert seen["verify"] == str(bundle)
    assert seen["proxies"].get("https") == "http://proxy.hospital.example:3128"
    assert seen["authorization"] == f"Bearer {server.key}"


def test_a_certificate_failure_says_how_to_point_to_a_bundle(transport, monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.exceptions.SSLError("certificate verify failed")

    monkeypatch.setattr(transport._session, "request", refuse)
    with pytest.raises(NetworkError) as caught:
        transport.json("GET", "/api/v1/me")
    assert caught.value.extra["reason"] == "certificate"
    assert "REQUESTS_CA_BUNDLE" in messages.text("net.certificate")
    assert messages.text("net.certificate") in str(caught.value)


# ── A8: the API switched off, or the key dies, during a run ───────────────────
@pytest.mark.parametrize("status,code", [(401, "key_revoked"), (401, "key_expired"),
                                         (403, "api_disabled"), (503, "api_unavailable")])
def test_the_key_dying_mid_run_stops_and_names_the_queued_tasks(make_client, server, scans,
                                                                monkeypatch, status, code):
    client = make_client()
    real = client._submit

    def after_first(body, key):
        answer = real(body, key)
        server.auth_refusal = (status, code, {})
        return answer

    monkeypatch.setattr(client, "_submit", after_first)
    with pytest.raises(Exception) as caught:
        client.run([scans(f"s{i}.nii.gz") for i in range(4)], model=MODEL, budget="personal")
    assert caught.value.code == code
    assert len(caught.value.tasks) >= 1 and len(caught.value.tasks) == len(server.tasks)
    expected = cli.EXIT_UNAVAILABLE if code == "api_unavailable" else cli.EXIT_AUTH
    assert cli.exit_status(caught.value) == expected


def test_the_command_line_names_the_tasks_queued_before_the_key_died(env, server, stub_anon,
                                                                     no_sleep, monkeypatch):
    from nalvera.client import Client
    real = Client._submit

    def after_first(self, body, key):
        answer = real(self, body, key)
        server.auth_refusal = (401, "key_revoked", {})
        return answer

    monkeypatch.setattr(Client, "_submit", after_first)
    files = []
    for name in ("a", "b", "c"):
        path = env / f"{name}.nii.gz"
        path.write_bytes(scan_bytes(name.encode()))
        files.append(str(path))
    out, err = io.StringIO(), io.StringIO()
    status = cli.main(["run", *files, "--model", MODEL, "--budget", "personal", "--no-wait"],
                      stdin=io.StringIO(), stdout=out, stderr=err)
    assert status == cli.EXIT_AUTH
    for task_id in server.tasks:
        assert task_id in out.getvalue()


def test_an_expired_key_mid_upload_is_not_retried(transport, server, tmp_path):
    data = scan_bytes(b"patient", 2500)
    path = tmp_path / "scan-CCCCCCCC.nii.gz"
    path.write_bytes(data)
    server.inject("PATCH", "/api/v1/uploads/", status=401, body=error_body("key_expired"),
                  times=10)
    with pytest.raises(AuthenticationError) as caught:
        Uploader(transport).upload(str(path), filename=path.name, size=len(data),
                                   display_name="x")
    assert caught.value.code == "key_expired"
    assert len(server.calls("PATCH")) == 1


# ── A9: a package too old for the platform ────────────────────────────────────
def test_a_package_too_old_stops_before_any_work(client, server, scans):
    server.minimum_client_version = "9.0.0"
    with pytest.raises(ClientTooOld) as caught:
        client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert caught.value.code == "client_too_old"
    assert caught.value.extra["minimum_version"] == "9.0.0"
    assert stub_anonymizer.prepared == [] and server.calls("POST", "/api/v1/uploads") == []
    with pytest.raises(ClientTooOld):
        client.generate("nv_generate_ctmr", "rflow-ct", {"mode": "image-only"},
                        budget="personal")
    assert server.calls("POST", "/api/v1/tasks") == []
    assert cli.exit_status(caught.value) == cli.EXIT_AUTH


def test_an_older_minimum_version_lets_the_run_go(client, server, scans):
    server.minimum_client_version = "0.0.9"
    assert client.run(scans("a.nii.gz"), model=MODEL, budget="personal")


# ── A10: fields and codes from a newer server ─────────────────────────────────
def test_unknown_fields_are_kept_aside_and_ignored(client, server, scans, monkeypatch):
    real = server._task_object

    def newer(record):
        return {**real(record), "brand_new_field": {"x": 1}, "state_detail": "warming"}

    monkeypatch.setattr(server, "_task_object", newer)
    task = client.run(scans("a.nii.gz"), model=MODEL, budget="personal")[0]
    assert task.raw["brand_new_field"] == {"x": 1}
    assert client.wait([task])[0].state == "completed"
