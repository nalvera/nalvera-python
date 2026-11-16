"""The transport: HTTPS rule, no redirects, timeouts, pacing, retries."""
from __future__ import annotations

import pytest

from fake_server import error_body, make_key
from nalvera import _http, auth
from nalvera._version import __version__
from nalvera.errors import (
    ConfigError,
    InsufficientCredits,
    InvalidRequest,
    NalveraError,
    NetworkError,
    PermissionDenied,
    RateLimited,
    ServiceUnavailable,
    UnexpectedResponse,
)


@pytest.fixture
def transport(env, server, fake_time):
    creds = auth.Credentials(email=server.email, api_key=auth.ApiKey(server.key))
    t = _http.Transport(server.url, creds, clock=fake_time.clock, sleep=fake_time.sleep,
                        rand=lambda: 0.5)
    yield t
    t.close()


# ── the address ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("url,expected", [
    ("https://nalvera.ai", "https://nalvera.ai"),
    ("https://nalvera.ai/", "https://nalvera.ai"),
    ("https://Nalvera.AI:8443", "https://nalvera.ai:8443"),
    ("http://localhost:8000", "http://localhost:8000"),
    ("http://127.0.0.1:5000/", "http://127.0.0.1:5000"),
])
def test_accepted_addresses(url, expected):
    assert _http.validate_base_url(url) == expected


@pytest.mark.parametrize("url", ["http://nalvera.ai", "http://192.168.1.20", "http://[::1]:80"])
def test_plain_http_is_refused_except_for_local_hosts(url):
    with pytest.raises(ConfigError) as caught:
        _http.validate_base_url(url)
    assert caught.value.code == "insecure_base_url"


@pytest.mark.parametrize("url", ["ftp://nalvera.ai", "nalvera.ai", "https://nalvera.ai/api",
                                 "https://user:pw@nalvera.ai", "https://nalvera.ai?x=1", ""])
def test_malformed_addresses_are_refused(url):
    with pytest.raises(ConfigError) as caught:
        _http.validate_base_url(url)
    assert caught.value.code == "invalid_base_url"


# ── what every request carries ────────────────────────────────────────────────
def test_sign_in_headers_and_user_agent(transport, server):
    transport.json("GET", "/api/v1/me")
    request = server.calls("GET", "/api/v1/me")[0]
    assert request["headers"]["User-Agent"] == f"nalvera-python/{__version__}"
    assert request["headers"]["Authorization"] == f"Bearer {server.key}"
    assert request["headers"]["X-Nalvera-Email"] == server.email
    assert server.key not in request["path"] and server.key not in request["query"]


def test_every_request_has_a_timeout_checks_certificates_and_never_redirects(transport,
                                                                             monkeypatch):
    seen = []
    real = transport._session.request

    def spy(method, url, **kwargs):
        seen.append(kwargs)
        return real(method, url, **kwargs)

    monkeypatch.setattr(transport._session, "request", spy)
    transport.json("GET", "/api/v1/me")
    transport.json("POST", "/api/v1/estimate",
                   json={"model": "totalsegmentator_ct", "shape": [4, 4, 4]}, idempotent=True)
    assert len(seen) == 2
    for kwargs in seen:
        assert kwargs["timeout"] and kwargs["verify"] is True
        assert kwargs["allow_redirects"] is False


def test_a_redirect_is_not_followed(transport, server):
    server.inject("GET", "/api/v1/me", status=302,
                  headers={"Location": "https://elsewhere.example/steal"})
    with pytest.raises(UnexpectedResponse) as caught:
        transport.json("GET", "/api/v1/me")
    assert caught.value.code == "unexpected_redirect"
    assert len(server.calls()) == 1


def test_a_netrc_entry_cannot_replace_the_key(transport, server, tmp_path, monkeypatch):
    netrc = tmp_path / ".netrc"
    netrc.write_text("machine 127.0.0.1 login someone password other\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    transport.json("GET", "/api/v1/me")
    assert server.calls()[-1]["headers"]["Authorization"] == f"Bearer {server.key}"


# ── retries ───────────────────────────────────────────────────────────────────
def test_rate_limited_is_retried_after_retry_after(transport, server, fake_time):
    server.inject("GET", "/api/v1/me", status=429, times=2,
                  body=error_body("rate_limited", retry_after=60), headers={"Retry-After": "60"})
    assert transport.json("GET", "/api/v1/me")["email"]
    assert len(server.calls("GET", "/api/v1/me")) == 3
    assert 120 <= fake_time.slept <= 125


@pytest.mark.parametrize("code", ["temporarily_unavailable", "queue_full", "storage_full"])
def test_busy_server_codes_are_retried(transport, server, code):
    server.inject("GET", "/api/v1/budgets", status=503, body=error_body(code, retry_after=5),
                  headers={"Retry-After": "5"})
    assert "free" in transport.json("GET", "/api/v1/budgets")
    assert len(server.calls("GET", "/api/v1/budgets")) == 2


def test_insufficient_credits_stops_at_once(transport, server, fake_time):
    server.inject("POST", "/api/v1/tasks", status=402,
                  body=error_body("insufficient_credits", budget="personal"))
    with pytest.raises(InsufficientCredits) as caught:
        transport.json("POST", "/api/v1/tasks", json={}, headers={"Idempotency-Key": "k"},
                       idempotent=True)
    assert caught.value.code == "insufficient_credits" and caught.value.status == 402
    assert caught.value.extra["budget"] == "personal"
    assert len(server.calls("POST", "/api/v1/tasks")) == 1 and fake_time.sleeps == []


@pytest.mark.parametrize("status,code,cls", [
    (403, "scope_missing", PermissionDenied), (400, "invalid_request", InvalidRequest),
    (503, "api_unavailable", ServiceUnavailable)])
def test_codes_marked_no_retry_are_not_retried(transport, server, status, code, cls):
    server.inject("GET", "/api/v1/models", status=status, body=error_body(code))
    with pytest.raises(cls):
        transport.json("GET", "/api/v1/models")
    assert len(server.calls("GET", "/api/v1/models")) == 1


def test_total_waiting_is_capped(transport, server, fake_time):
    server.inject("GET", "/api/v1/me", status=429, times=100,
                  body=error_body("rate_limited", retry_after=600),
                  headers={"Retry-After": "600"})
    with pytest.raises(RateLimited) as caught:
        transport.json("GET", "/api/v1/me")
    assert caught.value.retry_after == 600
    assert fake_time.slept <= transport.retry.max_total_wait
    assert len(server.calls("GET", "/api/v1/me")) <= 3


def test_a_dropped_connection_is_retried_for_a_safe_request(transport, server):
    server.inject("GET", "/api/v1/me", drop=True)
    assert transport.json("GET", "/api/v1/me")["email"]
    assert len(server.calls("GET", "/api/v1/me")) == 2


def test_a_dropped_connection_is_not_repeated_for_an_unsafe_request(transport, server):
    server.inject("POST", "/api/v1/uploads", drop=True)
    with pytest.raises(NetworkError) as caught:
        transport.json("POST", "/api/v1/uploads", json={"filename": "a.nii.gz", "size": 5},
                       idempotent=False)
    assert caught.value.code == "network_error"
    assert len(server.calls("POST", "/api/v1/uploads")) == 1


def test_a_proxy_error_page_is_retried_only_when_safe(transport, server):
    server.inject("GET", "/api/v1/me", status=502, body=b"<html>bad gateway</html>")
    assert transport.json("GET", "/api/v1/me")["email"]
    server.inject("POST", "/api/v1/estimate", status=502, body=b"<html>bad gateway</html>")
    with pytest.raises(ServiceUnavailable):
        transport.json("POST", "/api/v1/estimate", json={}, idempotent=False)
    assert len(server.calls("POST", "/api/v1/estimate")) == 1


def test_an_unreachable_server_is_a_network_error(env, fake_time):
    creds = auth.Credentials(email="a@b.org", api_key=auth.ApiKey(make_key()))
    t = _http.Transport("http://127.0.0.1:9", creds, clock=fake_time.clock,
                        sleep=fake_time.sleep)
    with pytest.raises(NetworkError):
        t.json("GET", "/api/v1/me")
    assert fake_time.slept <= t.retry.max_total_wait


def test_an_unknown_code_shows_the_servers_detail(transport, server):
    server.inject("GET", "/api/v1/me", status=418,
                  body={"code": "brand_new_code", "detail": "Something new happened."})
    with pytest.raises(NalveraError) as caught:
        transport.json("GET", "/api/v1/me")
    assert caught.value.code == "brand_new_code"
    assert str(caught.value) == "Something new happened."


# ── pacing ────────────────────────────────────────────────────────────────────
def test_requests_are_paced_below_the_limit(transport, fake_time):
    transport.set_limits({"requests_per_minute": 120, "task_submits_per_minute": 20})
    for _ in range(5):
        transport.json("GET", "/api/v1/me")
    assert len(fake_time.sleeps) == 4
    assert all(s >= 60.0 / 120 for s in fake_time.sleeps)


def test_task_submits_have_their_own_slower_pace(transport, server, fake_time):
    transport.set_limits({"requests_per_minute": 120, "task_submits_per_minute": 20})
    for i in range(3):
        try:
            transport.json("POST", "/api/v1/tasks", json={"model": "x"},
                           headers={"Idempotency-Key": f"k{i}"}, idempotent=True)
        except InvalidRequest:
            pass
    assert fake_time.slept >= 2 * 60.0 / 20


def test_pacer_without_a_limit_does_not_wait(fake_time):
    pacer = _http.Pacer(fake_time.clock, fake_time.sleep)
    for _ in range(10):
        pacer.wait()
    assert fake_time.sleeps == []
