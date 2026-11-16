"""More of what the package guards against: the key in no traceback in any
mode, a labelled generation never charged twice, the line a stopped call
keeps, every way an answer can fail, the spending cap with two upload
workers, the wait for an upload slot, tables, a signed-out client, allowed
folders at the moment of use, lab ids from the platform. The real-kernel
cases are in tests/test_notebooks.py, the source archive in
tests/test_safeguards.py."""
from __future__ import annotations

import datetime
import http.client
import json
import os
import re
import ssl
import sys
import threading
import time
import traceback
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests
import urllib3

import nalvera
from fake_server import error_body
from nalvera import _http, auth, display, errors, messages, report
from nalvera import client as client_module
from nalvera.errors import ConfigError, NalveraError, NetworkError
from test_notebook_reporter import FakeHandle, FakeKernel

MODEL = "totalsegmentator_ct"
GEN = {"mode": "image-only", "output_size": [256, 256, 128], "spacing": [1.5, 1.5, 2.0]}
SHAPED = "nvk_" + "A" * 57
#: Code of the libraries a request passes through: none of their frames may
#: stay in a traceback the package raises from a request.
_LIBRARIES = ("/requests/", "/urllib3/", "/http/client.py", "/ssl.py", "/socket.py")


def _charged(server, before=100.0):
    return round(before - server.personal_credits, 2)


class _OneWorker(client_module.ThreadPoolExecutor):
    def __init__(self, max_workers, **kwargs):
        super().__init__(max_workers=1, **kwargs)


@pytest.fixture
def no_env(env, monkeypatch):
    monkeypatch.delenv("NALVERA_EMAIL")
    monkeypatch.delenv("NALVERA_API_KEY")
    return env


@pytest.fixture
def kernel(monkeypatch):
    """A fake Jupyter kernel recording what is displayed (as in
    tests/test_notebook_reporter.py)."""
    state = types.SimpleNamespace(kernel=FakeKernel(), displays=[], handles=[], plain=[])
    shell = types.SimpleNamespace(kernel=state.kernel, parent_header={}, execution_count=1)
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    shown = types.ModuleType("IPython.display")

    def display_(bundle, raw=False, display_id=None):
        if display_id is None:
            state.plain.append(bundle)
            return None
        state.displays.append(bundle)
        handle = FakeHandle(bundle)
        state.handles.append(handle)
        return handle

    shown.display = display_
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", shown)
    return state


@pytest.fixture
def transport(env, server, fake_time):
    creds = auth.Credentials(email=server.email, api_key=auth.ApiKey(server.key))
    made = _http.Transport(server.url, creds, clock=fake_time.clock, sleep=fake_time.sleep,
                           rand=lambda: 0.5)
    yield made
    made.close()


# ── the key is in no traceback ────────────────────────────────────────────────
def _beyond_this_file(tb):
    """The traceback without this test file's own frames (a test may hold
    the key on purpose)."""
    while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
        tb = tb.tb_next
    return tb


def _chain(error):
    seen = []
    while error is not None and all(error is not e for e in seen):
        seen.append(error)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return seen


def _frames(error):
    out = []
    for item in _chain(error):
        tb = _beyond_this_file(item.__traceback__)
        while tb is not None:
            out.append(tb.tb_frame)
            tb = tb.tb_next
    return out


def _every_local(error) -> str:
    """Every variable of every frame of the error and of each exception
    chained to it, as ``repr`` shows it (Sentry, a debugger)."""
    shown = []
    for frame in _frames(error):
        for value in frame.f_locals.values():
            try:
                shown.append(repr(value))
            except Exception:
                pass
    return "\n".join(shown)


def _stdlib(error) -> str:
    """The traceback with every frame's locals, as the standard library
    prints it with ``capture_locals``."""
    return "".join(traceback.TracebackException(
        type(error), error, _beyond_this_file(error.__traceback__),
        capture_locals=True).format())


def _pytest(error) -> str:
    """The traceback as ``pytest -l`` prints it."""
    info = pytest.ExceptionInfo.from_exc_info(
        (type(error), error, _beyond_this_file(error.__traceback__) or error.__traceback__))
    return str(info.getrepr(showlocals=True, style="long"))


def _verbose(error) -> str:
    """The traceback as IPython prints it in ``%xmode Verbose``; nothing
    where IPython is not installed (no notebook can show one there)."""
    try:
        from IPython.core import ultratb
    except ImportError:
        return ""
    try:
        printer = ultratb.VerboseTB(theme_name="nocolor")       # IPython 9
    except TypeError:
        printer = ultratb.VerboseTB(color_scheme="NoColor")     # IPython 8
    text = printer.text(type(error), error, _beyond_this_file(error.__traceback__))
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _rich(error) -> str:
    """The traceback as ``rich`` prints it with ``show_locals``."""
    rich_traceback = pytest.importorskip("rich.traceback")
    from rich.console import Console
    console = Console(record=True, width=200, file=open(os.devnull, "w"))
    console.print(rich_traceback.Traceback.from_exception(
        type(error), error, _beyond_this_file(error.__traceback__), show_locals=True))
    return console.export_text()


def _assert_no_key(error, key) -> None:
    for show in (_every_local, _stdlib, _pytest):
        assert key not in show(error) and key[12:] not in show(error), show.__name__
    library = [f.f_code.co_filename for f in _frames(error)
               if any(part in f.f_code.co_filename.replace("\\", "/") for part in _LIBRARIES)]
    assert library == []


def _break_the_next_answer(monkeypatch, error):
    """The person presses Interrupt (or something else fails) while the
    platform is answering: raised inside http.client, once."""
    real = http.client.HTTPConnection.getresponse

    def broken(self, *args, **kwargs):
        monkeypatch.setattr(http.client.HTTPConnection, "getresponse", real)
        raise error

    monkeypatch.setattr(http.client.HTTPConnection, "getresponse", broken)


def test_the_header_that_holds_the_key_shows_only_its_prefix_and_is_sent_whole(transport,
                                                                              server):
    prepared = requests.Request("GET", server.url + "/api/v1/me",
                                auth=transport._auth).prepare()
    value = prepared.headers["Authorization"]
    assert value == f"Bearer {server.key}" and value.encode("latin-1") == \
        f"Bearer {server.key}".encode("ascii")
    for shown in (repr(value), repr(prepared.headers), repr(dict(prepared.headers))):
        assert server.key not in shown and server.key[:12] in shown
    with pytest.raises(TypeError):
        import pickle
        pickle.dumps(value)
    transport.json("GET", "/api/v1/me")
    server.inject("GET", "/api/v1/me", drop=True, after=True)
    transport.json("GET", "/api/v1/me")                          # sent again after a retry
    assert [c["headers"]["Authorization"] for c in server.calls("GET", "/api/v1/me")] == \
        [f"Bearer {server.key}"] * 3


@pytest.mark.parametrize("through", ["client", "module function"])
def test_an_interrupt_while_the_platform_answers_keeps_no_frame_that_holds_the_key(
        env, server, monkeypatch, through):
    if through == "client":
        made = nalvera.Client(reporter=nalvera.NullReporter())
        call = made.models
    else:
        nalvera.login(reporter=nalvera.NullReporter())        # the key from the environment
        call = nalvera.models
    call()
    _break_the_next_answer(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt) as caught:
        call(refresh=True)
    assert type(caught.value) is KeyboardInterrupt
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    _assert_no_key(caught.value, server.key)
    assert server.key[12:] not in _verbose(caught.value)


def test_another_error_while_the_platform_answers_is_the_packages_without_its_frames(
        env, server, monkeypatch):
    made = nalvera.Client(reporter=nalvera.NullReporter())
    made.me()
    _break_the_next_answer(monkeypatch, ValueError("an unreadable status line"))
    with pytest.raises(NalveraError) as caught:
        made.models(refresh=True)
    assert caught.value.code == "request_failed" and caught.value.extra["cause"] == "ValueError"
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    _assert_no_key(caught.value, server.key)
    assert server.key[12:] not in _verbose(caught.value)


def test_a_traceback_with_rich_and_show_locals_holds_no_key(env, server, monkeypatch):
    made = nalvera.Client(reporter=nalvera.NullReporter())
    made.me()
    _break_the_next_answer(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt) as caught:
        made.models(refresh=True)
    assert server.key[12:] not in _rich(caught.value)


@pytest.mark.parametrize("email", ["jan.kowalski@uniwersytet-łodz.example", "josé@example.org",
                                   "a\x07b@example.org", "a@example.org‮",
                                   "a@exa\nmple.org", "a@exa\x7fmple.org"])
def test_an_email_with_a_control_or_non_ascii_character_is_refused_before_any_request(
        no_env, server, email):
    with pytest.raises(ConfigError) as caught:
        nalvera.login(email, server.key, reporter=nalvera.ListReporter())
    assert caught.value.code == "invalid_email" and server.requests == []
    _assert_no_key(caught.value, server.key)


def test_an_email_from_the_environment_is_checked_too(env, server, monkeypatch):
    monkeypatch.setenv("NALVERA_EMAIL", "jan@uniwersytet-łodz.example")
    with pytest.raises(ConfigError) as caught:
        nalvera.Client(reporter=nalvera.NullReporter())
    assert caught.value.code == "invalid_email" and server.requests == []


# ── a client that is signed out signs nothing ─────────────────────────────────
def test_after_logout_the_client_login_returned_signs_nothing(env, server):
    made = nalvera.login(server.email, server.key, reporter=nalvera.NullReporter())
    nalvera.logout()
    before = len(server.requests)
    with pytest.raises(NalveraError) as caught:
        made.me(refresh=True)
    assert caught.value.code == "signed_out" and len(server.requests) == before
    key = made._transport.credentials.api_key
    with pytest.raises(ConfigError) as revealed:
        key.reveal()
    assert revealed.value.code == "signed_out"
    assert key.prefix == server.key[:12] and server.key not in repr(made)


def test_a_closed_client_signs_nothing_and_closing_twice_is_harmless(make_client, server):
    made = make_client()
    made.me()
    made.close()
    made.close()
    before = len(server.requests)
    for call in (lambda: made.models(refresh=True), lambda: made.tasks(),
                 lambda: made.task("abc")):
        with pytest.raises(NalveraError) as caught:
            call()
        assert caught.value.code == "signed_out"
    assert len(server.requests) == before


def test_a_second_login_signs_out_the_first(no_env, server):
    first = nalvera.login(server.email, server.key, reporter=nalvera.NullReporter())
    second = nalvera.login(server.email, server.key, reporter=nalvera.NullReporter())
    with pytest.raises(NalveraError):
        first.me(refresh=True)
    assert second.me(refresh=True)["email"]


# ── every way an answer can fail ──────────────────────────────────────────────
def test_a_submit_answer_cut_short_is_asked_again_and_charged_once(make_client, server, scans):
    server.inject("POST", "/api/v1/tasks", after=True, truncate=True)
    result = make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert len(server.tasks) == 1 and result[0].charged_now
    assert result.charged_credits == _charged(server) == server.credits_per_task


def test_a_read_cut_short_is_tried_again(make_client, server):
    made = make_client()
    made.me()
    server.inject("GET", "/api/v1/budgets", after=True, truncate=True)
    assert made.budgets()["personal"]["available_credits"] == server.personal_credits
    assert len(server.calls("GET", "/api/v1/budgets")) == 2


def test_answers_cut_short_every_time_end_as_a_network_error(make_client, server):
    made = make_client()
    made.me()
    server.inject("GET", "/api/v1/budgets", after=True, truncate=True, times=100)
    with pytest.raises(NetworkError) as caught:
        made.budgets()
    assert caught.value.extra["reason"] == "failed"
    assert caught.value.__context__ is None


def test_an_answer_that_cannot_be_decoded_may_have_run_and_is_tried_again(transport, server):
    server.inject("GET", "/api/v1/me", status=200, body=b"not gzip",
                  headers={"Content-Encoding": "gzip", "Content-Type": "application/json"})
    transport.json("GET", "/api/v1/me")
    assert transport.last_may_have_run and transport.last_attempts == 2


def _tls_error(reason):
    inner = urllib3.exceptions.SSLError(reason)
    return requests.exceptions.SSLError(
        urllib3.exceptions.MaxRetryError(None, "/api/v1/me", reason=inner))


@pytest.mark.parametrize("reason, shown, repeated", [
    (ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify "
                                     "failed"), "certificate", False),
    (ssl.SSLError(1, "[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed"), "tls",
     True),
    (ssl.SSLEOFError(8, "EOF occurred in violation of protocol"), "tls", True)])
def test_a_certificate_that_fails_is_told_apart_from_a_tls_error_while_reading(
        transport, server, monkeypatch, reason, shown, repeated):
    real = transport._session.request
    calls = []

    def failing_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _tls_error(reason)
        return real(*args, **kwargs)

    monkeypatch.setattr(transport._session, "request", failing_once)
    if repeated:
        transport.json("GET", "/api/v1/me")
        assert transport.last_may_have_run and len(calls) == 2
        return
    with pytest.raises(NetworkError) as caught:
        transport.json("GET", "/api/v1/me")
    assert caught.value.extra["reason"] == shown and len(calls) == 1
    assert not transport.last_may_have_run


def _self_signed(folder: Path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    import ipaddress
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = folder / "cert.pem", folder / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return cert_path, key_path


def test_a_platform_whose_certificate_fails_verification_is_the_certificate_error(
        env, server, tmp_path, fake_time):
    cert, key = _self_signed(tmp_path)

    class Quiet(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            pass                                 # the refused handshake is the point

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert), str(key))
    httpd = Server(("127.0.0.1", 0), Quiet)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        creds = auth.Credentials(email=server.email, api_key=auth.ApiKey(server.key))
        made = _http.Transport(f"https://127.0.0.1:{httpd.server_address[1]}", creds,
                               clock=fake_time.clock, sleep=fake_time.sleep)
        with pytest.raises(NetworkError) as caught:
            made.json("GET", "/api/v1/me")
        assert caught.value.extra["reason"] == "certificate" and not made.last_may_have_run
        made.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_coded_server_error_may_have_run(transport, server):
    server.inject("GET", "/api/v1/me", status=500, after=True, body=error_body("server_error"))
    with pytest.raises(errors.ServerError):
        transport.json("GET", "/api/v1/me")
    assert transport.last_may_have_run


def test_a_503_that_says_the_task_is_queued_counts_its_charge(make_client, server, scans):
    """The platform queued and charged the task, then could not build its
    answer: 503 ``temporarily_unavailable`` with ``"queued": true``. The
    retry with the same key is this run's own task, and its charge counts."""
    server.inject("POST", "/api/v1/tasks", status=503, after=True,
                  body=error_body("temporarily_unavailable", retry_after=1, queued=True))
    result = make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert len(server.tasks) == 1 and len(server.calls("POST", "/api/v1/tasks")) == 2
    assert result[0].charged_now
    assert result.charged_credits == _charged(server) == server.credits_per_task


def test_the_queued_mark_alone_makes_a_coded_503_count_as_run(transport, server):
    server.inject("GET", "/api/v1/me", status=503,
                  body=error_body("temporarily_unavailable", retry_after=1, queued="yes"))
    transport.json("GET", "/api/v1/me")                   # "yes" is no mark: retried, not counted
    assert transport.last_attempts == 2 and not transport.last_may_have_run
    server.inject("GET", "/api/v1/me", status=503,
                  body=error_body("temporarily_unavailable", retry_after=1, queued=True))
    transport.json("GET", "/api/v1/me")
    assert transport.last_attempts == 2 and transport.last_may_have_run


@pytest.mark.parametrize("status, code", [(503, "api_unavailable"), (402, "insufficient_credits")])
def test_a_coded_refusal_did_not_run(transport, server, status, code):
    server.inject("GET", "/api/v1/me", status=status, body=error_body(code))
    with pytest.raises(NalveraError):
        transport.json("GET", "/api/v1/me")
    assert not transport.last_may_have_run


def test_after_a_server_error_on_a_submit_no_other_task_is_sent(make_client, server, scans):
    """The platform queued and charged the first task, then answered 500: its
    charge stays held, and the second worker sends nothing."""
    server.inject("POST", "/api/v1/tasks", status=500, after=True, body=error_body("server_error"))
    server.patch_delay = 0.05
    with pytest.raises(errors.ServerError):
        make_client().run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL,
                          budget="personal", max_credits=10)
    assert len(server.calls("POST", "/api/v1/tasks")) == 1


# ── the spending cap with two upload workers ──────────────────────────────────
@pytest.mark.parametrize("lost", [False, True])
def test_the_cap_holds_with_two_upload_workers_when_a_charge_beats_its_estimate(
        make_client, server, scans, monkeypatch, lost):
    """Estimated 1.0 per task, charged 1.5, cap 2.0."""
    server.credits_per_task = 1.0
    made = make_client()
    real = made._upload_all

    def charge_more(job, **kwargs):
        server.credits_per_task = 1.5
        return real(job, **kwargs)

    monkeypatch.setattr(made, "_upload_all", charge_more)
    real_submit = made._submit
    submits = []

    def slow_first_submit(body, key):
        # The first submit takes a moment, so the other worker, its upload
        # done, comes to its own submit while this one is on its way.
        submits.append(1)
        if len(submits) == 1:
            time.sleep(0.3)
        return real_submit(body, key)

    monkeypatch.setattr(made, "_submit", slow_first_submit)
    if lost:
        server.inject("POST", "/api/v1/tasks", drop=True, after=True)
    server.patch_delay = 0.05                       # both workers upload at the same time
    result = made.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL,
                      budget="personal", max_credits=2.0)
    assert _charged(server) == result.charged_credits == 1.5
    assert len(result) == 1 and [n.code for n in result.not_queued] == ["max_credits_exceeded"]
    assert server.max_active_patches == 2          # the uploads still ran side by side
    assert not [u for u in server.uploads.values() if u["state"] == "ready"]


def test_without_a_cap_two_submits_may_run_at_once(make_client, server, scans):
    result = make_client().run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL,
                               budget="personal")
    assert len(result) == 2 and result.charged_credits == 2 * server.credits_per_task


# ── a labelled generation is never charged twice ──────────────────────────────
def _lose_every_submit_answer(server):
    server.inject("POST", "/api/v1/tasks", drop=True, after=True, times=100)


def test_a_labelled_generation_whose_answers_were_lost_is_returned_by_a_rerun(make_client,
                                                                              server):
    _lose_every_submit_answer(server)
    with pytest.raises(NetworkError) as first:
        make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                               label="phantom-01")
    assert messages.text("generate.may_be_queued") not in str(first.value)
    server.faults.clear()
    again = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                                   label="phantom-01")
    assert len(server.tasks) == 1 and _charged(server) == server.credits_per_task
    assert again.id in server.tasks and again.earlier and not again.charged_now


def test_a_labelled_generation_stopped_during_the_retry_wait_is_returned_by_a_rerun(
        make_client, server):
    server.inject("POST", "/api/v1/tasks", drop=True, after=True)
    made = make_client()

    def interrupted(seconds):
        raise KeyboardInterrupt

    made._transport.sleep = interrupted
    with pytest.raises(KeyboardInterrupt):
        made.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal", label="phantom-01")
    again = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                                   label="phantom-01")
    assert len(server.tasks) == 1 and again.id in server.tasks
    assert _charged(server) == server.credits_per_task


def test_force_and_other_settings_queue_a_new_generation(make_client, server):
    made = make_client()
    first = made.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                          label="phantom-01")
    again = made.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                          label="phantom-01")
    forced = made.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                           label="phantom-01", force=True)
    other = made.generate("nv_generate_ctmr", "rflow-ct", {**GEN, "seed": 8}, budget="personal",
                          label="phantom-01")
    assert again.id == first.id and again.earlier
    assert len({first.id, forced.id, other.id}) == 3 and len(server.tasks) == 3


def _fail(server, task_id):
    server.tasks[task_id]["script"] = [("failed", None)]
    server.tasks[task_id]["step"] = 0


def test_a_labelled_generation_that_failed_is_queued_again(make_client, server):
    made = make_client()
    first = made.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                          label="phantom-01")
    _fail(server, first.id)
    again = made.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                          label="phantom-01")
    assert again.id != first.id and not again.earlier and len(server.tasks) == 2
    _fail(server, again.id)
    third = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                                   label="phantom-01")
    assert third.id not in (first.id, again.id) and len(server.tasks) == 3


def test_a_failed_generation_the_record_does_not_hold_is_not_returned(make_client, server,
                                                                      tmp_path, monkeypatch):
    first = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                                   label="phantom-01")
    _fail(server, first.id)
    monkeypatch.setenv("NALVERA_MAPPING", str(tmp_path / "another_computer.csv"))
    again = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                                   label="phantom-01")
    assert again.id != first.id and len(server.tasks) == 2 and again.charged_now


def test_an_unlabelled_generation_that_may_have_been_queued_says_so(make_client, server):
    _lose_every_submit_answer(server)
    with pytest.raises(NetworkError) as caught:
        make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal")
    assert messages.text("generate.may_be_queued") in str(caught.value)


def test_an_unlabelled_generation_stopped_after_a_lost_answer_says_so(make_client, server, out):
    server.inject("POST", "/api/v1/tasks", drop=True, after=True)
    made = make_client()

    def interrupted(seconds):
        raise KeyboardInterrupt

    made._transport.sleep = interrupted
    with pytest.raises(KeyboardInterrupt):
        made.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal")
    assert messages.text("generate.may_be_queued") in out.getvalue()


def test_an_unlabelled_generation_refused_with_a_code_says_nothing_more(make_client, server):
    server.inject("POST", "/api/v1/tasks", status=402,
                  body=error_body("insufficient_credits", budget="personal"))
    with pytest.raises(errors.InsufficientCredits) as caught:
        make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal")
    assert messages.text("generate.may_be_queued") not in str(caught.value)


def test_the_generation_key_holds_the_account_and_is_a_valid_key():
    keys = {client_module._generation_key(email, "https://nalvera.ai", "phantom-01",
                                          "nv_generate_ctmr", "rflow-ct", "d" * 64, retired)
            for email, retired in (("a@x.org", ()), ("b@x.org", ()), ("a@x.org", ("t1",)))}
    assert len(keys) == 3
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", k) for k in keys)
    assert client_module._generation_key("A@X.org", "https://nalvera.ai", "phantom-01",
                                         "nv_generate_ctmr", "rflow-ct", "d" * 64, ()) in keys


# ── a call that raises or is stopped keeps one line ───────────────────────────
def _final(kernel):
    shown = []
    for number, handle in enumerate(kernel.handles):
        shown.append((handle.updates[-1] if handle.updates else
                      kernel.displays[number])["text/plain"])
    return "\n".join(shown)


def _spend_elsewhere_after_the_estimate(made, server, monkeypatch, credits=1.3):
    real = made._upload_all

    def spend(job, **kwargs):
        server.personal_credits = credits          # one task at 1.25
        return real(job, **kwargs)

    monkeypatch.setattr(made, "_upload_all", spend)


def test_a_run_that_raises_keeps_one_line_of_what_it_queued_and_charged(
        kernel, make_client, server, scans, monkeypatch):
    monkeypatch.setattr(client_module, "ThreadPoolExecutor",
                        lambda max_workers, **kw: _OneWorker(1, **kw))
    made = make_client(reporter=report.NotebookReporter(), output=None)
    _spend_elsewhere_after_the_estimate(made, server, monkeypatch)
    with pytest.raises(nalvera.InsufficientCredits) as caught:
        made.run([scans(f"t{i}.nii.gz") for i in range(3)], model=MODEL, budget="personal")
    [task] = caught.value.tasks
    final = _final(kernel)
    line = messages.text("run.stopped_with_tasks", count=1, ids=task.id, credits="1.25")
    assert final.count(line) == 1 and final.count(task.id) == 1
    assert messages.text("upload.done", name="t0.nii.gz") not in final     # progress stays gone


def test_a_stopped_run_keeps_the_line_and_names_the_call_once(kernel, make_client, server,
                                                              scans, monkeypatch):
    monkeypatch.setattr(client_module, "ThreadPoolExecutor",
                        lambda max_workers, **kw: _OneWorker(1, **kw))
    made = make_client(reporter=report.NotebookReporter(), output=None)
    real_submit = made._submit
    calls = []

    def submit_then_stop(body, key):
        if calls:
            raise KeyboardInterrupt
        calls.append(1)
        return real_submit(body, key)

    monkeypatch.setattr(made, "_submit", submit_then_stop)
    with pytest.raises(nalvera.RunInterrupted) as caught:
        made.run([scans(f"t{i}.nii.gz") for i in range(3)], model=MODEL, budget="personal")
    [task] = caught.value.tasks
    final = _final(kernel)
    assert messages.text("run.stopped_with_tasks", count=1, ids=task.id,
                         credits="1.25") in final
    follow = messages.fragment("follow")
    assert (final + str(caught.value)).count(follow) == 1


def test_a_stopped_run_that_queued_nothing_keeps_no_line(kernel, make_client, server, scans,
                                                         monkeypatch):
    made = make_client(reporter=report.NotebookReporter(), output=None)

    def stop(body, key):
        raise KeyboardInterrupt

    monkeypatch.setattr(made, "_submit", stop)
    with pytest.raises(nalvera.RunInterrupted) as caught:
        made.run(scans("t.nii.gz"), model=MODEL, budget="personal")
    assert caught.value.tasks == [] and "run.stopped_with_tasks" not in _final(kernel)
    assert messages.fragment("follow") in str(caught.value)


def test_plain_output_keeps_each_queued_line_and_adds_none(make_client, server, scans,
                                                           monkeypatch, out):
    monkeypatch.setattr(client_module, "ThreadPoolExecutor",
                        lambda max_workers, **kw: _OneWorker(1, **kw))
    made = make_client()
    _spend_elsewhere_after_the_estimate(made, server, monkeypatch)
    with pytest.raises(nalvera.InsufficientCredits) as caught:
        made.run([scans(f"t{i}.nii.gz") for i in range(3)], model=MODEL, budget="personal")
    [task] = caught.value.tasks
    shown = out.getvalue()
    assert task.id in shown
    assert messages.text("run.stopped_with_tasks", count=1, ids=task.id,
                         credits="1.25") not in shown


def test_a_wait_that_raises_keeps_one_line_of_the_tasks_still_going(kernel, make_client,
                                                                    server, scans):
    made = make_client(reporter=report.NotebookReporter(), output=None)
    tasks = made.run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL, budget="personal")
    server.inject("GET", "/api/v1/tasks", status=401, body=error_body("key_revoked"), times=10)
    with pytest.raises(errors.AuthenticationError):
        made.wait(tasks)
    final = _final(kernel)
    ids = messages.text("list.separator").join(t.id for t in tasks)
    assert final.count(messages.text("wait.stopped", count=2, ids=ids)) == 1


def test_a_generation_stopped_after_its_answer_keeps_one_line(kernel, make_client, server,
                                                              monkeypatch):
    made = make_client(reporter=report.NotebookReporter(), output=None)

    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(made, "_record_generation", stop)
    with pytest.raises(KeyboardInterrupt):
        made.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal", label="phantom-01")
    [task_id] = list(server.tasks)
    assert messages.text("run.stopped_with_tasks", count=1, ids=task_id,
                         credits="1.25") in _final(kernel)


# ── tables ────────────────────────────────────────────────────────────────────
def test_a_key_shaped_value_is_redacted_in_every_cell_and_note():
    assert SHAPED not in display.cell(SHAPED) and display.cell(SHAPED).startswith(SHAPED[:12])
    assert SHAPED not in display.note("found " + SHAPED)
    tables = [display.ModelList([{"id": SHAPED, "name": SHAPED, "tasks": [{"id": SHAPED}]}]),
              display.Account({"email": SHAPED, "key": {"prefix": SHAPED, "name": SHAPED}}),
              display.ScanRecord([{"code": SHAPED, "original_name": SHAPED}]),
              client_module.Task(id=SHAPED, state="queued", name=SHAPED)]
    for table in tables:
        assert SHAPED[12:] not in repr(table) + table._repr_html_()


@pytest.mark.parametrize("char", ["­", "᠎", "\U000e0000", "\U000e0041", "\U000e007f"])
def test_invisible_characters_become_question_marks(char):
    assert errors.printable(f"a{char}b") == "a?b"
    assert display.cell(f"a{char}b") == "a?b"


# ── what a shared notebook carries ────────────────────────────────────────────
def _helpers():
    # Imported when used: the module puts the repository's src/ first on
    # sys.path, and a run of these tests on another copy of the source
    # (a mutation check) must keep importing that copy.
    import examples_helpers
    return examples_helpers


def _notebook(name):
    return json.loads((_helpers().NOTEBOOKS / name).read_text(encoding="utf-8"))["cells"]


@pytest.mark.parametrize("name", ["quickstart.ipynb", "many_scans_and_budgets.ipynb",
                                  "generate_synthetic_ct.ipynb"])
def test_where_a_typed_key_ends_up_is_said_before_the_sign_in_cell(name):
    cells = _notebook(name)
    login_cell = _helpers().NOTEBOOK_LOGIN
    [login] = [n for n, c in enumerate(cells)
               if c["cell_type"] == "code" and login_cell in "".join(c["source"])]
    before = cells[login - 1]
    assert before["cell_type"] == "markdown" and "`nalvera.login()`" in "".join(before["source"])
    assert not [c for c in cells[login + 1:] if c["cell_type"] == "markdown"
                and "`nalvera.login()`" in "".join(c["source"])]


def test_the_many_scans_notebook_ends_by_showing_where_the_record_is():
    cells = _notebook("many_scans_and_budgets.ipynb")
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    assert code[-1] == "nalvera.scans().path"
    assert not [c for c in code if c.strip() == "nalvera.scans()"]


def test_storing_the_key_is_shown_as_the_call_that_asks_for_it():
    text = (_helpers().EXAMPLES / "set_api_key.py").read_text(encoding="utf-8")
    assert "nalvera.login(remember=True)" in text
    assert not re.search(r"nalvera\.login\([^)]+,\s*remember=True\)", text)
    assert messages.fragment("login_remember", messages.PYTHON) == "nalvera.login(remember=True)"


# ── allowed folders at the moment of use ──────────────────────────────────────
@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symbolic links")
def test_a_dicom_file_swapped_for_a_link_after_listing_is_refused_where_it_is_linked(
        env, server, monkeypatch, fake_time):
    pytest.importorskip("pydicom")
    import anonymize_helpers as helpers
    inside, outside = env / "allowed" / "study", env / "private_elsewhere"
    inside.mkdir(parents=True)
    outside.mkdir()
    helpers.write_dicom_series(str(inside), series_number=3, plants=helpers.DICOM_PLANTS,
                               slices=4, seed=3)
    helpers.write_dicom_series(str(outside), series_number=7, plants=helpers.DICOM_PLANTS,
                               slices=4, seed=11)
    real_expand = client_module._expand_paths

    def swap_after_listing(*args, **kwargs):
        found = real_expand(*args, **kwargs)
        victim = inside / sorted(os.listdir(inside))[0]
        victim.unlink()
        victim.symlink_to(outside / sorted(os.listdir(outside))[0])
        return found

    monkeypatch.setattr(client_module, "_expand_paths", swap_after_listing)
    made = nalvera.Client(allowed_folders=[str(env / "allowed")], quiet=True, interactive=False,
                          clock=fake_time.clock, sleep=fake_time.sleep)
    try:
        with pytest.raises(ConfigError) as caught:
            made.run(str(inside), model=MODEL, budget="personal")
    finally:
        made.close()
    assert caught.value.code == "path_not_allowed" and not server.uploads


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symbolic links")
def test_the_converter_links_each_files_checked_real_path(tmp_path, monkeypatch):
    from nalvera.anonymize import AnonymizeError, dicom
    target = tmp_path / "real.dcm"
    target.write_bytes(b"\x00" * 128 + b"DICM")
    link = tmp_path / "link.dcm"
    link.symlink_to(target)
    seen = []
    private = tmp_path / "private"
    private.mkdir()
    # No converter program: convert stops after linking, which is what is tested.
    monkeypatch.setattr(dicom, "find_dcm2niix", lambda scan_folder=None: str(tmp_path / "none"))
    with pytest.raises(AnonymizeError):
        dicom.convert([str(link)], str(private), check=seen.append)
    assert seen == [os.path.realpath(target)]
    assert os.readlink(private / "in" / "000000.dcm") == os.path.realpath(target)


def test_a_refused_file_stops_the_conversion_before_anything_is_linked(tmp_path):
    from nalvera.anonymize import dicom
    (tmp_path / "a.dcm").write_bytes(b"\x00" * 128 + b"DICM")
    private = tmp_path / "private"
    private.mkdir()

    def refuse(path):
        raise ConfigError("outside", code="path_not_allowed")

    with pytest.raises(ConfigError):
        dicom.convert([str(tmp_path / "a.dcm")], str(private), check=refuse)
    assert list(private.iterdir()) == []


# ── lab ids from the platform ─────────────────────────────────────────────────
# The strings below that read like code are hostile answers of a platform,
# inert data that is never run: the point is that they are refused.
@pytest.mark.parametrize("lab_id", ["12\"); import os; os.system(\"x", 12.5, "12", True, -3, None])
def test_a_lab_id_that_is_no_integer_is_refused_before_anything_is_sent(make_client, server,
                                                                       scans, lab_id):
    server.add_lab(12, "Imaging Lab", 100.0)
    server.labs[0]["id"] = lab_id
    with pytest.raises(errors.UnexpectedResponse) as caught:
        make_client().run(scans("a.nii.gz"), model=MODEL)
    assert caught.value.code == "unexpected_response" and not server.uploads
    assert "os.system" not in str(caught.value)


def test_a_suggested_budget_of_another_shape_is_refused(make_client, server):
    server.add_lab(12, "Imaging Lab", 100.0)
    real = server._budgets
    server._budgets = lambda: {**real(), "suggested": 'lab:12"); x("'}
    with pytest.raises(errors.UnexpectedResponse):
        make_client().budgets()


def test_an_estimate_naming_a_budget_of_another_shape_is_refused(make_client, server, scans,
                                                                 monkeypatch):
    made = make_client()
    real = made._transport.json

    def answer(method, path, **kwargs):
        body = real(method, path, **kwargs)
        if path == "/api/v1/estimate":
            body["budgets"] = [{"budget": 'lab:1"); os.system("x', "available_credits": 1.0}]
        return body

    monkeypatch.setattr(made._transport, "json", answer)
    with pytest.raises(errors.UnexpectedResponse):
        made.estimate_run(scans("a.nii.gz"), model=MODEL)


def test_whole_lab_ids_still_work(make_client, server, scans):
    server.add_lab(12, "Imaging Lab", 100.0)
    budgets = make_client().budgets()
    assert "lab:12" in repr(budgets)
    result = make_client().run(scans("a.nii.gz"), model=MODEL, budget="lab:12")
    assert result[0].charged["source"] == "lab"


# ── too_many_uploads: a long wait is never waited in silence ──────────────────
def test_a_long_wait_for_an_upload_slot_stops_at_once_and_says_when(make_client, server,
                                                                    scans, fake_time):
    server.inject("POST", "/api/v1/uploads", status=429, times=100,
                  headers={"Retry-After": "21600"},
                  body=error_body("too_many_uploads", retry_after=21600, limit=4))
    with pytest.raises(errors.RateLimited) as caught:
        make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert caught.value.code == "too_many_uploads" and caught.value.retry_after == 21600
    assert fake_time.slept < 60
    assert errors.describe_wait(21600) in str(caught.value)
    assert str(caught.value).count(errors.describe_wait(21600)) == 1


def test_a_short_wait_for_an_upload_slot_is_announced_then_waited(make_client, server, scans,
                                                                  fake_time):
    server.inject("POST", "/api/v1/uploads", status=429, headers={"Retry-After": "60"},
                  body=error_body("too_many_uploads", retry_after=60, limit=4))
    reporter = nalvera.ListReporter()
    result = make_client(reporter=reporter).run(scans("a.nii.gz"), model=MODEL,
                                                budget="personal")
    assert len(result) == 1 and fake_time.slept >= 60
    [wait] = [e for e in reporter.events if e.key == "run.retry_wait"]
    assert wait.fields["seconds"] >= 60
