"""The module functions: nalvera.login, nalvera.logout and one
function per public Client method, signed in once per process."""
from __future__ import annotations

import inspect
import io
import json
import stat
import sys
import traceback

import pytest

import nalvera
from fake_server import make_key
from nalvera import _session, messages
from nalvera.errors import AuthenticationError, ConfigError, NoCredentials

MODEL = "totalsegmentator_ct"


class AskingReporter(nalvera.ListReporter):
    """A reporter that can ask: answers from a list, notes which were hidden."""

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)
        self.asked = []

    def can_ask(self):
        return True

    def ask(self, prompt, *, hidden=False):
        self.asked.append((prompt, hidden))
        return self.answers.pop(0) if self.answers else None


@pytest.fixture
def no_env(env, monkeypatch):
    monkeypatch.delenv("NALVERA_EMAIL")
    monkeypatch.delenv("NALVERA_API_KEY")
    return env


# ── the surface ───────────────────────────────────────────────────────────────
def test_every_public_client_method_has_a_module_function_with_its_signature_and_doc():
    public = {n for n, m in inspect.getmembers(nalvera.Client, inspect.isfunction)
              if not n.startswith("_")}
    assert public - set(_session.NOT_DELEGATED) == set(_session.DELEGATED) | set(_session.OWN)
    for name in public - set(_session.NOT_DELEGATED):
        function = getattr(nalvera, name)
        method = getattr(nalvera.Client, name)
        expected = inspect.signature(method)
        expected = expected.replace(parameters=list(expected.parameters.values())[1:])
        assert inspect.signature(function) == expected, name
        assert function.__doc__ == method.__doc__ and function.__doc__, name
        assert name in nalvera.__all__
    for name in ("login", "logout"):
        assert name in nalvera.__all__ and getattr(nalvera, name).__doc__


def test_the_docstrings_people_read_name_every_argument_and_no_command():
    for name in ("login", "run", "wait", "download", "generate"):
        function = getattr(nalvera, name)
        doc = inspect.getdoc(function)
        for parameter in inspect.signature(function).parameters:
            if parameter != "client_options":
                assert f"``{parameter}``" in doc, (name, parameter)
        assert "Example::" in doc, name
        assert "nalvera login" not in doc and "--" not in doc, name
    module_doc = nalvera.__doc__
    assert "nalvera.login(" in module_doc and "nalvera login" not in module_doc


def test_importing_the_package_loads_no_http_library(tmp_path):
    import subprocess
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "src"
    code = ("import sys, nalvera; assert 'requests' not in sys.modules; "
            "assert 'IPython' not in sys.modules; print('ok')")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(src), "PATH": ""}, cwd=tmp_path, timeout=60)
    assert done.returncode == 0 and "ok" in done.stdout, done.stderr


# ── login ─────────────────────────────────────────────────────────────────────
def test_login_checks_the_key_with_one_request_and_signs_in_the_process(no_env, server, out):
    client = nalvera.login(server.email, server.key, output=out)
    assert len(server.calls("GET", "/api/v1/me")) == 1
    assert isinstance(client, nalvera.Client) and _session.current() is client
    listed = nalvera.models()
    assert listed and server.calls("GET", "/api/v1/models")[0]["headers"]["Authorization"] \
        == f"Bearer {server.key}"


def test_a_failed_login_leaves_the_earlier_sign_in_in_place(no_env, server, out):
    first = nalvera.login(server.email, server.key, output=out)
    server.auth_refusal = (401, "invalid_credentials", {})
    with pytest.raises(AuthenticationError):
        nalvera.login(server.email, make_key(), output=out)
    assert _session.current() is first


def test_a_second_login_replaces_and_closes_the_first(no_env, server, out, monkeypatch):
    first = nalvera.login(server.email, server.key, output=out)
    closed = []
    monkeypatch.setattr(first, "close", lambda: closed.append(True))
    second = nalvera.login(server.email, server.key, output=out)
    assert closed == [True] and _session.current() is second


@pytest.mark.parametrize("kwargs", [{"email": "a@b.org"}, {"api_key": "nvk_x"}])
def test_login_with_one_of_the_two_is_a_usage_error(no_env, server, kwargs):
    with pytest.raises(ConfigError) as caught:
        nalvera.login(**kwargs)
    assert caught.value.code == "login_needs_both" and server.requests == []


def test_login_without_arguments_asks_when_it_can_the_key_hidden(no_env, server):
    reporter = AskingReporter([server.email, server.key])
    nalvera.login(reporter=reporter)
    assert [hidden for _prompt, hidden in reporter.asked] == [False, True]
    assert reporter.keys() == ["login.ok_platform"]
    assert len(server.calls("GET", "/api/v1/me")) == 1


def test_login_without_arguments_and_nobody_to_ask_uses_the_environment(env, server):
    reporter = nalvera.ListReporter()
    client = nalvera.login(reporter=reporter)
    assert client.email == server.email and _session.current() is client


def test_login_without_anything_found_raises_the_steps(no_env, server):
    with pytest.raises(NoCredentials):
        nalvera.login(reporter=nalvera.ListReporter())
    assert server.requests == []


def test_remember_stores_and_logout_forget_removes(no_env, server, out):
    nalvera.login(server.email, server.key, remember=True, output=out)
    stored = no_env / "config" / "credentials.json"
    assert json.loads(stored.read_text())["email"] == server.email
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600
    nalvera.logout()
    assert stored.exists()                                      # forget=False keeps it
    again = nalvera.login(reporter=nalvera.ListReporter())     # from what was stored
    assert again.email == server.email
    assert nalvera.logout(forget=True) is True
    assert not stored.exists()
    with pytest.raises(NoCredentials):
        nalvera.me()


def test_logout_closes_the_client_and_the_next_function_signs_in_again(env, server, out,
                                                                       monkeypatch):
    client = nalvera.login(server.email, server.key, output=out)
    closed = []
    monkeypatch.setattr(client, "close", lambda: closed.append(True))
    nalvera.logout()
    assert closed == [True]
    assert nalvera.me()["email"].lower() == server.email.lower()                # the environment
    assert _session.current() is not client


def test_without_login_the_first_function_uses_the_environment(env, server):
    assert nalvera.me()["email"].lower() == server.email.lower()
    assert isinstance(_session.current(), nalvera.Client)


def test_the_process_client_is_closed_when_the_interpreter_exits(env, server, out,
                                                                 monkeypatch):
    client = nalvera.login(server.email, server.key, output=out)
    closed = []
    monkeypatch.setattr(client, "close", lambda: closed.append(True))
    assert _session._closer_registered
    _session._close_at_exit()
    assert closed == [True] and _session._client is None


# ── the key's protections on the new paths ────────────────────────────────────
def _locals_text(error: BaseException) -> str:
    """What a traceback printed with locals shows of the package's frames
    (the test's own frame holds the key on purpose)."""
    def ours(filename: str) -> bool:
        return "/nalvera/" in filename.replace("\\", "/")

    frames = []
    tb = error.__traceback__
    while tb is not None:
        if ours(tb.tb_frame.f_code.co_filename):
            frames += [repr(v) for v in tb.tb_frame.f_locals.values()]
        tb = tb.tb_next
    shown, keep = [], False
    for line in traceback.TracebackException.from_exception(
            error, capture_locals=True).format():
        if line.lstrip().startswith("File "):
            keep = ours(line)
        if keep:
            shown.append(line)
    return "".join(shown) + "".join(frames)


def test_the_key_is_in_no_repr_event_or_display(no_env, server):
    reporter = nalvera.ListReporter()
    client = nalvera.login(server.email, server.key, reporter=reporter)
    shown = [repr(client), str(client), repr(nalvera.me()), nalvera.me()._repr_html_(),
             repr(nalvera.limits()), nalvera.budgets()._repr_html_()]
    shown += [e.text + json.dumps(e.fields, default=str) for e in reporter.events]
    for text in shown:
        assert server.key not in text and server.key[12:] not in text


def test_the_key_is_in_no_exception_or_traceback_locals(no_env, server):
    server.auth_refusal = (401, "invalid_credentials", {})
    with pytest.raises(AuthenticationError) as caught:
        nalvera.login(server.email, server.key, reporter=nalvera.ListReporter())
    assert server.key not in str(caught.value) + repr(caught.value)
    assert server.key not in _locals_text(caught.value)
    # A last character other than the real one: the checksum fails every time.
    wrong = server.key[:-1] + ("x" if server.key[-1] != "x" else "y")
    with pytest.raises(ConfigError) as malformed:
        nalvera.login(server.email, wrong, reporter=nalvera.ListReporter())
    assert server.key[12:-1] not in _locals_text(malformed.value)


def test_an_asked_key_is_in_no_traceback_locals(no_env, server):
    server.auth_refusal = (401, "invalid_credentials", {})
    reporter = AskingReporter([server.email, server.key])
    with pytest.raises(AuthenticationError) as caught:
        nalvera.login(reporter=reporter)
    assert server.key not in _locals_text(caught.value)


def test_the_python_dialect_is_the_default_and_messages_name_python():
    assert messages.dialect() == messages.PYTHON
    text = messages.text("error.no_credentials")
    assert "nalvera.login(" in text and "nalvera login" not in text


def test_login_texts_go_to_the_reporter_and_nothing_else_prints(no_env, server, capsys):
    reporter = nalvera.ListReporter()
    nalvera.login(server.email, server.key, reporter=reporter)
    assert capsys.readouterr().out == "" and reporter.keys() == ["login.ok_platform"]


def test_signing_in_says_it_once_with_the_platform_only_when_it_is_not_the_usual_one(
        no_env, server, monkeypatch):
    reporter = nalvera.ListReporter()
    nalvera.login(server.email, server.key, reporter=reporter)
    [event] = reporter.events
    assert event.fields["platform"] == server.url and server.key[:12] in event.text
    assert event.fields["expires"] == server.key_expires_at[:10]
    # At https://nalvera.ai the line names no platform.
    monkeypatch.setattr(_session, "DEFAULT_BASE_URL", server.url)
    reporter = nalvera.ListReporter()
    nalvera.login(server.email, server.key, reporter=reporter)
    assert reporter.keys() == ["login.ok"] and server.url not in reporter.events[0].text


def test_login_output_option_prints_its_lines_there(no_env, server):
    buffer = io.StringIO()
    nalvera.login(server.email, server.key, output=buffer)
    shown = buffer.getvalue()
    assert server.email.lower() in shown.lower() and server.key not in shown
