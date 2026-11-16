"""Credentials, and the three ways to give a key (an example of setting the
API key relies on them)."""
from __future__ import annotations

import argparse
import sys
import types

import pytest

from fake_server import make_key
from nalvera import Client, auth, cli, messages
from nalvera.errors import ConfigError, NoCredentials
from test_cli import run_cli


# ── D1: the key is never a command-line argument ──────────────────────────────
def test_no_command_takes_a_key_as_an_option():
    parser = cli.build_parser()
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for name, sub in subparsers.choices.items():
        for action in sub._actions:
            for option in action.option_strings:
                assert "key" not in option.lower(), (name, option)
    assert run_cli("login", "--api-key", make_key())[0] == cli.EXIT_USAGE


# ── D2: values pasted with quotes, spaces or a newline ────────────────────────
@pytest.mark.parametrize("wrap", ['"{}"', "'{}'", "  {}\n", "{}\r\n", ' "{}" '])
def test_a_key_from_the_environment_is_cleaned(env, server, monkeypatch, wrap):
    monkeypatch.setenv("NALVERA_API_KEY", wrap.format(server.key))
    monkeypatch.setenv("NALVERA_EMAIL", f'  "{server.email}"\n')
    with Client(quiet=True) as client:
        assert client.me()["email"] == server.email.lower()
    assert server.calls("GET", "/api/v1/me")[-1]["headers"]["Authorization"] == \
        f"Bearer {server.key}"


def test_a_mistyped_key_is_reported_without_any_request(env, server, monkeypatch):
    typo = server.key[:20] + ("A" if server.key[20] != "A" else "B") + server.key[21:]
    monkeypatch.setenv("NALVERA_API_KEY", typo)
    with pytest.raises(ConfigError) as caught:
        Client(quiet=True)
    assert caught.value.code == "invalid_key_format"
    assert server.requests == [] and typo not in str(caught.value)
    status, _out, err = run_cli("whoami")
    assert status == cli.EXIT_AUTH and server.requests == [] and typo not in err


# ── D3: a credentials file others can read ────────────────────────────────────
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_a_readable_credentials_file_is_refused_with_the_fixing_command(cli_dialect, env,
                                                                       server,
                                                                        monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    monkeypatch.delenv("NALVERA_EMAIL")
    path = auth.credentials_path()
    auth.store(server.email, auth.ApiKey(server.key), use_keyring=False)
    path.chmod(0o644)
    status, _out, err = run_cli("whoami")
    assert status == cli.EXIT_AUTH
    assert messages.text("error.credentials_file_insecure", path=str(path)) in err
    assert f"chmod 600 {path}" in err and server.requests == []


# ── D4: no keychain backend ───────────────────────────────────────────────────
def _fake_keyring_module(backend_priority=None, set_fails=False):
    keyring = types.ModuleType("keyring")
    backends = types.ModuleType("keyring.backends")
    fail = types.ModuleType("keyring.backends.fail")

    class FailKeyring:
        priority = 0

    fail.Keyring = FailKeyring

    class Working:
        priority = 1

    keyring.get_keyring = lambda: FailKeyring() if backend_priority is None else Working()

    def set_password(*args):
        if set_fails:
            raise RuntimeError("No recommended backend was available")

    keyring.set_password = set_password
    keyring.get_password = lambda *args: None
    backends.fail = fail
    keyring.backends = backends
    return {"keyring": keyring, "keyring.backends": backends, "keyring.backends.fail": fail}


@pytest.mark.parametrize("scene", ["no_backend", "backend_fails"])
def test_without_a_keychain_the_file_is_used_without_a_traceback(tmp_path, monkeypatch,
                                                                 scene):
    modules = _fake_keyring_module(backend_priority=None if scene == "no_backend" else 1,
                                   set_fails=scene == "backend_fails")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    key = make_key()
    assert auth.store("user@lab.org", auth.ApiKey(key), directory=tmp_path) == "file"
    creds = auth.resolve(env={}, directory=tmp_path)
    assert creds.api_key.reveal() == key and creds.source == "file"


# ── D5: when the key ends ─────────────────────────────────────────────────────
@pytest.mark.parametrize("server_date,warned", [
    ("Wed, 10 Nov 2027 10:00:00 GMT", True),           # 5 days left
    ("Wed, 06 Oct 2027 10:00:00 GMT", False),          # 40 days left
])
def test_whoami_and_login_warn_when_the_key_ends_soon(cli_dialect, env, server, monkeypatch,
                                                     no_sleep,
                                                      server_date, warned):
    # Measured against the platform's clock, never this computer's (A7).
    server.date_header = server_date
    warning = messages.text("whoami.expires_soon", days=5)
    _status, out, _err = run_cli("whoami")
    assert "2027-11-15" in out and (warning in out) is warned
    monkeypatch.delenv("NALVERA_API_KEY")
    status, out, _err = run_cli("login", "--email", server.email, secret=server.key)
    assert status == 0 and "2027-11-15" in out
    assert (any(line.startswith(warning[:20]) for line in out.splitlines())) is warned


# ── the three ways to give a key, and logout ──────────────────────────────────
def test_the_three_ways_to_give_a_key_and_logout(env, server, monkeypatch, no_sleep):
    # 1. Arguments (for a program reading a secret store).
    monkeypatch.delenv("NALVERA_API_KEY")
    monkeypatch.delenv("NALVERA_EMAIL")
    with Client(email=server.email, api_key=server.key, quiet=True) as client:
        assert client.me()["key"]["prefix"] == server.key[:12]
    # 2. The environment.
    monkeypatch.setenv("NALVERA_EMAIL", server.email)
    monkeypatch.setenv("NALVERA_API_KEY", server.key)
    assert run_cli("whoami")[0] == 0
    # 3. nalvera login stores it; whoami then needs nothing else.
    monkeypatch.delenv("NALVERA_API_KEY")
    monkeypatch.delenv("NALVERA_EMAIL")
    assert run_cli("login", "--email", server.email, secret=server.key)[0] == 0
    assert run_cli("whoami")[0] == 0
    with Client(quiet=True) as client:
        assert client.key_prefix == server.key[:12]
    # logout removes it: no key is left on this computer.
    assert run_cli("logout")[0] == 0
    with pytest.raises(NoCredentials):
        Client(quiet=True)
    assert not auth.credentials_path().exists()
