"""Credentials: the key's offline check, where credentials come from, storage."""
from __future__ import annotations

import os
import pickle
import stat
import sys

import pytest

from fake_server import ALPHABET, make_key
from nalvera import auth
from nalvera.errors import ConfigError, NoCredentials


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in ("NALVERA_EMAIL", "NALVERA_API_KEY", "NALVERA_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(auth, "_keyring", lambda: None)
    return tmp_path


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def set_password(self, service, user, secret):
        self.store[(service, user)] = secret

    def get_password(self, service, user):
        return self.store.get((service, user))

    def delete_password(self, service, user):
        del self.store[(service, user)]


# ── the key's shape and checksum ──────────────────────────────────────────────
def test_a_key_built_like_the_servers_is_well_formed():
    for _ in range(50):
        assert auth.is_well_formed(make_key())


def test_any_one_character_typo_is_caught_offline():
    key = make_key()
    for position in range(4, len(key)):
        wrong = ALPHABET[(ALPHABET.index(key[position]) + 1) % len(ALPHABET)]
        assert not auth.is_well_formed(key[:position] + wrong + key[position + 1:])


@pytest.mark.parametrize("value", ["", "nvk_", "nvk_abc", None, 12, "xyz_" + "a" * 57])
def test_shapes_that_are_not_keys(value):
    assert not auth.is_well_formed(value)


def test_missing_or_extra_character_is_caught():
    key = make_key()
    assert not auth.is_well_formed(key[:-1])
    assert not auth.is_well_formed(key + "A")


def test_known_checksum_vector():
    # CRC-32 of the ASCII body, base62, left-padded to 6 (backend/api_keys.py).
    body = "nvk_" + "A" * 51
    import zlib
    number, digits = zlib.crc32(body.encode("ascii")), []
    while number:
        number, rem = divmod(number, 62)
        digits.append(ALPHABET[rem])
    assert auth.key_checksum(body) == "".join(reversed(digits)).rjust(6, "0")


# ── the key never shows itself ────────────────────────────────────────────────
def test_api_key_repr_and_str_show_only_the_prefix():
    key = make_key()
    held = auth.ApiKey(key)
    for shown in (repr(held), str(held), f"{held}", repr([held]), repr({"k": held})):
        assert key not in shown and key[12:] not in shown
        assert key[:12] in shown
    assert held.reveal() == key


def test_api_key_cannot_be_pickled():
    with pytest.raises(TypeError):
        pickle.dumps(auth.ApiKey(make_key()))


def test_credentials_repr_hides_the_key():
    key = make_key()
    creds = auth.Credentials(email="a@b.org", api_key=auth.ApiKey(key))
    assert key not in repr(creds) and key not in str(creds)


def test_a_malformed_key_names_its_source_without_showing_it(clean_env):
    bad = make_key()[:-1] + "!"
    with pytest.raises(ConfigError) as caught:
        auth.resolve("a@b.org", bad)
    assert caught.value.code == "invalid_key_format"
    assert bad not in str(caught.value)


# ── where credentials come from ───────────────────────────────────────────────
def test_arguments_come_before_the_environment(clean_env, monkeypatch):
    env_key, arg_key = make_key(), make_key()
    monkeypatch.setenv("NALVERA_EMAIL", "env@lab.org")
    monkeypatch.setenv("NALVERA_API_KEY", env_key)
    creds = auth.resolve("arg@lab.org", arg_key)
    assert (creds.email, creds.api_key.reveal(), creds.source) == ("arg@lab.org", arg_key,
                                                                   "argument")


def test_environment_comes_before_stored_credentials(clean_env, monkeypatch):
    stored, env_key = make_key(), make_key()
    auth.store("user@lab.org", auth.ApiKey(stored), directory=clean_env, use_keyring=False)
    monkeypatch.setenv("NALVERA_API_KEY", env_key)
    creds = auth.resolve(env={"NALVERA_API_KEY": env_key}, directory=clean_env)
    assert creds.api_key.reveal() == env_key and creds.email == "user@lab.org"
    assert creds.source == "environment"


def test_keychain_comes_before_the_file(clean_env, monkeypatch):
    ring = FakeKeyring()
    monkeypatch.setattr(auth, "_keyring", lambda: ring)
    in_ring = make_key()
    assert auth.store("user@lab.org", auth.ApiKey(in_ring), directory=clean_env) == "keyring"
    record = (clean_env / "credentials.json").read_text()
    assert in_ring not in record                       # the file holds the email only
    creds = auth.resolve(env={}, directory=clean_env)
    assert creds.api_key.reveal() == in_ring and creds.source == "keyring"


def test_the_file_is_used_last(clean_env):
    stored = make_key()
    assert auth.store("user@lab.org", auth.ApiKey(stored), directory=clean_env,
                      use_keyring=False) == "file"
    creds = auth.resolve(env={}, directory=clean_env)
    assert creds.api_key.reveal() == stored and creds.source == "file"


def test_a_stored_key_is_not_paired_with_another_accounts_email(clean_env):
    auth.store("user@lab.org", auth.ApiKey(make_key()), directory=clean_env, use_keyring=False)
    with pytest.raises(NoCredentials):
        auth.resolve(env={"NALVERA_EMAIL": "someone.else@lab.org"}, directory=clean_env)


def test_nothing_found_raises_no_credentials_with_the_steps(clean_env):
    with pytest.raises(NoCredentials) as caught:
        auth.resolve(env={}, directory=clean_env)
    assert caught.value.code == "no_credentials"
    from nalvera import messages
    assert str(caught.value) == messages.text("error.no_credentials")


def test_whitespace_around_a_pasted_key_is_ignored(clean_env):
    key = make_key()
    assert auth.resolve("a@b.org", f"  {key}\n").api_key.reveal() == key


# ── storage ───────────────────────────────────────────────────────────────────
@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_the_credentials_file_is_private(clean_env):
    folder = clean_env / "cfg"
    auth.store("user@lab.org", auth.ApiKey(make_key()), directory=folder, use_keyring=False)
    assert stat.S_IMODE(os.stat(folder).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(folder / "credentials.json").st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_a_file_others_can_read_is_refused(clean_env):
    auth.store("user@lab.org", auth.ApiKey(make_key()), directory=clean_env, use_keyring=False)
    os.chmod(clean_env / "credentials.json", 0o644)
    with pytest.raises(ConfigError) as caught:
        auth.resolve(env={}, directory=clean_env)
    assert caught.value.code == "credentials_file_insecure"


def test_forget_removes_file_and_keychain_entry(clean_env, monkeypatch):
    ring = FakeKeyring()
    monkeypatch.setattr(auth, "_keyring", lambda: ring)
    auth.store("user@lab.org", auth.ApiKey(make_key()), directory=clean_env)
    assert auth.forget(directory=clean_env) is True
    assert ring.store == {} and not (clean_env / "credentials.json").exists()
    assert auth.forget(directory=clean_env) is False


def test_config_dir_follows_the_setting(monkeypatch, tmp_path):
    assert auth.config_dir({"NALVERA_CONFIG_DIR": str(tmp_path)}) == tmp_path
    if sys.platform != "win32":
        assert auth.config_dir({"XDG_CONFIG_HOME": str(tmp_path)}) == tmp_path / "nalvera"
