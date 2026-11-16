"""Shared fixtures: the fake server, a stand-in anonymizer, a fake clock."""
from __future__ import annotations

import gzip
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import stub_anonymizer  # noqa: E402
from fake_server import FakeNalvera  # noqa: E402

_ENV = ("NALVERA_EMAIL", "NALVERA_API_KEY", "NALVERA_BUDGET", "NALVERA_BASE_URL",
        "NALVERA_MAPPING", "NALVERA_CONFIG_DIR", "NALVERA_ENCRYPT", "NALVERA_MAX_CREDITS")


class FakeTime:
    """A clock that moves only when the code sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: List[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now += float(seconds)

    @property
    def slept(self) -> float:
        return sum(self.sleeps)


class TTYStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _signed_out():
    """Each test starts and ends without a process sign-in (nalvera.login)
    and with texts in the Python dialect."""
    from nalvera import messages
    yield
    session = sys.modules.get("nalvera._session")
    if session is not None:
        session.logout()
    messages.set_dialect(messages.PYTHON)


@pytest.fixture
def cli_dialect():
    """For a test of the command line: texts the test builds are rendered
    for the command line, as the command renders its own."""
    from nalvera import messages
    previous = messages.set_dialect(messages.CLI)
    yield
    messages.set_dialect(previous)


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


@pytest.fixture
def no_sleep(monkeypatch):
    """For command-line tests (real clock): pacing and polls do not wait."""
    import time
    slept: List[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    return slept


@pytest.fixture
def server():
    fake = FakeNalvera().start()
    yield fake
    fake.stop()


@pytest.fixture
def stub_anon(monkeypatch):
    """Replace nalvera.anonymize with the stand-in for this test."""
    import nalvera
    stub_anonymizer.prepared.clear()
    module = stub_anonymizer.as_module()
    monkeypatch.setitem(sys.modules, "nalvera.anonymize", module)
    monkeypatch.setattr(nalvera, "anonymize", module, raising=False)
    return module


@pytest.fixture
def env(monkeypatch, tmp_path, server):
    """Settings pointing at the fake server; no stored credentials, no keychain."""
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NALVERA_EMAIL", server.email)
    monkeypatch.setenv("NALVERA_API_KEY", server.key)
    monkeypatch.setenv("NALVERA_BASE_URL", server.url)
    monkeypatch.setenv("NALVERA_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("NALVERA_MAPPING", str(tmp_path / "nalvera_scans.csv"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    from nalvera import auth
    monkeypatch.setattr(auth, "_keyring", lambda: None)
    # Each test gets its own temporary folder, so the clean-up of stale
    # run folders only ever sees this test's.
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def out() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def make_client(env, stub_anon, fake_time, out):
    """A Client against the fake server with an instant clock."""
    from nalvera import Client
    made = []

    def make(**kwargs):
        kwargs.setdefault("output", out)
        kwargs.setdefault("interactive", False)
        kwargs.setdefault("clock", fake_time.clock)
        kwargs.setdefault("sleep", fake_time.sleep)
        client = Client(**kwargs)
        made.append(client)
        return client

    yield make
    for client in made:
        client.close()


@pytest.fixture
def client(make_client):
    return make_client()


def scan_bytes(tag: bytes = b"scan", size: int = 2500) -> bytes:
    """Gzip-looking bytes of about ``size`` (several upload parts). The same
    tag always gives the same bytes: the gzip header's time stamp is fixed,
    so a scan written twice in a test is the same scan across a second's
    boundary too."""
    raw = gzip.compress(tag * 10, mtime=0)
    return (raw + b"\x00" * size)[:max(size, len(raw))]


@pytest.fixture
def scans(env):
    """Write scan files under the test folder; returns a helper."""

    def write(name: str, data: bytes = None) -> Path:
        path = env / "input" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(scan_bytes(name.encode()) if data is None else data)
        return path

    return write
