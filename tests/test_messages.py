"""messages.py is the one home of user-facing text."""
from __future__ import annotations

import re
from pathlib import Path

from nalvera import messages

SRC = Path(__file__).resolve().parent.parent / "src" / "nalvera"
CORE = ["__init__.py", "__main__.py", "_http.py", "_session.py", "auth.py", "budget.py", "cli.py",
        "client.py", "display.py", "encrypt.py", "errors.py", "generation.py", "report.py",
        "settings.py", "upload.py"]


def _source() -> str:
    return "\n".join((SRC / name).read_text(encoding="utf-8") for name in CORE)


def test_every_key_the_code_uses_has_a_text():
    used = set(re.findall(r"""(?:text|say|_say|has)\(\s*["']([a-z_]+\.[A-Za-z0-9_.:]+)["']""",
                          _source()))
    assert used, "the scan found no keys"
    missing = sorted(k for k in used if not messages.has(k))
    assert missing == []


def test_keys_built_from_codes_exist():
    for state in ("queued", "processing", "completed", "failed", "cancelled"):
        assert messages.has(f"wait.state.{state}")
    for scope in ("tasks:read", "results:read", "tasks:write"):
        assert messages.has(f"scope.{scope}")
    for code in ("not_found", "not_a_scan", "unreadable", "too_large", "dicom_tool_missing",
                 "conversion_failed", "verification_failed", "invalid_label"):
        assert messages.has(f"anonymize.{code}")


def test_texts_say_task_never_job():
    for key, value in messages.MESSAGES.items():
        assert not re.search(r"\bjobs?\b", value, re.IGNORECASE), key


def test_a_missing_placeholder_never_crashes():
    assert messages.text("error.scope_missing") is not None
    assert messages.text("time.minutes", m="x", s="y")          # a format spec, wrong type
    assert messages.text("no.such.key") == "no.such.key"


def test_every_placeholder_is_documented_beside_its_text():
    source = (SRC / "messages.py").read_text(encoding="utf-8")
    for key, value in messages.MESSAGES.items():
        for name in re.findall(r"{(\w+)", value):
            at = source.index(f'"{key}"')
            window = source[max(0, source.rfind("\n\n", 0, at) - 2000):at]
            assert "{" + name + "}" in window or "{" + name in window, (key, name)
