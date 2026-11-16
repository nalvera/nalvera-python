"""examples/mcp/: one configuration snippet per assistant app, a README of the
tools, and a script that starts the server with the SDK's client. Each
snippet starts ``python -m nalvera.mcp`` and holds no key; the script runs against the
fake server."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys

import pytest

import examples_helpers as helpers
from nalvera.mcp import load_manifest
from test_mcp_support import SRC, fast, sdk_ready

FOLDER = helpers.EXAMPLES / "mcp"
COVERED: set = set()
#: The start line of every snippet: ``python -m nalvera.mcp``.
START = ["-m", "nalvera.mcp"]
SNIPPETS = {"claude_code.mcp.json": "mcpServers", "claude_desktop_config.json": "mcpServers",
            "cursor_mcp.json": "mcpServers", "vscode_mcp.json": "servers"}


def covers(*names: str):
    COVERED.update(names)
    return lambda test: test


def _files():
    return sorted(p for p in FOLDER.iterdir() if p.is_file() and not p.name.startswith("."))


def _no_key(text: str, name: str) -> None:
    assert "nvk_" not in text, name
    assert not re.search(r"[0-9A-Za-z]{40,}", text), name
    assert not re.search(r"\bjobs?\b", text, re.IGNORECASE), name


@covers(*SNIPPETS)
@pytest.mark.parametrize("name", sorted(SNIPPETS))
def test_each_json_snippet_starts_the_server_and_holds_no_key(name):
    text = (FOLDER / name).read_text(encoding="utf-8")
    _no_key(text, name)
    data = json.loads(text)
    assert list(data) == [SNIPPETS[name]]
    entry = data[SNIPPETS[name]]["nalvera"]
    assert entry["command"] == "python" and entry["args"] == START
    assert set(entry.get("env") or {}) <= {"NALVERA_MCP_ALLOWED_DIRS"}
    if name in ("cursor_mcp.json", "vscode_mcp.json"):
        assert entry["type"] == "stdio"


@covers("codex_config.toml")
def test_the_codex_snippet(tmp_path):
    text = (FOLDER / "codex_config.toml").read_text(encoding="utf-8")
    _no_key(text, "codex_config.toml")
    header = [line for line in text.splitlines()[:5] if line.startswith("#")]
    assert 0 < len(header) <= 4
    assert helpers.shell_comment_problems("codex_config.toml", text) == []
    if sys.version_info >= (3, 11):
        import tomllib
        server = tomllib.loads(text)["mcp_servers"]["nalvera"]
        assert server["command"] == "python" and server["args"] == START
        assert set(server["env"]) == {"NALVERA_MCP_ALLOWED_DIRS"}
    else:
        assert '[mcp_servers.nalvera]' in text and 'args = ["-m", "nalvera.mcp"]' in text


@covers("README.md")
def test_the_readme_names_every_tool_once_and_links_every_snippet():
    text = (FOLDER / "README.md").read_text(encoding="utf-8")
    _no_key(text, "README.md")
    rows = re.findall(r"^\| `([a-z_]+)` \|", text, re.MULTILINE)
    assert sorted(rows) == sorted(t["name"] for t in load_manifest()["tools"])
    linked = set(re.findall(r"\]\(([^)#]+)\)", text))
    assert {n for n in linked if "/" not in n} == {p.name for p in _files()} - {"README.md"}
    assert all((FOLDER / n).resolve().is_file() for n in linked)
    assert "../set_api_key.py" in linked
    problems = [helpers.command_problem(c) for c in helpers.markdown_commands(text)]
    assert not [p for p in problems if p]
    settings = re.findall(r"^([A-Z][A-Z0-9_]+)\s+default:\s*(\S+)", text, re.MULTILINE)
    assert settings and all(d.startswith("unset") for _, d in settings)
    assert {n for n, _ in settings} <= set(helpers.setting_names())


@covers("list_tools.py")
def test_the_script_follows_the_examples_rules():
    source = (FOLDER / "list_tools.py").read_text(encoding="utf-8")
    _no_key(source, "list_tools.py")
    tree = ast.parse(source, feature_version=(3, 10))
    doc = ast.get_docstring(tree) or ""
    assert doc and len([line for line in doc.splitlines() if line.strip()]) <= 4
    assert re.search(r"\bRun\b", doc) and re.search(r"spends nothing", doc, re.IGNORECASE)
    assert len(source.splitlines()) <= 70
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"asyncio", "os", "sys", "mcp"}


@pytest.mark.skipif(not sdk_ready(), reason="needs the mcp extra")
def test_the_script_lists_the_tools_of_a_running_server(server, tmp_path):
    fast(server)
    server.free_tasks_left = 3
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NALVERA_", "PYTHONPATH"))}
    env.update({"NALVERA_EMAIL": server.email, "NALVERA_API_KEY": server.key,
                "NALVERA_BASE_URL": server.url, "NALVERA_CONFIG_DIR": str(tmp_path / "config"),
                "NO_PROXY": "127.0.0.1,localhost", "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": str(tmp_path)})
    # Started from the source folder, the server's `python -m nalvera` finds
    # this tree (an installed package needs nothing).
    done = subprocess.run([sys.executable, str(FOLDER / "list_tools.py")], cwd=str(SRC),
                          env=env, capture_output=True, text=True, timeout=120,
                          stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stderr[-3000:]
    names = [line.split(":")[0] for line in done.stdout.splitlines() if ":" in line]
    assert set(t["name"] for t in load_manifest()["tools"]) <= set(names)
    assert "3" in done.stdout.splitlines()[-1]
    assert server.key not in done.stdout + done.stderr
    assert not server.calls("POST")


def test_every_file_has_a_test():
    assert {p.name for p in _files()} == COVERED
