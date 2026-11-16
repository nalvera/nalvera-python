"""mcp_tools.json, the one source of the tools' names, descriptions, schemas,
annotations and texts. The entries the hosted server also offers stay equal
to the Platform's copy (compared when that copy is beside this repository)."""
from __future__ import annotations

import json
import re

import pytest

from nalvera.mcp import LOCAL, MANIFEST_PATH, load_manifest
from test_mcp_support import platform_manifest, sdk_ready

MANIFEST = load_manifest()
TOOLS = {t["name"]: t for t in MANIFEST["tools"]}
LOCAL_ONLY = {"run_scan", "get_run", "download_result"}
SPENDING = {"generate", "run_scan"}
HINTS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


def test_twelve_tools_all_offered_here_three_only_here():
    assert len(TOOLS) == 12
    assert {n for n, t in TOOLS.items() if LOCAL in t["servers"]} == set(TOOLS)
    assert {n for n, t in TOOLS.items() if t["servers"] == [LOCAL]} == LOCAL_ONLY
    assert all(set(t["servers"]) == {"local", "hosted"} for n, t in TOOLS.items()
               if n not in LOCAL_ONLY)


def test_every_tool_is_described_and_typed():
    for name, tool in TOOLS.items():
        assert tool["title"] and 1 <= len(re.findall(r"[.!?](\s|$)", tool["description"])) <= 3
        for schema in ("inputSchema", "outputSchema"):
            assert tool[schema]["type"] == "object", (name, schema)
            assert tool[schema]["additionalProperties"] is False, (name, schema)
        assert set(tool["annotations"]) == set(HINTS)
        assert all(isinstance(tool["annotations"][h], bool) for h in HINTS)


def test_spending_tools_need_a_budget_and_a_cap_and_never_take_a_key():
    for name in SPENDING:
        required = set(TOOLS[name]["inputSchema"]["required"])
        assert {"budget", "max_credits"} <= required, name
        budget = TOOLS[name]["inputSchema"]["properties"]["budget"]["pattern"]
        for value in ("personal", "free", "lab:12"):
            assert re.fullmatch(budget, value)
        assert not re.fullmatch(budget, "lab:0") and not re.fullmatch(budget, "ask")
    assert "request_id" in TOOLS["generate"]["inputSchema"]["required"]
    for tool in TOOLS.values():
        names = set(tool["inputSchema"]["properties"]) | set(tool.get("localInputs") or {})
        assert not names & {"key", "api_key", "token", "email", "password", "authorization"}


def test_the_texts_say_task_and_reasons_and_notes_are_named():
    text = json.dumps(MANIFEST, ensure_ascii=False)
    assert not re.search(r"\bjobs?\b", text, re.IGNORECASE)
    assert set(MANIFEST["reasons"]) >= {"over_max_credits", "series_to_confirm", "in_progress"}
    assert "ai_output" in MANIFEST["notes"]


def test_the_shared_entries_equal_the_platforms_copy():
    path = platform_manifest()
    if not path.is_file():
        pytest.skip(f"the Platform's manifest is not at {path}")
    platform = json.loads(path.read_text(encoding="utf-8"))
    theirs = {t["name"]: t for t in platform["tools"]}
    assert set(theirs) == set(TOOLS)
    for name, tool in TOOLS.items():
        if "hosted" not in tool["servers"]:
            continue
        mine, other = dict(tool), dict(theirs[name])
        # The local server owns estimate_cost's path input (localInputs).
        mine.pop("localInputs", None)
        other.pop("localInputs", None)
        assert mine == other, name
    for block in ("errors", "reasons", "notes"):
        for key, value in platform[block].items():
            assert MANIFEST[block].get(key) == value, (block, key)
    assert MANIFEST["server"] == platform["server"]
    assert MANIFEST["idempotency_key_prefix"] == platform["idempotency_key_prefix"]


def test_the_manifest_ships_with_the_package():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    assert '"nalvera.mcp" = ["mcp_tools.json"]' in text
    assert MANIFEST_PATH.name == "mcp_tools.json"


@pytest.mark.skipif(not sdk_ready(), reason="needs the mcp extra")
def test_the_registered_tools_equal_the_manifest():
    from nalvera.mcp.server import build_server, local_entries
    from test_mcp_support import Session

    class _Tools:                      # only what build_server reads
        manifest = MANIFEST
    listed = Session.__new__(Session)
    listed.server = build_server(_Tools())
    listed.mode = "auto"
    tools = listed.list_tools().tools
    entries = local_entries(MANIFEST)
    assert [t.name for t in tools] == [e["name"] for e in entries]
    for tool, entry in zip(tools, entries):
        assert tool.input_schema == entry["inputSchema"]
        assert tool.output_schema == entry["outputSchema"]
        assert (tool.title, tool.description) == (entry["title"], entry["description"])
    estimate = next(e for e in entries if e["name"] == "estimate_cost")
    assert estimate["inputSchema"]["properties"]["path"] == TOOLS["estimate_cost"][
        "localInputs"]["path"]
