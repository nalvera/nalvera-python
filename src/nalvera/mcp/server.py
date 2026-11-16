"""The local MCP server over standard input and output (``nalvera mcp``).

Built on the MCP Python SDK (``mcp``, low-level ``Server``), which serves
both protocol eras over stdio: revision 2026-07-28 (no handshake;
``server/discover``) and the handshake revisions (``initialize``), the
client's first request deciding which.

Standard output belongs to the protocol. While the server runs, the SDK's
stdio transport points file descriptor 1 at standard error and descriptor 0
at the null device, so a stray ``print`` or a child process's output never
reaches the client; the package's own messages go to a reporter that shows
nothing; this module's log lines go to standard error (never a key, never a
tool's arguments or results).

Shutdown, when standard input closes or on SIGTERM: runs stop at their next
pause, their temporary folders are removed, and the process exits within a
few seconds.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import signal
import threading
import time
from typing import Any, Dict, List, Optional

import anyio
import mcp_types as types
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from mcp.server.caching import CacheHint
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError

from .._version import __version__
from . import LOCAL
from .tools import LocalTools, ToolFailure, clean_stale

logger = logging.getLogger("nalvera.mcp")

#: No tool call runs longer than this (seconds).
TOOL_TIMEOUT = 30.0
#: How long runs get to stop at shutdown (seconds).
SHUTDOWN_SECONDS = 4.0
#: The tool list changes only with the package version.
LIST_TTL_MS = 3_600_000


def local_entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The manifest's entries this server offers, in manifest order, with
    the local-only inputs (``localInputs``) merged into their input schema."""
    entries = []
    for entry in manifest["tools"]:
        if LOCAL not in entry.get("servers", ()):
            continue
        entry = copy.deepcopy(entry)
        extra = entry.pop("localInputs", None)
        if extra:
            entry["inputSchema"].setdefault("properties", {}).update(extra)
        entries.append(entry)
    return entries


def mcp_tool(entry: Dict[str, Any]) -> types.Tool:
    return types.Tool(
        name=entry["name"], title=entry["title"], description=entry["description"],
        input_schema=entry["inputSchema"], output_schema=entry["outputSchema"],
        annotations=types.ToolAnnotations(
            read_only_hint=entry["annotations"]["readOnlyHint"],
            destructive_hint=entry["annotations"]["destructiveHint"],
            idempotent_hint=entry["annotations"]["idempotentHint"],
            open_world_hint=entry["annotations"]["openWorldHint"]))


class RedactKeys(logging.Filter):
    """Replaces anything key-shaped in a log record with its prefix: a
    second wall behind never logging one."""

    _RX = re.compile(r"(nvk_[0-9A-Za-z]{8})[0-9A-Za-z]+")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "nvk_" in message:
            record.msg = self._RX.sub(r"\1…[redacted]", message)
            record.args = None
        return True


def configure_logging() -> None:
    """This server's log lines and the SDK's warnings on standard error."""
    ours, sdk = logging.getLogger("nalvera.mcp"), logging.getLogger("mcp")
    ours.setLevel(logging.INFO)
    sdk.setLevel(logging.WARNING)
    if any(isinstance(f, RedactKeys) for f in ours.filters):
        return
    for target in (ours, sdk):
        target.addFilter(RedactKeys())
    handler = logging.StreamHandler()                 # standard error
    handler.setFormatter(logging.Formatter("nalvera mcp %(levelname)s %(message)s"))
    handler.addFilter(RedactKeys())
    ours.addHandler(handler)
    sdk.addHandler(handler)
    ours.propagate = False
    sdk.propagate = False


def _result(failed: bool, data: Dict[str, Any]) -> types.CallToolResult:
    # The text block is exactly the structure as JSON, so a model that reads
    # only text sees every part as a named field. An error has no
    # structured content: output schemas describe the success shape.
    text = [types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]
    if failed:
        return types.CallToolResult(content=text, is_error=True)
    return types.CallToolResult(content=text, structured_content=data)


def build_server(tools: LocalTools, *, tool_timeout: float = TOOL_TIMEOUT) -> Server:
    """The SDK server offering the manifest's local tools, run by ``tools``."""
    manifest = tools.manifest
    entries = local_entries(manifest)
    listed = [mcp_tool(entry) for entry in entries]
    schemas = {entry["name"]: entry["inputSchema"] for entry in entries}
    validators = {name: Draft202012Validator(schema) for name, schema in schemas.items()}

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=listed)

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        name = params.name
        if name not in validators:
            raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {name[:80]}")
        started = time.monotonic()
        arguments = params.arguments or {}
        problem = best_match(validators[name].iter_errors(arguments))
        if problem is not None:
            where = ".".join(str(p) for p in problem.absolute_path) or "arguments"
            failed, data = True, {"error": tools.error_fields(
                ToolFailure("invalid_arguments", f"{where}: {problem.message}"[:500]))}
        else:
            try:
                with anyio.fail_after(tool_timeout):
                    failed, data = await anyio.to_thread.run_sync(
                        tools.call, name, arguments, abandon_on_cancel=True)
            except TimeoutError:
                failed, data = True, {"error": tools.error_fields(ToolFailure("timeout"))}
        outcome = data["error"]["code"] if failed else "ok"
        logger.info("tool=%s outcome=%s ms=%d", name, outcome, (time.monotonic() - started) * 1000)
        return _result(failed, data)

    server = manifest["server"]
    return Server(server["name"], version=__version__, title=server["title"],
                  instructions=server["instructions"],
                  cache_hints={"tools/list": CacheHint(ttl_ms=LIST_TTL_MS, scope="public")},
                  get_tool_input_schema=schemas.get,
                  on_list_tools=list_tools, on_call_tool=call_tool)


async def serve(tools: LocalTools, *, stdin: Any = None, stdout: Any = None) -> None:
    """Serve the protocol until standard input closes (or the given
    ``stdin``/``stdout`` streams, for tests)."""
    server = build_server(tools)
    async with stdio_server(stdin, stdout) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(tools: Optional[LocalTools] = None) -> int:
    """Run the server; return the exit status (0) once it stopped."""
    configure_logging()
    clean_stale()                # temporary folders a killed process left
    tools = tools or LocalTools()
    if tools.allowed:
        logger.info("started (%s): %d allowed folder(s)", __version__, len(tools.allowed))
    else:
        logger.warning("started (%s): no allowed folder; set NALVERA_MCP_ALLOWED_DIRS to use "
                       "files", __version__)
    stopping = threading.Event()

    def on_sigterm(signum: int, frame: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=_stop_and_exit, args=(tools,), daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, on_sigterm)
    except (ValueError, OSError, AttributeError):      # not the main thread, or no SIGTERM
        pass
    try:
        anyio.run(serve, tools)
    finally:
        stopping.set()
        alive = tools.close(SHUTDOWN_SECONDS)
    logger.info("stopped")
    if alive:
        # A run is stuck inside a request: its temporary files are gone;
        # end without waiting for it.
        logging.shutdown()
        os._exit(0)
    return 0


def _stop_and_exit(tools: LocalTools) -> None:
    try:
        tools.close(SHUTDOWN_SECONDS)
        logger.info("stopped (SIGTERM)")
        logging.shutdown()
    finally:
        os._exit(0)


__all__ = ["build_server", "serve", "main", "local_entries", "mcp_tool", "TOOL_TIMEOUT"]
