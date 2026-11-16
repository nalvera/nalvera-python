"""The local MCP server: ``python -m nalvera.mcp`` (or the command ``nalvera mcp``).

An AI assistant starts it on the researcher's own computer and speaks the
Model Context Protocol with it over standard input and output. It offers the
tools listed in ``mcp_tools.json`` (the one source of their names,
descriptions, schemas and texts, shared with the hosted server at
https://mcp.nalvera.ai/mcp), scan runs included: scans are anonymized on this
computer as in every other use of the package.

The server signs in with the key ``nalvera.login(..., remember=True)`` stored (or
``NALVERA_EMAIL`` and ``NALVERA_API_KEY``): no key is ever a tool argument
or part of a result. It reads and writes files only inside the folders of
``NALVERA_MCP_ALLOWED_DIRS``, by default the folder it was started in.

Modules: ``tools`` (what each tool does, as data in and data out),
``runs`` (``run_scan`` in the background, with its state on disk) and
``server`` (the MCP SDK's server over stdio). Importing this package, or
``nalvera``, never imports the SDK, which comes with the ``mcp`` extra
(``pip install "nalvera[mcp]"``).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional

MANIFEST_PATH = Path(__file__).with_name("mcp_tools.json")
LOCAL = "local"


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    """The manifest (``mcp_tools.json``) as a dict."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def unavailable_reason() -> Optional[str]:
    """None when the server can run here; ``"sdk"`` when the ``mcp`` extra
    is not installed. Imports nothing."""
    for module in ("mcp", "mcp_types", "anyio", "jsonschema"):
        try:
            if importlib.util.find_spec(module) is None:
                return "sdk"
        except (ImportError, ValueError):
            return "sdk"
    return None


__all__ = ["MANIFEST_PATH", "LOCAL", "load_manifest", "unavailable_reason"]
