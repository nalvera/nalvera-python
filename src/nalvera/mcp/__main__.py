"""``python -m nalvera.mcp``: start the local MCP server for an AI assistant.

The assistant's app runs this line; the server speaks the Model Context
Protocol on standard input and output until standard input closes. Without
the ``mcp`` extra one line goes to standard error and the status is 2.
"""
from __future__ import annotations

import sys
from typing import Callable, Optional


def main(error: Optional[Callable[[str], None]] = None) -> int:
    """Run the server; returns the exit status. ``error`` (default: a line
    on standard error) shows why the server cannot start."""
    from .. import messages
    from . import unavailable_reason
    reason = unavailable_reason()
    if reason is not None:
        text = messages.text(f"error.mcp_{reason}")
        if error is not None:
            error(text)
        else:
            sys.stderr.write(text + "\n")
        return 2
    from . import server
    return server.main()


if __name__ == "__main__":
    raise SystemExit(main())
