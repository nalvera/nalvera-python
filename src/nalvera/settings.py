"""Every setting the package reads from the environment, in one place.

``SETTINGS`` lists them as data (name, default, meaning), so documentation,
``nalvera --help`` and tools built on the package show the same list.
``get(name)`` is how the package reads one: a name that is not listed here
is a programming error (a test also checks the source for unlisted names).

The meaning of each setting is a text in ``nalvera.messages`` under
``setting.<NAME>``; its default is described under ``setting.<NAME>.default``.
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from . import messages


@dataclass(frozen=True)
class Setting:
    """One environment setting. ``read_by`` is ``"nalvera"`` for the
    package's own settings, or the library that reads it (``"requests"``
    reads the proxy and certificate settings)."""
    name: str
    read_by: str = "nalvera"

    @property
    def meaning(self) -> str:
        """What the setting does (a text of ``nalvera.messages``)."""
        return messages.text(f"setting.{self.name}")

    @property
    def default(self) -> str:
        """What applies when the setting is not set."""
        return messages.text(f"setting.{self.name}.default")


#: The settings, in the order ``nalvera --help`` shows them.
SETTINGS: Tuple[Setting, ...] = (
    Setting("NALVERA_EMAIL"),
    Setting("NALVERA_API_KEY"),
    Setting("NALVERA_BUDGET"),
    Setting("NALVERA_MAX_CREDITS"),
    Setting("NALVERA_ENCRYPT"),
    Setting("NALVERA_MAPPING"),
    Setting("NALVERA_BASE_URL"),
    Setting("NALVERA_CONFIG_DIR"),
    Setting("NALVERA_DCM2NIIX"),
    Setting("NALVERA_MCP_ALLOWED_DIRS"),
    Setting("NALVERA_MCP_SHOW_NAMES"),
    Setting("XDG_CONFIG_HOME"),
    Setting("APPDATA"),
    Setting("HTTPS_PROXY", read_by="requests"),
    Setting("REQUESTS_CA_BUNDLE", read_by="requests"),
)

NAMES = frozenset(s.name for s in SETTINGS)


def get(name: str, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The value of the setting ``name`` from ``env`` (default: the process
    environment), or None when it is unset or blank."""
    if name not in NAMES:
        raise KeyError(f"{name} is not a listed setting")
    value = (os.environ if env is None else env).get(name)
    if value is None or not str(value).strip():
        return None
    return value


def describe(width: int = 79) -> str:
    """The settings as text lines for ``nalvera --help``, wrapped at
    ``width`` (default 79) characters."""
    lines = []
    for setting in SETTINGS:
        block = messages.text("help.setting_line", name=setting.name, meaning=setting.meaning,
                              default=setting.default)
        for line in block.splitlines():
            indent = line[:len(line) - len(line.lstrip())]
            lines.append(textwrap.fill(line.strip(), width=width, initial_indent=indent,
                                       subsequent_indent=indent) if line.strip() else line)
    return "\n".join(lines)


__all__ = ["Setting", "SETTINGS", "get", "describe"]
