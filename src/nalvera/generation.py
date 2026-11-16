"""Generations: tasks that make new images from settings, with no input scan
(MAISI-v2, model ``nv_generate_ctmr``, tasks ``rflow-ct`` and
``rflow-mr-brain``).

The settings go to the server as they are; the server checks them and its
explanation of a refusal is shown unchanged. ``nalvera models --model
nv_generate_ctmr`` lists what each task accepts.

A setting accepted by the server, for example::

    {"mode": "image-only", "output_size": [256, 256, 128],
     "spacing": [1.5, 1.5, 2.0], "num_output_samples": 1, "seed": 7}
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, Iterable, List, Tuple

from . import messages
from .errors import ConfigError

_NAME_RX = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def parse_param(item: str) -> Tuple[str, Any]:
    """One ``name=value`` setting. The value is read as JSON when it can be
    (numbers, ``true``/``false``, ``null``, lists such as ``[256,256,128]``,
    quoted strings); ``True``/``False`` are accepted too; anything else is
    kept as text (``mode=image-only``)."""
    name, sep, raw = str(item).partition("=")
    name = name.strip()
    if not sep or not _NAME_RX.fullmatch(name):
        raise ConfigError(messages.text("error.invalid_param", param=item), code="invalid_param")
    raw = raw.strip()
    if raw in ("True", "False"):
        return name, raw == "True"
    try:
        return name, json.loads(raw)
    except ValueError:
        return name, raw


def parse_params(items: Iterable[str]) -> Dict[str, Any]:
    """Several ``name=value`` settings as a dict; a name given twice is an
    error."""
    out: Dict[str, Any] = {}
    for item in items or ():
        name, value = parse_param(item)
        if name in out:
            raise ConfigError(messages.text("error.duplicate_param", name=name),
                              code="duplicate_param")
        out[name] = value
    return out


def new_idempotency_key() -> str:
    """A fresh key for one logical generation: kept across the retries of
    that request, never reused for another."""
    return "gen-" + uuid.uuid4().hex


def describe(generation: Dict[str, Any]) -> List[Tuple[str, str]]:
    """A generation task's settings help (``tasks[].generation`` of
    ``GET /api/v1/models``) as (name, allowed values) pairs for display."""
    out: List[Tuple[str, str]] = []
    for key, value in (generation or {}).items():
        out.append((str(key), _show(value)))
    return out


def _show(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_show(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value) and any(" " in v for v in value):
            return " | ".join(value)
        return json.dumps(list(value))
    return json.dumps(value) if not isinstance(value, str) else value
