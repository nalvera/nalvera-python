"""Neutral scan codes and the labels a user may choose.

A file name can hold a patient name, so it never leaves the computer: each
scan gets a random code, and that code is the name the platform shows. The
user may add a label of their own; it is their text and reaches the platform,
so it is limited to characters that cannot carry a path or markup.
"""
from __future__ import annotations

import re
import secrets
from typing import Optional

from .errors import AnonymizeError

# Crockford base32: no I, L, O or U, so a code read aloud or retyped is unambiguous.
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
CODE_PREFIX = "scan-"
CODE_LENGTH = 8
CODE_RX = re.compile(r"scan-[0-9A-HJKMNP-TV-Z]{8}")

LABEL_MAX = 60
# ASCII only, spelled out: \w would admit any Unicode letter.
_LABEL_RX = re.compile(r"[A-Za-z0-9 ._-]{1,60}")


def new_code() -> str:
    """A random scan code such as ``scan-7F3K2Q9A``: 8 Crockford base32
    characters (40 bits) drawn from the operating system's secure generator."""
    return CODE_PREFIX + "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))


def is_code(value: object) -> bool:
    """True when ``value`` is a scan code in the exact form ``new_code`` makes."""
    return isinstance(value, str) and CODE_RX.fullmatch(value) is not None


def filename_for(code: str) -> str:
    """The file name a scan is uploaded under: its code plus ``.nii.gz``."""
    return code + ".nii.gz"


def clean_label(label: Optional[str]) -> Optional[str]:
    """The label with surrounding spaces removed, or None when none was given
    (None or only spaces). Raises AnonymizeError(``invalid_label``) for
    anything but 1-60 ASCII letters, digits, spaces, dots, dashes and
    underscores."""
    if label is None:
        return None
    if not isinstance(label, str):
        raise AnonymizeError("invalid_label")
    label = label.strip(" ")
    if not label:
        return None
    if _LABEL_RX.fullmatch(label) is None:
        raise AnonymizeError("invalid_label")
    return label
