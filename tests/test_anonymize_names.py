"""Scan codes and labels (nalvera.anonymize.names)."""
from __future__ import annotations

import pytest

from nalvera.anonymize import AnonymizeError, names


def test_code_shape():
    for _ in range(500):
        code = names.new_code()
        assert names.is_code(code)
        assert code.startswith("scan-") and len(code) == 13
        assert set(code[5:]) <= set(names.ALPHABET)
        assert not set(code[5:]) & set("ILOU")


def test_codes_are_random_and_spread():
    codes = {names.new_code() for _ in range(5000)}
    assert len(codes) == 5000
    used = {c for code in codes for c in code[5:]}
    assert used == set(names.ALPHABET)


def test_code_comes_from_secrets(monkeypatch):
    calls = []

    def fixed(alphabet):
        calls.append(alphabet)
        return "Z"

    monkeypatch.setattr(names.secrets, "choice", fixed)
    assert names.new_code() == "scan-ZZZZZZZZ"
    assert len(calls) == 8


def test_is_code_refuses_lookalikes():
    for value in ("scan-7F3K2Q9", "scan-7F3K2Q9AA", "scan-7F3K2Q9I", "scan-7f3k2q9a",
                  "SCAN-7F3K2Q9A", "scan_7F3K2Q9A", " scan-7F3K2Q9A", None, 7):
        assert not names.is_code(value)


def test_filename_for():
    assert names.filename_for("scan-7F3K2Q9A") == "scan-7F3K2Q9A.nii.gz"


@pytest.mark.parametrize("label, expected", [
    (None, None),
    ("", None),
    ("   ", None),
    ("P012 baseline", "P012 baseline"),
    ("  P012  ", "P012"),
    ("a", "a"),
    ("x" * 60, "x" * 60),
    ("visit_2.follow-up", "visit_2.follow-up"),
])
def test_valid_labels(label, expected):
    assert names.clean_label(label) == expected


@pytest.mark.parametrize("label", [
    "x" * 61, "P012/baseline", "..\\up", "Zoë", "name\twith tab", "line\nbreak",
    "<b>", "a;b", "a:b", "ａ", 12, b"P012",
])
def test_invalid_labels(label):
    with pytest.raises(AnonymizeError) as caught:
        names.clean_label(label)
    assert caught.value.code == "invalid_label"
