"""README.md, the page PyPI shows, is held by the code: every marked block
is its example file (or lines of it) byte for byte, every generated table is
what ``tests/examples_helpers.py`` builds, and the text names only functions,
arguments, settings and exceptions that exist, in Python terms only.

After changing an example, a message or a signature: ``python
tests/examples_helpers.py`` rewrites the marked parts (and leaves the prose)."""
from __future__ import annotations

import builtins
import inspect
import re

import pytest

import examples_helpers as helpers

TEXT = helpers.PACKAGE_README.read_text(encoding="utf-8")
FENCE_RX = re.compile(r"^```\w*\n.*?^```$", re.M | re.S)


def _prose(text: str = TEXT) -> str:
    """The README without its fenced blocks and without the marker lines."""
    text = FENCE_RX.sub("", text)
    return re.sub(r"^\[//\]: # \(.*\)$", "", text, flags=re.M)


def _code_spans(text: str) -> list:
    return re.findall(r"`([^`\n]+)`", _prose(text))


def test_the_markers_are_there():
    quotes = list(helpers.QUOTE_RX.finditer(TEXT))
    tables = {m.group("name") for m in helpers.TABLE_RX.finditer(TEXT)}
    assert len(quotes) >= 15
    assert tables == {"errors-platform", "errors-package", "errors-anonymizer", "functions",
                      "examples", "settings"}


def test_every_marked_block_is_its_file_byte_for_byte():
    for match in helpers.QUOTE_RX.finditer(TEXT):
        expected = helpers.quoted_lines(match.group("path"), match.group("first"),
                                        match.group("last"))
        assert match.group("body") == expected + "\n", match.group("path")


def test_every_generated_table_is_current():
    """Also holds the functions table to ``inspect.signature`` and the
    settings table to examples/README.md's. The platform's errors are
    checked by the next test."""
    tables = helpers.readme_tables(TEXT)
    for match in helpers.TABLE_RX.finditer(TEXT):
        name = match.group("name")
        if name in tables:
            assert match.group("body") == tables[name], (
                f"{name}: run python tests/examples_helpers.py")


def test_the_platform_error_table_is_current():
    platform = helpers.platform_dir()
    if platform is None:
        pytest.skip("the Platform source (backend/api_messages.py) is not next to this package")
    assert helpers.platform_status_problems(platform) == []
    body = next(m.group("body") for m in helpers.TABLE_RX.finditer(TEXT)
                if m.group("name") == "errors-platform")
    assert body == helpers.errors_platform_table(platform)


def test_rewriting_changes_nothing():
    assert helpers.rewrite_readme(TEXT) == TEXT


def test_the_functions_table_covers_every_module_function_with_a_description():
    import nalvera
    body = next(m.group("body") for m in helpers.TABLE_RX.finditer(TEXT)
                if m.group("name") == "functions")
    rows = [line for line in body.splitlines()[2:]]
    names = [re.match(r"^\| `nalvera\.([\w.]+)\(", row).group(1) for row in rows]
    assert set(nalvera.FUNCTIONS) <= set(names)
    assert all(row.rstrip(" |").split(" | ")[-1].strip() for row in rows)
    for name, row in zip(names, rows):
        shown, function = helpers.readme_function(name)
        assert f"`{helpers.signature_text(function, shown)}`" in row


def test_the_settings_table_is_the_examples_index_table():
    body = next(m.group("body") for m in helpers.TABLE_RX.finditer(TEXT)
                if m.group("name") == "settings")
    index = helpers.README.read_text(encoding="utf-8")
    assert body in index


def test_python_only_no_command_and_no_option():
    """The one mention of the command line is the closing line with its link."""
    problems = helpers.python_text_problems("README.md", TEXT)
    assert problems == []
    assert TEXT.count("https://nalvera.ai/command-line") == 1
    assert TEXT.rstrip().endswith("https://nalvera.ai/command-line")


def test_every_function_argument_setting_and_exception_named_exists():
    import nalvera
    import nalvera.anonymize as anonymize
    prose = _prose()
    for name in set(re.findall(r"\bnalvera\.(?!ai\b)([A-Za-z_]\w*)(?:\.(\w+))?", prose)):
        head, attribute = name
        assert hasattr(nalvera, head) or head in ("anonymize", "mcp"), head
        if head == "anonymize" and attribute:
            assert hasattr(anonymize, attribute), attribute
    settings = {s.name for s in nalvera.SETTINGS}
    for setting in set(re.findall(r"\bNALVERA_[A-Z_]+\b", TEXT)):
        assert setting in settings, setting
    arguments = set()
    for function in list(helpers.calls().values()) + [nalvera.Client.run_in_background,
                                                       nalvera.logout]:
        arguments |= set(inspect.signature(function).parameters)
    for span in _code_spans(TEXT):
        for argument in re.findall(r"(?<![\w.\"'])([a-z_][a-z0-9_]*)=", span):
            assert argument in arguments, (argument, span)
    for span in _code_spans(TEXT):
        match = re.match(r"^(?:nalvera\.)?([A-Z][a-z][A-Za-z0-9]*)(?:\.(\w+))?$", span)
        if match and match.group(1) not in dir(builtins):          # None, True, ...
            assert hasattr(nalvera, match.group(1)) or hasattr(anonymize, match.group(1)), span


def test_no_relative_link_and_no_html():
    prose = _prose()
    for target in re.findall(r"\]\(([^)]*)\)", prose):
        assert re.match(r"https?://", target), target
    outside_code = re.sub(r"`[^`\n]*`", "", prose)
    assert not re.search(r"<[A-Za-z!/]", outside_code), re.search(r"<[A-Za-z!/].{0,40}",
                                                                  outside_code).group(0)
