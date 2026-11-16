"""One source, two dialects: a text names what a person types
through a fragment, rendered as Python in the library and as a command on
the command line; neither rendering names the other interface's forms."""
from __future__ import annotations

import argparse
import inspect
import io
import re

import pytest

import nalvera
from nalvera import cli, messages

COMMAND_RX = re.compile(r"(?<![\w./\\-])nalvera (?:{})\b".format("|".join(
    name for action in cli.build_parser()._actions
    if isinstance(action, argparse._SubParsersAction) for name in action.choices)))
OPTION_RX = re.compile(r"(?<![\w-])--[a-z][a-z-]*")
CALL_RX = re.compile(r"\bnalvera\.\w+\(")
KEYWORD_RX = re.compile(r"(?<![\w.%-])[a-z_]+=")
#: Install lines are the same in both interfaces.
INSTALL_RX = re.compile(r"pip install[^.:\n]*(?:\.(?=\w)[^.:\n]*)*")


def _keys():
    return [k for k in messages.MESSAGES if not k.startswith("help.")]


def _without_install(text: str) -> str:
    return INSTALL_RX.sub("", text)


def _parser_options():
    out = {}
    for action in cli.build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                out[name] = {flag for a in sub._actions for flag in a.option_strings}
    return out


def test_no_text_holds_a_command_or_an_option_outside_a_fragment():
    for key in _keys():
        raw = _without_install(re.sub(r"\{%\w+\}", "", messages.MESSAGES[key]))
        assert not COMMAND_RX.search(raw), key
        assert not OPTION_RX.search(raw), key
        assert not CALL_RX.search(raw), key
        assert "Python:" not in raw, key


def test_every_fragment_has_both_renderings_and_every_one_is_used():
    used = {name for text in messages.MESSAGES.values()
            for name in re.findall(r"\{%(\w+)\}", text)}
    assert used <= set(messages.FRAGMENTS), used - set(messages.FRAGMENTS)
    assert set(messages.FRAGMENTS) <= used, set(messages.FRAGMENTS) - used
    for name, renderings in messages.FRAGMENTS.items():
        assert len(renderings) == 2 and all(r.strip() for r in renderings), name
        assert renderings[0] != renderings[1], name


def _public_parameters():
    names = set()
    for function in [getattr(nalvera, f) for f in nalvera.FUNCTIONS] + [nalvera.Client]:
        names |= set(inspect.signature(function).parameters)
    return names


def test_every_python_rendering_names_a_function_and_arguments_that_exist():
    parameters = _public_parameters()
    for name, (python, _command) in messages.FRAGMENTS.items():
        for function, inside in re.findall(r"\bnalvera\.(\w+)\(([^)]*)\)", python):
            assert function in nalvera.FUNCTIONS, (name, function)
            signature = inspect.signature(getattr(nalvera, function))
            for keyword in re.findall(r"\b(\w+)=", inside):
                assert keyword in signature.parameters, (name, function, keyword)
        bare = re.sub(r"\bnalvera\.\w+\([^)]*\)", "", python)
        bare = re.sub(r"os\.chmod\([^)]*\)", "", bare)
        for keyword in re.findall(r"\b(\w+)=", bare):
            assert keyword in parameters, (name, keyword)
        if python.startswith("python -m "):
            import importlib.util
            assert importlib.util.find_spec(python.split()[-1] + ".__main__"), name
        assert not COMMAND_RX.search(python) and not OPTION_RX.search(python), name


def test_every_command_line_rendering_parses():
    options = _parser_options()
    every = set().union(*options.values())
    settings = {s.name for s in nalvera.SETTINGS}
    for name, (_python, command) in messages.FRAGMENTS.items():
        assert not CALL_RX.search(command), name
        found = re.findall(r"(?<![\w./-])nalvera ([a-z]+)((?: --?[a-z][a-z-]*(?: \S+)?)*)",
                           command)
        for sub, rest in found:
            assert sub in options, (name, sub)
            for flag in re.findall(r"--[a-z][a-z-]*", rest):
                assert flag in options[sub], (name, sub, flag)
        if not found:
            for flag in re.findall(r"--[a-z][a-z-]*", command):
                assert flag in every, (name, flag)
            for word in re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", command):
                assert word in settings, (name, word)


@pytest.mark.parametrize("dialect", messages.DIALECTS)
def test_each_rendering_names_only_its_own_interface(dialect):
    fields = {"ids": "abc123", "model": "m", "budget": "lab:12", "id": "total", "path": "/x"}
    for key in _keys():
        text = _without_install(messages.render(key, dialect, **fields))
        if dialect == messages.PYTHON:
            assert not COMMAND_RX.search(text), (key, text)
            assert not OPTION_RX.search(text), (key, text)
        else:
            assert not CALL_RX.search(text), (key, text)
            assert not [k for k in KEYWORD_RX.findall(text) if k != "name="], (key, text)


def test_the_library_renders_python_and_the_command_line_its_own(env, server, monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    assert messages.dialect() == messages.PYTHON
    with pytest.raises(nalvera.NoCredentials) as caught:
        nalvera.Client()
    assert messages.fragment("login", messages.PYTHON) in str(caught.value)
    err = io.StringIO()
    assert cli.main(["whoami"], stdin=io.StringIO(), stdout=io.StringIO(), stderr=err) == 3
    assert messages.fragment("login", messages.CLI) in err.getvalue()
    assert messages.fragment("login", messages.PYTHON) not in err.getvalue()
    assert messages.dialect() == messages.PYTHON                  # restored after the command


def test_settings_texts_name_no_command():
    for setting in nalvera.SETTINGS:
        for text in (setting.meaning, setting.default):
            assert not COMMAND_RX.search(text) and not CALL_RX.search(text), setting.name


def test_where_the_package_has_its_own_text_the_server_detail_is_not_shown():
    from nalvera.errors import from_response
    for code in ("invalid_credentials", "insufficient_credits", "scope_missing", "not_ready",
                 "rate_limited", "no_free_tasks", "api_disabled", "key_expired"):
        error = from_response(400, {"code": code, "detail": "Run nalvera login (server text)"})
        assert "server text" not in str(error), code
