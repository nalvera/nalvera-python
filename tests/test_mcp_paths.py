"""The local MCP server's folders: every path is resolved, symbolic links
included, and must lie inside an allowed folder before anything is read or
written. Calls the tools directly (no MCP SDK needed)."""
from __future__ import annotations

import os

import pytest

from nalvera.mcp import tools as tools_mod
from test_mcp_support import make_tools, nifti, prepare_env

MODEL = "totalsegmentator_ct"
needs_links = pytest.mark.skipif(not hasattr(os, "symlink"), reason="symbolic links")


@pytest.fixture
def mcp_env(server, tmp_path, monkeypatch):
    return prepare_env(server, tmp_path, monkeypatch)


@pytest.fixture
def local_tools(mcp_env):
    tools = make_tools(mcp_env)
    yield tools
    tools.close(2.0)


def _error(tools, name, arguments):
    failed, data = tools.call(name, arguments)
    assert failed, data
    return data["error"]


def _outside(env):
    secret = env.root / "private"
    secret.mkdir(exist_ok=True)
    return nifti(env, "../private/secret.nii.gz", seed=9)


# ── which folders are allowed ─────────────────────────────────────────────────
def test_the_setting_lists_folders_and_leaves_out_missing_ones(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    value = os.pathsep.join([str(a), " ", str(tmp_path / "missing"), str(b), str(a)])
    assert tools_mod.allowed_folders(value, "/") == [str(a.resolve()), str(b.resolve())]
    assert tools_mod.allowed_folders("", str(a)) == []


def test_by_default_the_start_folder_unless_it_is_the_root_or_home(tmp_path, monkeypatch):
    assert tools_mod.allowed_folders(None, str(tmp_path)) == [str(tmp_path.resolve())]
    assert tools_mod.allowed_folders(None, os.path.abspath(os.sep)) == []
    monkeypatch.setenv("HOME", str(tmp_path))
    assert tools_mod.allowed_folders(None, str(tmp_path)) == []


def test_the_environment_setting_is_read(mcp_env, monkeypatch):
    other = mcp_env.root / "scans"
    other.mkdir()
    monkeypatch.setenv(tools_mod.ALLOWED_DIRS_SETTING, str(other))
    tools = tools_mod.LocalTools(config_dir=str(mcp_env.config))
    try:
        assert tools.allowed == [str(other.resolve())]
        # The record stays in the configuration folder, out of the project.
        assert tools.mapping.parent == mcp_env.config / "records"
    finally:
        tools.close(1.0)


def test_with_no_allowed_folder_every_path_is_refused(mcp_env):
    tools = make_tools(mcp_env, allowed=[])
    try:
        inside = nifti(mcp_env, "ct.nii.gz")
        assert _error(tools, "run_scan", {"paths": [str(inside)], "model": MODEL,
                                          "budget": "personal", "max_credits": 5})["code"] \
            == "path_not_allowed"
        assert _error(tools, "download_result", {"task_id": "abc123"})["code"] == \
            "path_not_allowed"
    finally:
        tools.close(1.0)


# ── what is refused, before anything is read ──────────────────────────────────
def test_a_path_outside_is_refused_before_reading(local_tools, mcp_env, monkeypatch):
    secret = _outside(mcp_env)
    read = []
    anonymize = __import__("nalvera.anonymize", fromlist=["inspect"])
    monkeypatch.setattr(anonymize, "inspect", lambda *a, **k: read.append(a))
    for name, arguments in (
            ("run_scan", {"paths": [str(secret)], "model": MODEL, "budget": "personal",
                          "max_credits": 5}),
            ("estimate_cost", {"model": MODEL, "path": str(secret)}),
            ("download_result", {"task_id": "abc123", "folder": str(secret.parent)})):
        assert _error(local_tools, name, arguments)["code"] == "path_not_allowed"
    assert read == [] and mcp_env.server.requests == []
    assert local_tools.runs.runs == {}


def test_dot_dot_cannot_climb_out(local_tools, mcp_env):
    _outside(mcp_env)
    error = _error(local_tools, "estimate_cost",
                   {"model": MODEL, "path": str(mcp_env.work / ".." / "private" /
                                                "secret.nii.gz")})
    assert error["code"] == "path_not_allowed"
    assert _error(local_tools, "estimate_cost", {"model": MODEL,
                                                 "path": "../private/secret.nii.gz"})[
        "code"] == "path_not_allowed"


@needs_links
def test_a_link_inside_that_points_out_is_refused(local_tools, mcp_env):
    secret = _outside(mcp_env)
    os.symlink(secret, mcp_env.work / "innocent.nii.gz")
    assert _error(local_tools, "run_scan", {"paths": ["innocent.nii.gz"], "model": MODEL,
                                            "budget": "personal", "max_credits": 5})[
        "code"] == "path_not_allowed"


@needs_links
def test_a_folder_holding_a_link_that_points_out_is_refused(local_tools, mcp_env):
    secret = _outside(mcp_env)
    study = mcp_env.work / "study"
    nifti(mcp_env, "study/ct.nii.gz")
    (study / "series").mkdir()
    os.symlink(secret, study / "series" / "IM0001")
    assert _error(local_tools, "run_scan", {"paths": ["study"], "model": MODEL,
                                            "budget": "personal", "max_credits": 5})[
        "code"] == "path_not_allowed"
    assert mcp_env.server.requests == []


@needs_links
def test_a_link_that_stays_inside_is_followed(local_tools, mcp_env):
    real = nifti(mcp_env, "data/ct.nii.gz")
    os.symlink(real, mcp_env.work / "alias.nii.gz")
    assert local_tools.resolve("alias.nii.gz") == str(real.resolve())


def test_a_missing_path_inside_and_outside(local_tools, mcp_env):
    assert _error(local_tools, "estimate_cost", {"model": MODEL, "path": "nothing.nii.gz"})[
        "code"] == "path_not_found"
    # Outside, the answer is the same whether or not the path exists.
    assert _error(local_tools, "estimate_cost", {"model": MODEL,
                                                 "path": "/no/such/file.nii.gz"})["code"] \
        == "path_not_allowed"


def test_a_result_folder_may_be_new_but_must_be_inside(local_tools, mcp_env):
    assert local_tools.resolve("results/new", must_exist=False) == \
        str(mcp_env.work / "results" / "new")
    with pytest.raises(tools_mod.ToolFailure) as caught:
        local_tools.resolve(str(mcp_env.root / "results"), must_exist=False)
    assert caught.value.code == "path_not_allowed"


def test_error_details_name_no_local_folder(local_tools, mcp_env):
    from nalvera import errors
    message = (f"The folder {mcp_env.work / 'study' / 'P01'} holds NIfTI and DICOM files; "
               f"temporary files went to {mcp_env.temp / 'nalvera-run-x'}.")
    detail = local_tools.error_fields(errors.ConfigError(message, code="mixed_folder"))["detail"]
    assert str(mcp_env.work) not in detail and str(mcp_env.temp) not in detail
    assert os.path.join("study", "P01") in detail and "nalvera-run-x" in detail


# ── every code a tool can answer has an instruction ──────────────────────────
def test_every_error_code_the_tools_use_has_a_text(local_tools):
    import re
    from pathlib import Path
    folder = Path(tools_mod.__file__).parent
    source = "\n".join(p.read_text(encoding="utf-8") for p in folder.glob("*.py"))
    used = set(re.findall(r"""ToolFailure\(\s*["']([a-z_]+)["']""", source))
    used |= set(re.findall(r"""return\s+["']([a-z_]+)["'],""", source))
    texts = local_tools.manifest["errors"]
    assert used and sorted(used - set(texts)) == []
    assert "series_not_found" in texts and "timeout" in texts


def test_the_packages_errors_map_to_codes_with_instructions(local_tools):
    from nalvera import errors
    cases = [
        (errors.NoCredentials(), "no_credentials"),
        (errors.NetworkError("down"), "api_unreachable"),
        (errors.UnexpectedResponse("odd", code="unexpected_redirect"), "unexpected_answer"),
        (errors.EncryptionError("no key", code="encrypt_no_trusted_key"), "encryption_error"),
        (errors.ConfigError("bad key", code="invalid_key_format"), "invalid_credentials"),
        (errors.ConfigError("no model", code="unknown_model"), "invalid_request"),
        (errors.ConfigError("disk", code="temp_space_low"), "local_error"),
        (errors.PermissionDenied(code="scope_missing", extra={"required_scope": "tasks:write"}),
         "scope_missing"),
        (errors.from_response(402, {"code": "insufficient_credits", "budget": "lab:12",
                                    "detail": "Too few."}), "insufficient_credits"),
        (errors.from_response(409, {"code": "a_newer_code", "detail": "New."}), "a_newer_code"),
        (RuntimeError("boom with nvk_secret"), "internal_error"),
    ]
    for error, code in cases:
        fields = local_tools.error_fields(error)
        assert fields["code"] == code, (error, fields)
        assert fields["message"] in local_tools.manifest["errors"].values()
        assert "nvk_" not in str(fields)
    assert local_tools.error_fields(cases[7][0])["required_scope"] == "tasks:write"
    assert local_tools.error_fields(cases[8][0])["budget"] == "lab:12"
