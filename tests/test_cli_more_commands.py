"""More of the command line: `download` with no task id, `status --all`,
`generate --estimate`, `anonymize`, and `--ids` for scripts. Assertions are
on requests, files, exit statuses and message keys, never on wording."""
from __future__ import annotations

import csv
import os
import stat
from pathlib import Path

import pytest

import examples_helpers as helpers
from conftest import scan_bytes
from nalvera import cli, messages
from test_cli import run_cli

pytestmark = pytest.mark.usefixtures("cli_dialect")

MODEL, TASK = "totalsegmentator_ct", "total"
GEN = ["--model", "nv_generate_ctmr", "--task", "rflow-ct", "--param", "mode=image-only",
       "--param", "output_size=[256,256,128]", "--param", "spacing=[1.5,1.5,2.0]"]
CODE_FILE = r"scan-[0-9A-HJKMNP-TV-Z]{8}\.nii\.gz"
POSIX_NOT_ROOT = os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() != 0


def _files(folder: Path):
    return sorted(p.name for p in folder.iterdir() if not p.name.startswith(".")) \
        if folder.exists() else []


def _queue(env, *names, extra=()):
    paths = []
    for name in names:
        path = env / name
        path.write_bytes(scan_bytes(name.encode()))
        paths.append(str(path))
    status, out, err = run_cli("run", *paths, "--model", MODEL, "--budget", "personal",
                               "--no-wait", *extra)
    assert status == 0, err
    return out, err


# ── nalvera download with no task id ──────────────────────────────────────────
def test_download_without_ids_saves_each_new_result_of_the_local_record_once(
        env, server, stub_anon, no_sleep):
    server.script = [("completed", None)]
    website = server.add_task("completed")                   # not in the local record
    _queue(env, "a.nii.gz", "b.nii.gz")
    status, _out, err = run_cli("download", "--out", str(env / "dl"))
    assert status == 0, err
    assert _files(env / "dl") == [f"a__{MODEL}__{TASK}.nii.gz", f"b__{MODEL}__{TASK}.nii.gz"]
    assert not server.calls("GET", f"/api/v1/tasks/{website}/result")
    fetched = len(server.calls("GET", "/result"))
    assert run_cli("download", "--out", str(env / "dl"))[0] == 0
    assert len(server.calls("GET", "/result")) == fetched              # nothing twice
    (env / "dl" / f"a__{MODEL}__{TASK}.nii.gz").unlink()
    assert run_cli("download", "--out", str(env / "dl"))[0] == 0
    assert len(server.calls("GET", "/result")) == fetched + 1          # deleted: fetched again
    assert _files(env / "dl") == [f"a__{MODEL}__{TASK}.nii.gz", f"b__{MODEL}__{TASK}.nii.gz"]
    assert run_cli("download", _task_of(env, "a.nii.gz"), "--out", str(env / "dl"))[0] == 0
    assert len(server.calls("GET", "/result")) == fetched + 1          # saved: not again


def _task_of(env, original_name: str) -> str:
    rows = _mapping_rows(Path(os.environ["NALVERA_MAPPING"]))
    return next(r["task_id"] for r in rows if r["original_name"] == original_name
                and r["task_id"])


def test_download_without_ids_leaves_running_tasks_for_a_later_call(env, server, stub_anon,
                                                                   no_sleep):
    server.script = [("completed", None)]
    _queue(env, "a.nii.gz", "b.nii.gz")
    later = _task_of(env, "b.nii.gz")
    server.tasks[later]["script"] = [("processing", None)]
    status, _out, _err = run_cli("download", "--out", str(env / "dl"))
    assert status == 0 and _files(env / "dl") == [f"a__{MODEL}__{TASK}.nii.gz"]
    server.tasks[later]["script"] = [("completed", None)]
    assert run_cli("download", "--out", str(env / "dl"))[0] == 0
    assert _files(env / "dl") == [f"a__{MODEL}__{TASK}.nii.gz", f"b__{MODEL}__{TASK}.nii.gz"]


def test_download_without_ids_and_without_a_local_record_is_a_usage_error(env, server):
    assert not Path(os.environ["NALVERA_MAPPING"]).exists()
    status, _out, err = run_cli("download")
    assert status == cli.EXIT_USAGE and err
    assert server.requests == []


# ── nalvera status --all ──────────────────────────────────────────────────────
def test_status_all_lists_finished_tasks_newest_first(env, server):
    ids = [server.add_task(state) for state in ("completed", "failed", "cancelled", "queued")]
    status, out, _err = run_cli("status", "--all")
    assert status == 0
    [listing] = server.calls("GET", "/api/v1/tasks")
    assert "state=all" in listing["query"] and "limit=50" in listing["query"]
    positions = [out.index(task_id) for task_id in reversed(ids)]
    assert positions == sorted(positions)                       # newest first


def test_status_all_takes_a_limit_and_status_alone_is_unchanged(env, server):
    for state in ("completed", "failed", "queued"):
        server.add_task(state)
    status, out, _err = run_cli("status", "--all", "--limit", "2")
    assert status == 0 and "limit=2" in server.calls("GET", "/api/v1/tasks")[-1]["query"]
    assert server.order[0] not in out and server.order[2] in out
    status, out, _err = run_cli("status")
    assert status == 0
    assert "state=active" in server.calls("GET", "/api/v1/tasks")[-1]["query"]


@pytest.mark.parametrize("limit", ["0", "101", "many"])
def test_a_limit_outside_the_listing_range_is_a_usage_error(env, server, limit):
    assert run_cli("status", "--all", "--limit", limit)[0] == cli.EXIT_USAGE
    assert server.requests == []


def test_the_limit_follows_the_platforms_listing():
    import inspect

    import fake_server
    from nalvera import Client
    assert cli.STATUS_LIMIT_RANGE == fake_server.TASK_LIST_LIMIT      # checked against api_v1
    assert cli.STATUS_LIMIT_DEFAULT == inspect.signature(Client.tasks).parameters["limit"].default


# ── nalvera generate --estimate ───────────────────────────────────────────────
def test_generate_estimate_queues_nothing_and_needs_no_budget(env, server):
    server.add_lab(12, "Imaging Lab", 40.0)               # a lab, no terminal, no --budget
    status, out, err = run_cli("generate", *GEN, "--estimate")
    assert status == 0, err
    assert len(server.calls("POST", "/api/v1/estimate")) == 1
    assert not server.calls("POST", "/api/v1/tasks")
    assert "1.25" in out and "lab:12" in out and "personal" in out


def test_generate_estimate_of_an_unknown_task_is_a_usage_error(env, server):
    status, _out, _err = run_cli("generate", "--model", "nv_generate_ctmr", "--task", "nope",
                                 "--estimate")
    assert status == cli.EXIT_USAGE and not server.calls("POST")


# ── nalvera anonymize ─────────────────────────────────────────────────────────
@pytest.fixture
def no_key(env, monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    monkeypatch.delenv("NALVERA_EMAIL")
    return env


def _mapping_rows(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_anonymize_writes_only_the_copies_and_records_them_on_this_computer(no_key, server):
    import re

    import nibabel as nib
    import numpy as np
    one = helpers.write_nifti(no_key / "in" / "ct_scan.nii.gz", seed=1, descrip=b"Doe^Jane")
    helpers.write_nifti(no_key / "in" / "cohort" / "a.nii.gz", seed=2)
    helpers.write_nifti(no_key / "in" / "cohort" / "b.nii", seed=3)
    status, out, err = run_cli("anonymize", str(one), str(no_key / "in" / "cohort"),
                               "--out", str(no_key / "anon"))
    assert status == 0, err
    every = sorted(p.name for p in (no_key / "anon").iterdir())      # hidden files too
    assert len(every) == 3 and all(re.fullmatch(CODE_FILE, n) for n in every)
    rows = _mapping_rows(Path(os.environ["NALVERA_MAPPING"]))        # the record
    assert sorted(r["original_name"] for r in rows) == ["a.nii.gz", "b.nii", "ct_scan.nii.gz"]
    assert sorted(r["code"] + ".nii.gz" for r in rows) == every
    first = next(r for r in rows if r["original_name"] == "ct_scan.nii.gz")
    copy = nib.load(str(no_key / "anon" / (first["code"] + ".nii.gz")))
    assert copy.header["descrip"].item() == b""
    assert np.array_equal(np.asarray(copy.dataobj), np.asarray(nib.load(str(one)).dataobj))
    assert "Doe" not in out + err
    assert server.requests == []


def test_anonymize_without_a_record_setting_records_in_the_configuration_folder(
        no_key, server, monkeypatch):
    monkeypatch.delenv("NALVERA_MAPPING")
    scan = helpers.write_nifti(no_key / "ct_scan.nii.gz", seed=1)
    assert run_cli("anonymize", str(scan), "--out", str(no_key / "anon"))[0] == 0
    [record] = list((no_key / "config" / "records").glob("*.csv"))
    assert _mapping_rows(record)[0]["original_name"] == "ct_scan.nii.gz"
    assert stat.S_IMODE((no_key / "config" / "records").stat().st_mode) == 0o700
    assert [p.suffix for p in (no_key / "anon").iterdir()] == [".gz"]
    assert not list(no_key.glob("*.csv"))                       # nothing in the working folder


def test_anonymize_never_overwrites_and_adds_to_its_record(no_key, server):
    scan = helpers.write_nifti(no_key / "ct_scan.nii.gz", seed=1)
    out = no_key / "anon"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    for _ in range(2):
        assert run_cli("anonymize", str(scan), "--out", str(out))[0] == 0
    assert (out / "keep.txt").read_text() == "mine"
    assert len([n for n in _files(out) if n.endswith(".nii.gz")]) == 2
    assert len(_mapping_rows(Path(os.environ["NALVERA_MAPPING"]))) == 2


def test_anonymize_into_the_folder_of_the_runs_mapping_records_there(no_key, server,
                                                                     monkeypatch):
    scan = helpers.write_nifti(no_key / "ct_scan.nii.gz", seed=1)
    monkeypatch.setenv("NALVERA_MAPPING", str(no_key / "nalvera_scans.csv"))
    assert run_cli("anonymize", str(scan), "--out", str(no_key))[0] == 0
    [row] = _mapping_rows(no_key / "nalvera_scans.csv")
    assert row["original_name"] == "ct_scan.nii.gz" and not row["task_id"]


def test_anonymize_goes_on_past_a_file_that_is_not_a_scan(no_key, server):
    good = helpers.write_nifti(no_key / "good.nii.gz", seed=1)
    bad = no_key / "notes.nii.gz"
    bad.write_bytes(b"not a scan")
    status, out, _err = run_cli("anonymize", str(good), str(bad), "--out", str(no_key / "anon"))
    assert status == cli.EXIT_FAILED and "notes.nii.gz" in out
    assert len([n for n in _files(no_key / "anon") if n.endswith(".nii.gz")]) == 1


def test_anonymize_needs_an_output_folder(no_key, server):
    scan = helpers.write_nifti(no_key / "ct_scan.nii.gz", seed=1)
    assert run_cli("anonymize", str(scan))[0] == cli.EXIT_USAGE


@pytest.mark.skipif(not POSIX_NOT_ROOT, reason="needs POSIX permissions and a non-root user")
def test_anonymize_into_a_folder_that_cannot_be_written_is_a_usage_error(no_key, server):
    scan = helpers.write_nifti(no_key / "ct_scan.nii.gz", seed=1)
    locked = no_key / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert run_cli("anonymize", str(scan), "--out", str(locked))[0] == cli.EXIT_USAGE
    finally:
        locked.chmod(stat.S_IRWXU)


# ── --ids: only the queued task ids on standard output ────────────────────────
def test_run_no_wait_ids_prints_only_the_ids(env, server, stub_anon, no_sleep):
    out, err = _queue(env, "a.nii.gz", "b.nii.gz", extra=("--ids",))
    ids = out.splitlines()
    assert sorted(ids) == sorted(server.order) and len(ids) == 2
    assert ids == [_task_of(env, "a.nii.gz"), _task_of(env, "b.nii.gz")]    # in scan order
    assert messages.text("run.no_wait_hint", ids=" ".join(ids)) in err


def test_generate_no_wait_ids_prints_only_the_id(env, server, no_sleep):
    status, out, err = run_cli("generate", *GEN, "--budget", "personal", "--no-wait", "--ids")
    assert status == 0, err
    assert out.splitlines() == server.order and len(server.order) == 1


def test_ids_of_tasks_queued_before_a_run_stopped_still_reach_standard_output(
        env, server, stub_anon, no_sleep):
    server.personal_credits = 1.3                          # pays for one task of two
    paths = []
    for name in ("a.nii.gz", "b.nii.gz"):
        path = env / name
        path.write_bytes(scan_bytes(name.encode()))
        paths.append(str(path))
    status, out, _err = run_cli("run", *paths, "--model", MODEL, "--budget", "personal",
                                "--no-wait", "--ids")
    assert status == cli.EXIT_CREDITS
    assert out.splitlines() == server.order and len(server.order) == 1


def test_run_ids_while_waiting_prints_the_ids_and_saves_the_results(env, server, stub_anon,
                                                                   no_sleep):
    scan = env / "a.nii.gz"
    scan.write_bytes(scan_bytes(b"a"))
    status, out, err = run_cli("run", str(scan), "--model", MODEL, "--budget", "personal",
                               "--ids", "--out", str(env / "res"))
    assert status == 0, err
    assert out.splitlines() == server.order
    assert _files(env / "res") == [f"a__{MODEL}__{TASK}.nii.gz"]
