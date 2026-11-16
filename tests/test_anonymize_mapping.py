"""The local mapping file (nalvera.anonymize.mapping)."""
from __future__ import annotations

import csv
import os
import stat
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Optional

import pytest

from nalvera.anonymize import AnonymizeError, Mapping
from nalvera.anonymize.mapping import COLUMNS

FAKE_KEY = "nvk_" + "A1b2C3d4" + "x" * 43 + "000000"


@dataclass
class Scan:
    code: str
    original_path: str
    original_name: str
    source: str = "nifti"
    series: Optional[str] = None
    label: Optional[str] = None
    sha256: str = "ab" * 32
    sha256_original: Optional[str] = "cd" * 32


def _scan(code="scan-7F3K2Q9A", name="Hendrickx_Marieke.nii.gz", **extra) -> Scan:
    return Scan(code=code, original_path="/data/" + name, original_name=name, **extra)


def _raw(path) -> str:
    with open(path, encoding="utf-8-sig") as fh:
        return fh.read()


def test_nothing_is_created_by_reading(tmp_path):
    mapping = Mapping(str(tmp_path / "nalvera_scans.csv"))
    assert mapping.rows() == []
    assert mapping.find(code="scan-7F3K2Q9A") == []
    assert os.listdir(tmp_path) == []


def test_created_private_with_a_header_and_a_notice(tmp_path):
    path = tmp_path / "nalvera_scans.csv"
    Mapping(str(path)).record_scan(_scan())
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    with open(path, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    assert tuple(header) == COLUMNS
    notice = tmp_path / "nalvera_scans.README.txt"
    assert notice.exists() and "nalvera_scans.csv" in notice.read_text()
    # No temporary file is left behind.
    assert set(os.listdir(tmp_path)) == {".nalvera_scans.csv.lock", "nalvera_scans.csv",
                                         "nalvera_scans.README.txt"}


def test_mode_stays_private_after_changes(tmp_path):
    path = tmp_path / "m.csv"
    mapping = Mapping(str(path))
    mapping.record_scan(_scan())
    os.chmod(path, 0o644)
    mapping.record_task("scan-7F3K2Q9A", "t1", "totalsegmentator_ct", "total")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_scan_task_result_round_trip(tmp_path):
    mapping = Mapping(str(tmp_path / "m.csv"))
    scan = _scan(series="3", label="P012 baseline", source="dicom")
    mapping.record_scan(scan)
    mapping.record_task(scan.code, "task-1", "totalsegmentator_ct", "total")
    mapping.record_result("task-1", str(tmp_path / "out" / "a.nii.gz"))
    [row] = Mapping(str(tmp_path / "m.csv")).rows()
    assert row["code"] == scan.code
    assert row["original_path"] == scan.original_path
    assert row["original_name"] == scan.original_name
    assert row["source"] == "dicom" and row["series"] == "3"
    assert row["label"] == "P012 baseline"
    assert row["sha256"] == scan.sha256 and row["sha256_original"] == scan.sha256_original
    assert row["created_at"].endswith("Z")
    assert (row["task_id"], row["model"], row["task"]) == ("task-1", "totalsegmentator_ct",
                                                           "total")
    assert row["result_path"] == str(tmp_path / "out" / "a.nii.gz")


def test_more_tasks_and_results(tmp_path):
    mapping = Mapping(str(tmp_path / "m.csv"))
    scan = _scan()
    mapping.record_scan(scan)
    mapping.record_scan(scan)                                   # recorded once
    mapping.record_task(scan.code, "t1", "totalsegmentator_ct", "total")
    mapping.record_task(scan.code, "t2", "moose", "clin_ct_organs")
    mapping.record_task(scan.code, "t2", "moose", "clin_ct_organs")   # recorded once
    mapping.record_result("t2", "/r/one.nii.gz")
    mapping.record_result("t2", "/r/two.nii.gz")
    mapping.record_result("t2", "/r/two.nii.gz")                # recorded once
    rows = mapping.rows()
    assert [(r["task_id"], r["result_path"]) for r in rows] == [
        ("t1", ""), ("t2", "/r/one.nii.gz"), ("t2", "/r/two.nii.gz")]
    assert all(r["original_name"] == scan.original_name for r in rows)
    assert len(mapping.find(code=scan.code)) == 3
    assert len(mapping.find(task_id="t2")) == 2
    assert len(mapping.find(original=scan.original_name)) == 3
    assert len(mapping.find(original=scan.original_path, task_id="t1")) == 1
    assert mapping.find(code="scan-0000000Z") == []


def test_generation_has_no_scan(tmp_path):
    mapping = Mapping(str(tmp_path / "m.csv"))
    mapping.record_task(None, "gen-1", "nv_generate_ctmr", "rflow-ct")
    mapping.record_result("gen-1", "/r/gen.nii.gz")
    [row] = mapping.rows()
    assert row["code"] == "" and row["task_id"] == "gen-1"
    assert row["result_path"] == "/r/gen.nii.gz"


def test_result_for_an_unknown_task(tmp_path):
    mapping = Mapping(str(tmp_path / "m.csv"))
    mapping.record_result("web-task", "/r/x.nii.gz")
    assert mapping.find(task_id="web-task")[0]["result_path"] == "/r/x.nii.gz"


@pytest.mark.parametrize("name", [
    '=HYPERLINK("http://x","y").nii.gz', "+cmd.nii.gz", "-2+3.nii.gz", "@SUM(A1).nii.gz",
    "\tTab.nii.gz", "\rCR.nii.gz", "'quoted.nii.gz", "''twice.nii.gz", "plain.nii.gz",
    "Müller, Zoë \"x\"\nline2.nii.gz",
])
def test_formulas_are_neutralised(tmp_path, name):
    path = tmp_path / "m.csv"
    mapping = Mapping(str(path))
    mapping.record_scan(_scan(name=name))
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    for cell in rows[1]:
        assert not cell.startswith(("=", "+", "-", "@", "\t", "\r"))
    assert mapping.rows()[0]["original_name"] == name


def test_api_key_never_written(tmp_path):
    path = tmp_path / "m.csv"
    mapping = Mapping(str(path))
    mapping.record_scan(_scan(name=f"scan {FAKE_KEY}.nii.gz"))
    mapping.record_task("scan-7F3K2Q9A", FAKE_KEY, "m", "t")
    assert FAKE_KEY not in _raw(path)
    assert FAKE_KEY[12:] not in _raw(path)
    assert "nvk_A1b2C3d4" in _raw(path)


def test_refuses_a_file_it_did_not_write(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("name;value\nHendrickx;1\n")
    with pytest.raises(AnonymizeError) as caught:
        Mapping(str(path)).record_scan(_scan())
    assert caught.value.code == "mapping_unreadable"
    assert path.read_text() == "name;value\nHendrickx;1\n"


def test_keeps_columns_added_in_a_spreadsheet(tmp_path):
    path = tmp_path / "m.csv"
    mapping = Mapping(str(path))
    mapping.record_scan(_scan())
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    rows[0].append("notes")
    rows[1].append("follow up in May")
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    mapping.record_task("scan-7F3K2Q9A", "t1", "m", "t")
    [row] = mapping.rows()
    assert row["notes"] == "follow up in May"
    assert row["task_id"] == "t1"


def test_never_holds_the_scan_bytes_or_key(tmp_path):
    """The columns are the fixed ones; nothing else is written."""
    path = tmp_path / "m.csv"
    Mapping(str(path)).record_scan(_scan())
    with open(path, encoding="utf-8-sig", newline="") as fh:
        assert tuple(next(csv.reader(fh))) == COLUMNS


WRITER = textwrap.dedent("""
    import contextlib, os, sys, time
    from dataclasses import dataclass
    from nalvera.anonymize import Mapping

    if os.environ.get("WRITER_NO_LOCK"):        # only to show the test has teeth
        Mapping._locked = lambda self: contextlib.nullcontext()

    @dataclass
    class Scan:
        code: str
        original_path: str
        original_name: str
        source: str = "nifti"
        series: str = None
        label: str = None
        sha256: str = "0" * 64
        sha256_original: str = None

    path, worker, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    mapping = Mapping(path)
    # Start together: announce readiness, then wait for the go file.
    open(path + ".ready-" + worker, "w").close()
    deadline = time.monotonic() + 60
    while not os.path.exists(path + ".go") and time.monotonic() < deadline:
        time.sleep(0.005)
    for index in range(count):
        code = "scan-W%sN%05d" % (worker, index)
        mapping.record_scan(Scan(code, "/d/" + code, code + ".nii.gz"))
        mapping.record_task(code, "task-" + code, "m", "t")
""")


def run_writers(tmp_path, workers: int, count: int, extra_env=None) -> str:
    """Run ``workers`` processes that each record ``count`` scans and tasks
    into one mapping file, all starting at the same moment."""
    import time

    script = tmp_path / "writer.py"
    script.write_text(WRITER)
    path = str(tmp_path / "shared.csv")
    env = dict(os.environ, **(extra_env or {}))
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")]
        + [p for p in sys.path if p])
    procs = [subprocess.Popen([sys.executable, str(script), path, str(w), str(count)], env=env)
             for w in range(workers)]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not all(
            os.path.exists(f"{path}.ready-{w}") for w in range(workers)):
        time.sleep(0.01)
    open(path + ".go", "w").close()
    for proc in procs:
        assert proc.wait(timeout=120) == 0
    return path


def test_concurrent_writers_lose_nothing(tmp_path):
    workers, count = 6, 20
    path = run_writers(tmp_path, workers, count)
    rows = Mapping(str(path)).rows()
    assert len(rows) == workers * count
    assert len({r["code"] for r in rows}) == workers * count
    assert all(r["task_id"] == "task-" + r["code"] for r in rows)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        lines = list(csv.reader(fh))
    assert lines[0] == list(COLUMNS)
    assert len(lines) == 1 + workers * count
