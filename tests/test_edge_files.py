"""Files on the user's computer."""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

import nalvera
import stub_anonymizer
from conftest import TTYStringIO, scan_bytes
from nalvera import cli, messages
from nalvera import client as client_module
from nalvera.errors import ConfigError, ConfirmationRequired, RunInterrupted
from nalvera.report import StreamReporter
from nalvera.upload import Uploader

MODEL, TASK = "totalsegmentator_ct", "total"
POSIX_NOT_ROOT = os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() != 0


def _dicom_folder(base: Path, name: str, series=("s1", "s2", "s3")) -> Path:
    folder = base / name
    folder.mkdir(parents=True)
    for s in series:
        (folder / f"{s}.dcm").write_bytes(scan_bytes(s.encode()))
    return folder


# ── a folder is not always DICOM ──────────────────────────────────────────────
def test_a_folder_of_nifti_files_gives_one_task_per_top_level_file(client, server, env, out):
    folder = env / "input" / "cohort"
    (folder / "deeper").mkdir(parents=True)
    (folder / "a.nii.gz").write_bytes(scan_bytes(b"a"))
    (folder / "b.nii").write_bytes(scan_bytes(b"b"))
    (folder / "readme.txt").write_text("notes")
    (folder / "deeper" / "c.nii.gz").write_bytes(scan_bytes(b"c"))
    tasks = client.run(folder, model=MODEL, budget="personal")
    assert sorted(t.original_name for t in tasks) == ["a.nii.gz", "b.nii"]
    assert messages.text("run.folder_nifti", path=str(folder), count=2) in out.getvalue()


def test_a_folder_with_nifti_and_dicom_files_stops_and_names_it(client, server, env):
    folder = env / "input" / "mixed"
    folder.mkdir(parents=True)
    (folder / "a.nii.gz").write_bytes(scan_bytes(b"a"))
    (folder / "x.dcm").write_bytes(scan_bytes(b"x"))
    with pytest.raises(ConfigError) as caught:
        client.run(folder, model=MODEL, budget="personal")
    assert caught.value.code == "mixed_folder" and str(folder) in str(caught.value)
    assert stub_anonymizer.prepared == [] and server.calls("POST", "/api/v1/uploads") == []


def test_an_empty_folder_stops_and_names_it(client, server, env):
    folder = env / "input" / "empty"
    folder.mkdir(parents=True)
    (folder / "notes.txt").write_text("nothing here")
    with pytest.raises(ConfigError) as caught:
        client.run(folder, model=MODEL, budget="personal")
    assert caught.value.code == "no_scans_in_folder" and str(folder) in str(caught.value)
    assert cli.exit_status(caught.value) == cli.EXIT_USAGE


def test_a_file_that_is_not_a_scan_is_named_in_its_failure(client, server, scans):
    good = scans("good.nii.gz")
    bad = scans("photo.jpg", b"NOTASCAN jpeg")
    result = client.run([good, bad], model=MODEL, budget="personal")
    assert [(f.original_name, f.code) for f in result.failed] == [("photo.jpg", "not_a_scan")]


# ── the same file name in many folders ────────────────────────────────────────
def test_results_of_scans_with_one_name_carry_their_folder_and_never_overwrite(client, server,
                                                                               env):
    paths = []
    for patient in ("patient01", "patient02"):
        path = env / "input" / patient / "ct.nii.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(scan_bytes(patient.encode()))
        paths.append(path)
    done = client.wait(client.run(paths, model=MODEL, budget="personal"))
    saved = client.download(done, out=env / "results")
    assert sorted(p.name for p in saved) == [f"patient01_ct__{MODEL}__{TASK}.nii.gz",
                                             f"patient02_ct__{MODEL}__{TASK}.nii.gz"]
    fetched = len(server.calls("GET", "/result"))
    again = client.download(done, out=env / "results")              # a cell run twice
    assert list(again) == list(saved) and len(server.calls("GET", "/result")) == fetched
    saved[0].write_bytes(b"changed")                                # not ours any more
    third = client.download(done, out=env / "results")
    assert third[0].name == f"patient01_ct__{MODEL}__{TASK}-2.nii.gz" and third[1] == saved[1]
    assert saved[0].read_bytes() == b"changed" and all(p.exists() for p in saved)


# ── a cost surprise in a DICOM folder ─────────────────────────────────────────
def test_a_dicom_folder_with_several_scans_needs_a_yes_without_a_terminal(client, server, env):
    folder = _dicom_folder(env / "input", "Patient_DICOM")
    with pytest.raises(ConfirmationRequired) as caught:
        client.run(folder, model=MODEL, budget="personal")
    assert caught.value.code == "confirmation_required"
    assert caught.value.extra["count"] == 3 and caught.value.extra["credits"] == "3.75"
    assert server.calls("POST", "/api/v1/uploads") == []
    assert cli.exit_status(caught.value) == cli.EXIT_CREDITS


@pytest.mark.parametrize("how", ["yes", "max_credits"])
def test_yes_or_a_cap_goes_ahead(client, server, env, how):
    folder = _dicom_folder(env / "input", "Patient_DICOM")
    kwargs = {"yes": True} if how == "yes" else {"max_credits": 10}
    assert len(client.run(folder, model=MODEL, budget="personal", **kwargs)) == 3


@pytest.mark.parametrize("answer,queued", [("y\n", 3), ("n\n", 0), ("\n", 0)])
def test_a_terminal_is_asked(make_client, server, env, answer, queued):
    folder = _dicom_folder(env / "input", "Patient_DICOM")
    out = io.StringIO()
    client = make_client(interactive=True, input=io.StringIO(answer), output=out)
    if queued:
        assert len(client.run(folder, model=MODEL, budget="personal")) == queued
    else:
        with pytest.raises(ConfirmationRequired) as caught:
            client.run(folder, model=MODEL, budget="personal")
        assert caught.value.code == "confirmation_declined"
        assert server.calls("POST", "/api/v1/uploads") == []
    assert messages.text("run.confirm_dicom", count=3, credits="3.75") in out.getvalue()


def test_a_dicom_folder_with_one_scan_needs_no_question(client, server, env):
    folder = _dicom_folder(env / "input", "One_series", series=("s1",))
    assert len(client.run(folder, model=MODEL, budget="personal")) == 1


# ── what a conversion left out is reported (the anonymizer decides) ───────────
def test_dropped_volumes_left_out_files_and_a_partial_conversion_are_reported(
        make_client, server, env, stub_anon, monkeypatch):
    from types import SimpleNamespace
    folder = _dicom_folder(env / "input", "Patient_DICOM", series=("s3",))
    real = stub_anon.prepare

    class Result(list):
        pass

    def converted(path, **kwargs):
        made = Result(real(path, **kwargs))
        made.dropped = [SimpleNamespace(series="3", reason="uncorrected", kept="3_Tilt_1")]
        made.left_out, made.partial = 4, True
        return made

    monkeypatch.setattr(stub_anon, "prepare", converted)
    reporter = nalvera.ListReporter()
    make_client(reporter=reporter, output=None).run(folder, model=MODEL, budget="personal")
    events = {e.key: e.fields for e in reporter.events}
    assert events["run.dicom_dropped"]["series"] == "3"
    assert events["run.dicom_dropped"]["reason"] == messages.text("dropped.uncorrected")
    assert events["run.dicom_left_out"]["count"] == 4
    assert events["run.dicom_partial"]["count"] == 1


# ── disk and permissions, checked before any upload ───────────────────────────
@pytest.mark.skipif(not POSIX_NOT_ROOT, reason="POSIX permissions, not as root")
def test_a_mapping_folder_that_cannot_be_written(make_client, server, scans, env):
    locked = env / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        client = make_client(mapping=locked / "nalvera_scans.csv")
        with pytest.raises(ConfigError) as caught:
            client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    finally:
        locked.chmod(0o700)
    assert caught.value.code == "mapping_not_writable" and str(locked) in str(caught.value)
    assert server.calls("POST", "/api/v1/uploads") == []


def test_a_mapping_file_this_package_did_not_write_is_left_alone(make_client, server, scans,
                                                                 env):
    foreign = env / "my_spreadsheet.csv"
    foreign.write_text("patient,birthday\nDoe,1970\n")
    with pytest.raises(ConfigError) as caught:
        make_client(mapping=foreign).run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert caught.value.code == "mapping_unreadable"
    assert foreign.read_text() == "patient,birthday\nDoe,1970\n"
    assert server.calls("POST", "/api/v1/uploads") == []


@pytest.mark.skipif(not POSIX_NOT_ROOT, reason="POSIX permissions, not as root")
def test_an_output_folder_that_cannot_be_written_stops_before_uploading(env, server,
                                                                        stub_anon, no_sleep):
    from test_cli import run_cli
    scan = env / "a.nii.gz"
    scan.write_bytes(scan_bytes(b"a"))
    locked = env / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        status, _out, err = run_cli("run", str(scan), "--model", MODEL, "--budget", "personal",
                                    "--out", str(locked / "results"))
    finally:
        locked.chmod(0o700)
    assert status == cli.EXIT_USAGE and server.calls("POST", "/api/v1/uploads") == []


def test_too_little_temporary_space_stops_before_uploading(client, server, scans, monkeypatch):
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda path: real(path)._replace(free=10))
    with pytest.raises(ConfigError) as caught:
        client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert caught.value.code == "temp_space_low"
    assert server.calls("POST", "/api/v1/uploads") == []
    assert os.listdir(tempfile.gettempdir()) == []               # its folder was removed


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks")
def test_a_link_that_leaves_the_folder_is_refused(client, server, env):
    outside = env / "elsewhere.nii.gz"
    outside.write_bytes(scan_bytes(b"x"))
    folder = env / "input" / "cohort"
    folder.mkdir(parents=True)
    (folder / "a.nii.gz").write_bytes(scan_bytes(b"a"))
    os.symlink(outside, folder / "b.nii.gz")
    with pytest.raises(ConfigError) as caught:
        client.run(folder, model=MODEL, budget="personal")
    assert caught.value.code == "path_outside_folder"
    assert server.calls("POST", "/api/v1/uploads") == []


# ── left-over temporary folders ───────────────────────────────────────────────
def test_stale_run_folders_are_removed_at_the_start(client, server, scans):
    base = Path(tempfile.gettempdir())
    old, fresh, other = (base / "nalvera-run-old"), (base / "nalvera-run-fresh"), \
        (base / "someone-else-old")
    for folder in (old, fresh, other):
        folder.mkdir()
        (folder / "scan-AAAAAAAA.nii.gz").write_bytes(b"x")
    two_days_ago = time.time() - 2 * 86400
    os.utime(old, (two_days_ago, two_days_ago))
    os.utime(other, (two_days_ago, two_days_ago))
    client.run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert not old.exists() and fresh.exists() and other.exists()


# ── Ctrl-C ────────────────────────────────────────────────────────────────────
def test_ctrl_c_during_uploads_says_what_was_queued_and_cleans_up(client, server, scans,
                                                                  monkeypatch):
    real = Uploader.upload
    count = {"n": 0}

    def interrupted(self, path, **kwargs):
        count["n"] += 1
        if count["n"] == 2:
            raise KeyboardInterrupt
        return real(self, path, **kwargs)

    monkeypatch.setattr(Uploader, "upload", interrupted)
    with pytest.raises(RunInterrupted) as caught:
        client.run([scans(f"s{i}.nii.gz") for i in range(4)], model=MODEL, budget="personal")
    assert len(caught.value.tasks) <= 1 and len(caught.value.tasks) == len(server.tasks)
    assert os.listdir(tempfile.gettempdir()) == []
    assert isinstance(caught.value, KeyboardInterrupt)


def test_ctrl_c_on_the_command_line_names_the_queued_tasks(cli_dialect, env, server, stub_anon,
                                                            no_sleep,
                                                           monkeypatch):
    real = Uploader.upload
    count = {"n": 0}

    def interrupted(self, path, **kwargs):
        count["n"] += 1
        if count["n"] == 2:
            raise KeyboardInterrupt
        return real(self, path, **kwargs)

    monkeypatch.setattr(Uploader, "upload", interrupted)
    files = []
    for name in ("a", "b"):
        path = env / f"{name}.nii.gz"
        path.write_bytes(scan_bytes(name.encode()))
        files.append(str(path))
    out, err = io.StringIO(), io.StringIO()
    status = cli.main(["run", *files, "--model", MODEL, "--budget", "personal"],
                      stdin=io.StringIO(), stdout=out, stderr=err)
    assert status == cli.EXIT_INTERRUPTED
    assert messages.text("run.interrupted") in err.getvalue()
    for task_id in server.tasks:
        assert task_id in out.getvalue()


def test_ctrl_c_while_waiting_names_the_command_that_waits_again(cli_dialect, env, server,
                                                                no_sleep, monkeypatch):
    waiting = server.add_task("queued")
    from nalvera.client import Client

    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Client, "wait", interrupted)
    out, err = io.StringIO(), io.StringIO()
    status = cli.main(["wait", waiting], stdin=io.StringIO(), stdout=out, stderr=err)
    assert status == cli.EXIT_INTERRUPTED
    assert messages.text("wait.interrupted") in err.getvalue()
    assert messages.fragment("wait", messages.CLI) in err.getvalue()


def test_a_second_ctrl_c_exits_at_once():
    exits = []
    guard = cli._InterruptGuard(io.StringIO(), exit_now=exits.append)
    with pytest.raises(KeyboardInterrupt):
        guard.handler(2, None)
    assert exits == []
    guard.handler(2, None)
    assert exits == [cli.EXIT_INTERRUPTED]


# ── no mapping file, or another folder ────────────────────────────────────────
def test_status_wait_and_download_work_without_the_mapping(make_client, server, scans, env,
                                                           monkeypatch):
    task = make_client().run(scans("Doe_John.nii.gz"), model=MODEL, budget="personal")[0]
    (env / "nalvera_scans.csv").unlink()
    elsewhere = env / "other_folder"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    later = make_client()
    assert later.task(task.id).display_name == f"{task.code}.nii.gz"
    done = later.wait([task.id])
    saved = later.download(done, out=elsewhere / "results")
    assert [p.name for p in saved] == [f"{task.code}__{MODEL}__{TASK}.nii.gz"]
    assert "Doe_John" not in saved[0].name


# ── a console that cannot show every character ────────────────────────────────
def test_a_console_without_unicode_gets_question_marks_not_a_crash(env, monkeypatch):
    monkeypatch.delenv("NALVERA_API_KEY")
    monkeypatch.delenv("NALVERA_EMAIL")
    raw_out, raw_err = io.BytesIO(), io.BytesIO()
    out = io.TextIOWrapper(raw_out, encoding="cp1252")
    err = io.TextIOWrapper(raw_err, encoding="cp1252")
    status = cli.main(["whoami"], stdin=io.StringIO(), stdout=out, stderr=err)
    err.flush()
    assert status == cli.EXIT_AUTH
    shown = raw_err.getvalue().decode("cp1252")
    assert "⋮" in messages.text("error.no_credentials")         # the text that needed it
    assert "Settings ? API" in shown and "(?)" in shown


def test_the_reporter_replaces_what_a_stream_cannot_encode():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii")
    StreamReporter(output=stream).say("error.no_credentials")
    stream.flush()
    assert b"?" in raw.getvalue() and b"Settings" in raw.getvalue()


# ── odd file names stay local and harmless ────────────────────────────────────
@pytest.mark.skipif(os.name != "posix", reason="a newline in a file name")
def test_odd_file_names_are_shown_on_one_line_and_saved_safely(make_client, server, env):
    names = ["=1+1 \"quoted\" é.nii.gz", "two\nlines.nii.gz", "x" * 150 + ".nii.gz"]
    paths = []
    for name in names:
        path = env / "input" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(scan_bytes(name.encode()))
        paths.append(path)
    out = io.StringIO()
    client = make_client(output=out)
    done = client.wait(client.run(paths, model=MODEL, budget="personal"))
    saved = client.download(done, out=env / "results")
    assert len(saved) == 3
    for line in out.getvalue().splitlines():
        assert not line.startswith("lines.nii.gz")                # no forged line
    for path in saved:
        assert path.parent == env / "results"
        assert "\n" not in path.name and len(path.name) < 255
    assert all(server.tasks[t]["body"]["name"].startswith("scan-") for t in server.tasks)


# ── Jupyter and scripts with no terminal ──────────────────────────────────────
def test_without_a_terminal_progress_is_plain_lines(make_client, server, scans):
    out = io.StringIO()
    client = make_client(output=out)
    client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    assert "\x1b" not in out.getvalue()
    screen = TTYStringIO()
    tty_client = make_client(output=screen)
    tty_client.wait(tty_client.run(scans("b.nii.gz"), model=MODEL, budget="personal"))
    assert "\x1b[" in screen.getvalue()


def test_without_a_terminal_nothing_is_ever_read(make_client, server, scans):
    class NoInput(io.StringIO):
        def readline(self, *args):
            raise AssertionError("read from a stream with no terminal")

    server.add_lab(12, "Imaging Lab", 100.0)
    client = make_client(input=NoInput(), output=io.StringIO(), interactive=None)
    with pytest.raises(nalvera.BudgetChoiceRequired) as caught:
        client.run(scans("a.nii.gz"), model=MODEL)
    assert {o["budget"] for o in caught.value.options} == {"personal", "lab:12"}


# ── downloads ─────────────────────────────────────────────────────────────────
def test_a_download_that_arrives_short_is_fetched_again(client, server, env):
    task_id = server.add_task("completed")
    server.inject("GET", "/result", after=True, truncate=True)
    path = client.download(task_id, out=env / "results")[0]
    assert path.read_bytes() == server.result_bytes
    assert len(server.calls("GET", "/result")) == 2
    assert not [p for p in (env / "results").iterdir() if p.name.endswith(".part")]


def test_the_short_check_works_without_urllib3s_own_check(client, server, env, monkeypatch):
    # urllib3 1.x does not check Content-Length by itself; the package does.
    task_id = server.add_task("completed")
    real = client_module._expected_length
    # Announced one byte more than arrives, on every attempt.
    monkeypatch.setattr(client_module, "_expected_length",
                        lambda headers: (real(headers) or 0) + 1)
    with pytest.raises(nalvera.NetworkError):
        client.download(task_id, out=env / "results")
    assert len(server.calls("GET", "/result")) == 3
    assert not (env / "results").exists() or not list((env / "results").iterdir())


@pytest.mark.parametrize("disposition,expected", [
    ('attachment; filename="../../evil/report.txt"', ".txt"),
    ('attachment; filename="..\\\\..\\\\outputs.zip"', ".zip"),
    ("attachment; filename*=UTF-8''%2E%2E%2Fsegmentation.nii.gz", ".nii.gz"),
    ('attachment; filename="result.exe;rm -rf"', ".gz"),        # from the content type
])
def test_the_servers_file_name_gives_only_an_extension(client, server, env, disposition,
                                                       expected):
    task_id = server.add_task("completed")
    server.result_disposition = disposition
    path = client.download(task_id, out=env / "results")[0]
    assert path.parent == env / "results" and path.name.endswith(expected)
    assert ".." not in path.name and "/" not in path.name and "\\" not in path.name
    assert path.name.startswith(f"website-scan__{MODEL}__{TASK}")


# ── no dcm2niix on this platform ──────────────────────────────────────────────
def test_the_converter_is_optional_where_no_build_exists():
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    dependencies = text.split("dependencies = [", 1)[1].split("\n]", 1)[0]
    line = next(x for x in dependencies.splitlines() if x.strip().startswith('"dcm2niix'))
    assert "platform_machine" in line                      # installed only with a build
    assert 'dicom = ["dcm2niix' in text
    assert "pydicom" not in dependencies                   # a test tool, never a dependency


def test_nifti_works_without_the_converter(env, server, monkeypatch, fake_time):
    pytest.importorskip("nibabel")
    import nibabel
    import numpy
    monkeypatch.setitem(sys.modules, "dcm2niix", None)     # not importable
    monkeypatch.setenv("PATH", str(env / "no-programs-here"))
    monkeypatch.delenv("NALVERA_DCM2NIIX", raising=False)
    image = nibabel.Nifti1Image(numpy.zeros((4, 4, 4), dtype=numpy.int16), numpy.eye(4))
    path = env / "Doe_John.nii.gz"
    nibabel.save(image, str(path))
    from nalvera import Client
    client = Client(output=io.StringIO(), interactive=False, clock=fake_time.clock,
                    sleep=fake_time.sleep)
    task = client.run(path, model=MODEL, budget="personal")[0]
    assert task.original_name == "Doe_John.nii.gz" and task.code.startswith("scan-")
    client.close()
    assert messages.has("anonymize.dicom_tool_missing")
    assert "pip install dcm2niix" in messages.text("anonymize.dicom_tool_missing")
