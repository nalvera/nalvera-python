"""What the package guards against: a spending cap that free tasks could slip
past, a scan charged twice, a file read outside the allowed folders, a stored
key sent to another platform, a hostile server's names and numbers, an
assistant told more than it needs, and a source archive that packs more than
the files it names."""
from __future__ import annotations

import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import anonymize_helpers as helpers
import examples_helpers
import stub_anonymizer
from conftest import scan_bytes
from test_mcp_support import dicom_ready, make_tools, nifti, prepare_env

MODEL = "totalsegmentator_ct"
needs_links = pytest.mark.skipif(not hasattr(os, "symlink"), reason="symbolic links")
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
needs_dicom = pytest.mark.skipif(not dicom_ready(), reason="needs pydicom and dcm2niix")


# ── the spending cap holds when free tasks vanish ─────────────────────────────
def test_a_named_budget_and_a_cap_hold_when_free_tasks_vanish(make_client, server, scans,
                                                              monkeypatch):
    server.free_tasks_left = 2
    client = make_client()
    real = client._upload_all

    def free_tasks_spent_elsewhere(job, **kwargs):
        server.free_tasks_left = 0              # a website task used them a moment ago
        return real(job, **kwargs)

    monkeypatch.setattr(client, "_upload_all", free_tasks_spent_elsewhere)
    before = server.personal_credits
    result = client.run([scans(f"s{i}.nii.gz") for i in range(3)], model=MODEL,
                        budget="personal", max_credits=1.25)
    assert result.charged_credits <= 1.25
    assert round(before - server.personal_credits, 2) <= 1.25
    assert len(result) == 1
    assert [f.code for f in result.not_queued] == ["max_credits_exceeded"] * 2


def test_free_tasks_that_stay_pay_with_a_named_budget_and_a_cap(make_client, server, scans):
    server.free_tasks_left = 2
    result = make_client().run([scans(f"s{i}.nii.gz") for i in range(3)], model=MODEL,
                               budget="personal", max_credits=1.25)
    assert len(result) == 3 and result.charged_credits == 1.25
    sent = [record["budget"] for record in server.tasks.values()]
    assert sorted(sent) == ["free", "free", "personal"]


# ── two interrupted runs never charge one scan twice ──────────────────────────
@pytest.mark.parametrize("label", [None, "P012 baseline"])
def test_two_interrupted_runs_never_charge_one_scan_twice(make_client, server, scans,
                                                          monkeypatch, label):
    path = scans("a.nii.gz")
    first = make_client()
    real = first._submit

    def submit_then_die(body, key):
        real(body, key)
        raise KeyboardInterrupt

    monkeypatch.setattr(first, "_submit", submit_then_die)
    with pytest.raises(KeyboardInterrupt):
        first.run(path, model=MODEL, budget="personal", label=label)
    second = make_client()
    from nalvera.upload import Uploader
    original_upload = Uploader.upload

    def upload_dies(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Uploader, "upload", upload_dies)
    try:
        # Without a label the earlier task is found by its name and nothing is
        # uploaded; with one, the upload is interrupted.
        second.run(path, model=MODEL, budget="personal", label=label)
    except KeyboardInterrupt:
        pass
    monkeypatch.setattr(Uploader, "upload", original_upload)
    credits = server.personal_credits
    make_client().run(path, model=MODEL, budget="personal", label=label)
    assert len(server.tasks) == 1
    assert server.personal_credits == credits
    codes = {r["code"] for r in stub_anonymizer.Mapping(os.environ["NALVERA_MAPPING"]).rows()}
    assert len(codes) == 1                      # the scan kept its first code


def test_a_recovered_scan_is_sent_under_its_first_code(make_client, server, scans,
                                                      monkeypatch):
    path = scans("a.nii.gz")
    first = make_client()
    from nalvera.upload import Uploader
    original_upload = Uploader.upload

    def upload_dies(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Uploader, "upload", upload_dies)
    with pytest.raises(KeyboardInterrupt):
        first.run(path, model=MODEL, budget="personal")
    monkeypatch.setattr(Uploader, "upload", original_upload)
    rows = stub_anonymizer.Mapping(os.environ["NALVERA_MAPPING"]).rows()
    code = rows[0]["code"]
    result = make_client().run(path, model=MODEL, budget="personal")
    assert result[0].code == code
    assert [record["body"]["name"] for record in server.tasks.values()] == [f"{code}.nii.gz"]


def test_of_several_earlier_codes_the_first_is_tried(make_client, server, scans, monkeypatch):
    """A record an older version left: one labelled scan, two codes without a
    task. The first code's request landed; its key finds that task."""
    path = scans("a.nii.gz")
    first = make_client()
    real = first._submit

    def submit_then_die(body, key):
        real(body, key)
        raise KeyboardInterrupt

    monkeypatch.setattr(first, "_submit", submit_then_die)
    with pytest.raises(KeyboardInterrupt):
        first.run(path, model=MODEL, budget="personal", label="P012 baseline")
    mapping = stub_anonymizer.Mapping(os.environ["NALVERA_MAPPING"])
    [row] = mapping.rows()
    later = stub_anonymizer.PreparedScan(
        code="scan-LATERCDE", filename="scan-LATERCDE.nii.gz", path="-", size=1,
        sha256=row["sha256"], shape=(4, 4, 4), original_path=row["original_path"],
        original_name=row["original_name"], source="nifti", label="P012 baseline",
        sha256_original=row["sha256_original"])
    mapping.record_scan(later)
    credits = server.personal_credits
    result = make_client().run(path, model=MODEL, budget="personal", label="P012 baseline")
    assert len(server.tasks) == 1 and result[0].replayed
    assert server.personal_credits == credits


# ── a code collision never replays another scan's task ────────────────────────
def test_the_idempotency_key_holds_the_anonymized_files_digest():
    from nalvera import client as client_mod
    one = client_mod._scan_idempotency_key("scan-AAAAAAAA", "a" * 64, MODEL, "total")
    two = client_mod._scan_idempotency_key("scan-AAAAAAAA", "b" * 64, MODEL, "total")
    assert one != two


def test_a_reused_code_never_replays_another_scans_task(make_client, server, scans,
                                                        monkeypatch, tmp_path):
    monkeypatch.setattr(stub_anonymizer, "_new_code", lambda: "scan-SAMECODE")
    first = make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    # The record is lost (another computer, a deleted file): the next scan gets the
    # same code, and its submit must not be answered with the first scan's task.
    monkeypatch.setenv("NALVERA_MAPPING", str(tmp_path / "other.csv"))
    second = make_client().run(scans("b.nii.gz", scan_bytes(b"other")), model=MODEL,
                               budget="personal")
    assert second[0].id != first[0].id and not second[0].replayed


def test_a_code_the_record_holds_is_drawn_again(env, monkeypatch, tmp_path):
    from nalvera.anonymize import names, prepare
    drawn = iter(["scan-AAAAAAAA", "scan-BBBBBBBB"])
    monkeypatch.setattr(names, "new_code", lambda: next(drawn))
    import nalvera.anonymize as anonymize
    monkeypatch.setattr(anonymize, "new_code", names.new_code)
    path = examples_helpers.write_nifti(tmp_path / "ct.nii.gz")
    scans = prepare(path, workdir=str(tmp_path / "work"), avoid={"scan-AAAAAAAA"})
    assert scans[0].code == "scan-BBBBBBBB"


# ── a text query is part of a task's identity ─────────────────────────────────
def test_the_same_scan_with_another_text_is_a_new_task(make_client, server, scans):
    path = scans("ct.nii.gz")
    client = make_client()
    one = client.run(path, model="merlin", budget="personal", text="liver lesion")
    two = client.run(path, model="merlin", budget="personal", text="kidney stone")
    again = client.run(path, model="merlin", budget="personal", text="liver lesion")
    assert two[0].id != one[0].id and not two[0].earlier
    assert again[0].id == one[0].id and again[0].earlier
    assert len(server.tasks) == 2


# ── allowed folders cover every file read ─────────────────────────────────────
@needs_links
def test_a_dicom_link_out_of_the_allowed_folders_is_refused(make_client, server, env):
    inside, outside = env / "allowed", env / "elsewhere"
    (inside / "study").mkdir(parents=True)
    outside.mkdir()
    (outside / "img.dcm").write_bytes(scan_bytes(b"dicom"))
    os.symlink(outside / "img.dcm", inside / "study" / "img.dcm")
    from nalvera import ConfigError
    client = make_client(allowed_folders=[str(inside)])
    with pytest.raises(ConfigError) as caught:
        client.run(str(inside / "study"), model=MODEL, budget="personal")
    assert caught.value.code == "path_not_allowed"
    assert not server.uploads


@needs_links
@needs_dicom
def test_a_real_dicom_folder_with_links_out_is_refused(env, server):
    from nalvera import Client, ConfigError
    inside, outside = env / "allowed", env / "private_elsewhere"
    (inside / "study").mkdir(parents=True)
    outside.mkdir()
    helpers.write_dicom_series(str(outside), series_number=7, plants=helpers.DICOM_PLANTS,
                               slices=4, seed=11)
    for name in sorted(os.listdir(outside)):
        os.symlink(outside / name, inside / "study" / name)
    client = Client(allowed_folders=[str(inside)], quiet=True, interactive=False)
    try:
        with pytest.raises(ConfigError) as caught:
            client.run(str(inside / "study"), model=MODEL, budget="personal")
    finally:
        client.close()
    assert caught.value.code == "path_not_allowed" and not server.uploads


@pytest.fixture
def mcp_env(server, tmp_path, monkeypatch):
    return prepare_env(server, tmp_path, monkeypatch)


def test_the_mcp_tree_check_refuses_at_its_entry_limit(mcp_env, monkeypatch):
    from nalvera.mcp import tools as tools_mod
    big = mcp_env.work / "big"
    big.mkdir()
    for i in range(6):
        (big / f"a{i}").write_bytes(b"")
    monkeypatch.setattr(tools_mod, "_TREE_MAX_ENTRIES", 5)
    tools = make_tools(mcp_env)
    try:
        with pytest.raises(tools_mod.ToolFailure) as caught:
            tools.resolve(str(big))
    finally:
        tools.close(2.0)
    assert caught.value.code == "local_error" and caught.value.detail


# ── a stored key goes only to the platform it was stored for ──────────────────
def _stored(env, server, base_url):
    from nalvera import auth
    for name in ("NALVERA_EMAIL", "NALVERA_API_KEY"):
        os.environ.pop(name, None)
    auth.store(server.email, auth.make_key(server.key, "argument"), use_keyring=False,
               base_url=base_url)


def test_a_stored_key_is_not_sent_to_another_platform(env, server, monkeypatch):
    from nalvera import Client, NoCredentials
    _stored(env, server, "https://nalvera.ai")
    monkeypatch.setenv("NALVERA_BASE_URL", server.url)       # a project file points elsewhere
    with pytest.raises(NoCredentials) as caught:
        Client(quiet=True)
    assert caught.value.code == "stored_key_other_platform"
    assert not server.requests


def test_a_stored_key_goes_to_its_own_platform(env, server):
    from nalvera import Client
    _stored(env, server, server.url)
    with Client(quiet=True) as client:
        assert client.me()["email"].lower() == server.email.lower()


def test_a_key_given_as_an_argument_goes_where_it_is_pointed(env, server, monkeypatch):
    from nalvera import Client
    monkeypatch.delenv("NALVERA_API_KEY")
    with Client(api_key=server.key, quiet=True) as client:
        assert client.me()["email"].lower() == server.email.lower()


def test_login_with_remember_records_the_platform(env, server):
    import nalvera
    from nalvera import auth
    nalvera.login(server.email, server.key, remember=True, quiet=True)
    record = json.loads(auth.credentials_path().read_text(encoding="utf-8"))
    assert record["base_url"] == server.url


# ── the credentials file and the configuration folder ─────────────────────────
@posix_only
@needs_links
def test_a_planted_link_never_receives_the_key(env, server, tmp_path):
    from nalvera import auth
    folder = auth.config_dir()
    auth.private_dir(folder)
    target = tmp_path / "attacker.txt"
    for name in os.listdir(folder):
        os.remove(folder / name)
    for pid_like in range(os.getpid() - 2, os.getpid() + 3):
        try:
            os.symlink(target, folder / f".credentials.json.{pid_like}.tmp")
        except FileExistsError:
            pass
    auth.store(server.email, auth.make_key(server.key, "argument"), use_keyring=False,
               base_url=server.url)
    assert not target.exists()
    assert stat.S_IMODE(os.stat(auth.credentials_path()).st_mode) == 0o600


@posix_only
def test_a_configuration_folder_others_can_write_is_refused(env, server):
    from nalvera import ConfigError, auth
    folder = auth.config_dir()
    folder.mkdir(parents=True)
    os.chmod(folder, 0o777)
    with pytest.raises(ConfigError) as caught:
        auth.store(server.email, auth.make_key(server.key, "argument"), use_keyring=False,
                   base_url=server.url)
    assert caught.value.code == "config_dir_insecure"
    assert not auth.credentials_path().exists()


def test_a_plaintext_keyring_backend_is_not_a_keychain(monkeypatch):
    from nalvera import auth

    class PlaintextKeyring:
        priority = 1
    PlaintextKeyring.__module__ = "keyrings.alt.file"

    class Chainer:
        priority = 10
        backends = [PlaintextKeyring()]
    Chainer.__module__ = "keyring.backends.chainer"
    assert auth.is_plaintext_backend(PlaintextKeyring())
    assert auth.is_plaintext_backend(Chainer())

    class MacKeyring:
        priority = 5
    MacKeyring.__module__ = "keyring.backends.macOS"
    assert not auth.is_plaintext_backend(MacKeyring())
    # The keyring module with a plain-file backend is not taken as a keychain.
    import types
    keyring = types.ModuleType("keyring")
    backends = types.ModuleType("keyring.backends")
    fail = types.ModuleType("keyring.backends.fail")

    class FailKeyring:
        priority = 0
    fail.Keyring = FailKeyring
    backends.fail = fail
    keyring.backends = backends
    keyring.get_keyring = lambda: PlaintextKeyring()
    for name, module in (("keyring", keyring), ("keyring.backends", backends),
                         ("keyring.backends.fail", fail)):
        monkeypatch.setitem(sys.modules, name, module)
    assert auth._keyring() is None and auth.keyring_is_plaintext()


def test_login_says_a_plaintext_keyring_was_not_used(env, server, monkeypatch):
    import nalvera
    from nalvera import ListReporter, auth
    monkeypatch.setattr(auth, "keyring_is_plaintext", lambda: True)
    reporter = ListReporter()
    nalvera.login(server.email, server.key, remember=True, reporter=reporter)
    keys = reporter.keys()
    assert "login.keyring_plaintext" in keys and "login.stored_file" in keys
    assert json.loads(auth.credentials_path().read_text(encoding="utf-8"))["api_key"]


# ── a hostile server cannot hurt the computer through names or numbers ────────
@pytest.mark.parametrize("name", ["CON", "con.txt", "NUL", "COM1", "lpt9.nii", "AUX", "PRN",
                                  "CON.NUL"])
def test_reserved_device_names_are_rewritten(name):
    from nalvera import client as client_mod
    task = client_mod.Task.from_api({"id": "t1", "model": "m", "task": "t", "state": "completed"})
    base = client_mod._result_base(name, task, None)
    stem = base.split(".")[0].rstrip(" ").upper()          # what Windows reads as the device
    assert stem not in {"CON", "NUL", "COM1", "LPT9", "AUX", "PRN"}


def test_names_ending_in_a_dot_or_space_are_rewritten():
    from nalvera import client as client_mod
    assert not client_mod._safe("name. ").endswith((".", " "))
    assert not client_mod._safe("name...").endswith(".")


def test_the_length_cap_counts_bytes_of_the_final_name_with_part():
    from nalvera import client as client_mod
    task = client_mod.Task.from_api({"id": "t1", "model": "nv_generate_ctmr", "task": "rflow-ct",
                                     "state": "completed"})
    base = client_mod._result_base("é" * 200, task, "image")
    temp = f".{base}.12345678.part"
    final = f"{base}-99{'.nii.gz'}"
    assert len(temp.encode("utf-8")) <= 255 and len(final.encode("utf-8")) <= 255


@pytest.mark.parametrize("disposition,expected", [
    ('attachment; filename="x.exe"', ".bin"), ('attachment; filename="x.command"', ".bin"),
    ('attachment; filename="x.html"', ".bin"), ('attachment; filename="r.NII.GZ"', ".nii.gz"),
    ('attachment; filename="r.tar.gz"', ".tar.gz"), ('attachment; filename="r.zip"', ".zip"),
    ('attachment; filename="r.json"', ".json"), ('attachment; filename="r.txt"', ".txt"),
    ('attachment; filename="r.csv"', ".csv"), ('attachment; filename="r.gz"', ".gz"),
    ('attachment; filename="r.npy"', ".npy")])
def test_the_extension_comes_from_an_allow_list(disposition, expected):
    from nalvera import client as client_mod
    assert client_mod._extension({"Content-Disposition": disposition}) == expected


@pytest.mark.parametrize("value", ["1e400", "nan", "inf", "-5", "99999999999"])
def test_retry_after_is_bounded(value):
    from nalvera import errors
    error = errors.from_response(429, {"code": "rate_limited", "detail": "x",
                                       "retry_after": float(value)})
    assert 0 <= error.retry_after <= errors.MAX_RETRY_AFTER
    assert str(error)


@pytest.mark.parametrize("value", ["1e400", "nan", "99999999999"])
def test_a_retry_after_header_is_bounded(value):
    from nalvera import _http, errors

    class Answer:
        headers = {"Retry-After": value}
    seconds = _http._retry_after_header(Answer())
    assert seconds is None or 0 <= seconds <= errors.MAX_RETRY_AFTER


def test_numbers_from_an_answer_are_bounded():
    from nalvera import client as client_mod
    task = client_mod.Task.from_api({"id": "t", "state": "queued", "queue_position": 1e400,
                                     "waited_seconds": float("nan"),
                                     "running_seconds": -3})
    assert task.queue_position is None or task.queue_position >= 0
    assert task.waited_seconds is None or math.isfinite(task.waited_seconds)
    assert task.running_seconds is None or task.running_seconds >= 0


def test_an_absurd_price_counts_as_over_any_cap(make_client, server, scans):
    from nalvera import CreditCapExceeded
    server.credits_per_task = float("nan")
    with pytest.raises(CreditCapExceeded):
        make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal", max_credits=100)
    assert not server.uploads


def test_a_server_text_of_an_unknown_code_is_made_printable():
    from nalvera import errors
    error = errors.from_response(418, {"code": "brand_new", "detail": "a\x1b[2Jb‮c\x85d"})
    for bad in ("\x1b", "‮", "\x85"):
        assert bad not in str(error)


# ── the assistant server tells the assistant no more than it needs ────────────
def _run_scan(tools, mcp_env, name="Hendrickx_Marieke_CT.nii.gz", **extra):
    nifti(mcp_env, name, seed=3)
    arguments = {"paths": [name], "model": MODEL, "budget": "personal", "max_credits": 50,
                 **extra}
    failed, data = tools.call("run_scan", arguments)
    assert not failed, data
    end = time.monotonic() + 60
    while data["state"] in ("preparing", "waiting", "uploading") and time.monotonic() < end:
        time.sleep(0.05)
        data = tools.call("get_run", {"run_id": data["run_id"]})[1]
    return data


def test_run_results_carry_codes_not_original_names(mcp_env):
    tools = make_tools(mcp_env)
    try:
        data = _run_scan(tools, mcp_env)
    finally:
        tools.close(2.0)
    text = json.dumps(data)
    assert "Hendrickx" not in text
    assert data["scans"][0]["name"].startswith("scan-")
    assert data["scans"][0]["position"] == 1


def test_names_are_shown_when_the_person_accepts_it(mcp_env, monkeypatch):
    monkeypatch.setenv("NALVERA_MCP_SHOW_NAMES", "1")
    tools = make_tools(mcp_env)
    try:
        data = _run_scan(tools, mcp_env)
    finally:
        tools.close(2.0)
    assert data["scans"][0]["name"] == "Hendrickx_Marieke_CT.nii.gz"


def test_a_download_is_reported_as_a_relative_folder_and_a_count(mcp_env, server):
    tools = make_tools(mcp_env)
    try:
        data = _run_scan(tools, mcp_env)
        task_id = data["scans"][0]["task_id"]
        with server.lock:
            server.tasks[task_id]["step"] = len(server.tasks[task_id]["script"]) - 1
        failed, saved = tools.call("download_result", {"task_id": task_id})
    finally:
        tools.close(2.0)
    assert not failed, saved
    assert not os.path.isabs(saved["path"]) and "Hendrickx" not in json.dumps(saved)
    assert saved["path"] == "results" and saved["files"] == 1
    assert (mcp_env.work / "results").is_dir()


def test_the_show_names_setting_is_listed():
    import nalvera
    assert "NALVERA_MCP_SHOW_NAMES" in {s.name for s in nalvera.SETTINGS}


def test_a_label_equal_to_the_files_stem_is_refused(make_client, server, scans):
    from nalvera import ConfigError
    path = scans("Hendrickx_Marieke_CT.nii.gz")
    with pytest.raises(ConfigError) as caught:
        make_client().run(path, model=MODEL, budget="personal", label="hendrickx marieke ct")
    assert caught.value.code == "label_is_file_name"
    assert not server.uploads


def test_the_label_descriptions_say_it_leaves_the_computer():
    from nalvera.mcp import load_manifest
    manifest = load_manifest()
    for name in ("run_scan", "generate"):
        entry = next(t for t in manifest["tools"] if t["name"] == name)
        text = entry["inputSchema"]["properties"]["label"]["description"].lower()
        assert "patient" in text and "platform" in text


# ── argument patterns are checked with fullmatch ──────────────────────────────
@pytest.mark.parametrize("name,arguments", [
    ("get_task", {"task_id": "abc123\n"}),
    ("generate", {"model": "nv_generate_ctmr", "task": "rflow-ct", "params": {},
                  "budget": "personal\n", "max_credits": 1, "request_id": "abcdefgh"}),
    ("generate", {"model": "nv_generate_ctmr", "task": "rflow-ct", "params": {},
                  "budget": "personal", "max_credits": 1, "request_id": "abcdefgh\n"}),
    ("get_run", {"run_id": "run-1\n"}),
])
def test_a_trailing_newline_never_passes_a_pattern(mcp_env, server, name, arguments):
    tools = make_tools(mcp_env)
    try:
        failed, data = tools.call(name, arguments)
    finally:
        tools.close(2.0)
    assert failed and data["error"]["code"] == "invalid_arguments"
    assert not server.calls("POST", "/api/v1/tasks")


# ── 2(b). clean stops and the start-of-run clean-up ───────────────────────────
@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_sigterm_and_sighup_stop_the_command_like_one_ctrl_c(tmp_path, signame):
    code = ("import sys, time\n"
            "from nalvera import cli\n"
            "def slow(args, ctx):\n"
            "    try:\n"
            "        print('ready', flush=True); time.sleep(30)\n"
            "    finally:\n"
            "        open(sys.argv[1], 'w').write('cleaned')\n"
            "cli.cmd_scans = slow\n"
            "raise SystemExit(cli.main(['scans']))\n")
    marker = tmp_path / "marker"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")}
    child = subprocess.Popen([sys.executable, "-c", code, str(marker)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
    assert child.stdout.readline().strip() == "ready"
    child.send_signal(getattr(signal, signame))
    child.wait(timeout=20)
    assert child.returncode == 130
    assert marker.read_text() == "cleaned"


def _stale(folder: Path, prefix: str) -> Path:
    stale = folder / f"{prefix}leftover"
    stale.mkdir()
    (stale / "vol_s3.nii.gz").write_bytes(b"x")
    old = time.time() - 3 * 24 * 3600
    os.utime(stale, (old, old))
    return stale


@pytest.mark.parametrize("prefix", ["nalvera-run-", "nalvera-dicom-", "nalvera-inspect-"])
def test_save_anonymized_removes_residue_of_a_killed_process(env, tmp_path, prefix):
    import tempfile

    import nalvera
    stale = _stale(Path(tempfile.gettempdir()), prefix)
    scan = examples_helpers.write_nifti(tmp_path / "ct.nii.gz")
    nalvera.save_anonymized(scan, tmp_path / "copies")
    assert not stale.exists()


def test_login_removes_residue_of_a_killed_process(env, server):
    import tempfile

    import nalvera
    stale = _stale(Path(tempfile.gettempdir()), "nalvera-dicom-")
    nalvera.login(server.email, server.key, quiet=True)
    assert not stale.exists()


def test_the_mcp_server_start_removes_residue(mcp_env):
    from nalvera.mcp import tools as tools_mod
    stale = _stale(mcp_env.temp, "nalvera-run-")
    tools_mod.clean_stale()
    assert not stale.exists()


def test_a_folder_of_a_run_in_progress_is_kept(env):
    import tempfile

    from nalvera import client as client_mod
    fresh = Path(tempfile.mkdtemp(prefix="nalvera-dicom-"))
    client_mod._remove_stale_workdirs()
    assert fresh.exists()


# ── nothing of the package is written into the working folder ─────────────────
def test_no_file_of_the_package_lands_in_the_working_folder(server, tmp_path, monkeypatch):
    import tempfile

    from conftest import _ENV
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NALVERA_BASE_URL", server.url)
    monkeypatch.setenv("NALVERA_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    from nalvera import auth
    monkeypatch.setattr(auth, "_keyring", lambda: None)
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    work = tmp_path / "work"
    work.mkdir()
    scan = examples_helpers.write_nifti(tmp_path / "scans" / "ct.nii.gz")
    monkeypatch.chdir(work)
    server.limits.update(requests_per_minute=600000, task_submits_per_minute=600000,
                         min_poll_seconds=0.01)
    import nalvera
    nalvera.login(server.email, server.key, remember=True, quiet=True)
    nalvera.estimate_run(scan, model=MODEL)
    tasks = nalvera.run(scan, model=MODEL, budget="personal")
    nalvera.scans()
    nalvera.save_anonymized(scan, tmp_path / "copies")
    nalvera.wait(tasks, poll_seconds=0.01)
    assert os.listdir(work) == []
    nalvera.download(tasks)                      # the results folder, by name, is the user's
    assert os.listdir(work) == ["results"]


# ── the converter is found only where it is trusted ───────────────────────────
def test_a_relative_converter_setting_is_not_used(monkeypatch, tmp_path):
    from nalvera.anonymize import dicom
    fake = tmp_path / "dcm2niix"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NALVERA_DCM2NIIX", "dcm2niix")
    assert dicom.find_dcm2niix() is None
    monkeypatch.setenv("NALVERA_DCM2NIIX", str(fake))
    assert dicom.find_dcm2niix() == str(fake)


@posix_only
def test_a_converter_on_path_in_the_current_or_scan_folder_is_not_used(monkeypatch, tmp_path):
    from nalvera.anonymize import dicom
    planted = tmp_path / "scans"
    planted.mkdir()
    fake = planted / "dcm2niix"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.delenv("NALVERA_DCM2NIIX", raising=False)
    monkeypatch.setitem(sys.modules, "dcm2niix", None)     # no package build
    monkeypatch.setenv("PATH", str(planted) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.chdir(planted)
    assert dicom.find_dcm2niix() != str(fake)
    monkeypatch.chdir(tmp_path)
    assert dicom.find_dcm2niix(scan_folder=str(planted)) != str(fake)
    other = tmp_path / "tools"
    other.mkdir()
    trusted = other / "dcm2niix"
    trusted.write_text("#!/bin/sh\n")
    trusted.chmod(0o755)
    monkeypatch.setenv("PATH", str(other))
    assert dicom.find_dcm2niix(scan_folder=str(planted)) == str(trusted)


def test_the_dcm2niix_floor_is_the_measured_release():
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert text.count("dcm2niix>=1.0.20261029") == 2


# ── a resampled DICOM series is said and recorded ─────────────────────────────
def test_a_corrected_series_is_said_and_recorded(make_client, server, env, monkeypatch):
    from nalvera import ListReporter
    folder = env / "input" / "tilted"
    folder.mkdir(parents=True)
    (folder / "3.dcm").write_bytes(scan_bytes(b"tilt"))
    original = stub_anonymizer._one

    def corrected(*args, **kwargs):
        scan = original(*args, **kwargs)
        import dataclasses
        return dataclasses.replace(scan, corrections=("gantry_tilt",))

    monkeypatch.setattr(stub_anonymizer, "_one", corrected)
    reporter = ListReporter()
    make_client(reporter=reporter).run(str(folder), model=MODEL, budget="personal")
    assert "run.dicom_corrected" in reporter.keys()
    rows = stub_anonymizer.Mapping(os.environ["NALVERA_MAPPING"]).rows()
    assert rows[0]["corrections"] == "gantry_tilt"


def test_the_run_docstring_names_the_converted_volume():
    from nalvera import Client
    doc = " ".join(Client.run.__doc__.split())
    assert "converted volume" in doc and "tilt" in doc


# ── the source archive is an allow-list ───────────────────────────────────────
def _git_files(root: Path):
    done = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True)
    if done.returncode != 0:
        return None
    return {f for f in done.stdout.decode("utf-8").split("\0") if f}


def test_manifest_in_names_each_example_and_its_list_is_current():
    root = Path(__file__).resolve().parent.parent
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    assert manifest == examples_helpers.manifest_with_examples(manifest), \
        "run: python tests/examples_helpers.py"
    lines = [line for line in manifest.splitlines() if not line.startswith("#")]
    text = " ".join(lines)
    assert "graft" not in text and "_info" not in text
    assert not [line for line in lines if "examples" in line and not line.startswith("include ")]
    assert not [line for line in lines if line.startswith("include examples")
                and any(c in line for c in "*?[")]
    named = {line.split(" ", 1)[1] for line in lines if line.startswith("include examples/")}
    tracked = _git_files(root)
    if tracked is not None:
        assert named == {f for f in tracked if f.startswith("examples/")}


def _build_sdist(source: Path, out: Path) -> list:
    import tarfile
    pytest.importorskip("build")
    done = subprocess.run([sys.executable, "-m", "build", "--sdist", "--no-isolation",
                           "--outdir", str(out), str(source)], capture_output=True, text=True,
                          timeout=600)
    if done.returncode != 0 and "setuptools" in done.stderr:
        pytest.skip("setuptools cannot build here: " + done.stderr[-200:])
    assert done.returncode == 0, done.stderr[-2000:]
    archive = next(out.glob("*.tar.gz"))
    with tarfile.open(archive) as tar:
        return [m.name.split("/", 1)[1] for m in tar.getmembers() if m.isfile() and "/" in m.name]


def test_the_sdist_leaves_out_what_a_run_left_in_examples(tmp_path):
    import shutil
    root = Path(__file__).resolve().parent.parent
    copy = tmp_path / "pkg"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "*.egg-info", ".ruff_cache", "build", "dist", ".DS_Store"))
    examples = copy / "examples"
    (examples / "scans").mkdir()
    (examples / "scans" / "IM030000").write_bytes(b"\x00" * 128 + b"DICM")
    (examples / "results").mkdir()
    (examples / "results" / "ct__nv_reason_ct__report.txt").write_text("report")
    (examples / "results" / "vector.npy").write_bytes(b"\x93NUMPY")
    (examples / "results" / ".x.part").write_bytes(b"partial")
    (examples / "anonymized").mkdir()
    (examples / "anonymized" / "nalvera_scans.csv").write_text("code,task_id\n")
    (examples / "anonymized" / ".nalvera_scans.csv.lock").write_text("")
    (examples / "anonymized" / "nalvera_scans.README.txt").write_text("notice")
    # What a person leaves: Jupyter's
    # checkpoint of an opened notebook, a copy of it, a script and notes.
    checkpoints = examples / "notebooks" / ".ipynb_checkpoints"
    checkpoints.mkdir()
    (checkpoints / "quickstart-checkpoint.ipynb").write_text('{"cells": []}')
    (examples / "notebooks" / "quickstart-Copy1.ipynb").write_text('{"cells": []}')
    (examples / "my_cohort_run.py").write_text("import nalvera\n")
    (examples / "run_notes.md").write_text("notes\n")
    (examples / "command_line" / "my_job.sh").write_text("echo\n")
    (examples / "mcp" / "my_settings.json").write_text("{}\n")
    (copy / "tests" / ".ipynb_checkpoints").mkdir()
    (copy / "tests" / ".ipynb_checkpoints" / "test_x-checkpoint.py").write_text("x = 1\n")
    # The maintainer's own notes beside the package (_info/, ignored by git).
    (copy / "_info").mkdir(exist_ok=True)
    (copy / "_info" / "release_notes.md").write_text("notes\n")
    names = _build_sdist(copy, tmp_path / "dist")
    planted = ("examples/scans", "examples/results", "examples/anonymized",
               "examples/notebooks/.ipynb_checkpoints", "examples/notebooks/quickstart-Copy1",
               "examples/my_cohort_run.py", "examples/run_notes.md",
               "examples/command_line/my_job.sh", "examples/mcp/my_settings.json",
               "tests/.ipynb_checkpoints", "_info")
    assert not [n for n in names if n.startswith(planted)]
    assert "examples/notebooks/quickstart.ipynb" in names
    assert "examples/command_line/cli_walkthrough.sh" in names
    assert set(examples_helpers.example_files_by_index()) <= set(names)


def test_the_sdist_holds_exactly_the_tracked_files(tmp_path):
    root = Path(__file__).resolve().parent.parent
    tracked = _git_files(root)
    if tracked is None:
        pytest.skip("not a git checkout")
    import shutil
    copy = tmp_path / "pkg"
    copy.mkdir()
    for name in tracked:
        if (root / name).is_file():
            (copy / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, copy / name)
    names = set(_build_sdist(copy, tmp_path / "dist"))
    generated = {n for n in names if n == "PKG-INFO" or n == "setup.cfg"
                 or ".egg-info/" in n}
    shipped = names - generated
    # What lives in the repository only: the ignore list, the skill for AI
    # assistants and the repository's own workflows.
    expected = {n for n in tracked if (root / n).is_file()
                and n != ".gitignore"
                and not n.startswith(("skills/", ".github/"))}
    assert shipped == expected, (sorted(shipped - expected), sorted(expected - shipped))


def test_the_maintainers_notes_are_in_no_commit():
    """_info/ holds the maintainer's own notes. Git ignores the folder, so a
    commit made from this folder never carries it."""
    root = Path(__file__).resolve().parent.parent
    ignore = root / ".gitignore"
    if not ignore.is_file():
        pytest.skip("the source archive carries no .gitignore")
    assert "_info/" in ignore.read_text(encoding="utf-8").splitlines()
    tracked = _git_files(root)
    if tracked is not None:
        assert not [f for f in tracked if f.startswith("_info/")]


# ── the command-line examples ─────────────────────────────────────────────────
COMMAND_LINE = Path(__file__).resolve().parent.parent / "examples" / "command_line"


def test_the_slurm_job_writes_and_reads_no_key_file():
    text = (COMMAND_LINE / "slurm_job.sh").read_text(encoding="utf-8")
    code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not [line for line in code if "NALVERA_API_KEY" in line or "cat " in line]


@pytest.mark.skipif(os.name != "posix", reason="bash")
@pytest.mark.parametrize("base,refused", [("http://nalvera.example", True),
                                          ("ftp://x", True), ("http://127.0.0.1:1", False)])
def test_the_curl_example_refuses_an_address_without_https(base, refused):
    env = {**os.environ, "NALVERA_BASE_URL": base, "NALVERA_EMAIL": "a@b.org",
           "NALVERA_API_KEY": "nvk_" + "0" * 57}
    done = subprocess.run(["bash", str(COMMAND_LINE / "http_read_only.sh")], env=env,
                          capture_output=True, text=True, timeout=60)
    if refused:
        assert done.returncode == 2 and "nvk_" not in done.stdout + done.stderr
    else:
        assert "must start with https" not in done.stderr


def test_the_workflow_pins_the_package_version():
    from nalvera import __version__
    text = (COMMAND_LINE / "github_actions_workflow.yml").read_text(encoding="utf-8")
    assert f'pip install "nalvera=={__version__}"' in text and "--upgrade nalvera" not in text
