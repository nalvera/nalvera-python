"""Every file under examples/ runs, or is checked, against the fake server.

Python examples run as subprocesses with the real anonymizer, the server's
address and the credentials in the environment, in a temporary folder that
holds the placeholder inputs they name. The fake server lists the live
catalog and checks generation settings as the real route does. Assertions
are on what reached the server, the exit status and the files written, never
on wording. A file without a test fails test_every_example_has_a_test.
"""
from __future__ import annotations

import ast
import gzip
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

import examples_helpers as helpers
from fake_server import DEFAULT_SCRIPT, error_body

EXAMPLES = helpers.EXAMPLES
SRC = helpers.ROOT / "src"
CODE_RX = re.compile(r"scan-[0-9A-HJKMNP-TV-Z]{8}\.nii\.gz(\.nvlt)?")
COVERED: set = set()


def covers(*names: str):
    """Mark a test as the one that runs (or checks) these example files."""
    COVERED.update(names)
    return lambda test: test


def _dicom_ready() -> bool:
    if importlib.util.find_spec("pydicom") is None:
        return False
    from nalvera.anonymize import dicom
    return dicom.find_dcm2niix() is not None


needs_dicom = pytest.mark.skipif(not _dicom_ready(), reason="needs pydicom and dcm2niix")


class Scene:
    """A working folder, the fake server listing the live catalog, and the
    environment an example runs in (no keychain, no real home folder)."""

    def __init__(self, server, folder: Path):
        self.server = server
        self.root = folder
        self.dir = folder / "work"
        self.home = folder / "home"
        self.temp = folder / "tmp"
        self.bin = folder / "bin"
        self.config = folder / "config"
        for path in (self.dir, self.home, self.temp, self.bin):
            path.mkdir()
        for name, line in (("nalvera", f'exec "{sys.executable}" -m nalvera "$@"'),
                           ("python3", f'exec "{sys.executable}" "$@"')):
            shim = self.bin / name
            shim.write_text(f"#!/bin/sh\n{line}\n")
            shim.chmod(0o755)
        server.use_catalog()
        # The subprocess runs on the real clock: no pacing, short polls.
        server.limits.update(requests_per_minute=600000, task_submits_per_minute=600000,
                             min_poll_seconds=0.01)
        self.seed = 0
        drop = ("NALVERA_", "STANDIN_", "PYTHONPATH", "PYTHONHOME", "HTTPS_PROXY", "HTTP_PROXY",
                "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
        self.env.update({
            "NALVERA_EMAIL": server.email, "NALVERA_API_KEY": server.key,
            "NALVERA_BASE_URL": server.url, "NALVERA_CONFIG_DIR": str(self.config),
            "HOME": str(self.home), "TMPDIR": str(self.temp), "PYTHONPATH": str(SRC),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
        })

    # ── inputs ────────────────────────────────────────────────────────────────
    def nifti(self, name: str, **kwargs: Any) -> Path:
        self.seed += 1
        kwargs.setdefault("seed", self.seed)
        return helpers.write_nifti(self.dir / name, **kwargs)

    def dicom(self, name: str, series=(3,)) -> Path:
        self.seed += 10
        return helpers.write_dicom(self.dir / name, series, seed=self.seed)

    # ── running ───────────────────────────────────────────────────────────────
    def run(self, name: str, *, expect: int = 0, env: Dict[str, Any] = None,
            stdin: str = None, path: Path = None) -> subprocess.CompletedProcess:
        script = path or EXAMPLES / name
        command = ["bash", str(script)] if script.suffix == ".sh" else [sys.executable,
                                                                       str(script)]
        merged = dict(self.env)
        for key, value in (env or {}).items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = str(value)
        options: Dict[str, Any] = {"input": stdin} if stdin is not None else \
            {"stdin": subprocess.DEVNULL}
        done = subprocess.run(command, cwd=self.dir, env=merged, capture_output=True, text=True,
                              timeout=240, start_new_session=True, **options)
        assert done.returncode == expect, (
            f"{name} exited {done.returncode}, expected {expect}\n--- stdout\n"
            f"{done.stdout[-4000:]}\n--- stderr\n{done.stderr[-4000:]}")
        self.assert_neutral_names(required=False)       # after every run
        assert self.server.key not in done.stdout + done.stderr
        return done

    # ── what reached the server ───────────────────────────────────────────────
    def api_tasks(self) -> List[Dict[str, Any]]:
        with self.server.lock:
            return [self.server.tasks[t] for t in self.server.order
                    if self.server.tasks[t]["source"] == "api"]

    def assert_neutral_names(self, required: bool = True) -> None:
        """Every upload and every scan task the platform holds is named by a
        code; ``required``: there was an upload."""
        uploads = list(self.server.uploads.values())
        assert uploads or not required
        assert all(CODE_RX.fullmatch(u["filename"]) for u in uploads), \
            [u["filename"] for u in uploads]
        for record in self.api_tasks():
            body = record.get("body") or {}
            if body.get("upload_id"):
                assert CODE_RX.fullmatch(body.get("name") or ""), body.get("name")

    def files(self, folder: str) -> List[str]:
        path = self.dir / folder
        return sorted(p.name for p in path.iterdir() if not p.name.startswith(".")) \
            if path.exists() else []


@pytest.fixture
def ex(server, tmp_path) -> Scene:
    return Scene(server, tmp_path)


def _key_gone_from(scene: Scene) -> bool:
    return not (scene.config / "credentials.json").exists()


# ── Set the API key ───────────────────────────────────────────────────────────
@covers("set_api_key.py", "command_line/set_api_key.sh")
def test_the_ways_to_give_the_key_and_logout(ex):
    signed = f"Bearer {ex.server.key}"
    no_env = {"NALVERA_EMAIL": None, "NALVERA_API_KEY": None}

    def new_me_calls(since: int) -> List[Dict[str, Any]]:
        return ex.server.calls("GET", "/api/v1/me")[since:]

    # The environment, read by the script into nalvera.login(email, api_key).
    ex.run("set_api_key.py")
    assert [c["headers"]["Authorization"] for c in new_me_calls(0)] == [signed]
    # nalvera login, run by set_api_key.sh with the email and the key typed in.
    seen = len(new_me_calls(0))
    ex.run("command_line/set_api_key.sh", env=no_env,
           stdin=f"{ex.server.email}\n{ex.server.key}\n")
    stored = ex.config / "credentials.json"
    assert json.loads(stored.read_text())["email"] == ex.server.email
    assert stored.stat().st_mode & 0o077 == 0                  # readable by the user only
    assert len(new_me_calls(seen)) == 2                        # login checks, whoami
    seen = len(new_me_calls(0))
    ex.run("set_api_key.py", env=no_env)                       # nalvera.login(): stored key
    assert [c["headers"]["Authorization"] for c in new_me_calls(seen)] == [signed]
    # The logout the script shows removes the stored key.
    [logout] = [c for c in helpers.shell_commands(
        (EXAMPLES / "command_line" / "set_api_key.sh").read_text()) if c == ["logout"]]
    subprocess.run([str(ex.bin / "nalvera"), *logout], env=ex.env, cwd=ex.dir,
                   capture_output=True, check=True, start_new_session=True)
    assert _key_gone_from(ex)
    seen = len(ex.server.requests)
    ex.run("set_api_key.py", env=no_env, expect=1)
    assert len(ex.server.requests) == seen                     # no key: nothing was sent


# ── Start ─────────────────────────────────────────────────────────────────────
@covers("sign_in_and_check_account.py", "list_models_and_tasks.py",
        "check_budgets_and_free_tasks.py", "show_limits.py")
@pytest.mark.parametrize("name,route", [
    ("sign_in_and_check_account.py", "/api/v1/budgets"),
    ("list_models_and_tasks.py", "/api/v1/models"),
    ("check_budgets_and_free_tasks.py", "/api/v1/budgets"),
    ("show_limits.py", "/api/v1/me"),
])
def test_the_account_examples_read_and_spend_nothing(ex, name, route):
    ex.server.add_lab(12, "Imaging Lab", 40.0)
    ex.run(name)
    assert ex.server.calls("GET", route)
    assert not ex.server.calls("POST") and not ex.api_tasks()


def test_an_account_example_with_a_refused_key_fails_without_a_traceback_of_the_key(ex):
    ex.server.auth_refusal = (401, "key_revoked", {})
    done = ex.run("sign_in_and_check_account.py", expect=1)
    assert ex.server.key not in done.stdout + done.stderr


# ── One scan ──────────────────────────────────────────────────────────────────
@covers("run_one_scan.py")
def test_run_one_scan(ex):
    ex.nifti("ct_scan.nii.gz")
    ex.run("run_one_scan.py")
    [task] = ex.api_tasks()
    assert (task["model"], task["task"], task["budget"]) == ("totalsegmentator_ct", "total",
                                                           "personal")
    ex.assert_neutral_names()
    assert ex.files("results") == ["ct_scan__totalsegmentator_ct__total.nii.gz"]


@covers("run_uncompressed_nii.py")
def test_run_an_uncompressed_nii(ex):
    ex.nifti("ct_scan.nii")
    ex.run("run_uncompressed_nii.py")
    [upload] = ex.server.uploads.values()
    assert upload["filename"].endswith(".nii.gz") and bytes(upload["data"])[:2] == b"\x1f\x8b"
    assert ex.files("results") == ["ct_scan__totalsegmentator_ct__total.nii.gz"]


@covers("run_one_scan_with_label.py")
def test_run_one_scan_with_a_label(ex):
    ex.nifti("ct_scan.nii.gz")
    ex.run("run_one_scan_with_label.py")
    [task] = ex.api_tasks()
    assert task["body"]["label"] == "S012 baseline" and task["name"] == "S012_baseline.nii.gz"
    assert ex.files("results") == ["ct_scan__totalsegmentator_ct__total.nii.gz"]


@covers("run_one_scan_several_tasks.py")
def test_several_tasks_on_one_scan_upload_it_once_each(ex):
    ex.nifti("ct_scan.nii.gz")
    ex.run("run_one_scan_several_tasks.py")
    assert [t["task"] for t in ex.api_tasks()] == ["total", "lung_vessels", "body"]
    assert len(ex.server.uploads) == 3
    results = ex.files("results")
    assert len(results) == 3 and all(
        r.endswith(f"ct_scan__totalsegmentator_ct__{t}.nii.gz")
        for r, t in zip(results, ("body", "lung_vessels", "total")))


def test_several_tasks_of_one_scan_are_saved_under_the_scan_name(ex):
    ex.nifti("ct_scan.nii.gz")
    ex.run("run_one_scan_several_tasks.py")
    assert ex.files("results") == [f"ct_scan__totalsegmentator_ct__{t}.nii.gz"
                                   for t in ("body", "lung_vessels", "total")]


@needs_dicom
@covers("run_mixed_inputs.py")
def test_a_mix_of_inputs_in_one_run(ex):
    ex.nifti("ct_scan.nii.gz")
    ex.nifti("ct_scan_2.nii")
    ex.nifti("scans/a.nii.gz")
    ex.nifti("scans/b.nii.gz")
    ex.dicom("dicom_folder")
    ex.run("run_mixed_inputs.py")
    assert len(ex.api_tasks()) == 5
    ex.assert_neutral_names()
    assert len(ex.files("results")) == 5


# ── DICOM ─────────────────────────────────────────────────────────────────────
def _no_dicom_detail_uploaded(scene: Scene) -> None:
    import anonymize_helpers
    for upload in scene.server.uploads.values():
        plain = gzip.decompress(bytes(upload["data"]))
        assert not anonymize_helpers.leaks(plain, anonymize_helpers.DICOM_TOKENS)


@needs_dicom
@covers("run_dicom_folder.py")
def test_a_dicom_folder_is_converted_here_and_no_tag_is_uploaded(ex):
    ex.dicom("dicom_folder")
    ex.run("run_dicom_folder.py")
    assert len(ex.api_tasks()) == 1
    ex.assert_neutral_names()
    _no_dicom_detail_uploaded(ex)
    [result] = ex.files("results")
    assert result.startswith("dicom_folder_") and result.endswith("__total.nii.gz")


@needs_dicom
@covers("run_dicom_several_series.py")
def test_a_dicom_folder_with_several_series_gives_a_task_each(ex):
    ex.dicom("dicom_study", series=(3, 5))
    ex.run("run_dicom_several_series.py")
    assert len(ex.api_tasks()) == 2
    _no_dicom_detail_uploaded(ex)
    assert len(ex.files("results")) == 2


@needs_dicom
@covers("run_dicom_subjects_in_subfolders.py")
def test_subject_subfolders_name_their_results(ex):
    ex.dicom("dicom_subjects/subject_01")
    ex.dicom("dicom_subjects/subject_02")
    ex.run("run_dicom_subjects_in_subfolders.py")
    assert len(ex.api_tasks()) == 2
    results = ex.files("results")
    assert [r.split("_")[:2] for r in results] == [["subject", "01"], ["subject", "02"]]


@needs_dicom
@covers("inspect_dicom_before_running.py")
def test_inspecting_dicom_needs_no_key_and_sends_nothing(ex):
    ex.dicom("dicom_study", series=(3, 5))
    ex.run("inspect_dicom_before_running.py",
           env={"NALVERA_EMAIL": None, "NALVERA_API_KEY": None})
    assert ex.server.requests == []
    assert not list(ex.temp.glob("nalvera-inspect-*"))


# ── Many scans ────────────────────────────────────────────────────────────────
@covers("run_folder_of_scans.py")
def test_a_folder_of_scans_with_one_that_is_not_a_scan(ex):
    for name in ("a", "b", "c"):
        ex.nifti(f"scans/{name}.nii.gz")
    (ex.dir / "scans" / "notes.nii.gz").write_bytes(b"not a scan")
    ex.run("run_folder_of_scans.py")
    assert len(ex.api_tasks()) == 3
    assert ex.files("results") == [f"{n}__totalsegmentator_ct__total.nii.gz" for n in "abc"]


@covers("run_many_scans.py")
def test_many_scans_with_a_cap_and_scans_that_share_a_name(ex):
    ex.nifti("cohort/site_a/ct.nii.gz")
    ex.nifti("cohort/site_b/ct.nii.gz")
    ex.run("run_many_scans.py")
    tasks = ex.api_tasks()
    assert len(tasks) == 2 and {t["budget"] for t in tasks} == {"personal"}
    assert ex.files("results") == ["site_a_ct__totalsegmentator_ct__total.nii.gz",
                                   "site_b_ct__totalsegmentator_ct__total.nii.gz"]


@covers("run_cohort_with_summary_csv.py")
def test_the_cohort_summary_has_a_row_per_scan(ex):
    ex.nifti("cohort/x.nii.gz")
    ex.nifti("cohort/y.nii.gz")
    (ex.dir / "cohort" / "z.nii.gz").write_bytes(b"not a scan")
    ex.run("run_cohort_with_summary_csv.py")
    import csv
    with open(ex.dir / "summary.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    done = {r["original_name"]: r for r in rows if r["state"] == "completed"}
    assert set(done) == {"x.nii.gz", "y.nii.gz"}
    task_ids = {t["id"] for t in ex.api_tasks()}
    for row in done.values():
        assert CODE_RX.fullmatch(row["code"] + ".nii.gz") and row["task_id"] in task_ids
        assert Path(row["result_path"]).exists()
    [failed] = [r for r in rows if r["state"] != "completed"]
    assert failed["original_name"] == "z.nii.gz" and not failed["task_id"]


@covers("watch_folder_for_new_scans.py")
def test_the_watched_folder_queues_each_new_scan_once(ex):
    ex.nifti("inbox/a.nii.gz")
    ex.nifti("inbox/b.nii.gz")
    ex.run("watch_folder_for_new_scans.py")
    assert len(ex.api_tasks()) == 2
    ex.nifti("inbox/c.nii.gz")
    ex.run("watch_folder_for_new_scans.py")
    assert len(ex.api_tasks()) == 3
    seen = len(ex.server.requests)
    ex.run("watch_folder_for_new_scans.py")             # nothing new: no request at all
    assert len(ex.server.requests) == seen and len(ex.api_tasks()) == 3
    assert json.loads((ex.dir / "inbox_state.json").read_text()) == ["a.nii.gz", "b.nii.gz",
                                                                     "c.nii.gz"]


# ── Each model ────────────────────────────────────────────────────────────────
MODEL_SCRIPTS = [
    ("segment_ct_totalsegmentator.py", "ct_scan.nii.gz", "totalsegmentator_ct",
     ["total", "lung_vessels"]),
    ("segment_mr_totalsegmentator.py", "mr_scan.nii.gz", "totalsegmentator_mr",
     ["total_mr", "vertebrae_mr"]),
    ("segment_ct_moose.py", "ct_scan.nii.gz", "moose_v3_clinical",
     ["clin_ct_organs", "clin_ct_lungs"]),
    ("segment_mr_muscles_musclemap.py", "mr_scan.nii.gz", "musclemap", ["wholebody"]),
    ("segment_ct_nv_segment.py", "ct_scan.nii.gz", "nv_segment_ct", ["everything", "airway"]),
    ("segment_mr_mrsegmentator.py", "mr_scan.nii.gz", "mrsegmentator", ["anatomy", "body_comp"]),
]


@covers(*[m[0] for m in MODEL_SCRIPTS])
@pytest.mark.parametrize("name,scan,model,tasks", MODEL_SCRIPTS)
def test_each_segmentation_model(ex, name, scan, model, tasks):
    ex.nifti(scan)
    ex.run(name)
    assert [(t["model"], t["task"]) for t in ex.api_tasks()] == [(model, t) for t in tasks]
    stem = scan.split(".")[0]
    assert ex.files("results") == sorted(f"{stem}__{model}__{t}.nii.gz" for t in tasks)


@covers("ct_findings_and_text_query_merlin.py")
def test_merlin_text_result_and_text_query(ex):
    import numpy as np
    vector = io.BytesIO()
    np.save(vector, np.array([0.42]))
    ex.server.results[("merlin", "phenotype")] = (b"pleural effusion: likely\n", "result.txt")
    ex.server.results[("merlin", "contrastive")] = (vector.getvalue(), "result.npy")
    ex.nifti("ct_scan.nii.gz")
    ex.run("ct_findings_and_text_query_merlin.py")
    sent = [(t["task"], t["body"].get("text")) for t in ex.api_tasks()]
    assert sent == [("phenotype", None), ("contrastive", "pleural effusion")]
    assert ex.files("results") == ["ct_scan__merlin__contrastive.npy",
                                   "ct_scan__merlin__phenotype.txt"]


@covers("ct_report_nv_reason.py")
def test_nv_reason_reports_are_text(ex):
    for task in ("chest_report", "abdomen_report"):
        ex.server.results[("nv_reason_ct", task)] = (f"{task}: no findings\n".encode(), "r.txt")
    ex.nifti("ct_scan.nii.gz")
    ex.run("ct_report_nv_reason.py")
    assert [t["task"] for t in ex.api_tasks()] == ["chest_report", "abdomen_report"]
    assert ex.files("results") == ["ct_scan__nv_reason_ct__abdomen_report.txt",
                                   "ct_scan__nv_reason_ct__chest_report.txt"]


# ── Generation ────────────────────────────────────────────────────────────────
def _generation(scene: Scene) -> Dict[str, Any]:
    [task] = scene.api_tasks()
    assert task["model"] == "nv_generate_ctmr" and not scene.server.uploads
    return task


@covers("generate_ct.py")
def test_generate_ct(ex):
    ex.run("generate_ct.py")
    task = _generation(ex)
    assert task["task"] == "rflow-ct" and task["budget"] == "personal"
    assert task["body"]["params"] == {"mode": "image-only", "output_size": [256, 256, 128],
                                      "spacing": [1.5, 1.5, 2.0], "num_output_samples": 1,
                                      "seed": 7, "num_inference_steps": 30}
    assert ex.files("generated") == ["phantom-01__nv_generate_ctmr__rflow-ct.nii.gz"]


@covers("generate_mr.py")
def test_generate_mr_brain(ex):
    ex.run("generate_mr.py")
    task = _generation(ex)
    params = task["body"]["params"]
    assert task["task"] == "rflow-mr-brain" and params["contrasts"] == ["T1"]
    assert params["cfg_guidance_scale"] == 2.0 and params["mode"] == "image-only"
    assert ex.files("generated") == ["brain-t1-01__nv_generate_ctmr__rflow-mr-brain.nii.gz"]


@pytest.mark.parametrize("name,parts", [
    ("generate_ct_with_mask.py", ["image", "mask"]),
    ("generate_mr_cross_contrast.py", ["image", "image2", "image3"]),
])
@covers("generate_ct_with_mask.py", "generate_mr_cross_contrast.py")
def test_a_generation_with_several_files_downloads_as_one_zip(ex, name, parts):
    ex.run(name)
    task = _generation(ex)
    assert task["parts"] == parts
    [archive] = ex.files("generated")
    assert archive.endswith(".zip")
    with zipfile.ZipFile(ex.dir / "generated" / archive) as files:
        assert len(files.namelist()) == len(parts)


@covers("generate_several_volumes.py")
def test_several_volumes_in_one_task(ex):
    ex.run("generate_several_volumes.py")
    task = _generation(ex)
    assert task["body"]["params"]["num_output_samples"] == 3
    assert len(ex.files("generated/phantom-set")) == 3


@covers("download_one_part_of_a_generation.py")
def test_one_part_of_a_generation(ex):
    ex.run("download_one_part_of_a_generation.py")
    task = _generation(ex)
    assert ex.server.calls("GET", f"/api/v1/tasks/{task['id']}/result")[-1]["query"] == "part=mask"
    [saved] = ex.files("generated")
    assert saved == f"{task['id']}__nv_generate_ctmr__rflow-ct__mask.nii.gz"


@covers("estimate_a_generation.py")
def test_estimating_a_generation_queues_nothing(ex):
    ex.run("estimate_a_generation.py")
    assert ex.server.calls("POST", "/api/v1/estimate") and not ex.api_tasks()


def test_a_generation_setting_the_platform_refuses_fails_the_example(ex, tmp_path):
    """The fake checks generation settings as the real route: an example with
    a wrong one cannot pass."""
    broken = tmp_path / "broken_generate.py"
    broken.write_text((EXAMPLES / "generate_mr.py").read_text().replace('["T1"]', '["T1", "T2"]'))
    ex.run("broken_generate.py", path=broken, expect=1)
    assert not ex.api_tasks()


# ── Paying ────────────────────────────────────────────────────────────────────
def _scans(scene: Scene, count: int = 3) -> None:
    for number in range(count):
        scene.nifti(f"scans/s{number}.nii.gz")


@covers("estimate_before_uploading.py")
def test_the_estimate_uploads_nothing(ex):
    _scans(ex)
    ex.run("estimate_before_uploading.py")
    assert ex.server.calls("POST", "/api/v1/estimate")
    assert not ex.server.calls("POST", "/api/v1/uploads") and not ex.api_tasks()


@covers("use_free_tasks_only.py")
def test_free_tasks_only_queue_what_they_pay_for_and_charge_nothing(ex):
    ex.server.free_tasks_left = 2
    _scans(ex)
    ex.run("use_free_tasks_only.py")
    tasks = ex.api_tasks()
    assert len(tasks) == 2 and {t["budget"] for t in tasks} == {"free"}
    assert {t["charged"]["source"] for t in tasks} == {"free"}
    assert ex.server.personal_credits == 100.0


def test_free_tasks_only_with_none_left_upload_nothing(ex):
    _scans(ex, 1)
    ex.run("use_free_tasks_only.py", expect=1)
    assert not ex.server.calls("POST", "/api/v1/uploads") and not ex.api_tasks()
    assert ex.server.personal_credits == 100.0


@covers("pay_from_personal_budget.py")
def test_personal_budget_after_the_free_task(ex):
    ex.server.free_tasks_left = 1
    _scans(ex, 2)
    ex.run("pay_from_personal_budget.py")
    assert sorted(t["charged"]["source"] for t in ex.api_tasks()) == ["free", "personal"]
    assert ex.server.personal_credits == 100.0 - ex.server.credits_per_task


@covers("pay_from_lab_budget.py")
def test_a_lab_budget(ex):
    ex.server.add_lab(12, "Imaging Lab", 50.0)
    _scans(ex, 2)
    ex.run("pay_from_lab_budget.py")
    assert {t["budget"] for t in ex.api_tasks()} == {"lab:12"}
    assert ex.server.labs[0]["balance"] == 50.0 - 2 * ex.server.credits_per_task


@covers("run_folder_lab_budget.py")
def test_the_suggested_lab_pays(ex):
    ex.server.add_lab(14, "Small Lab", 5.0)
    ex.server.add_lab(12, "Imaging Lab", 50.0)
    _scans(ex, 2)
    ex.run("run_folder_lab_budget.py")
    assert {t["budget"] for t in ex.api_tasks()} == {"lab:12"}


def test_without_a_lab_the_suggested_lab_example_stops_before_uploading(ex):
    _scans(ex, 1)
    ex.run("run_folder_lab_budget.py", expect=1)
    assert not ex.server.calls("POST", "/api/v1/uploads")


@covers("stop_at_a_spending_cap.py")
def test_a_run_under_the_cap_goes_ahead(ex):
    _scans(ex)
    ex.run("stop_at_a_spending_cap.py")
    assert len(ex.api_tasks()) == 3


def test_a_run_over_the_cap_does_not_start(ex):
    ex.server.credits_per_task = 5.0
    _scans(ex)
    ex.run("stop_at_a_spending_cap.py", expect=1)
    assert not ex.server.calls("POST", "/api/v1/uploads") and not ex.api_tasks()
    assert ex.server.personal_credits == 100.0


# ── Queue and results ─────────────────────────────────────────────────────────
@covers("queue_now_collect_later.py")
def test_queue_now_and_collect_in_a_new_process(ex):
    _scans(ex, 2)
    ex.run("queue_now_collect_later.py")
    assert len(ex.api_tasks()) == 2 and (ex.dir / "queued_tasks.json").exists()
    assert not ex.server.calls("GET", "/result")
    ex.run("queue_now_collect_later.py")
    assert len(ex.api_tasks()) == 2 and not (ex.dir / "queued_tasks.json").exists()
    assert ex.files("results") == [f"s{n}__totalsegmentator_ct__total.nii.gz" for n in (0, 1)]


def _active_api_tasks(scene: Scene, count: int) -> List[str]:
    ids = []
    for _ in range(count):
        task_id = scene.server.add_task("queued")
        record = scene.server.tasks[task_id]
        record["source"], record["script"] = "api", list(DEFAULT_SCRIPT)
        ids.append(task_id)
    return ids


@covers("follow_queue.py")
def test_following_the_queue_until_nothing_waits(ex):
    ids = _active_api_tasks(ex, 2)
    ex.run("follow_queue.py")
    for task_id in ids:
        assert ex.server.calls("GET", f"/api/v1/tasks/{task_id}")
    assert all(ex.server._state(ex.server.tasks[t]) == "completed" for t in ids)


@covers("cancel_tasks.py")
def test_cancel_only_the_tasks_scripts_queued(ex):
    ids = _active_api_tasks(ex, 2)
    website = ex.server.add_task("queued")
    ex.server.tasks[website]["script"] = list(DEFAULT_SCRIPT)
    ex.run("cancel_tasks.py")
    cancelled = {c["path"].split("/")[4] for c in ex.server.calls("POST", "/cancel")}
    assert cancelled == set(ids)


@covers("download_results.py")
def test_downloading_again_fetches_nothing_twice(ex):
    ex.server.add_task("completed")
    ex.server.add_task("completed")
    ex.run("download_results.py")
    assert len(ex.files("results")) == 2
    fetched = len(ex.server.calls("GET", "/result"))
    ex.run("download_results.py")
    assert len(ex.files("results")) == 2 and len(ex.server.calls("GET", "/result")) == fetched


@covers("list_platform_tasks.py")
def test_listing_every_task_page_by_page(ex):
    for _ in range(3):
        ex.server.add_task("completed")
    ex.run("list_platform_tasks.py")
    [listed] = ex.server.calls("GET", "/api/v1/tasks")
    assert "limit=100" in listed["query"] and "offset=0" in listed["query"]


@covers("list_my_scans_and_tasks.py")
def test_the_local_record_needs_no_key_and_sends_no_request(ex):
    ex.nifti("ct_scan.nii.gz")
    ex.run("run_one_scan.py")
    seen = len(ex.server.requests)
    done = ex.run("list_my_scans_and_tasks.py", env={"NALVERA_API_KEY": None})
    assert len(ex.server.requests) == seen
    assert "ct_scan.nii.gz is ['scan-" in done.stdout


@covers("organ_volumes_from_result.py")
def test_label_volumes_in_millilitres(ex):
    import numpy as np
    labels = np.zeros((10, 10, 10), dtype=np.int16)
    labels[:2, :2, :2] = 1
    labels[5, 5, 5] = 5
    helpers.write_nifti(ex.dir / "results" / "ct_scan__totalsegmentator_ct__total.nii.gz",
                        data=labels, zooms=(2.0, 2.0, 2.5))
    ex.run("organ_volumes_from_result.py", env={"NALVERA_EMAIL": None, "NALVERA_API_KEY": None})
    import csv
    with open(ex.dir / "volumes.csv", newline="", encoding="utf-8") as fh:
        rows = [(int(r["label"]), int(r["voxels"]), float(r["ml"])) for r in csv.DictReader(fh)]
    assert rows == [(1, 8, 0.08), (5, 1, 0.01)]
    assert ex.server.requests == []


@covers("run_in_background.py")
def test_a_run_in_the_background(ex):
    _scans(ex, 2)
    ex.run("run_in_background.py")
    assert len(ex.api_tasks()) == 2


# ── Privacy ───────────────────────────────────────────────────────────────────
@covers("anonymize_only.py")
def test_anonymize_only_writes_a_clean_copy_and_sends_nothing(ex):
    import nibabel as nib
    import numpy as np

    import anonymize_helpers
    source = ex.nifti("ct_scan.nii.gz", descrip=b"Hendrickx^Marieke",
                      extension=b"Patient Hendrickx^Marieke")
    ex.run("anonymize_only.py", env={"NALVERA_EMAIL": None, "NALVERA_API_KEY": None})
    [written] = (ex.dir / "anonymized").iterdir()                # the copy, nothing else
    assert CODE_RX.fullmatch(written.name)
    before, after = nib.load(str(source)), nib.load(str(written))
    assert after.header["descrip"].item() == b"" and len(after.header.extensions) == 0
    assert np.array_equal(np.asarray(before.dataobj), np.asarray(after.dataobj))
    assert not anonymize_helpers.leaks(gzip.decompress(written.read_bytes()), ["Hendrickx"])
    assert ex.server.requests == []


@covers("command_line/anonymize_only.sh")
def test_the_anonymize_command_writes_only_clean_copies_and_sends_nothing(ex):
    import csv

    import anonymize_helpers
    ex.nifti("ct_scan.nii.gz", descrip=b"Hendrickx^Marieke")
    ex.nifti("scans/a.nii.gz")
    ex.nifti("scans/b.nii")
    ex.run("command_line/anonymize_only.sh",
           env={"NALVERA_EMAIL": None, "NALVERA_API_KEY": None})
    copies = list((ex.dir / "anonymized").iterdir())
    assert len(copies) == 3 and all(CODE_RX.fullmatch(p.name) for p in copies)
    for copy in copies:
        assert not anonymize_helpers.leaks(gzip.decompress(copy.read_bytes()), ["Hendrickx"])
    [record] = list((ex.config / "records").glob("*.csv"))
    with open(record, encoding="utf-8-sig") as fh:
        names = sorted(r["original_name"] for r in csv.DictReader(fh))
    assert names == ["a.nii.gz", "b.nii", "ct_scan.nii.gz"]
    assert ex.server.requests == [] and not list(ex.dir.glob("*.csv"))


@covers("encrypted_upload.py")
def test_encrypted_upload(ex):
    pytest.importorskip("cryptography")
    ex.server.enable_encryption()
    ex.nifti("ct_scan.nii.gz")
    ex.run("encrypted_upload.py")
    [upload] = ex.server.uploads.values()
    assert upload["filename"].endswith(".nii.gz.nvlt") and bytes(upload["data"])[:4] == b"NVLT"
    assert len(ex.api_tasks()) == 1 and ex.files("results")


@covers("keep_the_mapping_elsewhere.py")
def test_the_mapping_kept_in_a_private_folder(ex):
    ex.nifti("ct_scan.nii.gz")
    ex.run("keep_the_mapping_elsewhere.py")
    assert (ex.home / "private" / "nalvera_scans.csv").exists()
    assert not (ex.dir / "nalvera_scans.csv").exists()


@covers("limit_to_allowed_folders.py")
def test_a_path_outside_the_allowed_folders_is_never_read(ex):
    ex.nifti("data/ct_scan.nii.gz")
    ex.nifti("other/ct_scan.nii.gz")
    ex.run("limit_to_allowed_folders.py")
    assert len(ex.server.uploads) == 1 and len(ex.api_tasks()) == 1
    assert ex.files("data/results") == ["ct_scan__totalsegmentator_ct__total.nii.gz"]


# ── Automation and errors ─────────────────────────────────────────────────────
@covers("run_without_a_terminal.py")
@pytest.mark.parametrize("scene,status,queued", [
    ({}, 0, 2),
    ({"personal_credits": 0.0}, 4, 0),
    ({"auth_refusal": (401, "key_revoked", {})}, 3, 0),
    ({"auth_refusal": (403, "api_disabled", {})}, 3, 0),
])
def test_unattended_exit_statuses(ex, scene, status, queued):
    for name, value in scene.items():
        setattr(ex.server, name, value)
    _scans(ex, 2)
    ex.run("run_without_a_terminal.py", expect=status)
    assert len(ex.api_tasks()) == queued


@covers("resume_an_interrupted_run.py")
def test_starting_a_stopped_run_again_queues_and_charges_each_scan_once(ex):
    _scans(ex, 2)
    ex.server.inject("POST", "/api/v1/tasks", status=500, body=error_body("server_error"),
                     skip=1)
    ex.run("resume_an_interrupted_run.py", expect=1)
    assert len(ex.api_tasks()) == 1
    ex.run("resume_an_interrupted_run.py")
    tasks = ex.api_tasks()
    assert len(tasks) == 2 and len({t["upload"] for t in tasks}) == 2
    assert ex.server.personal_credits == 100.0 - 2 * ex.server.credits_per_task
    assert ex.files("results") == [f"s{n}__totalsegmentator_ct__total.nii.gz" for n in (0, 1)]


@covers("handle_errors.py")
@pytest.mark.parametrize("case,status,queued", [
    ("works", 0, 1),
    ("no_credits", 4, 0),
    ("key_expired", 3, 0),
    ("key_revoked", 3, 0),
    ("api_disabled", 3, 0),
    ("scan_rejected", 1, 0),
    ("rate_limited", 5, 0),
    ("connection_lost_mid_upload", 0, 1),
])
def test_each_error_takes_its_own_branch(ex, case, status, queued):
    server = ex.server
    if case == "no_credits":
        server.personal_credits = 0.0
    elif case in ("key_expired", "key_revoked"):
        server.auth_refusal = (401, case, {})
    elif case == "api_disabled":
        server.auth_refusal = (403, case, {})
    elif case == "scan_rejected":
        server.inject("PATCH", "/api/v1/uploads/", status=422,
                      body=error_body("scan_rejected", reason="a label mask"))
    elif case == "rate_limited":
        server.inject("POST", "/api/v1/tasks", status=429, times=20,
                      body=error_body("rate_limited", retry_after=3600),
                      headers={"Retry-After": "3600"})
    elif case == "connection_lost_mid_upload":
        server.inject("PATCH", "/api/v1/uploads/", drop=True)
    ex.nifti("ct_scan.nii.gz")
    ex.run("handle_errors.py", expect=status)
    assert len(ex.api_tasks()) == queued
    if case == "connection_lost_mid_upload":
        assert ex.server.calls("HEAD", "/api/v1/uploads/")       # resumed from the offset


@covers("command_line/slurm_job.sh", "command_line/github_actions_workflow.yml")
def test_the_cluster_and_ci_files_hold_no_key_and_their_commands_parse(ex):
    for name in ("slurm_job.sh", "github_actions_workflow.yml"):
        text = (EXAMPLES / "command_line" / name).read_text(encoding="utf-8")
        commands = helpers.shell_commands(text)
        assert commands and all(helpers.command_problem(c) is None for c in commands), name
    workflow = (EXAMPLES / "command_line" / "github_actions_workflow.yml").read_text(
        encoding="utf-8")
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        data = yaml.safe_load(workflow)
        steps = data["jobs"]["segment"]["steps"]
        env = next(s["env"] for s in steps if "env" in s)
        assert env["NALVERA_API_KEY"] == "${{ secrets.NALVERA_API_KEY }}"
    else:
        assert re.search(r"NALVERA_API_KEY: \$\{\{ secrets\.NALVERA_API_KEY \}\}", workflow)
    assert "secrets.NALVERA_API_KEY" in workflow


# ── Command line, other languages, notebook ───────────────────────────────────
@covers("command_line/cli_walkthrough.sh")
def test_the_walkthrough_runs_end_to_end(ex):
    ex.nifti("ct_scan.nii.gz")
    ex.nifti("ct_scan_2.nii.gz")
    ex.run("command_line/cli_walkthrough.sh", env={"NALVERA_EMAIL": None, "NALVERA_API_KEY": None},
           stdin=f"{ex.server.email}\n{ex.server.key}\n")
    tasks = ex.api_tasks()
    assert [t["model"] for t in tasks] == ["totalsegmentator_ct"] * 2 + ["nv_generate_ctmr"]
    assert ex.server.tasks[tasks[-1]["id"]]["cancel_requested"]      # its id, kept by --ids
    assert ex.server.calls("GET", f"/api/v1/tasks/{tasks[1]['id']}")  # waited for by its id
    assert set(ex.files("results")) == {"ct_scan__totalsegmentator_ct__total.nii.gz",
                                        "ct_scan_2__totalsegmentator_ct__total.nii.gz"}
    assert len(ex.server.calls("GET", "/result")) == 2                # each result once
    assert [c for c in ex.server.calls("GET", "/api/v1/tasks") if "state=all" in c["query"]]
    assert len(ex.server.calls("POST", "/api/v1/estimate")) >= 2      # the scan, the generation
    [copy] = [n for n in ex.files("anonymized") if n.endswith(".nii.gz")]
    assert CODE_RX.fullmatch(copy)
    assert _key_gone_from(ex)                           # the last command logs out


@covers("command_line/http_read_only.sh")
def test_the_curl_example_reads_with_the_key_on_standard_input(ex):
    if shutil.which("curl") is None:
        pytest.skip("curl is not installed")
    ex.run("command_line/http_read_only.sh")
    for route in ("/api/v1/me", "/api/v1/models", "/api/v1/tasks"):
        [call] = ex.server.calls("GET", route)
        assert call["headers"]["Authorization"] == f"Bearer {ex.server.key}"
    script = (EXAMPLES / "command_line" / "http_read_only.sh").read_text(encoding="utf-8")
    uses = [line for line in script.splitlines() if "NALVERA_API_KEY" in line
            and not line.lstrip().startswith("#")]
    assert len(uses) == 1 and uses[0].lstrip().startswith("printf")
    assert not ex.server.calls("POST")


def prepare_notebook_scene(scene: Scene, name: str) -> None:
    """The inputs and results a notebook needs (results nibabel can open)."""
    import io as io_mod

    import nibabel as nib
    import numpy as np
    labels = nib.Nifti1Image(np.arange(24, dtype=np.int16).reshape(2, 3, 4) % 3,
                             np.diag([1.5, 1.5, 2.0, 1.0]))
    buffer = io_mod.BytesIO()
    file_map = nib.Nifti1Image.make_file_map({"image": buffer})
    labels.to_file_map(file_map)
    data = gzip.compress(buffer.getvalue())
    scene.server.results[("totalsegmentator_ct", "total")] = (data, "result.nii.gz")
    scene.server.results[("nv_generate_ctmr", "rflow-ct")] = (data, "result.nii.gz")
    if name == "quickstart.ipynb":
        scene.nifti("ct_scan.nii.gz")
    elif name == "many_scans_and_budgets.ipynb":
        _scans(scene, 3)


NOTEBOOK_RUNS = {"quickstart.ipynb": (1, "results"), "many_scans_and_budgets.ipynb": (3, "results"),
                 "generate_synthetic_ct.ipynb": (1, "generated")}


@covers(*[f"notebooks/{n}" for n in NOTEBOOK_RUNS])
@pytest.mark.parametrize("name", sorted(NOTEBOOK_RUNS))
def test_each_notebook_runs_cell_by_cell_twice_and_stores_no_output(ex, tmp_path, name):
    path = EXAMPLES / "notebooks" / name
    _code, cells = helpers.notebook_code(path)
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [] and cell.get("execution_count") is None
    script = tmp_path / f"{name}.py"
    script.write_text(helpers.notebook_script(path, ex.server.email, ex.server.key),
                      encoding="utf-8")
    prepare_notebook_scene(ex, name)
    tasks, folder = NOTEBOOK_RUNS[name]
    for _ in range(2):                              # run again from the top: nothing twice
        ex.run(name, path=script)
        assert len(ex.api_tasks()) == tasks and len(ex.files(folder)) == tasks
    assert len(ex.server.calls("GET", "/result")) == tasks


@covers("command_line/cli_cheatsheet.md")
def test_the_cheatsheet_is_generated_from_the_parser(ex):
    text = (EXAMPLES / "command_line" / "cli_cheatsheet.md").read_text(encoding="utf-8")
    assert text == helpers.cheatsheet(), "run: python tests/examples_helpers.py"
    for command in helpers.subparsers():
        assert f"## `nalvera {command}" in text
    assert all(helpers.command_problem(c) is None for c in helpers.markdown_commands(text))


# ── Every file ────────────────────────────────────────────────────────────────
def _example_files() -> List[Path]:
    return sorted(p for folder in (EXAMPLES, helpers.NOTEBOOKS, helpers.COMMAND_LINE)
                  for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))


def _name(path: Path) -> str:
    """A file's name relative to examples/, as the tests cover it."""
    return path.relative_to(EXAMPLES).as_posix()


def _is_command_line(path: Path) -> bool:
    return path.parent == helpers.COMMAND_LINE


def _mcp_example_files() -> List[str]:
    """examples/mcp/ (the MCP server's snippets), tested in test_mcp_examples.py."""
    folder = EXAMPLES / "mcp"
    return sorted(f"mcp/{p.name}" for p in folder.iterdir()
                  if p.is_file() and not p.name.startswith(".")) if folder.is_dir() else []


@covers("README.md")
def test_the_index_lists_exactly_the_python_examples_and_its_table_is_current(ex):
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([^)#]+)\)", readme))
    assert linked == ({_name(p) for p in _example_files() if not _is_command_line(p)}
                      - {"README.md"}) | set(_mcp_example_files())
    assert readme == helpers.readme_with_table(readme), "run: python tests/examples_helpers.py"
    assert helpers.python_text_problems("README.md", readme) == []


@covers("command_line/README.md")
def test_the_command_line_index_lists_its_folder_and_names_only_commands(ex):
    readme = (helpers.COMMAND_LINE / "README.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([^)#]+)\)", readme))
    assert linked == {p.name for p in helpers.COMMAND_LINE.iterdir()
                      if p.is_file() and p.name != "README.md"}
    assert helpers.command_line_text_problems("README.md", readme) == []
    assert all(helpers.command_problem(c) is None for c in helpers.markdown_commands(readme))


def test_every_example_has_a_test():
    assert {_name(p) for p in _example_files()} == COVERED


def test_the_settings_table_covers_every_argument_and_setting_in_python_terms():
    import nalvera
    table = helpers.settings_table()
    for function in helpers.calls().values():
        for name in helpers.signature(function):
            assert f"| `{name}` (" in table, name
    for setting in nalvera.SETTINGS:
        assert f"`{setting.name}`" in table, setting.name
    assert helpers.python_text_problems("settings table", table) == []


def test_no_example_holds_a_key():
    for path in _example_files():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"nvk_[0-9A-Za-z]", text), path.name      # "nvk_..." only
        assert not re.search(r"[0-9A-Za-z]{40,}", text), path.name


def test_default_and_options_comments_match_the_code():
    problems = []
    for path in _example_files():
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            problems += helpers.comment_problems(path.name, text)
        elif path.suffix in (".sh", ".yml"):
            problems += helpers.shell_comment_problems(path.name, text)
            problems += [f"{path.name}: {p}" for c in helpers.shell_commands(text)
                         if (p := helpers.command_problem(c))]
        elif path.suffix == ".ipynb":
            problems += helpers.comment_problems(path.name, helpers.notebook_code(path)[0])
    assert not problems, "\n".join(problems)


def test_python_and_the_command_line_are_never_mixed():
    problems = []
    for path in _example_files() + [p for p in (EXAMPLES / "mcp").iterdir() if p.is_file()]:
        text = path.read_text(encoding="utf-8")
        if _is_command_line(path):
            problems += helpers.command_line_text_problems(_name(path), text)
        elif path.suffix == ".ipynb":
            for cell in helpers.notebook_code(path)[1]:
                problems += helpers.python_text_problems(_name(path), "".join(cell["source"]))
        elif path.parent.name == "mcp":
            # An assistant's own commands (claude mcp add --scope ...) may appear here;
            # a nalvera command may not.
            problems += [p for p in helpers.python_text_problems(_name(path), text)
                         if ": a command:" in p]
        else:
            problems += helpers.python_text_problems(_name(path), text)
    assert not problems, "\n".join(problems)


def test_the_checker_refuses_a_command_in_python_and_python_on_the_command_line():
    assert helpers.python_text_problems("x.py", "# then: nalvera run scans --model m")
    assert helpers.python_text_problems("x.py", 'budget="personal",  # or --budget personal')
    assert not helpers.python_text_problems("x.py", "# %pip install --upgrade nalvera")
    assert helpers.command_line_text_problems("x.sh", "python -c 'import nalvera'")
    assert helpers.command_line_text_problems("x.sh", "# nalvera.run(paths)")


def test_python_examples_follow_the_rules():
    allowed = {"nalvera", "numpy", "nibabel", "csv", "json", "os", "pathlib", "sys", "time",
               "zipfile"}
    for path in _example_files():
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, feature_version=(3, 10))
        doc = ast.get_docstring(tree) or ""
        assert doc and len([line for line in doc.splitlines() if line.strip()]) <= 4, path.name
        assert "Command line" not in doc, path.name
        assert len(source.splitlines()) <= 70, path.name
        imported, spends = set(), False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") in (
                    "run", "generate", "run_in_background"):
                # Spending credits: the budget is named, never left to a default.
                assert "budget" in {kw.arg for kw in node.keywords}, (path.name, node.lineno)
                spends = True
        assert imported <= allowed, (path.name, imported - allowed)
        assert not spends or re.search(r"spend|charges nothing", doc, re.IGNORECASE), \
            f"{path.name}: say whether it spends credits"


def test_shell_examples_live_in_their_folder_are_strict_and_pass_shellcheck():
    shellcheck = shutil.which("shellcheck")
    for path in _example_files():
        if path.suffix in (".sh", ".yml"):
            assert _is_command_line(path), path.name
        if path.suffix == ".yml":
            lines = path.read_text(encoding="utf-8").splitlines()
            header = [line for line in lines[:6] if line.startswith("#")]
            assert 0 < len(header) <= 4 and not lines[len(header)].startswith("#"), path.name
        if path.suffix != ".sh":
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "#!/usr/bin/env bash" and "set -euo pipefail" in lines, path.name
        header = lines[1:lines.index("set -euo pipefail")]
        assert len([h for h in header if not h.startswith("#SBATCH")]) <= 4, path.name
        assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
        if shellcheck:
            done = subprocess.run([shellcheck, str(path)], capture_output=True, text=True)
            assert done.returncode == 0, done.stdout


def test_texts_say_task_never_job():
    for path in _example_files():
        text = path.read_text(encoding="utf-8").replace("slurm_job.sh", "")
        text = re.sub(r"^jobs:$", "", text, flags=re.MULTILINE)
        assert not re.search(r"\bjobs?\b", text, re.IGNORECASE), path.name


# ── The fake server mirrors the real one ──────────────────────────────────────
def _platform() -> Path:
    candidates = [os.environ.get("NALVERA_PLATFORM_DIR") or "",
                  str(helpers.ROOT.parent / "Platform")]
    for candidate in candidates:
        if candidate and (Path(candidate) / "backend" / "routes" / "api_v1.py").exists():
            return Path(candidate)
    pytest.skip("the Platform source (backend/routes/api_v1.py) is not next to this package")


def _module_constants(path: Path) -> Dict[str, Any]:
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _generation_help_source(api_v1: Path) -> Dict[str, Dict[str, Any]]:
    """What api_v1._generation_help lists per task, read from its source:
    the literal parts (defaults, spacing, modes, rules); the lists built from
    jobs.py constants are compared through those constants."""
    # The function lived in api_v1.py; the Platform moves it to jobs.py.
    function = None
    for path, name in ((api_v1, "_generation_help"),
                       (api_v1.parent / "jobs.py", "generation_help")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next((n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if function is not None:
            break
    assert function is not None, "the Platform's generation help was not found"
    base: Dict[str, Any] = {}
    out: Dict[str, Dict[str, Any]] = {}

    def apply(statements, target):
        for node in statements:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                for key, value in zip(node.value.keys, node.value.values):
                    try:
                        target[key.value] = ast.literal_eval(value)
                    except ValueError:
                        pass
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                    and getattr(node.value.func, "attr", "") == "update":
                for kw in node.value.keywords:
                    try:
                        target[kw.arg] = ast.literal_eval(kw.value)
                    except ValueError:
                        pass
            elif isinstance(node, ast.AugAssign):
                target["rules"] = list(target.get("rules", [])) + ast.literal_eval(node.value)

    for node in function.body:
        if isinstance(node, ast.If):
            task = node.test.comparators[0].value
            out[task] = json.loads(json.dumps(base))
            apply(node.body, out[task])
        else:
            apply([node], base)
    return out


def test_the_fake_server_lists_what_the_real_route_lists():
    import fake_server
    root = _platform()
    jobs = _module_constants(root / "backend" / "routes" / "jobs.py")
    contrasts = _module_constants(root / "backend" / "gen_outputs.py")["NVGEN_MR_CONTRAST"]
    assert fake_server.NVGEN_XY_SIZES == list(jobs["_NVGEN_XY_SIZES"])
    assert fake_server.NVGEN_Z_SIZES == list(jobs["_NVGEN_Z_ENABLED"])
    assert fake_server.NVGEN_CT_REGIONS == list(jobs["_NVGEN_CT_REGIONS"])
    assert fake_server.NVGEN_CT_TUMOURS == list(jobs["_NVGEN_CT_TUMORS"])
    assert fake_server.NVGEN_CT_TUMOURS_BY_REGION == {
        k: list(v) for k, v in jobs["_NVGEN_CT_TUMORS_BY_REGION"].items()}
    assert fake_server.NVGEN_MR_CFG_DEFAULT == jobs["_NVGEN_MR_CFG_DEFAULT"]
    assert fake_server.NVGEN_MR_CONTRASTS == list(contrasts)
    real = _generation_help_source(root / "backend" / "routes" / "api_v1.py")
    for task, listed in real.items():
        mine = fake_server.generation_help(task)
        for key in ("spacing_mm", "modes", "rules", "skull_stripped"):
            if key in listed:
                assert mine[key] == listed[key], (task, key)
        defaults = dict(listed["defaults"])
        if task == "rflow-mr-brain":
            defaults["cfg_guidance_scale"] = jobs["_NVGEN_MR_CFG_DEFAULT"]
        assert mine["defaults"] == defaults, task


def test_the_fake_server_uses_the_real_schema_limits():
    import fake_server
    root = _platform()
    tree = ast.parse((root / "backend" / "schemas.py").read_text(encoding="utf-8"))
    request = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == "GenerateJobRequest")
    limits, defaults, required = {}, {}, []
    for node in request.body:
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        name, call = node.target.id, node.value
        if not isinstance(call, ast.Call):
            defaults[name] = ast.literal_eval(call)
            continue
        keywords = {kw.arg: kw.value for kw in call.keywords}
        if "ge" in keywords and "le" in keywords:
            limits[name] = (ast.literal_eval(keywords["ge"]), ast.literal_eval(keywords["le"]))
        if call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value is ...:
            required.append(name)
        if "default_factory" in keywords:
            defaults[name] = {"list": []}.get(keywords["default_factory"].id)
        elif call.args and not (isinstance(call.args[0], ast.Constant)
                                and call.args[0].value is ...):
            defaults[name] = ast.literal_eval(call.args[0])
    for name, (low, high) in fake_server.GENERATION_LIMITS.items():
        assert limits[name] == (low, high), name
    for name, value in fake_server.GENERATION_SCHEMA_DEFAULTS.items():
        assert defaults[name] == value, name
    assert set(fake_server.GENERATION_REQUIRED) <= set(required)
    source = (root / "backend" / "routes" / "api_v1.py").read_text(encoding="utf-8")
    states = re.search(r'state: str = Query\("all", pattern="\^\(([a-z|]+)\)\$"\)', source)
    limit = re.search(r"limit: int = Query\(50, ge=(\d+), le=(\d+)\)", source)
    assert tuple(states.group(1).split("|")) == fake_server.TASK_LIST_STATES
    assert (int(limit.group(1)), int(limit.group(2))) == fake_server.TASK_LIST_LIMIT


def test_the_fake_server_lists_the_live_catalog():
    import fake_server
    root = _platform()
    data = json.loads((root / "models" / "catalog" / "registry.compiled.json").read_text(
        encoding="utf-8"))
    models = data if isinstance(data, list) else data.get("models", data)
    models = list(models.values()) if isinstance(models, dict) else models
    real = {}
    for m in models:
        if m.get("deprecated") or m.get("visibility") == "private":
            continue
        tasks = [(s["task"], s["label"], bool(s.get("is_default")), s.get("input_modality"),
                  s.get("model_priceperMvoxels"),
                  s.get("output_extension") or m.get("output_extension"))
                 for s in m.get("submodels") or [] if not s.get("deprecated")]
        real[m["model_id"]] = (m["model_name"], m["model_version"], m.get("input_modality"),
                               m["model_type"], tasks)
    mine = {entry[0]: tuple(entry[1:]) for entry in fake_server.CATALOG}
    assert mine == real
    generative = {m["model_id"] for m in models if m.get("generative")}
    assert set(fake_server.GENERATIVE_MODELS) == generative
