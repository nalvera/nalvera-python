"""Result file names: the folder goes in front only where two results of one
download would otherwise share a name, and a generation keeps its label in
any client (the platform's task name carries it)."""
from __future__ import annotations

import pytest

from conftest import scan_bytes
from nalvera.errors import NetworkError

MODEL = "totalsegmentator_ct"
GEN = {"mode": "image-only", "output_size": [256, 256, 128], "spacing": [1.5, 1.5, 2.0]}


def _scan(env, relative: str, tag: bytes):
    path = env / "input" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scan_bytes(tag))
    return path


def _queue(client, path, task):
    [task_] = client.run(path, model=MODEL, task=task, budget="personal")
    return task_


def test_several_tasks_of_one_scan_keep_the_scan_name(client, env):
    scan = _scan(env, "work/ct_scan.nii.gz", b"one")
    tasks = [_queue(client, scan, task) for task in ("total", "lung_vessels")]
    saved = client.download(client.wait(tasks), out=env / "results")
    assert sorted(p.name for p in saved) == [f"ct_scan__{MODEL}__lung_vessels.nii.gz",
                                             f"ct_scan__{MODEL}__total.nii.gz"]


def test_only_the_results_that_would_collide_get_their_folder(client, env):
    first = _scan(env, "site_a/ct.nii.gz", b"a")
    second = _scan(env, "site_b/ct.nii.gz", b"b")
    tasks = [_queue(client, first, "total"), _queue(client, first, "lung_vessels"),
             _queue(client, second, "total")]
    saved = client.download(client.wait(tasks), out=env / "results")
    assert sorted(p.name for p in saved) == [f"ct__{MODEL}__lung_vessels.nii.gz",
                                             f"site_a_ct__{MODEL}__total.nii.gz",
                                             f"site_b_ct__{MODEL}__total.nii.gz"]


def test_results_that_differ_by_their_extension_keep_the_scan_name(client, env, server):
    first = _scan(env, "site_a/ct.nii.gz", b"a")
    second = _scan(env, "site_b/ct.nii.gz", b"b")
    tasks = [_queue(client, first, "total"), _queue(client, second, "total")]
    server.tasks[tasks[1].id]["result"] = (b"a text result\n", "report.txt")
    saved = client.download(client.wait(tasks), out=env / "results")
    assert sorted(p.name for p in saved) == [f"ct__{MODEL}__total.nii.gz",
                                             f"ct__{MODEL}__total.txt"]


def test_a_failed_download_still_saves_the_results_fetched_before_it(client, env, server):
    first = _scan(env, "site_a/ct.nii.gz", b"a")
    second = _scan(env, "site_b/ct.nii.gz", b"b")
    tasks = client.wait([_queue(client, first, "total"), _queue(client, second, "total")])
    server.inject("GET", f"/api/v1/tasks/{tasks[1].id}/result", drop=True, times=1000)
    with pytest.raises(NetworkError):
        client.download(tasks, out=env / "results")
    saved = sorted(p.name for p in (env / "results").iterdir() if not p.name.startswith("."))
    assert len(saved) == 1 and saved[0].endswith(f"ct__{MODEL}__total.nii.gz")


def test_a_generation_downloaded_by_another_client_keeps_its_label(make_client, env):
    first = make_client()
    task = first.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                          label="phantom 01")
    done = first.wait([task])
    mine = first.download(done, out=env / "mine")
    other = make_client().download([task.id], out=env / "other")
    assert [p.name for p in mine] == ["phantom_01__nv_generate_ctmr__rflow-ct.nii.gz"]
    assert [p.name for p in other] == [p.name for p in mine]


def test_a_generation_without_a_label_is_named_by_its_id_in_any_client(make_client, env,
                                                                        server):
    first = make_client()
    task = first.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal")
    assert server.tasks[task.id]["name"] == "NV-Generate rflow-ct (image-only)"
    done = first.wait([task])
    mine = first.download(done, out=env / "mine")
    other = make_client().download([task.id], out=env / "other")
    assert [p.name for p in mine] == [f"{task.id}__nv_generate_ctmr__rflow-ct.nii.gz"]
    assert [p.name for p in other] == [p.name for p in mine]
