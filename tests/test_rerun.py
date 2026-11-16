"""Running a notebook again from the top, after a restart, and where the
record lives: the same calls twice queue one task, charge once and save one
file."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import nalvera
import stub_anonymizer
from conftest import scan_bytes

MODEL, TASK = "totalsegmentator_ct", "total"
GEN = {"mode": "image-only", "output_size": [256, 256, 128], "spacing": [1.5, 1.5, 2.0]}


def _charged(server) -> float:
    return round(100.0 - server.personal_credits, 2)


# ── run ───────────────────────────────────────────────────────────────────────
def test_run_twice_queues_one_task_and_returns_it_marked_earlier(make_client, server, scans):
    paths = [scans("a.nii.gz"), scans("b.nii.gz")]
    first = make_client().run(paths, model=MODEL, budget="personal")
    charged = _charged(server)
    again = make_client().run(paths, model=MODEL, budget="personal")
    assert [t.id for t in again] == [t.id for t in first]          # in the scans' order
    assert all(t.earlier for t in again) and not any(t.earlier for t in first)
    assert [s.task_id for s in again.skipped] == [t.id for t in first]
    assert again.charged_credits == 0 and first.charged_credits == charged
    assert len(server.tasks) == 2 and len(server.uploads) == 2 and _charged(server) == charged


def test_run_with_a_new_scan_queues_only_the_new_one(make_client, server, scans):
    make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    again = make_client().run([scans("a.nii.gz"), scans("c.nii.gz")], model=MODEL,
                              budget="personal")
    assert [t.earlier for t in again] == [True, False] and len(server.tasks) == 2
    assert again.charged_credits == server.credits_per_task


def test_force_queues_it_again(make_client, server, scans):
    make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    again = make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal", force=True)
    assert not again[0].earlier and len(server.tasks) == 2


def test_the_repeat_check_works_from_any_working_folder(env, server, stub_anon, fake_time,
                                                        monkeypatch, tmp_path):
    monkeypatch.delenv("NALVERA_MAPPING")
    data = scan_bytes(b"same scan")
    tasks = []
    for folder in ("project_a", "project_b"):
        work = tmp_path / folder
        work.mkdir()
        (work / "ct.nii.gz").write_bytes(data)
        monkeypatch.chdir(work)
        with nalvera.Client(quiet=True, clock=fake_time.clock, sleep=fake_time.sleep) as client:
            tasks.append(client.run("ct.nii.gz", model=MODEL, budget="personal")[0])
        assert not [p for p in work.iterdir() if p.name != "ct.nii.gz"]   # no file of ours
    assert tasks[0].id == tasks[1].id and tasks[1].earlier and len(server.tasks) == 1
    records = list((env / "config" / "records").glob("*.csv"))
    assert len(records) == 1 and stat.S_IMODE(records[0].parent.stat().st_mode) == 0o700


def test_one_record_per_account_and_platform(env):
    from nalvera import auth
    one = auth.record_path("a@lab.org", "https://nalvera.ai", directory=env)
    assert one == auth.record_path("A@Lab.org", "https://nalvera.ai/", directory=env)
    assert one != auth.record_path("b@lab.org", "https://nalvera.ai", directory=env)
    assert one != auth.record_path("a@lab.org", "http://127.0.0.1:8000", directory=env)
    assert "a@lab.org" not in str(one) and one.parent == env / "records"


def test_a_record_chosen_with_mapping_keeps_working(make_client, server, scans, env):
    chosen = env / "elsewhere" / "my_record.csv"
    chosen.parent.mkdir()
    make_client(mapping=chosen).run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert stub_anonymizer.Mapping(str(chosen)).rows()
    again = make_client(mapping=chosen).run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert again[0].earlier and len(server.tasks) == 1


# ── generate ──────────────────────────────────────────────────────────────────
def test_generate_with_a_label_twice_queues_once(make_client, server):
    first = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                                   label="phantom-01")
    charged = _charged(server)
    again = make_client().generate("nv_generate_ctmr", "rflow-ct", dict(reversed(GEN.items())),
                                   budget="personal", label="phantom-01")
    assert again.id == first.id and again.earlier and len(server.tasks) == 1
    assert _charged(server) == charged and not server.calls("POST", "/api/v1/estimate")[1:]


def test_generate_with_other_settings_or_force_queues_again(make_client, server):
    make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                           label="phantom-01")
    other = make_client().generate("nv_generate_ctmr", "rflow-ct", {**GEN, "seed": 9},
                                   budget="personal", label="phantom-01")
    forced = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                                    label="phantom-01", force=True)
    assert len({other.id, forced.id}) == 2 and len(server.tasks) == 3
    assert not other.earlier and not forced.earlier


def test_generate_without_a_label_queues_every_time(make_client, server):
    for _ in range(2):
        make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal")
    assert len(server.tasks) == 2


def test_a_failed_generation_is_queued_again(make_client, server):
    server.script = [("failed", None)]
    first = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                                   label="phantom-01")
    again = make_client().generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                                   label="phantom-01")
    assert again.id != first.id and len(server.tasks) == 2


# ── download ──────────────────────────────────────────────────────────────────
def test_download_twice_fetches_once_and_returns_the_same_file(make_client, server, scans,
                                                               env):
    client = make_client()
    done = client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    first = client.download(done, out=env / "results")
    again = make_client().download(done, out=env / "results")
    assert list(again) == list(first) and len(server.calls("GET", "/result")) == 1
    assert [p.name for p in (env / "results").iterdir()] == [first[0].name]


def test_a_missing_result_is_fetched_again_under_its_name(make_client, server, scans, env):
    client = make_client()
    done = client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    first = client.download(done, out=env / "results")[0]
    first.unlink()
    again = client.download(done, out=env / "results")[0]
    assert again == first and again.exists() and len(server.calls("GET", "/result")) == 2


def test_a_short_result_is_fetched_again_and_kept_beside_it(make_client, server, scans, env):
    client = make_client()
    done = client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    first = client.download(done, out=env / "results")[0]
    first.write_bytes(first.read_bytes()[:5])
    again = client.download(done, out=env / "results")[0]
    assert again != first and len(server.calls("GET", "/result")) == 2
    assert again.stat().st_size > first.stat().st_size


def test_overwrite_fetches_anew(make_client, server, scans, env):
    client = make_client()
    done = client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    first = client.download(done, out=env / "results")[0]
    again = client.download(done, out=env / "results", overwrite=True)[0]
    assert again == first and len(server.calls("GET", "/result")) == 2


def test_two_different_results_never_share_a_name(make_client, server, env):
    for name in ("patient01", "patient02"):
        path = env / "input" / name / "ct.nii.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(scan_bytes(name.encode()))
    client = make_client()
    done = client.wait(client.run([env / "input" / "patient01" / "ct.nii.gz",
                                   env / "input" / "patient02" / "ct.nii.gz"], model=MODEL,
                                  budget="personal"))
    for _ in range(2):
        saved = client.download(done, out=env / "results")
        assert len({p.name for p in saved}) == 2
    assert len(list((env / "results").iterdir())) == 2


def test_a_generation_downloaded_twice_is_saved_once(make_client, server, env):
    client = make_client()
    task = client.wait([client.generate("nv_generate_ctmr", "rflow-ct", GEN,
                                        budget="personal", label="phantom-01")])
    client.download(task, out=env / "gen")
    make_client().download(task, out=env / "gen")
    assert len(list((env / "gen").iterdir())) == 1 and len(server.calls("GET", "/result")) == 1


# ── wait ──────────────────────────────────────────────────────────────────────
def test_wait_on_finished_tasks_returns_at_once_without_a_request(make_client, server, scans):
    client = make_client()
    done = client.wait(client.run([scans(f"s{i}.nii.gz") for i in range(5)], model=MODEL,
                                  budget="personal"))
    seen = len(server.requests)
    again = make_client().wait(done)
    assert [t.state for t in again] == ["completed"] * 5 and len(server.requests) == seen


def test_wait_on_many_ids_uses_the_listing(make_client, server, scans):
    client = make_client()
    ids = [t.id for t in client.wait(client.run([scans(f"s{i}.nii.gz") for i in range(5)],
                                                model=MODEL, budget="personal"))]
    seen = len(server.requests)
    make_client().wait(ids)
    new = server.requests[seen:]
    assert [r["path"] for r in new if r["method"] == "GET"].count("/api/v1/tasks") == 1
    assert not [r for r in new if r["path"].startswith("/api/v1/tasks/")]


def test_wait_without_tasks_follows_every_active_task_of_the_account(make_client, server):
    ids = [server.add_task("queued") for _ in range(2)]
    for task_id in ids:
        server.tasks[task_id]["script"] = [("queued", 1), ("completed", None)]
    done = make_client().wait()
    assert sorted(t.id for t in done) == sorted(ids)
    assert {t.state for t in done} == {"completed"}
    assert "state=active" in server.calls("GET", "/api/v1/tasks")[0]["query"]


# ── after a restart ───────────────────────────────────────────────────────────
def test_after_a_restart_tasks_wait_and_download_find_everything(make_client, server, scans,
                                                                 env):
    server.script = [("queued", 1), ("queued", 1), ("completed", None)]
    queued = make_client().run([scans("a.nii.gz"), scans("b.nii.gz")], model=MODEL,
                               budget="personal")
    # A new process: a new client, the same record, nothing kept in memory.
    fresh = make_client()
    listed = fresh.tasks()
    assert {t.id for t in listed} == {t.id for t in queued}
    assert {t.original_name for t in listed} == {"a.nii.gz", "b.nii.gz"}
    done = fresh.wait()
    assert {t.id for t in done} == {t.id for t in queued}
    saved = make_client().download(out=env / "results")
    assert sorted(p.name for p in saved) == [f"a__{MODEL}__{TASK}.nii.gz",
                                             f"b__{MODEL}__{TASK}.nii.gz"]
    assert len(make_client().download(out=env / "results")) == 0     # nothing new
    assert len(server.calls("GET", "/result")) == 2


def test_after_a_restart_a_generation_is_named_by_its_label(make_client, server, env):
    first = make_client()
    first.wait([first.generate("nv_generate_ctmr", "rflow-ct", GEN, budget="personal",
                               label="phantom-01")])
    saved = make_client().download(out=env / "gen")
    assert [p.name for p in saved] == ["phantom-01__nv_generate_ctmr__rflow-ct.nii.gz"]


def test_the_record_shows_scans_tasks_and_results(make_client, server, scans, env):
    client = make_client()
    done = client.wait(client.run(scans("a.nii.gz"), model=MODEL, budget="personal"))
    client.download(done, out=env / "results")
    record = make_client().scans()
    assert record.path == os.environ["NALVERA_MAPPING"]
    assert {r["original_name"] for r in record} == {"a.nii.gz"}
    assert any(r["result_path"] for r in record) and any(r["result_size"] for r in record)


# ── save_anonymized ───────────────────────────────────────────────────────────
def test_save_anonymized_leaves_the_copies_and_nothing_else(env, monkeypatch):
    import examples_helpers as helpers
    monkeypatch.delenv("NALVERA_MAPPING")
    monkeypatch.delenv("NALVERA_EMAIL")
    monkeypatch.delenv("NALVERA_API_KEY")
    scan = helpers.write_nifti(env / "in" / "Doe_Jane.nii.gz", seed=1, descrip=b"Doe^Jane")
    copies = nalvera.save_anonymized(scan, env / "share")
    every = sorted(p.name for p in (env / "share").iterdir())
    assert every == [copies[0].path.name] and copies[0].path.name.startswith("scan-")
    assert copies[0].shape == (8, 8, 6) and copies[0].removed
    record = copies.record
    assert record.parent == env / "config" / "records" and record.exists()
    assert "Doe_Jane" in record.read_text(encoding="utf-8-sig")
    assert "Doe" not in (env / "share" / every[0]).read_bytes().decode("latin-1")


def test_save_anonymized_never_works_inside_the_folder_it_is_given(env, monkeypatch):
    import tempfile

    import examples_helpers as helpers
    made = []
    real = tempfile.mkdtemp

    def watch(*args, **kwargs):
        made.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkdtemp", watch)
    scan = helpers.write_nifti(env / "ct.nii.gz", seed=2)
    nalvera.save_anonymized(scan, env / "share")
    assert made and all("dir" not in kwargs and len(args) < 3 for args, kwargs in made)


def test_save_anonymized_of_a_signed_in_client_uses_its_record(make_client, scans, env):
    client = make_client()
    copies = client.save_anonymized(scans("a.nii.gz"), env / "share")
    assert copies.record == Path(os.environ["NALVERA_MAPPING"])
    assert [p.name for p in (env / "share").iterdir()] == [copies[0].path.name]


def test_the_dicom_conversion_folder_is_made_in_the_system_temporary_folder():
    import inspect

    from nalvera import anonymize
    source = inspect.getsource(anonymize._from_dicom)
    assert "mkdtemp(prefix=DICOM_PREFIX)" in source and "dir=workdir" not in source
