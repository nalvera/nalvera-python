"""What a notebook cell shows: the values the functions
return keep their types for code and show a text table and an HTML table
with the same content. Every value from a server or a file name is escaped,
made printable and cut; no attribute, style or link is built from data."""
from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from nalvera import display
from nalvera.client import (
    Downloads,
    RunEstimate,
    RunResult,
    SavedCopies,
    SavedCopy,
    ScanEstimate,
    ScanFailure,
    SkippedScan,
    Task,
)

LONG = "x" * 10000
HOSTILE = ["<script>alert(1)</script>", '" onmouseover="alert(1)', "</td><td>forged",
           "two\nlines", "evil‮txt.exe", LONG, "<img src=x onerror=alert(1)>",
           "<a href=\"javascript:alert(1)\">link</a>", "<style>body{display:none}</style>"]


def _task(value: str) -> Task:
    return Task.from_api({"id": value, "state": value, "name": value, "model": value,
                          "task": value, "queue_position": None, "error": value},
                         original_name=value, label=value)


def _values(value: str):
    """Every returned type, each holding ``value`` in every field that comes
    from a server or a file name."""
    task = _task(value)
    lab = {"id": value, "name": value, "balance": 3.0, "member_status": "active"}
    yield display.TaskList([task])
    yield task
    yield display.ModelList([{"id": value, "name": value, "version": value,
                              "modality": [value], "tasks": [{"id": value, "name": value,
                                                              "default": True,
                                                              "price_eur_per_mvoxel": 1}]}])
    yield display.Budgets({"free": {"free_tasks_left": value},
                           "personal": {"available_credits": value}, "labs": [lab],
                           "suggested": value})
    yield display.Limits({value: value})
    yield display.Account({"email": value, "plan_label": value,
                           "key": {"prefix": value, "name": value, "scopes": [value],
                                   "expires_at": value}})
    yield display.ScanRecord([{"code": value, "original_name": value, "series": value,
                               "label": value, "task_id": value, "model": value,
                               "task": value, "result_path": value}], path=value)
    yield RunResult([task], failed=[ScanFailure(value, value, value, value)],
                    skipped=[SkippedScan(value, value, value, "other")],
                    not_queued=[ScanFailure(value, value, value, value)])
    yield RunEstimate(model=value, task=value,
                      scans=[ScanEstimate(value, value, (1, 2, 3), 1.0, series=value)],
                      skipped=[SkippedScan(value, value, value)],
                      failed=[ScanFailure(value, value, value, value)], credits=1.0,
                      free_count=1, paid_credits=0.0, free_period=False, free_tasks_left=1,
                      budgets=[{"budget": value, "name": value, "available_credits": value}],
                      suggested=value)
    yield Downloads([Path("/tmp") / value[:100]], skipped=[(value, value)])
    yield SavedCopies([SavedCopy(value, value, Path("/tmp") / value[:100], value, (1, 2, 3),
                                 [value], series=value)],
                      failed=[ScanFailure(value, value, value, value)],
                      record=Path("/tmp") / value[:100])


def _check_html(shown: str) -> None:
    assert "<script" not in shown and "<img" not in shown and "<style" not in shown
    assert "<a " not in shown
    for tag in re.findall(r"<[^>]*>", shown):
        assert re.fullmatch(r"</?(table|thead|tbody|tr|th|td|p)>", tag), tag   # no attributes
    assert "‮" not in shown and "\n" not in shown
    for cell in re.findall(r"<t[dh]>(.*?)</t[dh]>", shown):
        assert len(html.unescape(cell)) <= display.CELL_MAX


def _check_text(shown: str) -> None:
    assert "‮" not in shown and "\x1b" not in shown
    assert not re.search(r"[\x00-\x08\x0b-\x1f\x7f]", shown)
    for line in shown.splitlines():
        assert len(line) < 20 * (display.CELL_MAX + 2)


@pytest.mark.parametrize("value", HOSTILE)
def test_hostile_values_are_escaped_printable_and_cut(value):
    for item in _values(value):
        _check_html(item._repr_html_())
        _check_text(repr(item))


def test_the_text_table_has_the_content_of_the_html_table():
    for item in _values("Doe & <Jane>"):
        columns, rows, notes = item._table()
        text, shown = repr(item), html.unescape(item._repr_html_())
        for row in rows:
            for value in row:
                assert display.cell(value) in text and display.cell(value) in shown
        for note in notes:
            assert note in text and note in shown


def test_a_long_cell_is_cut_and_marked():
    assert len(display.cell(LONG)) == display.CELL_MAX and display.cell(LONG).endswith("…")


def test_the_types_stay_data_for_code():
    tasks = display.TaskList([_task("a")])
    assert isinstance(tasks, list) and tasks[0].id == "a"
    budgets = display.Budgets({"personal": {"available_credits": 1.0}})
    assert isinstance(budgets, dict) and budgets["personal"]["available_credits"] == 1.0
    import json
    assert json.loads(json.dumps(display.Account({"email": "a"}))) == {"email": "a"}


def test_what_the_functions_return_shows_as_tables(make_client, server, scans, env):
    client = make_client()
    for value in (client.models(), client.budgets(), client.limits(), client.me(),
                  client.tasks()):
        assert "<table>" in value._repr_html_() and "\n" in repr(value)
    result = client.run(scans("a.nii.gz"), model="totalsegmentator_ct", budget="personal")
    finished = client.wait(result)
    saved = client.download(finished, out=env / "results")
    for value in (result, finished, saved, finished[0], client.scans(),
                  client.estimate_run(scans("b.nii.gz"), model="totalsegmentator_ct")):
        assert "<table>" in value._repr_html_()
