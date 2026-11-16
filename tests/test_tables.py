"""What the tables show: a task's own modality, prices with the
digits they have, numbers alone in a number column, a heading on every
column, short paths, and the budget-choice error that names the suggestion."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from nalvera import budget as budget_mod
from nalvera import display, messages
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
from nalvera.errors import BudgetChoiceRequired

MRSEG = {"id": "mrsegmentator", "name": "MRSegmentator", "modality": ["MRI"], "tasks": [
    {"id": "anatomy", "name": "Anatomy (MRI)", "default": True, "modality": ["MRI"],
     "price_eur_per_mvoxel": 0.006},
    {"id": "anatomy_ct", "name": "Anatomy (CT)", "default": False, "modality": ["CT"],
     "price_eur_per_mvoxel": 0.01},
    {"id": "body_comp", "name": "Body composition", "default": False,
     "price_eur_per_mvoxel": 0.02}]}


def _rows(value):
    columns, rows, notes = value._table()
    return columns, [[display.cell(v) for v in row] for row in rows], notes


# ── a task's own modality ─────────────────────────────────────────────────────
def test_a_tasks_own_modality_wins_and_the_models_is_the_fallback():
    columns, rows, _notes = _rows(display.ModelList([MRSEG]))
    modality = list(columns).index("modality")
    assert [row[modality] for row in rows] == ["MRI", "CT", "MRI"]


def test_the_command_line_and_the_assistant_server_use_the_tasks_modality(cli_dialect):
    from nalvera.mcp.tools import models_out
    tasks = models_out([MRSEG], None)["models"][0]["tasks"]
    assert [t["modality"] for t in tasks] == [["MRI"], ["CT"], ["MRI"]]


# ── prices with the digits they have ──────────────────────────────────────────
@pytest.mark.parametrize("value,shown", [(0.006, "0.006"), (0.01, "0.01"), (0.02, "0.02"),
                                         (0.1, "0.10"), (1, "1.00"), (0.0125, "0.0125")])
def test_a_price_keeps_its_digits(value, shown):
    assert display.price_text(value) == shown


def test_the_models_table_shows_each_price_with_its_digits():
    columns, rows, _notes = _rows(display.ModelList([MRSEG]))
    price = list(columns).index("price")
    assert [row[price] for row in rows] == ["0.006", "0.01", "0.02"]


def test_credits_keep_two_decimals():
    assert display.cell(1.256) == "1.26" and display.cell(3.0) == "3.00"


def test_the_command_line_models_shows_each_price_with_its_digits(env, server, stub_anon,
                                                                  no_sleep):
    from test_cli import run_cli
    server.models = [MRSEG]
    status, out, _err = run_cli("models")
    assert status == 0
    for price in ("0.006", "0.01", "0.02"):
        assert f" {price} EUR" in out


def test_the_assistant_server_answers_the_exact_price():
    from nalvera.mcp.tools import models_out
    prices = [t["price_eur_per_mvoxel"] for t in models_out([MRSEG], None)["models"][0]["tasks"]]
    assert prices == [0.006, 0.01, 0.02]


# ── numbers alone in a number column; a note column only when needed ──────────
def _estimate(skipped=(), failed=()):
    return RunEstimate(model="m", task="t",
                       scans=[ScanEstimate("a.nii.gz", "/a", (4, 4, 4), 1.25)],
                       skipped=list(skipped), failed=list(failed), credits=1.25, free_count=0,
                       paid_credits=1.25, free_period=False, free_tasks_left=0,
                       budgets=[], suggested="personal")


def test_the_estimate_credits_column_holds_numbers_only_and_words_go_to_a_note():
    value = _estimate(skipped=[SkippedScan("b.nii.gz", "/b", "duplicate_in_run")],
                      failed=[ScanFailure("c.nii.gz", "/c", "not_a_scan", "Not a scan.")])
    columns, rows, _notes = value._table()
    assert list(columns) == ["scan", "shape", "credits", "note"]
    credits = [row[2] for row in rows]
    assert credits[0] == 1.25 and credits[1:] == [None, None]
    assert rows[0][3] is None and rows[1][3] and rows[2][3] == "Not a scan."


def test_without_words_the_estimate_has_no_note_column():
    columns, rows, _notes = _estimate()._table()
    assert list(columns) == ["scan", "shape", "credits"] and all(len(r) == 3 for r in rows)


def test_one_scan_has_no_total_line_that_repeats_its_row():
    _columns, _rows, notes = _estimate()._table()
    assert not [n for n in notes if "1.25" in n]


def test_the_budgets_and_the_suggestion_are_shown_only_when_there_is_a_choice():
    alone = [{"budget": "personal", "name": "Personal", "available_credits": 100.0}]
    lab = alone + [{"budget": "lab:12", "name": "Imaging Lab", "available_credits": 50.0}]
    value = _estimate()
    value.budgets, value.suggested = alone, "personal"
    assert value._table()[2] == []
    value.budgets, value.suggested = lab, "lab:12"
    notes = value._table()[2]
    assert len(notes) == 2 and messages.text("budget.mark_suggested").strip() in notes[1]
    budgets = {"personal": {"available_credits": 100.0}, "suggested": "personal", "labs": []}
    assert display.Budgets(budgets)._table()[2] == []
    budgets.update(suggested="lab:12", labs=[{"id": 12, "name": "Imaging Lab", "balance": 50.0,
                                              "member_status": "active"}])
    assert len(display.Budgets(budgets)._table()[2]) == 1


# ── every column has a heading ────────────────────────────────────────────────
def _task(**fields):
    data = {"id": "t1", "state": "queued", "name": "scan-AAAA.nii.gz", "model": "m",
            "task": "t", "queue_position": 2}
    return Task.from_api(data, **fields)


@pytest.mark.parametrize("make", [
    lambda: RunResult([_task(original_name="a.nii.gz",
                             charged={"source": "personal", "credits": 1.25},
                             charged_now=True)]),
    lambda: _estimate(failed=[ScanFailure("c.nii.gz", "/c", "not_a_scan", "Not a scan.")]),
    lambda: display.Account({"email": "a@b.org", "key": {"prefix": "nvk_aaaaaaaa",
                                                         "scopes": []}}),
    lambda: Downloads([Path("results") / "a__m__t.nii.gz"]),
    lambda: SavedCopies([SavedCopy("a.nii.gz", "/a", Path("/s/scan-A.nii.gz"), "scan-A",
                                   (1, 2, 3), ["the file name"])],
                        failed=[ScanFailure("c.nii.gz", "/c", "not_a_scan", "Not a scan.")]),
    lambda: display.ModelList([MRSEG]),
    lambda: display.Budgets({"personal": {"available_credits": 1}}),
    lambda: display.Limits({"min_poll_seconds": 5}), lambda: display.TaskList([_task()]),
    lambda: display.ScanRecord([{"code": "scan-A"}]),
])
def test_every_column_has_a_heading(make):
    value = make()
    columns, rows, _notes = value._table()
    for column in columns:
        assert display.header(column).strip(), column
    assert all(len(row) == len(columns) for row in rows)


def test_the_run_table_shows_the_charge_under_its_own_heading():
    item = _task(original_name="a.nii.gz", charged={"source": "personal", "credits": 1.25},
                 charged_now=True)
    columns, rows, notes = RunResult([item], charged_credits=1.25)._table()
    assert list(columns) == ["scan", "task_id", "state", "charge"]
    assert "1.25" in display.cell(rows[0][3])
    assert not [n for n in notes if "1.25" in n]          # one task: the row says it


def test_a_run_of_several_tasks_adds_the_total_charge():
    items = [_task(original_name=f"{n}.nii.gz", charged={"source": "personal", "credits": 1.25},
                   charged_now=True) for n in "ab"]
    items[1].id = "t2"
    _columns, _rows, notes = RunResult(items, charged_credits=2.5)._table()
    assert [n for n in notes if "2.50" in n]


# ── paths ─────────────────────────────────────────────────────────────────────
def test_a_path_in_the_working_folder_is_shown_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = tmp_path / "results" / "ct__totalsegmentator_ct__total.nii.gz"
    assert display.path_text(saved) == os.path.join("results", saved.name)


def test_a_long_path_is_cut_in_the_middle_keeping_the_file_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = "ct_scan__totalsegmentator_ct__total.nii.gz"
    deep = Path("/") / ("folder_" * 30) / "work" / "results" / name
    shown = display.path_text(deep)
    assert shown.endswith(os.sep + name) and "…" in shown
    assert len(shown) <= display.PATH_MAX and shown.startswith(os.sep + "folder_")


def test_the_download_table_shows_the_file_name_of_a_long_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = "ct_scan__totalsegmentator_ct__total.nii.gz"
    saved = Downloads([Path("/") / ("x" * 180) / name])
    assert name in saved._repr_html_() and name in repr(saved)


# ── the budget choice names the suggestion ────────────────────────────────────
OPTIONS = [{"budget": "personal", "name": "Personal", "available_credits": 13.2},
           {"budget": "lab:12", "name": "Imaging Lab", "available_credits": 100.0}]


def test_the_budget_choice_lists_balances_marks_the_suggestion_and_names_the_argument():
    error = BudgetChoiceRequired(OPTIONS, suggested="lab:12")
    text = str(error)
    assert error.suggested == "lab:12" and error.code == "budget_choice_required"
    lab_line = next(line for line in text.splitlines() if "lab:12" in line and "100.00" in line)
    assert messages.text("budget.mark_suggested").strip() in lab_line
    personal_line = next(line for line in text.splitlines() if "13.20" in line)
    assert messages.text("budget.mark_suggested").strip() not in personal_line
    assert 'budget="lab:12"' in text and "--budget" not in text


def test_on_the_command_line_the_budget_choice_names_the_option(cli_dialect):
    text = str(BudgetChoiceRequired(OPTIONS, suggested="lab:12"))
    assert "--budget lab:12" in text and "budget=" not in text


def test_without_a_suggestion_the_error_follows_the_platforms_rule():
    assert BudgetChoiceRequired(OPTIONS).suggested == "lab:12"
    assert BudgetChoiceRequired(OPTIONS[:1]).suggested == "personal"


def test_a_run_that_needs_a_choice_names_the_platforms_suggestion(make_client, server, scans):
    server.add_lab(12, "Imaging Lab", 100.0)
    server.add_lab(14, "Other Lab", 300.0)
    with pytest.raises(BudgetChoiceRequired) as caught:
        make_client().run(scans("a.nii.gz"), model="totalsegmentator_ct")
    assert caught.value.suggested == "lab:14" and 'budget="lab:14"' in str(caught.value)
    assert not server.calls("POST", "/api/v1/uploads")


def test_the_plan_passes_its_suggestion():
    budgets = {"personal": {"available_credits": 5.0}, "suggested": "lab:12",
               "labs": [{"id": 12, "name": "Imaging Lab", "balance": 100.0,
                         "member_status": "active"}]}
    with pytest.raises(BudgetChoiceRequired) as caught:
        budget_mod.plan(budgets, requested=None, interactive=False, count=1)
    assert caught.value.suggested == "lab:12"


def test_no_choice_text_repeats_a_balance():
    text = str(BudgetChoiceRequired(OPTIONS, suggested="lab:12"))
    assert len(re.findall(r"100\.00", text)) == 1 and len(re.findall(r"13\.20", text)) == 1
