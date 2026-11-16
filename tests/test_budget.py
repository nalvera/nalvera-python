"""Which budget pays: the rule and the question, with injected streams."""
from __future__ import annotations

import io

import pytest

from nalvera import budget
from nalvera.errors import BudgetChoiceRequired, ConfigError, PermissionDenied


def _budgets(labs=(), free=0, period=False, personal=13.2, suggested=None):
    active = [lab for lab in labs if lab["member_status"] == "active"]
    best = min(active, key=lambda lab: (-lab["balance"], lab["name"])) if active else None
    return {"free": {"period_active": period, "free_tasks_left": free},
            "personal": {"monthly_left": 3.2, "topup": 10.0, "available_credits": personal},
            "labs": list(labs),
            "suggested": suggested or (f"lab:{best['id']}" if best else "personal")}


def _lab(lab_id, name, balance, status="active", limit=None, spent=0.0):
    return {"id": lab_id, "name": name, "balance": balance, "member_status": status,
            "spending_limit": limit, "spent": spent}


LAB = _lab(12, "Imaging Lab", 100.0)


# ── the rule ──────────────────────────────────────────────────────────────────
def test_no_lab_and_no_budget_is_personal_without_a_question():
    plan = budget.plan(_budgets(), requested=None, interactive=True)
    assert (plan.budget, plan.ask) == ("personal", False)


def test_a_named_budget_is_used():
    assert budget.plan(_budgets([LAB]), requested="personal", interactive=False).budget \
        == "personal"
    assert budget.plan(_budgets([LAB]), requested="lab:12", interactive=False).budget == "lab:12"


def test_a_lab_that_is_not_an_active_membership_is_refused_before_uploading():
    suspended = _lab(13, "Old Lab", 500.0, status="suspended")
    for requested in ("lab:99", "lab:13"):
        with pytest.raises(PermissionDenied) as caught:
            budget.plan(_budgets([LAB, suspended]), requested=requested, interactive=True)
        assert caught.value.code == "lab_not_allowed"


def test_a_lab_and_no_terminal_require_a_choice():
    with pytest.raises(BudgetChoiceRequired) as caught:
        budget.plan(_budgets([LAB]), requested=None, interactive=False)
    assert caught.value.code == "budget_choice_required"
    assert [o["budget"] for o in caught.value.options] == ["personal", "lab:12"]


def test_yes_accepts_the_suggestion():
    big = _lab(14, "Big Lab", 900.0)
    plan = budget.plan(_budgets([LAB, big]), requested=None, interactive=False, assume_yes=True)
    assert plan.budget == "lab:14"


def test_a_lab_and_a_terminal_ask_later():
    plan = budget.plan(_budgets([LAB], free=2), requested=None, interactive=True)
    assert plan.ask is True and plan.budget is None and plan.free_tasks_left == 2


def test_ask_without_a_terminal_requires_a_choice():
    with pytest.raises(BudgetChoiceRequired):
        budget.plan(_budgets(), requested="ask", interactive=False)


def test_suspended_labs_are_never_offered():
    plan = budget.plan(_budgets([LAB, _lab(13, "Old", 999.0, "suspended")]), requested=None,
                       interactive=True)
    assert [o["budget"] for o in plan.options] == ["personal", "lab:12"]


def test_a_members_spending_limit_bounds_what_a_lab_offers():
    limited = _lab(15, "Limited", 1000.0, limit=10.0, spent=4.0)
    options = budget.options_from_budgets(_budgets([limited]))
    assert options[1]["available_credits"] == 6.0


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), (" Personal ", "personal"),
                                            ("lab:12", "lab:12"), ("ASK", "ask")])
def test_parse_budget(value, expected):
    assert budget.parse_budget(value) == expected


@pytest.mark.parametrize("value", ["team", "lab:", "lab:abc", "lab:0", "lab:-1", "wallet"])
def test_parse_budget_refuses_other_values(value):
    with pytest.raises(ConfigError) as caught:
        budget.parse_budget(value)
    assert caught.value.code == "invalid_budget"


# ── the question ──────────────────────────────────────────────────────────────
def _ask(answer, labs=(LAB,), **kwargs):
    plan = budget.plan(_budgets(list(labs), **kwargs), requested=None, interactive=True)
    stdin, stdout = io.StringIO(answer), io.StringIO()
    chosen = budget.ask(plan, credits=4.5, count=3, stdin=stdin, stdout=stdout)
    return chosen, stdout.getvalue()


def test_choosing_a_number_returns_that_budget_and_prints_how_to_skip():
    chosen, shown = _ask("1\n")
    assert chosen == "personal"
    assert 'budget="personal"' in shown and "NALVERA_BUDGET=personal" in shown


def test_enter_accepts_the_suggestion():
    chosen, shown = _ask("\n")
    assert chosen == "lab:12"
    assert 'budget="lab:12"' in shown and "NALVERA_BUDGET=lab:12" in shown


def test_the_question_shows_each_balance_and_the_runs_cost():
    _chosen, shown = _ask("\n")
    assert "4.50" in shown and "13.20" in shown and "100.00" in shown and "Imaging Lab" in shown


def test_a_wrong_answer_is_asked_again():
    chosen, _shown = _ask("7\nabc\n2\n")
    assert chosen == "lab:12"


def test_no_answer_at_all_requires_a_choice():
    with pytest.raises(BudgetChoiceRequired):
        _ask("")
    with pytest.raises(BudgetChoiceRequired):
        _ask("x\ny\nz\n")


def test_the_estimates_own_balances_are_shown():
    plan = budget.plan(_budgets([LAB]), requested=None, interactive=True)
    stdout = io.StringIO()
    budget.ask(plan, credits=1.0, count=1, stdin=io.StringIO("\n"), stdout=stdout,
               estimate_budgets=[{"budget": "lab:12", "available_credits": 42.0}])
    assert "42.00" in stdout.getvalue()
