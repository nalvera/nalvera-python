"""Which budget pays for a run.

The rule: free tasks first, then the paid budget the user chose; a
wallet is never picked silently while the account belongs to a lab.

The server uses free tasks first, always (an active free period, then the
account's own free tasks). The package decides what it SENDS:

1. A budget given by the user (``budget=``, ``--budget`` or
   ``NALVERA_BUDGET``) is sent as given: ``personal``, ``lab:<id>`` or
   ``free``. ``free`` means "use my free tasks and stop when they run out":
   the server queues a task only when a free task pays for it
   (``no_free_tasks`` otherwise) and never charges a budget.
2. No budget given, and free tasks cover every task of the run (the free
   period is active, or ``free_tasks_left`` is at least the number of
   tasks): ``free`` is sent and nothing is asked.
3. No budget given, and free tasks cover only part: the free part is sent
   as ``free``. For the rest, an account with no active lab pays from
   ``personal``; with a lab, ``yes`` takes the suggested budget (the active
   lab with the highest balance, as on the Submit page), a terminal gets
   one question (how many tasks are free, what the rest costs, each
   budget's balance), and without a terminal ``BudgetChoiceRequired``
   is raised before anything is uploaded.
4. ``no_free_tasks`` in the middle of a run (a free task was spent
   elsewhere): the remaining tasks follow rule 3.

``ask`` (as a budget) always asks, in a terminal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TextIO

from . import messages
from .errors import BudgetChoiceRequired, ConfigError, PermissionDenied, UnexpectedResponse

_LAB_RX = re.compile(r"lab:([1-9][0-9]{0,17})")
PERSONAL = "personal"
FREE = "free"
ASK = "ask"


def _unexpected() -> UnexpectedResponse:
    return UnexpectedResponse(messages.text("error.unexpected_response", status=200),
                              code="unexpected_response", status=200)


def lab_budget(lab_id: Any) -> str:
    """``lab:<id>`` for a lab id from the platform's answer. The text goes
    into an argument a person copies (``budget="lab:12"``), so an id that is
    not a whole number is refused as an answer this version cannot read
    (``UnexpectedResponse``)."""
    if isinstance(lab_id, bool) or not isinstance(lab_id, int) or not 1 <= lab_id < 10 ** 18:
        raise _unexpected()
    return f"lab:{lab_id}"


def _check_name(value: Any, allowed: Sequence[str]) -> None:
    """A budget name from an answer: one of ``allowed`` or ``lab:<id>``."""
    if not (value in allowed or (isinstance(value, str) and _LAB_RX.fullmatch(value))):
        raise _unexpected()


def checked_budgets(answer: Dict[str, Any]) -> Dict[str, Any]:
    """``GET /api/v1/budgets`` as it came, once every lab id is a whole
    number and the suggestion is ``personal`` or ``lab:<id>``."""
    for lab in answer.get("labs") or []:
        lab_budget(lab.get("id") if isinstance(lab, dict) else None)
    if answer.get("suggested") is not None:
        _check_name(answer.get("suggested"), (PERSONAL,))
    return answer


def checked_estimate(answer: Dict[str, Any]) -> Dict[str, Any]:
    """``POST /api/v1/estimate`` as it came, once every budget it names (and
    its suggestion) is ``personal``, ``free`` or ``lab:<id>``."""
    for option in answer.get("budgets") or []:
        _check_name(option.get("budget") if isinstance(option, dict) else None,
                    (PERSONAL, FREE))
    if answer.get("suggested") is not None:
        _check_name(answer.get("suggested"), (PERSONAL, FREE))
    return answer


def parse_budget(value: Optional[str]) -> Optional[str]:
    """``personal``, ``lab:<id>``, ``free`` or ``ask`` (trimmed, lower
    case); None when not set. Raises ``ConfigError``
    (``invalid_budget``) for anything else."""
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if not cleaned:
        return None
    if cleaned in (PERSONAL, FREE, ASK) or _LAB_RX.fullmatch(cleaned):
        return cleaned
    raise ConfigError(messages.text("error.invalid_budget", budget=value), code="invalid_budget")


def lab_available(lab: Dict[str, Any]) -> float:
    """What a member may still spend from a lab: its balance, bounded by
    their own spending limit when a manager set one."""
    available = float(lab.get("balance") or 0.0)
    limit = lab.get("spending_limit")
    if limit is not None:
        available = min(available, max(0.0, float(limit) - float(lab.get("spent") or 0.0)))
    return max(0.0, available)


def options_from_budgets(budgets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Personal and each ACTIVE lab, as ``{"budget", "name", "available_credits"}``."""
    out = [{"budget": PERSONAL, "name": messages.text("budget.personal_name"),
            "available_credits": float((budgets.get("personal") or {}).get("available_credits")
                                       or 0.0)}]
    for lab in budgets.get("labs") or []:
        if lab.get("member_status") == "active":
            out.append({"budget": lab_budget(lab.get("id")), "name": lab.get("name") or "",
                        "available_credits": round(lab_available(lab), 2)})
    return out


def free_cover(free: Dict[str, Any], count: int) -> int:
    """How many of ``count`` tasks the free tasks pay for, from a ``free``
    block (``period_active``, ``free_tasks_left``)."""
    if free.get("period_active"):
        return count
    try:
        left = int(free.get("free_tasks_left") or 0)
    except (TypeError, ValueError):
        left = 0
    return max(0, min(count, left))


def covered(costs: Sequence[float], available: float) -> int:
    """How many tasks, taken in order, ``available`` credits pay for."""
    total, count = 0.0, 0
    for cost in costs:
        total += float(cost)
        if total > float(available) + 1e-9:
            break
        count += 1
    return count


@dataclass
class BudgetPlan:
    """What a run sends. The first ``free_count`` tasks go as ``free``; the
    others as ``budget``, or, when ``ask`` is True, as the budget the
    question returns. ``budget`` is ``"free"`` when the free tasks pay for
    everything (or the user chose ``free``)."""
    budget: Optional[str]
    ask: bool
    suggested: str
    free_tasks_left: int
    free_period: bool
    options: List[Dict[str, Any]] = field(default_factory=list)
    free_count: int = 0
    requested: Optional[str] = None

    @property
    def has_lab(self) -> bool:
        """Whether an active lab could pay."""
        return len(self.options) > 1


def plan(budgets: Dict[str, Any], *, requested: Optional[str], interactive: bool,
         assume_yes: bool = False, count: Optional[int] = None) -> BudgetPlan:
    """Apply the module's rule to ``budgets`` (``GET /api/v1/budgets``) for a
    run of ``count`` tasks (None: not known yet, so no free coverage is
    assumed). Raises ``BudgetChoiceRequired``, or
    ``PermissionDenied`` (``lab_not_allowed``) for a lab that is not
    one of the account's active labs, before anything is uploaded."""
    requested = parse_budget(requested)
    options = options_from_budgets(budgets)
    suggested = budgets.get("suggested") or PERSONAL
    if suggested not in {o["budget"] for o in options}:
        suggested = PERSONAL
    free = budgets.get("free") or {}
    free_count = free_cover(free, count) if count is not None else 0
    base = {"suggested": suggested, "free_tasks_left": int(free.get("free_tasks_left") or 0),
            "free_period": bool(free.get("period_active")), "options": options,
            "requested": requested}
    has_lab = len(options) > 1
    if requested is not None and requested.startswith("lab:"):
        if requested not in {o["budget"] for o in options}:
            raise PermissionDenied(messages.text("error.lab_not_listed", budget=requested),
                                   code="lab_not_allowed", extra={"budget": requested})
        return BudgetPlan(budget=requested, ask=False, free_count=0, **base)
    if requested in (PERSONAL, FREE):
        return BudgetPlan(budget=requested, ask=False, free_count=0, **base)
    if requested is None and count is not None and count > 0 and free_count >= count:
        return BudgetPlan(budget=FREE, ask=False, free_count=free_count, **base)
    if requested is None and not has_lab:
        return BudgetPlan(budget=PERSONAL, ask=False, free_count=free_count, **base)
    if assume_yes and requested is None:
        return BudgetPlan(budget=suggested, ask=False, free_count=free_count, **base)
    if interactive:
        return BudgetPlan(budget=None, ask=True, free_count=free_count, **base)
    raise BudgetChoiceRequired(options, suggested=suggested)


def ask(plan_: BudgetPlan, *, credits: float, count: int, stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None, estimate_budgets: Optional[List[Dict[str, Any]]] = None,
        attempts: int = 3, reporter: Any = None, free_count: int = 0) -> str:
    """Ask once which budget pays for the tasks beyond the free ones.

    Shows the cost (``credits`` for ``count`` paid tasks, from
    ``POST /api/v1/estimate``), how many tasks are free (``free_count``),
    each budget's balance (``estimate_budgets`` when given: the server's own
    figures) and marks the suggestion; Enter accepts it. Prints the option
    and the setting that skip the question next time. Returns the budget.

    The question goes to ``reporter`` (a ``nalvera.Reporter`` that can
    ask); without one, to ``stdout`` and ``stdin``. Raises
    ``BudgetChoiceRequired`` when nobody answers."""
    if reporter is None:
        from .report import StreamReporter
        reporter = StreamReporter(output=stdout, input=stdin, interactive=True)
    options = _merge(plan_.options, estimate_budgets)
    if not reporter.can_ask():
        raise BudgetChoiceRequired(options, suggested=plan_.suggested)
    default = next((i for i, o in enumerate(options) if o["budget"] == plan_.suggested), 0)
    if free_count:
        reporter.say("budget.question_free", free=free_count, count=count,
                     credits=f"{credits:.2f}")
    else:
        reporter.say("budget.question", count=count, credits=f"{credits:.2f}")
    if not free_count:
        if plan_.free_period:
            reporter.say("budget.free_period_note")
        elif plan_.free_tasks_left:
            reporter.say("budget.free_note", count=plan_.free_tasks_left)
    for number, option in enumerate(options, start=1):
        mark = ""
        if number - 1 == default:
            mark = messages.text("budget.mark_suggested")
        elif option["available_credits"] < credits:
            mark = messages.text("budget.mark_short")
        reporter.say("budget.option", number=number, name=option["name"],
                     available=f"{option['available_credits']:.2f}", mark=mark)
    for _ in range(max(1, attempts)):
        line = reporter.ask(messages.text("budget.prompt", default=default + 1))
        if line is None:
            break                                   # end of input: nobody to answer
        answer = line.strip()
        if not answer:
            choice = default
        elif answer.isdigit() and 1 <= int(answer) <= len(options):
            choice = int(answer) - 1
        else:
            reporter.say("budget.invalid_choice", max=len(options))
            continue
        chosen = options[choice]["budget"]
        reporter.say("budget.skip_hint", budget=chosen)
        return chosen
    raise BudgetChoiceRequired(options, suggested=plan_.suggested)


def confirm(reporter: Any, key: str, /, **fields: Any) -> bool:
    """Ask a yes/no question (``key`` is its text); True only for yes."""
    if not reporter.can_ask():
        return False
    line = reporter.ask(messages.text(key, **fields) + messages.text("confirm.prompt"))
    return bool(line) and line.strip().lower() in ("y", "yes")


def _merge(options: List[Dict[str, Any]],
           estimate_budgets: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The options with the estimate's balances where it has them."""
    figures = {b.get("budget"): b for b in (estimate_budgets or []) if isinstance(b, dict)}
    out = []
    for option in options:
        merged = dict(option)
        if option["budget"] in figures:
            try:
                merged["available_credits"] = float(
                    figures[option["budget"]].get("available_credits"))
            except (TypeError, ValueError):
                pass
        out.append(merged)
    return out
