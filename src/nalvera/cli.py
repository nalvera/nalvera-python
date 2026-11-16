"""The ``nalvera`` command.

Exit statuses (also in ``nalvera --help`` and ``messages.py``, keys
``help.exit.*``):

0    all done
1    some scans failed or were rejected, or a task did not complete
2    wrong usage: an option, a path or a setting is not valid
3    sign-in or permission: no key, a key refused, expired or revoked, a
     permission the key lacks, API access off, a package too old
4    credits or budget: too few credits, no free task left, a budget choice
     or a confirmation needed, the spending cap reached
5    limits or the platform unavailable after retries, or no connection
130  interrupted (Ctrl-C)

Every command calls public functions of the package and prints: no
behaviour lives only here (``COMMAND_FUNCTIONS`` names them). Everything
goes through one ``nalvera.report.StreamReporter``, and texts are
rendered for the command line (``messages.set_dialect("cli")``) for the
command's run.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import textwrap
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO

from . import _session, auth, display, messages, settings
from . import generation as generate_mod
from ._version import __version__
from .client import (
    Client,
    RunResult,
    Task,
    check_output_folder,
    default_record_path,
    save_anonymized,
)
from .errors import (
    ApiDisabled,
    AuthenticationError,
    BudgetChoiceRequired,
    ClientTooOld,
    ConfigError,
    ConfirmationRequired,
    CreditCapExceeded,
    EncryptionError,
    InsufficientCredits,
    InvalidRequest,
    NalveraError,
    NetworkError,
    NoCredentials,
    PermissionDenied,
    RateLimited,
    RunInterrupted,
    ServerError,
    ServiceUnavailable,
    UnexpectedResponse,
)
from .report import StreamReporter, safe_write  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_AUTH = 3
EXIT_CREDITS = 4
EXIT_UNAVAILABLE = 5
EXIT_INTERRUPTED = 130
#: Names kept from the first version.
EXIT_ERROR = EXIT_FAILED
EXIT_PARTIAL = EXIT_FAILED

#: Every exit status and the messages key that explains it.
EXIT_STATUSES = OrderedDict([
    (EXIT_OK, "help.exit.0"), (EXIT_FAILED, "help.exit.1"), (EXIT_USAGE, "help.exit.2"),
    (EXIT_AUTH, "help.exit.3"), (EXIT_CREDITS, "help.exit.4"),
    (EXIT_UNAVAILABLE, "help.exit.5"), (EXIT_INTERRUPTED, "help.exit.130"),
])


#: Settings errors that concern the key itself: a sign-in problem (3).
_SIGN_IN_CODES = frozenset({"invalid_key_format", "credentials_file_insecure",
                            "credentials_file_unreadable"})

#: ``nalvera status --all --limit``: the platform's task listing takes 1 to
#: 100 tasks a page and lists 50 when not told (GET /api/v1/tasks).
STATUS_LIMIT_RANGE = (1, 100)
STATUS_LIMIT_DEFAULT = 50

#: The public functions of ``nalvera`` each command calls (``mcp``: the
#: server of ``nalvera.mcp``). A test holds that each exists and is
#: documented, so no behaviour lives only in this module.
COMMAND_FUNCTIONS: Dict[str, Sequence[str]] = OrderedDict([
    ("login", ("login",)), ("logout", ("logout",)),
    ("whoami", ("me", "budgets", "key_expiry", "encryption")), ("limits", ("limits",)),
    ("models", ("models",)), ("anonymize", ("save_anonymized",)),
    ("estimate", ("estimate_run",)), ("run", ("run", "wait", "download")),
    ("generate", ("generate", "estimate", "wait", "download")),
    ("status", ("tasks", "task", "wait")), ("wait", ("wait",)), ("download", ("download",)),
    ("cancel", ("cancel",)), ("scans", ("scans",)), ("mcp", ("nalvera.mcp",)),
])


def exit_status(error: BaseException) -> int:
    """The exit status for an error that ended a command."""
    if isinstance(error, KeyboardInterrupt):
        return EXIT_INTERRUPTED
    code = getattr(error, "code", "")
    if isinstance(error, (InsufficientCredits, BudgetChoiceRequired, CreditCapExceeded,
                          ConfirmationRequired)) or code == "lab_not_allowed":
        return EXIT_CREDITS
    if isinstance(error, (NoCredentials, AuthenticationError, PermissionDenied, ApiDisabled,
                          ClientTooOld, EncryptionError)) or code in _SIGN_IN_CODES:
        return EXIT_AUTH
    if code == "unexpected_redirect" or isinstance(error, (ConfigError, InvalidRequest)):
        return EXIT_USAGE
    if isinstance(error, (RateLimited, ServiceUnavailable, NetworkError, ServerError,
                          UnexpectedResponse)):
        return EXIT_UNAVAILABLE
    return EXIT_FAILED


class _Context:
    """The streams a command reads and writes, and the one reporter."""

    def __init__(self, stdin: TextIO, stdout: TextIO, stderr: TextIO,
                 secret_input: Optional[Callable[[str], str]]):
        self.stdin, self.stdout, self.stderr = stdin, stdout, stderr
        self.reporter = StreamReporter(output=stdout, input=stdin, errors=stderr)
        self.secret_input = secret_input
        self.interrupted_key = "cli.interrupted"
        self.print_ids_to: Optional[TextIO] = None

    def say(self, key: str, /, **fields: Any) -> None:
        self.reporter.say(key, **fields)

    def error(self, message: str) -> None:
        self.reporter.error(messages.text("cli.error", message=message))

    def ids_only(self) -> None:
        """``--ids``: standard output carries the queued task ids alone;
        every message and question goes to standard error."""
        self.reporter = StreamReporter(output=self.stderr, input=self.stdin, errors=self.stderr)
        self.print_ids_to = self.stdout

    def print_ids(self, tasks: Sequence[Task]) -> None:
        """The task ids, one per line, when ``--ids`` asked for them."""
        if self.print_ids_to is not None and tasks:
            safe_write(self.print_ids_to, "".join(f"{t.id}\n" for t in tasks))

    def client(self, **kwargs: Any) -> Client:
        return Client(reporter=self.reporter, **kwargs)


def _h(key: str) -> str:
    """A help text, escaped for argparse's %-formatting."""
    return messages.text(key).replace("%", "%%")


def _parser_kwargs(help_key: str) -> Dict[str, Any]:
    # A description is printed as it is; a help text is %-formatted.
    return {"help": _h(help_key), "description": messages.text(help_key), "add_help": False}


def _add_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-h", "--help", action="help", help=_h("help.help"))


def _epilog() -> str:
    lines = [messages.text("help.settings_title"), settings.describe(), "",
             messages.text("help.exit_title")]
    for status, key in EXIT_STATUSES.items():
        line = messages.text("help.exit_line", status=status, meaning=messages.text(key))
        indent = len(line) - len(line.lstrip(" ")) + 5
        lines.append(textwrap.fill(line, width=79, subsequent_indent=" " * indent))
    lines += ["", messages.text("help.epilog")]
    return "\n".join(lines)                     # printed as it is (no %-formatting)


def _add_encrypt(sub: argparse.ArgumentParser) -> None:
    group = sub.add_mutually_exclusive_group()
    group.add_argument("--encrypt", dest="encrypt", action="store_true", default=None,
                       help=_h("help.run.encrypt"))
    group.add_argument("--no-encrypt", dest="encrypt", action="store_false", default=None,
                       help=_h("help.run.no_encrypt"))


def _add_spending(sub: argparse.ArgumentParser, command: str) -> None:
    sub.add_argument("--budget", metavar="BUDGET", help=_h(f"help.{command}.budget"))
    sub.add_argument("--max-credits", metavar="N", type=_credits_arg, default=None,
                     help=_h(f"help.{command}.max_credits"))
    sub.add_argument("--yes", action="store_true", help=_h(f"help.{command}.yes"))


def _add_ids(sub: argparse.ArgumentParser, command: str) -> None:
    sub.add_argument("--ids", dest="ids_only", action="store_true",
                     help=_h(f"help.{command}.ids"))


def _limit_arg(value: str) -> int:
    low, high = STATUS_LIMIT_RANGE
    try:
        number = int(value)
    except ValueError:
        number = low - 1
    if not low <= number <= high:
        raise argparse.ArgumentTypeError(messages.text("error.invalid_limit", value=value,
                                                       low=low, high=high))
    return number


def _credits_arg(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        number = -1.0
    if not number >= 0:
        raise argparse.ArgumentTypeError(messages.text("error.invalid_max_credits", value=value))
    return number


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``nalvera`` command."""
    parser = argparse.ArgumentParser(prog="nalvera",
                                     description=messages.text("help.description"),
                                     epilog=_epilog(), add_help=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser._optionals.title = messages.text("help.options")
    _add_help(parser)
    parser.add_argument("--version", action="version", version=f"nalvera {__version__}",
                        help=_h("help.version"))
    commands = parser.add_subparsers(dest="command", metavar="COMMAND",
                                     title=messages.text("help.commands"))

    def command(name: str, help_key: str, handler: Callable[..., int]) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, **_parser_kwargs(help_key))
        sub._optionals.title = messages.text("help.options")
        _add_help(sub)
        sub.set_defaults(handler=handler)
        return sub

    sub = command("login", "help.login", cmd_login)
    sub.add_argument("--email", help=_h("help.login.email"))
    command("logout", "help.logout", cmd_logout)
    command("whoami", "help.whoami", cmd_whoami)
    command("limits", "help.limits", cmd_limits)
    sub = command("models", "help.models", cmd_models)
    sub.add_argument("--model", metavar="ID", help=_h("help.models.model"))

    sub = command("anonymize", "help.anonymize", cmd_anonymize)
    sub.add_argument("paths", nargs="+", metavar="PATH", help=_h("help.anonymize.paths"))
    sub.add_argument("--out", metavar="DIR", required=True, help=_h("help.anonymize.out"))

    sub = command("estimate", "help.estimate", cmd_estimate)
    sub.add_argument("paths", nargs="+", metavar="PATH", help=_h("help.estimate.paths"))
    sub.add_argument("--model", required=True, metavar="M", help=_h("help.estimate.model"))
    sub.add_argument("--task", metavar="T", help=_h("help.estimate.task"))
    sub.add_argument("--force", action="store_true", help=_h("help.estimate.force"))

    sub = command("run", "help.run", cmd_run)
    sub.add_argument("paths", nargs="+", metavar="PATH", help=_h("help.run.paths"))
    sub.add_argument("--model", required=True, metavar="M", help=_h("help.run.model"))
    sub.add_argument("--task", metavar="T", help=_h("help.run.task"))
    sub.add_argument("--label", metavar="L", help=_h("help.run.label"))
    sub.add_argument("--text", metavar="TEXT", help=_h("help.run.text"))
    _add_spending(sub, "run")
    _add_encrypt(sub)
    sub.add_argument("--force", action="store_true", help=_h("help.run.force"))
    sub.add_argument("--no-wait", action="store_true", help=_h("help.run.no_wait"))
    _add_ids(sub, "run")
    sub.add_argument("--out", metavar="DIR", default="results", help=_h("help.run.out"))

    sub = command("generate", "help.generate", cmd_generate)
    sub.add_argument("--model", required=True, metavar="M", help=_h("help.generate.model"))
    sub.add_argument("--task", required=True, metavar="T", help=_h("help.generate.task"))
    sub.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                     help=_h("help.generate.param"))
    sub.add_argument("--label", metavar="L", help=_h("help.generate.label"))
    sub.add_argument("--estimate", action="store_true", help=_h("help.generate.estimate"))
    sub.add_argument("--force", action="store_true", help=_h("help.generate.force"))
    _add_spending(sub, "generate")
    sub.add_argument("--no-wait", action="store_true", help=_h("help.generate.no_wait"))
    _add_ids(sub, "generate")
    sub.add_argument("--out", metavar="DIR", default="results", help=_h("help.generate.out"))

    sub = command("status", "help.status", cmd_status)
    sub.add_argument("ids", nargs="*", metavar="TASK_ID", help=_h("help.status.ids"))
    sub.add_argument("--watch", action="store_true", help=_h("help.status.watch"))
    sub.add_argument("--all", action="store_true", help=_h("help.status.all"))
    sub.add_argument("--limit", metavar="N", type=_limit_arg, default=STATUS_LIMIT_DEFAULT,
                     help=_h("help.status.limit"))
    sub = command("wait", "help.wait", cmd_wait)
    sub.add_argument("ids", nargs="*", metavar="TASK_ID", help=_h("help.wait.ids"))
    sub = command("download", "help.download", cmd_download)
    sub.add_argument("ids", nargs="*", metavar="TASK_ID", help=_h("help.download.ids"))
    sub.add_argument("--out", metavar="DIR", default="results", help=_h("help.download.out"))
    sub.add_argument("--part", metavar="PART", help=_h("help.download.part"))
    sub = command("cancel", "help.cancel", cmd_cancel)
    sub.add_argument("ids", nargs="+", metavar="TASK_ID", help=_h("help.cancel.ids"))
    command("scans", "help.scans", cmd_scans)
    command("mcp", "help.mcp", cmd_mcp)
    return parser


#: Signals that stop a command as one Ctrl-C does: a Slurm time limit or
#: ``timeout`` sends SIGTERM, a closed terminal SIGHUP. The clean-up then
#: runs (no anonymized copy or DICOM work folder is left behind).
_STOP_SIGNALS = tuple(getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP")
                      if hasattr(signal, name))


class _InterruptGuard:
    """The first Ctrl-C (or SIGTERM, or SIGHUP) stops the command
    cleanly (KeyboardInterrupt); a second one exits at once, even while
    clean-up is still running."""

    def __init__(self, stderr: TextIO, exit_now: Callable[[int], None] = os._exit):
        self.stderr = stderr
        self.exit_now = exit_now
        self.pressed = 0
        self._previous: Dict[int, Any] = {}
        self._installed = False

    def handler(self, signum: int, frame: Any) -> None:
        self.pressed += 1
        if self.pressed == 1:
            raise KeyboardInterrupt
        try:
            self.stderr.write("\n" + messages.text("cli.interrupted_twice") + "\n")
            self.stderr.flush()
        finally:
            self.exit_now(EXIT_INTERRUPTED)

    def __enter__(self) -> _InterruptGuard:
        if threading.current_thread() is threading.main_thread():
            for number in _STOP_SIGNALS:
                try:
                    self._previous[number] = signal.signal(number, self.handler)
                except (ValueError, OSError):
                    continue
            self._installed = bool(self._previous)
        return self

    def __exit__(self, *exc: Any) -> None:
        for number, previous in self._previous.items():
            signal.signal(number, previous)
        self._previous = {}


def _replace_unencodable(stream: Any) -> None:
    """A console that cannot show "⋮" or "→" gets "?" instead of an
    error."""
    try:
        stream.reconfigure(errors="replace")
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None, *, stdin: Optional[TextIO] = None,
         stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None,
         secret_input: Optional[Callable[[str], str]] = None) -> int:
    """Run the ``nalvera`` command with ``argv`` (default: the command line)
    and return its exit status (see the module docstring). Texts are
    rendered for the command line while it runs."""
    previous = messages.set_dialect(messages.CLI)
    try:
        return _main(argv, stdin=stdin, stdout=stdout, stderr=stderr, secret_input=secret_input)
    finally:
        messages.set_dialect(previous)


def _main(argv: Optional[Sequence[str]], *, stdin: Optional[TextIO], stdout: Optional[TextIO],
          stderr: Optional[TextIO], secret_input: Optional[Callable[[str], str]]) -> int:
    if stdout is None:
        _replace_unencodable(sys.stdout)
    if stderr is None:
        _replace_unencodable(sys.stderr)
    ctx = _Context(stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr, secret_input)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0) if isinstance(stop.code, int) else EXIT_USAGE
    if not getattr(args, "handler", None):
        parser.print_help(ctx.stdout)
        return EXIT_USAGE
    if getattr(args, "ids_only", False):
        ctx.ids_only()
    with _InterruptGuard(ctx.stderr):
        try:
            return args.handler(args, ctx)
        except KeyboardInterrupt as stop:
            ctx.reporter.error("\n" + messages.text(ctx.interrupted_key))
            queued = stop.tasks if isinstance(stop, RunInterrupted) else None
            if queued:
                ctx.say("run.no_wait_hint", ids=" ".join(t.id for t in queued))
                ctx.print_ids(queued)
            return EXIT_INTERRUPTED
        except NalveraError as error:
            ctx.error(str(error))
            queued = getattr(error, "tasks", None)
            if queued:
                ctx.say("run.stopped_queued", count=len(queued))
                ctx.say("run.no_wait_hint", ids=" ".join(t.id for t in queued))
                ctx.print_ids(queued)       # they run and cost credits: a script keeps them
            return exit_status(error)


# ── commands ──────────────────────────────────────────────────────────────────
def cmd_login(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera login``: read the email and the key (hidden), then
    ``nalvera.login(email, api_key, remember=True)`` checks them with the
    platform and stores them. The key is never a command-line argument."""
    asker = StreamReporter(output=ctx.stdout, input=ctx.stdin, errors=ctx.stderr,
                           interactive=True, secret_input=ctx.secret_input)
    email = (args.email or "").strip() or (asker.ask(messages.text("login.email_prompt"))
                                           or "").strip()
    if not email:
        raise NoCredentials()
    key = auth.make_key(asker.ask(messages.text("login.key_prompt"), hidden=True) or "",
                        "prompt")
    ctx.say("login.checking")
    _session.login(email, key.reveal(), remember=True, reporter=ctx.reporter)
    if settings.get("NALVERA_API_KEY") or settings.get("NALVERA_EMAIL"):
        ctx.say("login.env_overrides")
    return EXIT_OK


def cmd_logout(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera logout``: ``nalvera.logout(forget=True)``. No request."""
    ctx.say("logout.done" if _session.logout(forget=True) else "logout.nothing")
    return EXIT_OK


def cmd_whoami(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera whoami``: account, plan, key, permissions, end date, free
    tasks, budgets and whether uploads are encrypted."""
    with ctx.client() as client:
        me = client.me()
        budgets = client.budgets()
        expires, days = client.key_expiry()
        encryption = client.encryption()
    key = me.get("key") or {}
    ctx.say("whoami.account", email=me.get("email"), plan=me.get("plan_label") or me.get("plan"))
    ctx.say("whoami.key", prefix=key.get("prefix"), name=key.get("name"))
    ctx.say("whoami.expires", expires=_date(expires))
    _warn_expiry(ctx, days)
    scopes = [messages.text(f"scope.{s}") for s in key.get("scopes") or []
              if messages.has(f"scope.{s}")]
    ctx.say("whoami.scopes", scopes=messages.text("list.separator").join(scopes))
    free = budgets.get("free") or {}
    if free.get("period_active"):
        ctx.say("whoami.free_period")
    else:
        ctx.say("whoami.free", count=int(free.get("free_tasks_left") or 0))
    personal = budgets.get("personal") or {}
    ctx.say("whoami.personal", available=_credits(personal.get("available_credits")),
            monthly=_credits(personal.get("monthly_left")), topup=_credits(personal.get("topup")))
    for lab in budgets.get("labs") or []:
        ctx.say("whoami.lab", name=lab.get("name"), budget=f"lab:{lab.get('id')}",
                available=_credits(lab.get("balance")), status=lab.get("member_status"))
    if budgets.get("labs"):
        ctx.say("whoami.suggested", budget=budgets.get("suggested"))
    if encryption["required"]:
        ctx.say("whoami.encryption_required")
    elif not encryption["available"]:
        ctx.say("whoami.encryption_unavailable")
    elif encryption["default"]:
        ctx.say("whoami.encryption_on")
    else:
        ctx.say("whoami.encryption_off")
    if me.get("terms_reaccept_required"):
        ctx.say("whoami.terms")
    return EXIT_OK


def cmd_limits(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera limits``: the numbers of the key's limits."""
    with ctx.client() as client:
        limits = client.limits()
    for name in ("requests_per_minute", "task_submits_per_minute", "max_active_tasks",
                 "max_uploads_in_progress", "max_file_bytes", "chunk_bytes", "min_poll_seconds"):
        if name not in limits:
            continue
        value = limits[name]
        if name.endswith("_bytes"):
            value = int(value) // (1024 * 1024)
        ctx.say(f"limits.{name}", value=value)
    return EXIT_OK


def cmd_models(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera models [--model ID]``: models and tasks; with ``--model``,
    the settings a generation accepts."""
    with ctx.client() as client:
        listed = client.models()
    if args.model:
        listed = [m for m in listed if m.get("id") == args.model]
        if not listed:
            raise ConfigError(messages.text("error.unknown_model", model=args.model),
                              code="unknown_model")
    separator = messages.text("list.separator")
    for model in listed:
        modality: List[str] = []
        for task in model.get("tasks") or [{}]:      # each task's own, else the model's
            modality += [m for m in display.task_modality(model, task) if m not in modality]
        ctx.say("models.model", id=model.get("id"), name=model.get("name"),
                version=model.get("version") or "", modality=separator.join(modality) or "-")
        if model.get("generative"):
            ctx.say("models.generative_mark")
        for task in model.get("tasks") or []:
            mark = messages.text("models.default_mark") if task.get("default") else ""
            price = display.price_text(task.get("price_eur_per_mvoxel"))
            if price is None:
                ctx.say("models.task_no_price", id=task.get("id"), name=task.get("name"),
                        mark=mark)
            else:
                ctx.say("models.task", id=task.get("id"), name=task.get("name"), mark=mark,
                        price=price)
            if task.get("needs_text"):
                ctx.say("models.needs_text")
            if args.model and task.get("generation"):
                ctx.say("models.generation_intro")
                for setting, value in generate_mod.describe(task["generation"]):
                    ctx.say("models.generation_item", setting=setting, value=value)
    return EXIT_OK


def cmd_estimate(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera estimate``: what a run would cost, without uploading."""
    with ctx.client() as client:
        found = client.estimate_run(args.paths, model=args.model, task=args.task,
                                    force=args.force)
    for scan in found.scans:
        ctx.say("estimate.scan", name=scan.original_name, credits=f"{scan.credits:.2f}",
                shape=" x ".join(str(n) for n in scan.shape))
    if len(found.scans) > 1:                  # one scan: its line says what it costs
        ctx.say("estimate.total", count=len(found.scans), credits=f"{found.credits:.2f}")
    if found.free_count:
        ctx.say("estimate.free", free=found.free_count, paid=f"{found.paid_credits:.2f}")
    for option in found.budgets:
        mark = messages.text("budget.mark_suggested") if option.get("budget") == found.suggested \
            else ""
        ctx.say("estimate.budget", budget=option.get("budget"), name=option.get("name"),
                available=_credits(option.get("available_credits")), mark=mark)
    return EXIT_FAILED if found.failed else EXIT_OK


def cmd_run(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera run``: anonymize, upload and queue, then (unless
    ``--no-wait``) wait and download the results to ``--out``."""
    ctx.interrupted_key = "run.interrupted"
    if not args.no_wait:
        check_output_folder(args.out)
    with ctx.client() as client:
        tasks = client.run(args.paths, model=args.model, task=args.task, budget=args.budget,
                           label=args.label, text=args.text, yes=args.yes, encrypt=args.encrypt,
                           max_credits=args.max_credits, force=args.force)
        status = _run_status(tasks)
        ctx.print_ids(tasks)
        if args.no_wait or not tasks:
            if tasks:
                ctx.say("run.no_wait_hint", ids=" ".join(t.id for t in tasks))
            return status
        ctx.interrupted_key = "wait.interrupted"
        done = client.wait(tasks)
        saved = client.download(done, out=args.out)
    if status != EXIT_OK:
        return status
    return EXIT_FAILED if saved.skipped else EXIT_OK


def _run_status(result: RunResult) -> int:
    if result.failed:
        return EXIT_FAILED
    if result.not_queued:
        return EXIT_CREDITS
    return EXIT_OK


def cmd_generate(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera generate``: queue a generation with ``--param`` settings,
    then (unless ``--no-wait``) wait and download it. ``--estimate`` shows
    the cost per budget and queues nothing."""
    ctx.interrupted_key = "run.interrupted"
    params = generate_mod.parse_params(args.param)
    if args.estimate:
        return _estimate_generation(args, params, ctx)
    if not args.no_wait:
        check_output_folder(args.out)
    with ctx.client() as client:
        task = client.generate(args.model, args.task, params, budget=args.budget,
                               label=args.label, yes=args.yes, max_credits=args.max_credits,
                               force=args.force)
        ctx.print_ids([task])
        if args.no_wait:
            ctx.say("run.no_wait_hint", ids=task.id)
            return EXIT_OK
        ctx.interrupted_key = "wait.interrupted"
        done = client.wait([task])
        saved = client.download(done, out=args.out)
    return EXIT_FAILED if saved.skipped else EXIT_OK


def _estimate_generation(args: argparse.Namespace, params: Dict[str, Any],
                         ctx: _Context) -> int:
    """``nalvera generate --estimate``: what the generation would cost, from
    each budget, and whether a free task pays for it. Needs no budget."""
    with ctx.client() as client:
        found = client.estimate(args.model, args.task, params=params)
    credits = float(found.get("credits") or 0.0)
    ctx.say("estimate.total", count=1, credits=f"{credits:.2f}")
    if (found.get("free") or {}).get("available"):
        ctx.say("estimate.free_one")
    suggested = found.get("suggested")
    for option in found.get("budgets") or []:
        mark = messages.text("budget.mark_suggested") if option.get("budget") == suggested \
            else ""
        ctx.say("estimate.budget", budget=option.get("budget"), name=option.get("name"),
                available=_credits(option.get("available_credits")), mark=mark)
    return EXIT_OK


def cmd_status(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera status``: one line per task (the ones given, or all waiting
    or running; ``--all`` adds finished tasks, newest first, with their ids);
    ``--watch`` keeps following them."""
    ctx.interrupted_key = "wait.interrupted"
    listing = args.all and not args.ids
    with ctx.client() as client:
        if listing:
            tasks = client.tasks(state="all", limit=args.limit)
        else:
            tasks = _tasks_or_active(client, args.ids)
        if not tasks:
            ctx.say("status.none_at_all" if listing else "status.none")
            return EXIT_OK
        if args.watch:
            done = client.wait(tasks)
            return EXIT_OK if all(t.state == "completed" for t in done) else EXIT_FAILED
        for task in tasks:
            if listing:
                ctx.say("status.listed", task_id=task.id, line=client._wait_line(task))
            else:
                ctx.say("wait.line", **client._wait_fields(task))
    if listing:
        return EXIT_OK              # a listing: a task that failed long ago is no failure now
    return EXIT_FAILED if any(t.state == "failed" for t in tasks) else EXIT_OK


def cmd_wait(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera wait``: wait until the tasks finish."""
    ctx.interrupted_key = "wait.interrupted"
    with ctx.client() as client:
        tasks = _tasks_or_active(client, args.ids)
        done = client.wait(tasks)
    return EXIT_OK if all(t.state == "completed" for t in done) else EXIT_FAILED


def cmd_download(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera download``: save finished tasks' results under the original
    names. Without task ids: every completed task of the record on this
    computer whose result is not saved yet (``nalvera.download()``)."""
    with ctx.client() as client:
        saved = client.download(args.ids or None, out=args.out, part=args.part)
    return EXIT_FAILED if args.ids and saved.skipped else EXIT_OK


def cmd_cancel(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera cancel``: cancel each task given; one refusal does not stop
    the others."""
    status = EXIT_OK
    with ctx.client() as client:
        for task_id in args.ids:
            try:
                task = client.cancel(task_id)
            except NalveraError as error:
                ctx.error(str(error))
                status = max(status, exit_status(error))
                continue
            ctx.say("cancel.done", task_id=task.id, state=client._wait_fields(task)["state"])
    return status


def cmd_scans(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera scans``: the record on this computer (code, original name,
    label, tasks, result files) from ``nalvera.scans()``. No request."""
    rows = _session.scans()
    path = rows.path
    if not rows:
        ctx.say("scans.none", path=str(path))
        return EXIT_OK
    ctx.say("scans.header", path=str(path))
    grouped: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row.get("code") or ""), []).append(row)
    for code, group in grouped.items():
        first = group[0]
        name = first.get("original_name") or ""
        if first.get("series"):
            name = messages.text("run.series_name", name=name, series=first.get("series"))
        label = first.get("label")
        ctx.say("scans.scan", code=code, name=name,
                label=messages.text("scans.label", label=label) if label else "")
        seen_tasks, seen_results = set(), set()
        for row in group:
            task_id = row.get("task_id")
            if task_id and task_id not in seen_tasks:
                seen_tasks.add(task_id)
                ctx.say("scans.task", task_id=task_id, model=row.get("model") or "",
                        task=row.get("task") or "")
            result = row.get("result_path")
            if result and result not in seen_results:
                seen_results.add(result)
                ctx.say("scans.result", path=result)
    return EXIT_OK


def cmd_anonymize(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera anonymize PATH... --out DIR``: ``nalvera.save_anonymized``
    writes the anonymized copies, exactly the files a run would upload, into
    DIR, which then holds the copies only; the record that links each code
    to its original stays in the record on this computer. No key, no
    request."""
    copies = save_anonymized(args.paths, args.out, mapping=default_record_path(),
                             reporter=ctx.reporter)
    return EXIT_FAILED if copies.failed else EXIT_OK


def cmd_mcp(args: argparse.Namespace, ctx: _Context) -> int:
    """``nalvera mcp``: the local MCP server for an AI assistant (the same as
    ``python -m nalvera.mcp``), speaking the protocol on standard input and
    output until standard input closes. Needs the ``mcp`` extra; without it,
    one line on standard error and status 2."""
    from .mcp import __main__ as mcp_main
    with messages.using(messages.PYTHON):
        return mcp_main.main(error=ctx.reporter.error)


# ── helpers ───────────────────────────────────────────────────────────────────
def _tasks_or_active(client: Client, ids: List[str]) -> List[Task]:
    if ids:
        return [client.task(task_id) for task_id in ids]
    return client.tasks(state="active", limit=100)


def _warn_expiry(ctx: _Context, days: Optional[int]) -> None:
    """D5: a key that ends within two weeks is pointed out."""
    from .client import KEY_WARN_DAYS
    if days is not None and days <= KEY_WARN_DAYS:
        ctx.say("whoami.expires_soon", days=max(0, days))


def _date(value: Any) -> str:
    return str(value)[:10] if value else "-"


def _credits(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
