"""nalvera: run Nalvera.AI tasks on medical scans from Python.

Made for Jupyter notebooks and Python scripts::

    import nalvera

    nalvera.login("you@university.org", "nvk_...")   # checks the key with Nalvera.AI
    nalvera.models()                                 # a table in a notebook
    tasks = nalvera.run("ct_scan.nii.gz", model="totalsegmentator_ct", budget="personal")
    nalvera.wait(tasks)                              # queue position, time waited, time running
    paths = nalvera.download(tasks, out="results")

Scans are anonymized on this computer before any byte is uploaded; the
platform sees a neutral code (such as ``scan-7F3K2Q9A``) and results are
saved under the original names. A key typed into a notebook cell is saved
in the ``.ipynb`` file and in IPython's history file: ``nalvera.login()``
with no arguments asks for it instead. Running a notebook again from the
top queues, charges and downloads nothing twice, except a generation
without a label.

The functions: ``login``, ``logout``, ``me``, ``limits``, ``models``,
``budgets``, ``encryption``, ``key_expiry``, ``estimate_run``,
``estimate``, ``run``, ``run_in_background``, ``generate``, ``wait``,
``download``, ``tasks``, ``task``, ``cancel``, ``scans``,
``save_anonymized``; ``help(nalvera.run)`` describes each argument.
``nalvera.Client`` is for a script with several accounts. Without a
``login``, the environment (``NALVERA_EMAIL``, ``NALVERA_API_KEY``) or a
stored login is used; ``nalvera.SETTINGS`` lists every setting. Errors are
subclasses of ``nalvera.NalveraError`` with a ``code``.
Documentation: https://nalvera.ai/python-api

Importing the package makes no network request.
"""
from typing import Any, List

from ._version import __version__
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
    NoFreeTasks,
    NotCancellable,
    NotFound,
    NotReady,
    PermissionDenied,
    PreparationError,
    RateLimited,
    RunInterrupted,
    ScanRejected,
    ServerError,
    ServiceUnavailable,
    TooLarge,
    UnexpectedResponse,
    WaitInterrupted,
)
from .report import (
    Event,
    ListReporter,
    NotebookReporter,
    NullReporter,
    Reporter,
    StreamReporter,
)
from .settings import SETTINGS, Setting

# Loaded on first use (PEP 562), so `import nalvera` stays light and loads no
# HTTP library.
_LAZY = {"Client", "Task", "RunResult", "Downloads", "ScanFailure", "SkippedScan",
         "RunEstimate", "ScanEstimate", "BackgroundRun", "SavedCopy", "SavedCopies"}
#: The module functions (``nalvera._session``), loaded on first use too.
FUNCTIONS = ("login", "logout", "me", "limits", "key_expiry", "encryption", "models",
             "budgets", "estimate", "estimate_run", "run", "run_in_background", "generate",
             "wait", "download", "tasks", "task", "cancel", "scans", "save_anonymized")


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import client
        return getattr(client, name)
    if name in FUNCTIONS:
        from . import _session
        return _session.FUNCTIONS[name]
    raise AttributeError(f"module 'nalvera' has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "__version__", "SETTINGS", "Setting",
    *FUNCTIONS,
    "Client", "Task", "RunResult", "Downloads", "ScanFailure", "SkippedScan",
    "RunEstimate", "ScanEstimate", "BackgroundRun", "SavedCopy", "SavedCopies",
    "Reporter", "StreamReporter", "NotebookReporter", "NullReporter", "ListReporter", "Event",
    "NalveraError", "NoCredentials", "ConfigError", "BudgetChoiceRequired", "PreparationError",
    "NetworkError", "UnexpectedResponse", "AuthenticationError", "PermissionDenied",
    "ApiDisabled", "InsufficientCredits", "NoFreeTasks", "CreditCapExceeded",
    "ConfirmationRequired", "EncryptionError", "InvalidRequest", "ScanRejected", "TooLarge",
    "NotFound", "NotReady", "NotCancellable", "RateLimited", "ServiceUnavailable",
    "ServerError", "ClientTooOld", "RunInterrupted", "WaitInterrupted",
]
