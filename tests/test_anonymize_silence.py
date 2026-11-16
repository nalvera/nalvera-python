"""Silence, a computer without the converter, and what the module depends
on.

An MCP server speaks its protocol on standard input and output. This
module never prints, logs or warns, never reads standard input, and the
converter it starts gets neither stream: its output is captured.
NIfTI input works with no dcm2niix importable or on PATH; only DICOM
input needs it (``dicom_tool_missing``).
The documentation page and the examples are generated from the docstrings,
so each public function names every argument it takes.
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import inspect as pyinspect
import logging
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize_helpers as h  # noqa: E402
import nalvera.anonymize as anonymize  # noqa: E402
from nalvera.anonymize import (  # noqa: E402
    AnonymizeError,
    dicom,
    errors,
    mapping,
    names,
    nifti,
    prepare,
    verify,
)

MODULES = (anonymize, dicom, errors, mapping, names, nifti, verify)
PACKAGE_SRC = os.path.dirname(os.path.dirname(os.path.abspath(anonymize.__file__)))

needs_dcm2niix = pytest.mark.skipif(
    os.environ.get("NALVERA_DCM2NIIX") is None and dicom.find_dcm2niix() is None,
    reason="dcm2niix is not installed (pip install dcm2niix)")


class _NoStdin:
    """Stands in for sys.stdin: any use fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"standard input was used ({name})")


@contextlib.contextmanager
def _assert_silent(capfd, monkeypatch):
    """Fails when anything reaches file descriptor 1 or 2 (this process or a
    child), a warning is issued, a log record is made, or stdin is used."""
    capfd.readouterr()
    monkeypatch.setattr(sys, "stdin", _NoStdin())
    records = []

    class Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    root = logging.getLogger()
    handler, level = Catch(logging.DEBUG), root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""
    assert [str(w.message) for w in caught] == []
    assert [r.getMessage() for r in records] == []


def _nifti_inputs(folder) -> list:
    """Paths of NIfTI files that take each branch of the writer."""
    exact = np.diag([0.5, 0.5, 2.0, 1.0])
    data = np.random.default_rng(2).integers(0, 3000, size=(8, 7, 5)).astype(np.int16)
    made = []
    for name, args in (("plain.nii.gz", dict(slope=1.0, inter=-1024.0)),
                       ("v2.nii.gz", dict(version=2, qform=exact, sform=exact)),
                       ("nan.nii.gz", dict(slope=float("nan"))),
                       ("big.nii.gz", dict(endian=">"))):
        payload, _ = h.nifti_bytes(data, **args)
        path = os.path.join(str(folder), name)
        with open(path, "wb") as fh:
            fh.write(h.gzip_bytes(payload, name=b"Hendrickx.nii"))
        made.append(path)
    return made


# ---- silence -------------------------------------------------------------------------

def test_nifti_prepare_and_inspect_are_silent(tmp_path, capfd, monkeypatch):
    sources = tmp_path / "in"
    sources.mkdir()
    paths = _nifti_inputs(sources)
    refused = sources / "broken.nii.gz"
    refused.write_bytes(h.gzip_bytes(b"Hendrickx^Marieke\n" * 40))
    with _assert_silent(capfd, monkeypatch):
        for path in paths:
            prepare(path, workdir=str(tmp_path / "work"))
            anonymize.inspect(path)
        with pytest.raises(AnonymizeError):
            prepare(str(refused), workdir=str(tmp_path / "work"))
        anonymize.inspect(str(sources))
        mapping.Mapping(str(tmp_path / "m.csv")).rows()


@needs_dcm2niix
def test_dicom_prepare_and_inspect_are_silent(tmp_path, capfd, monkeypatch):
    """The real converter prints its version, every series and warnings
    quoting DICOM values: none of it may reach this process's streams."""
    folder = tmp_path / "study"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS,
                         positions=h.tilted_positions(6), orientation=h.tilted_orientation(),
                         extra={"GantryDetectorTilt": 15.0})
    h.write_secondary_capture(str(folder), plants=h.DICOM_PLANTS, text="Hendrickx^Marieke")
    screenshot_only = tmp_path / "screenshot"
    screenshot_only.mkdir()
    h.write_secondary_capture(str(screenshot_only), plants=h.DICOM_PLANTS, text="Hendrickx")
    with _assert_silent(capfd, monkeypatch):
        scans = prepare(str(folder), workdir=str(tmp_path / "work"))
        anonymize.inspect(str(folder), convert=True, workdir=str(tmp_path / "inspect"))
        with pytest.raises(AnonymizeError):
            prepare(str(screenshot_only), workdir=str(tmp_path / "work"))
    assert len(scans) == 1


@contextlib.contextmanager
def _pipe_on_stdin():
    """File descriptor 0 is a pipe for the duration (as for an MCP server),
    so a child that inherited it would see a pipe, not the null device."""
    saved = os.dup(0)
    read_end, write_end = os.pipe()
    os.dup2(read_end, 0)
    try:
        yield
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read_end, write_end):
            os.close(fd)


def test_converter_gets_no_terminal_streams(tmp_path, capfd, monkeypatch):
    """The stand-in prints patient details on stdout and stderr and records
    what its streams are: stdin is the null device (never this process's
    input, here a pipe), stdout and stderr are pipes this module reads;
    nothing reaches the terminal."""
    folder = tmp_path / "series"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)
    log = h.install_standin(tmp_path, monkeypatch)
    with _assert_silent(capfd, monkeypatch), _pipe_on_stdin():
        prepare(str(folder), workdir=str(tmp_path / "work"))
    assert log()["streams"] == {"stdin": "null", "stdout": "pipe", "stderr": "pipe"}


def _calls_and_imports(module):
    tree = ast.parse(pyinspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            yield "call", node.func.id
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "sys":
            yield "sys", node.attr
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield "import", alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield "import", (node.module or "").split(".")[0]


def test_source_never_prints_reads_or_logs():
    for module in MODULES:
        for kind, name in _calls_and_imports(module):
            assert not (kind == "call" and name in ("print", "input", "breakpoint")), \
                (module.__name__, name)
            assert not (kind == "sys" and name in ("stdin", "stdout", "stderr", "__stdout__",
                                                   "__stderr__", "__stdin__")), \
                (module.__name__, name)
            assert not (kind == "import" and name in ("logging", "warnings", "getpass",
                                                      "builtins")), (module.__name__, name)


# ---- no converter on this computer ---------------------------------------------------

def test_nifti_works_without_the_converter(tmp_path, monkeypatch):
    h.hide_converter(tmp_path, monkeypatch)
    assert dicom.find_dcm2niix() is None
    sources = tmp_path / "in"
    sources.mkdir()
    paths = _nifti_inputs(sources)
    for path in paths:
        [scan] = prepare(path, workdir=str(tmp_path / "work"))
        assert scan.source == "nifti"
        assert anonymize.inspect(path).scans[0].error is None
    assert len(prepare(str(sources), workdir=str(tmp_path / "work"))) == len(paths)


def test_only_dicom_needs_the_converter(tmp_path, monkeypatch):
    folder = tmp_path / "series"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=3, plants=h.DICOM_PLANTS)
    h.hide_converter(tmp_path, monkeypatch)
    work = tmp_path / "work"
    for path in (folder, folder / sorted(os.listdir(folder))[0]):
        with pytest.raises(AnonymizeError) as caught:
            prepare(str(path), workdir=str(work))
        assert caught.value.code == "dicom_tool_missing"
        assert os.listdir(work) == []
    found = anonymize.inspect(str(folder))                  # no conversion asked: no converter
    assert found.kind == "dicom" and found.needs_conversion
    with pytest.raises(AnonymizeError) as caught:
        anonymize.inspect(str(folder), convert=True)
    assert caught.value.code == "dicom_tool_missing"


def test_a_broken_converter_package_does_not_break_nifti(tmp_path, monkeypatch):
    """A ``dcm2niix`` package that fails on import (no build for this
    platform) is treated as no converter."""
    fake = tmp_path / "site" / "dcm2niix"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("raise RuntimeError('no build for this platform')\n")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.delitem(sys.modules, "dcm2niix", raising=False)
    monkeypatch.delenv("NALVERA_DCM2NIIX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    assert dicom.find_dcm2niix() is None
    [path] = _nifti_inputs(tmp_path)[:1]
    assert len(prepare(path, workdir=str(tmp_path / "work"))) == 1


def test_importing_the_module_does_not_load_the_converter():
    code = (f"import sys; sys.path.insert(0, {PACKAGE_SRC!r}); import nalvera.anonymize; "
            "print('dcm2niix' in sys.modules)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120, check=True)
    assert done.stdout.split() == ["False"]


# ---- pydicom is no dependency ----------------------------------------------------------

def test_works_with_pydicom_absent(tmp_path):
    """pydicom leaves the dependencies: with it unimportable, a NIfTI file is
    still prepared. (nibabel imports pydicom when it is installed, for its
    own optional DICOM support, so its presence in sys.modules proves
    nothing.)"""
    payload, _ = h.nifti_bytes(np.zeros((5, 4, 3), np.int16))
    source = tmp_path / "scan.nii"
    source.write_bytes(payload)
    code = (f"import sys; sys.modules['pydicom'] = None; sys.path.insert(0, {PACKAGE_SRC!r}); "
            "from nalvera.anonymize import prepare; "
            f"print(prepare({str(source)!r}, workdir={str(tmp_path / 'work')!r})[0].shape)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "(5, 4, 3)"


def test_no_source_file_imports_pydicom():
    found = []
    for root, _dirs, files in os.walk(os.path.join(PACKAGE_SRC, "nalvera")):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                found += [(path, m) for m in modules if m.split(".")[0] == "pydicom"]
    assert found == []


# ---- docstrings name every argument ----------------------------------------------------

def _named(doc: str, name: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", doc) \
        is not None


def _parameters(function):
    return [p.name for p in pyinspect.signature(function).parameters.values()
            if p.name not in ("self", "cls")
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def test_docstrings_name_every_argument():
    missing = []
    for module in MODULES:
        for name, value in vars(module).items():
            if name.startswith("_") or getattr(value, "__module__", None) != module.__name__:
                continue
            if pyinspect.isfunction(value):
                missing += [f"{module.__name__}.{name}({p})" for p in _parameters(value)
                            if not _named(value.__doc__ or "", p)]
            elif pyinspect.isclass(value):
                for attribute, member in vars(value).items():
                    if pyinspect.isfunction(member) and not attribute.startswith("_"):
                        missing += [f"{module.__name__}.{name}.{attribute}({p})"
                                    for p in _parameters(member)
                                    if not _named(member.__doc__ or "", p)]
    # Classes a caller builds or reads: every field or constructor argument.
    for name in anonymize.__all__:
        value = getattr(anonymize, name)
        if not pyinspect.isclass(value):
            continue
        doc = value.__doc__ or ""
        if dataclasses.is_dataclass(value):
            wanted = [f.name for f in dataclasses.fields(value)]
        elif "__init__" in vars(value):
            wanted = _parameters(value.__init__)
        else:
            wanted = []
        missing += [f"{name}({p})" for p in wanted if not _named(doc, p)]
    assert missing == []
