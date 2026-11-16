"""prepare(): the interface the rest of the package calls."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import os
import socket
import stat
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize_helpers as h  # noqa: E402
import nalvera.anonymize as anonymize  # noqa: E402
from nalvera.anonymize import (  # noqa: E402
    AnonymizeError,
    Mapping,
    PreparedScan,
    dicom,
    errors,
    mapping,
    names,
    nifti,
    prepare,
    verify,
)

MODULES = (anonymize, dicom, errors, mapping, names, nifti, verify)


@pytest.fixture
def scan_file(tmp_path):
    data = np.random.default_rng(1).integers(0, 4000, size=(10, 9, 5)).astype(np.int16)
    payload, _ = h.nifti_bytes(data, slope=1.0, inter=-1024.0,
                               fields={"descrip": b"Hendrickx^Marieke"})
    path = tmp_path / "Hendrickx_Marieke.nii.gz"
    path.write_bytes(h.gzip_bytes(payload, name=b"Hendrickx.nii"))
    return path


def _sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_interface():
    assert set(anonymize.__all__) == {"prepare", "inspect", "PreparedScan", "PreparedScans",
                                      "DroppedVolume", "Inspection", "FoundScan", "Mapping",
                                      "AnonymizeError"}
    fields = [f.name for f in dataclasses.fields(PreparedScan)]
    assert fields[:11] == ["code", "filename", "path", "size", "sha256", "shape",
                           "original_path", "original_name", "source", "series", "label"]
    parameters = inspect.signature(prepare).parameters
    assert list(parameters) == ["path", "workdir", "label", "max_bytes", "avoid", "check"]
    assert parameters["workdir"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["label"].default is None and parameters["max_bytes"].default is None
    assert parameters["avoid"].default is None and parameters["check"].default is None
    parameters = inspect.signature(anonymize.inspect).parameters
    assert list(parameters) == ["path", "convert", "workdir"]
    assert parameters["convert"].default is False and parameters["workdir"].default is None
    assert issubclass(anonymize.PreparedScans, list)
    for method in ("record_scan", "record_task", "record_result", "find", "rows"):
        assert callable(getattr(Mapping, method))


def test_prepared_scan(scan_file, tmp_path):
    work = tmp_path / "work"
    [scan] = prepare(scan_file, workdir=work, label="P012 baseline")
    assert names.is_code(scan.code)
    assert scan.filename == scan.code + ".nii.gz"
    assert scan.path == os.path.join(str(work), scan.filename)
    assert stat.S_IMODE(os.stat(scan.path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(work).st_mode) == 0o700
    assert scan.size == os.path.getsize(scan.path)
    assert scan.sha256 == _sha256(scan.path)
    assert scan.shape == (10, 9, 5)
    assert scan.original_path == str(scan_file)
    assert scan.original_name == "Hendrickx_Marieke.nii.gz"
    assert scan.source == "nifti"
    assert scan.series is None
    assert scan.label == "P012 baseline"
    assert scan.sha256_original == _sha256(scan_file)
    assert os.listdir(work) == [scan.filename]


def test_relative_paths(scan_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    [scan] = prepare(scan_file.name, workdir="work")
    assert scan.original_path == str(scan_file)
    assert scan.path == os.path.join(str(tmp_path), "work", scan.filename)


def test_every_call_gets_new_codes(scan_file, tmp_path):
    work = tmp_path / "work"
    codes = {prepare(scan_file, workdir=work)[0].code for _ in range(5)}
    assert len(codes) == 5
    assert len(os.listdir(work)) == 5


def test_invalid_label_writes_nothing(scan_file, tmp_path):
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(scan_file, workdir=work, label="P012/../x")
    assert caught.value.code == "invalid_label"
    assert not work.exists()


def test_not_found(tmp_path):
    with pytest.raises(AnonymizeError) as caught:
        prepare(tmp_path / "absent.nii.gz", workdir=tmp_path / "work")
    assert caught.value.code == "not_found"


def test_too_large(scan_file, tmp_path):
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(scan_file, workdir=work, max_bytes=100)
    assert caught.value.code == "too_large"
    assert os.listdir(work) == []


def test_max_bytes_that_fits(scan_file, tmp_path):
    [scan] = prepare(scan_file, workdir=tmp_path / "work", max_bytes=10 ** 6)
    assert scan.size <= 10 ** 6


def test_failed_check_leaves_nothing(scan_file, tmp_path, monkeypatch):
    real = verify._check_header

    def broken(block, expected):
        real(block, expected)
        raise verify._Failed("forced")

    monkeypatch.setattr(verify, "_check_header", broken)
    work = tmp_path / "work"
    with pytest.raises(AnonymizeError) as caught:
        prepare(scan_file, workdir=work)
    assert caught.value.code == "verification_failed"
    assert os.listdir(work) == []


def test_interrupt_leaves_nothing(scan_file, tmp_path, monkeypatch):
    def interrupted(path, info):
        raise KeyboardInterrupt

    monkeypatch.setattr(anonymize, "verify_clean", interrupted)
    work = tmp_path / "work"
    with pytest.raises(KeyboardInterrupt):
        prepare(scan_file, workdir=work)
    assert os.listdir(work) == []


def test_write_error_passes_through_and_cleans_up(scan_file, tmp_path, monkeypatch):
    real = nifti._create

    def full_disk(path):
        handle = real(path)

        class Full:
            def __getattr__(self, name):
                return getattr(handle, name)

            def write(self, data):
                raise OSError(28, "No space left on device")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                handle.close()

        return Full()

    monkeypatch.setattr(nifti, "_create", full_disk)
    work = tmp_path / "work"
    with pytest.raises(OSError) as caught:
        prepare(scan_file, workdir=work)
    assert not isinstance(caught.value, AnonymizeError)
    assert os.listdir(work) == []


def test_no_network(scan_file, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    [scan] = prepare(scan_file, workdir=tmp_path / "work")
    folder = tmp_path / "dicom"
    folder.mkdir()
    h.write_dicom_series(str(folder), series_number=2, plants=h.DICOM_PLANTS)
    if dicom.find_dcm2niix():
        assert len(prepare(folder, workdir=tmp_path / "work")) == 1
    Mapping(str(tmp_path / "m.csv")).record_scan(scan)


def test_no_network_module_imported():
    banned = {"socket", "http", "urllib", "requests", "ssl"}
    for module in MODULES:
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not imported & banned, (module.__name__, imported)


def test_python310_syntax():
    for module in MODULES:
        source = inspect.getsource(module)
        ast.parse(source, feature_version=(3, 10))


def test_public_functions_have_docstrings():
    for module in MODULES:
        assert module.__doc__, module.__name__
        for name, value in vars(module).items():
            if name.startswith("_") or getattr(value, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(value) or inspect.isclass(value):
                assert value.__doc__, f"{module.__name__}.{name}"
            if inspect.isclass(value):
                for attribute, member in vars(value).items():
                    if inspect.isfunction(member) and not attribute.startswith("_"):
                        assert member.__doc__, f"{module.__name__}.{name}.{attribute}"


def test_error_text_comes_from_the_package_messages(monkeypatch):
    import types

    fake = types.ModuleType("nalvera.messages")
    fake.MESSAGES = {"anonymize.not_found": "replaced text"}
    monkeypatch.setitem(sys.modules, "nalvera.messages", fake)
    assert str(AnonymizeError("not_found")) == "replaced text"
    # The technical detail stays on the exception, out of the sentence.
    error = AnonymizeError("unreadable", "why")
    assert error.detail == "why" and "why" not in str(error)
    monkeypatch.setitem(sys.modules, "nalvera.messages", None)
    assert str(AnonymizeError("not_found"))
