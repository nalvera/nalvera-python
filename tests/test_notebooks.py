"""The notebooks of examples/notebooks/ in a real Jupyter kernel: each
runs against the fake server, shows tables, no terminal escape
code, no key and no email but the account's; a cell that needs a budget
choice, with input refused as nbconvert refuses it, raises
BudgetChoiceRequired naming the argument. Skipped without ipykernel and
nbclient (the dev extra)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient", reason="needs nbclient (the dev extra)")
pytest.importorskip("ipykernel", reason="needs ipykernel (the dev extra)")
nbformat = pytest.importorskip("nbformat", reason="needs nbformat (installed with nbclient)")

import examples_helpers as helpers  # noqa: E402
from test_examples import NOTEBOOK_RUNS, Scene, prepare_notebook_scene  # noqa: E402

SRC = helpers.ROOT / "src"
EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@pytest.fixture
def kernel_scene(server, tmp_path, monkeypatch):
    """A Scene whose environment the kernel inherits."""
    scene = Scene(server, tmp_path)
    for name in list(os.environ):
        if name.startswith("NALVERA_"):
            monkeypatch.delenv(name)
    for name, value in scene.env.items():
        monkeypatch.setenv(name, value)
    return scene


def execute(path: Path, scene: Scene, *, allow_errors: bool = False):
    """``path`` executed in a fresh kernel in the scene's folder, signed in
    as the fake account; the login cell shows its placeholder again."""
    notebook = nbformat.read(str(path), as_version=4)
    login = None
    for cell in notebook.cells:
        if cell.cell_type == "code" and helpers.NOTEBOOK_LOGIN in cell.source:
            login = cell
            cell.source = cell.source.replace(
                helpers.NOTEBOOK_LOGIN,
                f"nalvera.login({scene.server.email!r}, {scene.server.key!r})")
    client = nbclient.NotebookClient(notebook, timeout=240, kernel_name="python3",
                                     allow_errors=allow_errors,
                                     resources={"metadata": {"path": str(scene.dir)}})
    client.execute()
    if login is not None:
        login.source = login.source.replace(
            f"nalvera.login({scene.server.email!r}, {scene.server.key!r})", helpers.NOTEBOOK_LOGIN)
    return notebook


def _outputs(notebook) -> str:
    return json.dumps([cell.get("outputs", []) for cell in notebook.cells])


@pytest.mark.parametrize("name", sorted(NOTEBOOK_RUNS))
def test_each_notebook_runs_in_a_kernel_and_shows_tables(kernel_scene, name):
    prepare_notebook_scene(kernel_scene, name)
    notebook = execute(helpers.NOTEBOOKS / name, kernel_scene)
    shown = _outputs(notebook)
    assert "text/html" in shown and "<table>" in shown
    assert "\\u001b" not in shown and "\x1b" not in shown
    assert kernel_scene.server.key not in shown and kernel_scene.server.key[12:] not in shown
    assert {e.lower() for e in EMAIL_RX.findall(shown)} <= {kernel_scene.server.email.lower()}
    tasks, folder = NOTEBOOK_RUNS[name]
    assert len(kernel_scene.api_tasks()) == tasks and len(kernel_scene.files(folder)) == tasks
    errors = [o for cell in notebook.cells for o in cell.get("outputs", [])
              if o.get("output_type") == "error"]
    assert errors == []


def _cell(notebook, start: str):
    [cell] = [c for c in notebook.cells if c.cell_type == "code" and c.source.startswith(start)]
    return cell


def _kinds(cell):
    return [o["output_type"] for o in cell.outputs]


def _text(output) -> str:
    if output["output_type"] == "stream":
        return output["text"]
    return output["data"].get("text/plain", "")


def _each_idea_once(cell) -> None:
    """No line of a cell's output appears twice, and no line of its display
    is also in its returned table."""
    lines = [line.strip() for o in cell.outputs for line in _text(o).splitlines()
             if line.strip()]
    assert len(lines) == len(set(lines)), lines
    shown = [_text(o) for o in cell.outputs if o["output_type"] == "display_data"]
    tables = [_text(o) for o in cell.outputs if o["output_type"] == "execute_result"]
    for line in "\n".join(shown).splitlines():
        assert not any(line and line in table for table in tables), line


def test_the_quickstart_shows_one_display_and_one_result_per_call(kernel_scene):
    from nalvera import auth, display, messages
    server = kernel_scene.server
    prepare_notebook_scene(kernel_scene, "quickstart.ipynb")
    server.script = [("queued", 2), ("queued", 1), ("processing", None), ("completed", None)]
    notebook = execute(helpers.NOTEBOOKS / "quickstart.ipynb", kernel_scene)

    login = _cell(notebook, "import nalvera")
    assert _kinds(login) == ["stream"]                  # one confirmation, no table after it
    [line] = _text(login.outputs[0]).splitlines()
    assert server.key[:12] in line and server.email.lower() in line.lower()
    assert server.key_expires_at[:10] in line and server.url in line

    assert _kinds(_cell(notebook, "nalvera.models()")) == ["execute_result"]

    estimate = _cell(notebook, "SCAN = ")
    assert _kinds(estimate) == ["display_data", "execute_result"]
    assert _text(estimate.outputs[0]) == messages.text("run.pixel_notice")

    run = _cell(notebook, "tasks = nalvera.run(")
    assert _kinds(run) == ["display_data", "execute_result"]
    record = display.path_text(auth.record_path(server.email, server.url,
                                                directory=kernel_scene.config), cut=False)
    assert _text(run.outputs[0]).splitlines() == [messages.text("run.pixel_notice"),
                                                  messages.text("run.record", path=record)]
    charge = f"{server.credits_per_task:.2f}"
    assert sum(_text(o).count(charge) for o in run.outputs) == 1

    wait = _cell(notebook, "finished = nalvera.wait(")
    assert _kinds(wait) == ["display_data", "execute_result"]
    assert wait.outputs[0]["data"]["text/plain"] == ""        # the progress is gone
    assert messages.text("wait.state.completed") in _text(wait.outputs[1])

    download = _cell(notebook, "paths = nalvera.download(")
    assert _kinds(download) == ["execute_result"]
    assert os.path.join("results", "ct_scan__totalsegmentator_ct__total.nii.gz") in \
        _text(download.outputs[0])

    for cell in (login, estimate, run, wait, download):
        _each_idea_once(cell)


def test_many_scans_show_one_display_and_one_result_per_call(kernel_scene):
    prepare_notebook_scene(kernel_scene, "many_scans_and_budgets.ipynb")
    notebook = execute(helpers.NOTEBOOKS / "many_scans_and_budgets.ipynb", kernel_scene)
    for start in ("FOLDER = ", "result = nalvera.run(", "finished = nalvera.wait("):
        cell = _cell(notebook, start)
        assert _kinds(cell) == ["display_data", "execute_result"], start
        _each_idea_once(cell)
    for start in ("nalvera.budgets()", "nalvera.download(", "nalvera.scans()"):
        cell = _cell(notebook, start)
        assert _kinds(cell) == ["execute_result"], start


def _run_cells(scene, sources, history=":memory:"):
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    client = nbclient.NotebookClient(notebook, timeout=240, kernel_name="python3",
                                     allow_errors=True, ipython_hist_file=history,
                                     resources={"metadata": {"path": str(scene.dir)}})
    client.execute()
    return notebook


INTERRUPT_WHILE_ANSWERING = '''import http.client
_answer = http.client.HTTPConnection.getresponse
def _interrupted(self, *args, **kwargs):
    # The person presses Interrupt while the platform is answering.
    http.client.HTTPConnection.getresponse = _answer
    raise KeyboardInterrupt
http.client.HTTPConnection.getresponse = _interrupted
'''


def test_a_verbose_traceback_in_a_kernel_holds_no_key(kernel_scene):
    """%xmode Verbose prints the variables of every frame: an interrupt while
    the platform answers, and an email the HTTP library cannot send, leave no
    frame that holds the key. The key comes from the
    environment, and no failing line of a cell names it or ``os.environ``
    (IPython 8 shows a cell's own variables, and the environment holds the
    key)."""
    server = kernel_scene.server
    notebook = _run_cells(kernel_scene, [
        "%xmode Verbose",
        "import os, nalvera\nnalvera.login()",
        INTERRUPT_WHILE_ANSWERING + "nalvera.models(refresh=True)",
        "os.environ['NALVERA_EMAIL'] = 'jan.kowalski@uniwersytet-łodz.example'",
        "nalvera.login()",
    ])
    shown = _outputs(notebook)
    errors = [o["ename"] for cell in notebook.cells for o in cell.outputs
              if o.get("output_type") == "error"]
    assert errors == ["KeyboardInterrupt", "ConfigError"]
    assert server.key not in shown and server.key[12:] not in shown
    assert server.calls("GET", "/api/v1/models")              # the request did go out


def _record_rows(scene):
    from nalvera import auth
    from nalvera.anonymize import Mapping
    path = auth.record_path(scene.server.email, scene.server.url, directory=scene.config)
    return Mapping(str(path)).rows() if path.exists() else []


@pytest.mark.parametrize("name", sorted(NOTEBOOK_RUNS))
def test_no_saved_output_pairs_a_platform_code_with_an_original_name(kernel_scene, name):
    """A notebook as shipped, run and saved, holds no table that links the
    platform's codes to the scans' original names."""
    prepare_notebook_scene(kernel_scene, name)
    notebook = execute(helpers.NOTEBOOKS / name, kernel_scene)
    shown = _outputs(notebook)
    rows = [r for r in _record_rows(kernel_scene) if r.get("code") and r.get("original_name")]
    if name != "generate_synthetic_ct.ipynb":
        assert rows                                    # there was something to pair
    paired = [(r["code"], r["original_name"]) for r in rows
              if r["code"] in shown and r["original_name"] in shown]
    assert paired == []


def test_a_budget_choice_without_input_raises_naming_the_argument(kernel_scene):
    kernel_scene.server.add_lab(12, "Imaging Lab", 100.0)
    kernel_scene.nifti("ct_scan.nii.gz")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(
        "import nalvera\n" + helpers.NOTEBOOK_LOGIN),
        nbformat.v4.new_code_cell('nalvera.run("ct_scan.nii.gz", model="totalsegmentator_ct")')]
    path = kernel_scene.root / "choice.ipynb"
    nbformat.write(notebook, str(path))
    executed = execute(path, kernel_scene, allow_errors=True)
    [error] = [o for o in executed.cells[1].outputs if o.get("output_type") == "error"]
    assert error["ename"] == "BudgetChoiceRequired"
    assert 'budget=' in error["evalue"] and "--budget" not in error["evalue"]
    assert not kernel_scene.server.calls("POST", "/api/v1/uploads")


RUN = 'nalvera.run("{scan}", model="{model}"{budget})'


def test_ten_likely_errors_read_as_instructions_in_a_kernel(kernel_scene, tmp_path):
    """The ten errors a notebook user meets most, raised in one kernel cell by
    cell (the fake server changes between cells): each is the expected
    exception, and its text names Python, never a command or a key."""
    from fake_server import make_key
    server = kernel_scene.server
    kernel_scene.nifti("ct_scan.nii.gz")
    (kernel_scene.dir / "notes.nii.gz").write_bytes(b"not a scan")
    other_key = make_key()
    login = f"nalvera.login({server.email!r}, {server.key!r})"
    scan = RUN.format(scan="ct_scan.nii.gz", model="totalsegmentator_ct",
                      budget=', budget="personal"')

    def state(**changes):
        def apply():
            for name, value in changes.items():
                setattr(server, name, value)
        return apply

    def add_lab():
        server.add_lab(12, "Imaging Lab", 100.0)

    def tiny_files():
        server.limits["max_file_bytes"] = 100

    # (case, what the server does before the cell, code, expected exception)
    cases = [
        ("no key", None, 'import os\nos.environ.pop("NALVERA_API_KEY", None)\nnalvera.models()',
         "NoCredentials"),
        ("wrong key", None, f"nalvera.login({server.email!r}, {other_key!r})",
         "AuthenticationError"),
        ("signed in", None, login, None),
        ("API off", state(auth_refusal=(403, "api_disabled", {})),
         "nalvera.models(refresh=True)", "ApiDisabled"),
        ("no permission", state(auth_refusal=None, scopes=["tasks:read"]),
         "nalvera.me(refresh=True)\n" + scan, "PermissionDenied"),
        ("no credits", state(scopes=["tasks:read", "results:read", "tasks:write"],
                             personal_credits=0.0),
         "nalvera.me(refresh=True)\n" + scan, "InsufficientCredits"),
        ("budget choice", add_lab,
         RUN.format(scan="ct_scan.nii.gz", model="totalsegmentator_ct", budget=""),
         "BudgetChoiceRequired"),
        ("unknown model", state(personal_credits=100.0),
         RUN.format(scan="ct_scan.nii.gz", model="nope", budget=', budget="personal"'),
         "ConfigError"),
        ("file not found", None,
         RUN.format(scan="missing.nii.gz", model="totalsegmentator_ct",
                    budget=', budget="personal"'), "PreparationError"),
        ("not a scan", None,
         RUN.format(scan="notes.nii.gz", model="totalsegmentator_ct",
                    budget=', budget="personal"'), "PreparationError"),
        ("too large", tiny_files, "nalvera.me(refresh=True)\n" + scan, "PreparationError"),
    ]
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell("import nalvera")]
    notebook.cells += [nbformat.v4.new_code_cell(code) for _n, _s, code, _e in cases]
    client = nbclient.NotebookClient(notebook, timeout=240, kernel_name="python3",
                                     allow_errors=True,
                                     resources={"metadata": {"path": str(kernel_scene.dir)}})
    shown = []
    with client.setup_kernel():
        client.execute_cell(notebook.cells[0], 0)
        for index, (name, before, _code, expected) in enumerate(cases, start=1):
            if before is not None:
                before()
            client.execute_cell(notebook.cells[index], index)
            errors = [o for o in notebook.cells[index].outputs
                      if o.get("output_type") == "error"]
            if expected is None:
                assert errors == [], (name, errors)
                continue
            assert [e["ename"] for e in errors] == [expected], (name, errors)
            text = errors[0]["evalue"]
            shown.append(f"{name}: {errors[0]['ename']}: {text}")
            assert "nalvera login" not in text and not re.search(r"(?<!\w)--[a-z]", text), \
                (name, text)
            assert server.key not in text and other_key not in text, name
    # A traceback quotes the cell's own line (a key typed into the cell);
    # every other line of every output holds no key.
    lines = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            text = json.dumps({k: v for k, v in output.items() if k != "traceback"})
            lines.append(text)
            # A traceback block holds several lines, coloured token by token
            # (IPython 8 colours inside "nalvera.login("): uncoloured, line by line.
            for block in output.get("traceback", []):
                plain = re.sub(r"\x1b\[[0-9;]*m", "", block)
                lines += [line for line in plain.splitlines() if "nalvera.login(" not in line]
    every = "\n".join(lines)
    assert server.key[12:] not in every and other_key[12:] not in every
    (tmp_path / "ten_errors.txt").write_text("\n\n".join(shown), encoding="utf-8")
    if os.environ.get("ERRORS_REVIEW_LOG"):
        Path(os.environ["ERRORS_REVIEW_LOG"]).write_text("\n\n".join(shown), encoding="utf-8")
