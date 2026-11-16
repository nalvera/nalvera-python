""".github/workflows/release.yml publishes a release to PyPI with no stored
password or token: PyPI accepts the upload because it comes from this
repository, this workflow file and a named environment. These checks hold
what keeps that safe. They read the file as text; the package has no YAML
library. The file lives in the repository only (the source archive does not
pack it)."""
from __future__ import annotations

import re

import pytest

import examples_helpers as helpers

WORKFLOW = helpers.ROOT / ".github" / "workflows" / "release.yml"

pytestmark = pytest.mark.skipif(not WORKFLOW.is_file(),
                                reason="the source archive carries no .github folder")


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job(name: str) -> str:
    """One job's block: from its key to the next job's key."""
    jobs = _text().split("\njobs:\n", 1)[1]
    blocks = re.split(r"^  (?=[a-z][a-z0-9_-]*:\n)", jobs, flags=re.M)
    (block,) = [b for b in blocks if b.startswith(f"{name}:\n")]
    return block


def test_the_workflow_reads_the_repository_and_nothing_more():
    head = _text().split("\njobs:\n", 1)[0]
    assert "\npermissions:\n  contents: read\n" in head
    assert "write" not in head


def test_no_secret_is_used():
    text = _text()
    assert "secrets." not in text
    assert "password" not in text and "token:" not in text.replace("id-token:", "")


def test_only_the_two_publish_jobs_may_ask_for_an_identity_token():
    assert _text().count("id-token: write") == 2
    assert "id-token" not in _job("build")
    for name in ("testpypi", "pypi"):
        assert "    permissions:\n      id-token: write\n" in _job(name)


def test_the_checkout_keeps_no_credentials():
    build = _job("build")
    assert "actions/checkout@" in build
    assert "persist-credentials: false" in build
    assert "actions/checkout@" not in _job("pypi") + _job("testpypi")


def test_a_published_release_goes_to_pypi_through_its_environment():
    job = _job("pypi")
    assert "    if: github.event_name == 'release'\n" in job
    assert "    environment:\n      name: pypi\n" in job
    assert "    needs: build\n" in job
    assert "repository-url" not in job                      # the action's default: pypi.org


def test_a_manual_run_goes_to_testpypi_through_its_environment():
    job = _job("testpypi")
    assert "    if: github.event_name == 'workflow_dispatch'\n" in job
    assert "    environment:\n      name: testpypi\n" in job
    assert "    needs: build\n" in job
    assert "repository-url: https://test.pypi.org/legacy/" in job


def test_the_workflow_runs_for_a_published_release_and_by_hand_only():
    triggers = _text().split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
    assert triggers == "  release:\n    types: [published]\n  workflow_dispatch:\n"


def test_a_release_whose_tag_differs_from_the_version_stops_before_publishing():
    build = _job("build")
    assert 'test -f "dist/nalvera-${GITHUB_REF_NAME#v}-py3-none-any.whl"' in build
    assert "twine check --strict dist/*" in build


def test_no_step_puts_a_value_from_the_event_into_a_command():
    """A tag or a title written by a person never becomes part of a script:
    the one value a command reads comes from an environment variable."""
    assert "${{" not in _text()


def test_every_action_is_pinned_to_one_commit():
    """A tag or a branch of an action can be moved to other code; a commit
    cannot. Each reference is a full commit id with its version beside it."""
    uses = re.findall(r"^\s+- uses: (.+)$", _text(), flags=re.M)
    assert len(uses) == 7
    for reference in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+", reference), reference
    # One commit per action: the same action is never pinned twice differently.
    pins = {}
    for reference in uses:
        name, commit = reference.split(" # ")[0].split("@")
        assert pins.setdefault(name, commit) == commit, name
