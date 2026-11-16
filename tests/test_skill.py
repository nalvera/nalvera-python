"""skills/nalvera/SKILL.md, the instructions an AI coding assistant loads, is
held by the package: its front matter follows the Agent Skills format, its
code block is the example file byte for byte, its first rules are the
README's, and every function, code, model and example it names exists.

The file lives in the repository only (the source archive does not pack it).
After changing ``examples/run_one_scan.py``, the README's rules for
assistants, its models table or its list of what is not offered, change the
skill to match: these tests say where."""
from __future__ import annotations

import inspect
import re

import examples_helpers as helpers
import nalvera
from nalvera import errors, messages

SKILL = helpers.ROOT / "skills" / "nalvera" / "SKILL.md"
TEXT = SKILL.read_text(encoding="utf-8")
README = helpers.PACKAGE_README.read_text(encoding="utf-8")
FRONT, BODY = TEXT.split("\n---\n", 1)
FENCE_RX = re.compile(r"^```\w*\n.*?^```$", re.M | re.S)
PROSE = re.sub(r"^\[//\]: # \(.*\)$", "", FENCE_RX.sub("", BODY), flags=re.M)


def _section(text: str, heading: str) -> str:
    return text.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def _items(block: str) -> list:
    return [line for line in block.splitlines() if line.startswith("- ")]


def _rows(block: str) -> list:
    return [line for line in block.splitlines() if line.startswith("| ")]


# ── the format ────────────────────────────────────────────────────────────────
def test_the_front_matter_follows_the_agent_skills_format():
    assert FRONT.startswith("---\n")
    fields = dict(line.split(": ", 1) for line in FRONT.splitlines()[1:])
    assert list(fields) == ["name", "description", "license"]
    # The name equals the folder's, in lower-case letters, digits and hyphens.
    assert fields["name"] == SKILL.parent.name == "nalvera"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"])
    description = fields["description"]
    # One plain line any YAML reader takes as a string, within the format's
    # 1,024 characters, saying what the skill does and when to use it.
    assert 0 < len(description) <= 1024, len(description)
    assert ": " not in description and " #" not in description
    assert not description.startswith(('"', "'", "[", "{", ">", "|", "*", "&", "!"))
    assert "<" not in description and ">" not in description
    assert "Use when" in description
    assert fields["license"] == "Apache-2.0"


def test_the_file_stays_short():
    assert len(TEXT.splitlines()) <= 150


# ── one source: the example and the README ───────────────────────────────────
def test_the_code_block_is_the_example_byte_for_byte():
    (quote,) = helpers.QUOTE_RX.finditer(TEXT)
    assert (quote.group("path"), quote.group("first"), quote.group("last")) == (
        "examples/run_one_scan.py", "6", "25")
    assert quote.group("body") == helpers.quoted_lines(
        quote.group("path"), quote.group("first"), quote.group("last")) + "\n"
    assert quote.group(0) in README                 # the README shows the same lines
    compile(quote.group("body"), "SKILL.md", "exec")


def test_the_first_rules_are_the_readmes_word_for_word():
    assistants = README.split("## For AI coding assistants", 1)[1].split("\n## ", 1)[0]
    readme_rules = _items(re.search(r"^Rules:\n\n((?:- .*\n)+)", assistants, re.M).group(1))
    skill_rules = _items(_section(BODY, "Rules"))
    assert len(readme_rules) == 6
    assert skill_rules[:6] == readme_rules


def test_the_models_and_what_is_not_offered_are_the_readmes():
    readme_models = [row for row in _rows(README) if re.match(r"\| [^|]+ \| `[a-z0-9_]+` \|", row)]
    skill_models = [row for row in _rows(_section(BODY, "Models")) if "`model` value" not in row
                    and not row.startswith("|---")]
    assert skill_models and skill_models == readme_models[:len(skill_models)]
    assert len(skill_models) == len(readme_models)
    assert _items(_section(BODY, "Not offered (yet)")) == _items(
        _section(README, "What is not offered (yet)"))


# ── every name exists ─────────────────────────────────────────────────────────
def test_every_function_setting_and_class_it_names_exists():
    names = set(re.findall(r"\bnalvera\.(?!ai\b)([A-Za-z_]\w*)", TEXT)) - {"mcp"}
    assert names and names <= set(nalvera.__all__), sorted(names - set(nalvera.__all__))
    assert (helpers.ROOT / "src" / "nalvera" / "mcp" / "__main__.py").is_file()
    settings = {s.name for s in nalvera.SETTINGS}
    used = set(re.findall(r"\bNALVERA_[A-Z_]+\b", TEXT))
    assert used <= settings, sorted(used - settings)
    classes = set(re.findall(r"`([A-Z][a-z][A-Za-z0-9]*)`", PROSE))
    assert classes <= set(nalvera.__all__), sorted(classes - set(nalvera.__all__))


def test_every_error_code_it_names_exists():
    error_classes = [c for c in vars(errors).values()
                     if inspect.isclass(c) and issubclass(c, errors.NalveraError)]
    known = ({key[len("error."):] for key in messages.MESSAGES if key.startswith("error.")}
             | set(errors._CLASS_BY_CODE) | {c.default_code for c in error_classes})
    named = []
    for row in _rows(_section(BODY, "Errors")):
        named += re.findall(r"`([a-z_]+)`", row.split("|")[1])
    assert {"invalid_credentials", "api_disabled", "scope_missing", "insufficient_credits",
            "no_free_tasks", "budget_choice_required", "max_credits_exceeded",
            "scan_rejected"} <= set(named)
    assert set(named) <= known, sorted(set(named) - known)
    # What the rules say about retrying is what the package does.
    assert "rate_limited" in errors.RETRYABLE_CODES
    assert not {"insufficient_credits", "no_free_tasks"} & set(errors.RETRYABLE_CODES)


def test_every_example_file_it_names_exists():
    files = set(re.findall(r"`([a-z0-9_]+\.py)`", PROSE))
    assert len(files) >= 10
    missing = sorted(name for name in files if not (helpers.EXAMPLES / name).is_file())
    assert not missing, missing


# ── what an assistant must never learn from it ────────────────────────────────
def test_it_holds_placeholders_only_and_no_other_commands():
    assert set(re.findall(r"nvk_[A-Za-z0-9_.]+", TEXT)) == {"nvk_..."}
    assert set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", TEXT)) == {"you@university.org"}
    assert "mcp.nalvera.ai" not in TEXT                 # the hosted server is not offered here
    commands = set(re.findall(r"`((?:pip|python|nalvera) [^`]+)`", PROSE))
    assert commands == {"pip install nalvera", 'pip install "nalvera[mcp]"',
                        "python -m nalvera.mcp"}, commands
    assert not re.search(r"\bjobs?\b", PROSE, re.I)
    assert helpers.python_text_problems("skills/nalvera/SKILL.md", PROSE) == []
