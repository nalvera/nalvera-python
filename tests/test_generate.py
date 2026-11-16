"""Generations: --param parsing, the request, idempotency, server messages."""
from __future__ import annotations

import pytest

from fake_server import error_body
from nalvera import generation as generate
from nalvera.errors import ConfigError, InvalidRequest

CT = {"mode": "image-only", "output_size": [256, 256, 128], "spacing": [1.5, 1.5, 2.0],
      "num_output_samples": 1, "seed": 7}


# ── --param ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("item,expected", [
    ("seed=7", ("seed", 7)),
    ("cfg_guidance_scale=2.5", ("cfg_guidance_scale", 2.5)),
    ("skull_stripped=true", ("skull_stripped", True)),
    ("skull_stripped=False", ("skull_stripped", False)),
    ("output_size=[256,256,128]", ("output_size", [256, 256, 128])),
    ("spacing=[1.5, 1.5, 2.0]", ("spacing", [1.5, 1.5, 2.0])),
    ("mode=image-only", ("mode", "image-only")),
    ('anatomy_list=["liver","spleen"]', ("anatomy_list", ["liver", "spleen"])),
    ("body_region=null", ("body_region", None)),
    ('label="007"', ("label", "007")),
    ("x= 3 ", ("x", 3)),
])
def test_parse_param(item, expected):
    assert generate.parse_param(item) == expected


@pytest.mark.parametrize("item", ["seed", "=7", "7seed=1", "a b=1", "a-b=1"])
def test_a_malformed_setting_is_refused(item):
    with pytest.raises(ConfigError) as caught:
        generate.parse_param(item)
    assert caught.value.code == "invalid_param"


def test_a_setting_given_twice_is_refused():
    with pytest.raises(ConfigError) as caught:
        generate.parse_params(["seed=1", "seed=2"])
    assert caught.value.code == "duplicate_param"


def test_describe_lists_every_setting():
    pairs = dict(generate.describe({"modes": ["image-only", "image-mask"],
                                    "spacing_mm": {"min": 0.5, "max": 5.0},
                                    "rules": ["X equal to Y", "at least 256 mm"]}))
    assert set(pairs) == {"modes", "spacing_mm", "rules"}
    assert "image-mask" in pairs["modes"] and "0.5" in pairs["spacing_mm"]


# ── the request ───────────────────────────────────────────────────────────────
def test_generate_sends_the_settings_through(client, server):
    task = client.generate("nv_generate_ctmr", "rflow-ct", CT, budget="personal",
                           label="phantom A")
    body = server.tasks[task.id]["body"]
    assert body["params"] == CT and body["model"] == "nv_generate_ctmr"
    assert body["task"] == "rflow-ct" and body["label"] == "phantom A"
    assert "upload_id" not in body
    assert task.label == "phantom A" and task.state == "queued"


def test_each_generation_has_its_own_key_kept_across_retries(client, server):
    server.inject("POST", "/api/v1/tasks", drop=True, after=True)
    first = client.generate("nv_generate_ctmr", "rflow-ct", CT, budget="personal")
    second = client.generate("nv_generate_ctmr", "rflow-ct", CT, budget="personal")
    keys = [r["headers"]["Idempotency-Key"] for r in server.calls("POST", "/api/v1/tasks")]
    assert len(keys) == 3 and keys[0] == keys[1] != keys[2]
    assert first.id != second.id and len(server.tasks) == 2


def test_the_servers_explanation_is_shown_as_it_is(client, server):
    detail = "params.output_size: output_size is [X, Y, Z] with X equal to Y"
    server.inject("POST", "/api/v1/tasks", status=400,
                  body=error_body("invalid_request", detail=detail))
    with pytest.raises(InvalidRequest) as caught:
        client.generate("nv_generate_ctmr", "rflow-ct", {**CT, "output_size": [256, 128, 128]},
                        budget="personal")
    assert detail in str(caught.value)
    assert len(server.calls("POST", "/api/v1/tasks")) == 1


def test_a_scan_model_is_refused_for_a_generation(client, server):
    with pytest.raises(ConfigError) as caught:
        client.generate("totalsegmentator_ct", "total", CT, budget="personal")
    assert caught.value.code == "model_needs_scan"


def test_the_question_uses_the_generations_estimate(make_client, server):
    import io
    server.add_lab(12, "Imaging Lab", 100.0)
    client = make_client(interactive=True, input=io.StringIO("2\n"), output=io.StringIO())
    task = client.generate("nv_generate_ctmr", "rflow-ct", CT)
    assert server.tasks[task.id]["body"]["budget"] == "lab:12"
    estimate = server.calls("POST", "/api/v1/estimate")
    assert len(estimate) == 1


def test_a_generation_result_is_named_after_its_label(client, server, env):
    task = client.generate("nv_generate_ctmr", "rflow-ct", CT, budget="personal",
                           label="phantom A")
    path = client.download(client.wait([task]), out=env / "results")[0]
    assert path.name == "phantom_A__nv_generate_ctmr__rflow-ct.nii.gz"


def test_without_a_label_the_task_id_names_the_result(client, server, env):
    task = client.generate("nv_generate_ctmr", "rflow-ct", CT, budget="personal")
    path = client.download(client.wait([task]), out=env / "results")[0]
    assert path.name == f"{task.id}__nv_generate_ctmr__rflow-ct.nii.gz"
