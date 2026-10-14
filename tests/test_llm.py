import json

import numpy as np
import pytest

from noise_injector.corruptions.llm_ops import apply_llm_op
from noise_injector.llm import FakeLLMBackend, LLMCache, cached_chat
from noise_injector.render import make_recipe, render_scenario
from noise_injector.spec import LLM_FAMILIES

from .fixtures import make_fake_nuscenes_root


def test_fake_backend_offline():
    b = FakeLLMBackend(canned="A red truck, not a white van.")
    out = b.chat("sys", "user")
    assert out == "A red truck, not a white van."
    assert len(b.calls) == 1


def test_cache_roundtrip(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    b = FakeLLMBackend(canned="cached-value")
    v1 = cached_chat(b, cache, "sys", "question")
    v2 = cached_chat(b, cache, "sys", "question")
    assert v1 == v2 == "cached-value"
    data = json.load(open(tmp_path / "c.json"))
    assert any(v == "cached-value" for v in data.values())


def test_llm_qa_rewrite_changes_answers(tmp_path):
    qa = {"planning": [{"Q": "What should the ego do?", "A": "Decelerate gradually without braking.", "C": None}]}
    b = FakeLLMBackend(canned="Accelerate through the intersection.")
    detail = apply_llm_op("llm_qa_rewrite", qa, {"rate": 1.0, "style": "plausible_wrong"}, np.random.default_rng(0), b, None)
    assert qa["planning"][0]["A"] == "Accelerate through the intersection."
    assert detail[0]["old"] == "Decelerate gradually without braking."


def test_llm_description_drift(tmp_path):
    koi = {"<c1,CAM_FRONT,1.0,1.0>": {"Category": "Vehicle", "Status": None, "Visual_description": "navy SUV", "2d_bbox": [0, 0, 10, 10]}}
    b = FakeLLMBackend(canned="yellow SUV")
    detail = apply_llm_op("llm_description_drift", koi, {"rate": 1.0}, np.random.default_rng(1), b, None)
    assert koi["<c1,CAM_FRONT,1.0,1.0>"]["Visual_description"] == "yellow SUV"
    assert detail[0]["old"] == "navy SUV"


def test_llm_op_requires_backend():
    qa = {"planning": [{"Q": "Q", "A": "A", "C": None}]}
    with pytest.raises(ValueError, match="requires an LLM backend"):
        apply_llm_op("llm_qa_rewrite", qa, {"rate": 1.0}, np.random.default_rng(0), None, None)


def test_render_with_llm_backend_end_to_end(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    from noise_injector.adapters.base import get_adapter

    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    recipe = make_recipe("llm_qa_rewrite", "llm_qa_rewrite", [5])[0]
    recipe["ops"][0]["params_override"] = {"rate": 1.0, "style": "plausible_wrong"}
    out = tmp_path / "off"
    backend = FakeLLMBackend(canned="rewritten wrong answer")
    cache = LLMCache(tmp_path / "cache.json")
    m = render_scenario(sc, recipe, out, seed=5, root=str(root), backend=backend, cache=cache)
    ev = m["injected_events"][0]
    assert ev["family"] == "llm_qa_rewrite"
    assert ev["backend"] == "fake-llm"
    ann = json.load(open(out / "annotations.json"))
    for fid in ann["key_frames"]:
        assert ann["key_frames"][fid]["QA"]["planning"][0]["A"] == "rewritten wrong answer"


def test_manifest_redacts_paths(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    from noise_injector.adapters.base import get_adapter

    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    out = tmp_path / "off"
    render_scenario(sc, make_recipe("fog", "fog", [2])[0], out, seed=1, root="/Users/someone/secret/DriveLM")
    m = json.load(open(out / "manifest.json"))
    assert m["parent"]["root"] == "<dataset-root>"
    assert "/Users" not in json.dumps(m)
