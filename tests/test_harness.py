import json

import pytest

from noise_injector.harness import discover_files, render_harness
from noise_injector.render import load_recipe

from .fixtures import make_fake_harness


def make_recipe_inline(name, ops):
    return {"name": name, "ops": ops}


def test_discover_skips_excluded(tmp_path):
    root = make_fake_harness(tmp_path)
    prompts, configs = discover_files(root)
    names = [str(p) for p in prompts + configs]
    assert "prompts/simple_prompt.txt" in names
    assert "cot_prompt.txt" in names
    assert "config.json" in names
    assert not any("runs/" in n for n in names)
    assert not any(n.endswith(".pt") for n in names)


def test_harness_copy_originals_untouched(tmp_path):
    root = make_fake_harness(tmp_path)
    out = tmp_path / "offspring"
    recipe = make_recipe_inline("mix", [
        {"family": "prompt_section_delete", "severity": 3},
        {"family": "vocab_drop", "severity": 2},
        {"family": "confidence_drift", "severity": 2},
    ])
    original_prompt = (root / "prompts" / "simple_prompt.txt").read_text()
    original_cfg = (root / "config.json").read_text()

    manifest = render_harness(root, recipe, out, seed=42)

    assert (root / "prompts" / "simple_prompt.txt").read_text() == original_prompt
    assert (root / "config.json").read_text() == original_cfg

    cfg = json.load(open(out / "config.json"))
    assert cfg["arms"][0]["detector_vocab"] != ["car", "pedestrian", "bus"]
    assert cfg["arms"][0]["detector_conf"] > 0.25

    corrupted = (out / "prompts" / "simple_prompt.txt").read_text()
    assert corrupted != original_prompt
    assert len(manifest["injected_events"]) == 4
    fams = [e["family"] for e in manifest["injected_events"]]
    assert fams.count("prompt_section_delete") == 2
    assert "vocab_drop" in fams and "confidence_drift" in fams
    assert all(e["layer"] == "F" for e in manifest["injected_events"])


def test_harness_instruction_injection(tmp_path):
    root = make_fake_harness(tmp_path)
    out = tmp_path / "off"
    recipe = make_recipe_inline("inject", [{"family": "prompt_instruction_injection", "severity": 2}])
    render_harness(root, recipe, out, seed=1)
    new = (out / "prompts" / "simple_prompt.txt").read_text()
    assert "ignore earlier instructions" in new or "verified safe" in new or "deprecated" in new


def test_harness_max_tokens_starve(tmp_path):
    root = make_fake_harness(tmp_path)
    out = tmp_path / "off"
    recipe = make_recipe_inline("starve", [{"family": "max_tokens_starve", "severity": 5}])
    render_harness(root, recipe, out, seed=2)
    cfg = json.load(open(out / "config.json"))
    assert cfg["generation"]["max_tokens"] <= 4096 * 0.05 + 16
    assert cfg["arms"][0]["max_tokens"] <= 1600 * 0.05 + 16


def test_harness_determinism(tmp_path):
    root = make_fake_harness(tmp_path)
    recipe = make_recipe_inline("d", [{"family": "prompt_garble", "severity": 3}])
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    m1 = render_harness(root, recipe, o1, seed=5)
    m2 = render_harness(root, recipe, o2, seed=5)
    assert m1["injected_events"] == m2["injected_events"]
    for rel in ("prompts/simple_prompt.txt", "cot_prompt.txt"):
        assert (o1 / rel).read_bytes() == (o2 / rel).read_bytes()


def test_all_harness_ops_run(tmp_path):
    root = make_fake_harness(tmp_path)
    for op in ["prompt_truncation", "prompt_format_break", "vocab_misspell", "vocab_swap",
               "max_tokens_starve", "quality_drift", "temperature_drift"]:
        out = tmp_path / f"off_{op}"
        recipe = make_recipe_inline(op, [{"family": op, "severity": 3}])
        m = render_harness(root, recipe, out, seed=3)
        assert m["injected_events"], op
