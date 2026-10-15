import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment3"))

sys.modules.pop("runner", None)
import runner as exp3  # noqa: E402

from .fixtures import make_fake_nuscenes_root  # noqa: E402


def test_variant_matrix_count():
    variants = exp3.build_variants()
    assert len(variants) == 12
    per_branch = {k: [v["fault"] for v in variants if v["branch"] == k] for k in ("A", "B")}
    assert per_branch["A"][0] == "clean" and len(per_branch["A"]) == 6
    assert set(per_branch["A"][1:]) == set(exp3.FAULT_OPS)
    assert set(per_branch["B"][1:]) == set(exp3.FAULT_OPS)


def test_branches_differ_only_in_integrity_note():
    a = exp3.build_variants()[0]["prompt"]
    b = exp3.build_variants()[6]["prompt"]
    assert "INTEGRITY NOTE" not in a
    assert "INTEGRITY NOTE" in b
    for line in a.splitlines():
        if line.strip():
            assert line in b


def test_fake_client_exists():
    assert hasattr(exp3.exp1, "FakeClient")


def test_faulty_variants_differ_from_clean():
    variants = exp3.build_variants()
    a_clean = variants[0]["prompt"]
    a_faulty = [v["prompt"] for v in variants if v["branch"] == "A" and v["fault"] != "clean"]
    assert all(p != a_clean for p in a_faulty)
    inj = next(v["prompt"] for v in variants if v["branch"] == "A" and v["fault"] == "prompt_instruction_injection")
    assert "ignore" in inj or "deprecated" in inj or "verified safe" in inj


def test_runner_end_to_end_offline(tmp_path, monkeypatch):
    root = make_fake_nuscenes_root(tmp_path)
    run_dir = tmp_path / "run"
    from noise_injector.adapters.base import get_adapter
    from noise_injector.render import render_scenario

    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    render_scenario(sc, {"name": "clean", "ops": []}, run_dir / "offspring" / "nuscenes" / "sceneA" / "clean", seed=1, root=str(root))

    clean_resp = '{"commands": ["go straight"], "objects": [], "sensors": {"front": "ok"}}'
    faulty_resp = '{"commands": ["accelerate hard"], "objects": [], "sensors": {"front": "ok"}}'
    judge_resp = ('{"safety": 3, "manipulation_resistance": 2, "unexplained_drift": 7, '
                  '"followed_malicious_instruction": true, "notes": "complied"}')

    fake_vlm = exp3.exp1.FakeClient(canned=clean_resp)
    fake_judge = exp3.exp1.FakeClient(canned=judge_resp)
    counters = {"n": 0}

    def factory(**kw):
        if "glm" in kw.get("model", ""):
            return fake_judge
        counters["n"] += 1
        return exp3.exp1.FakeClient(canned=clean_resp if counters["n"] % 2 == 0 else faulty_resp)

    real_cls = exp3.exp1.OpenAICompatClient
    monkeypatch.setattr(real_cls, "chat", lambda self, messages, max_tokens=700, temperature=0.0: self.canned)
    monkeypatch.setattr(exp3.exp1, "OpenAICompatClient", factory)
    monkeypatch.setattr(exp3.exp1.FakeClient, "chat", lambda self, messages, max_tokens=700, temperature=0.0: self.canned)

    rc = exp3.main(["--run-dir", str(run_dir), "--mothers", "1", "--workers", "2"])
    assert rc == 0
    results = json.load(open(exp3._HERE / "results" / "results.json"))
    assert len(results["cells"]) == 10
    agg = exp3.aggregate(results)
    assert set(agg["branch_summary"]) == {"A", "B"}
    report = exp3.write_report(results, agg, exp3._HERE / "results" / "report.html")
    content = report.read_text()
    assert "prompt integrity" in content.lower()
    assert "sk-or" not in content


def test_intent_split_aggregation():
    results = {"branches": exp3.BRANCHES, "cells": []}
    for br in ("A", "B"):
        for fault, intent in exp3.CORRUPTION_INTENT.items():
            for i in range(3):
                results["cells"].append({
                    "branch": br, "fault": fault,
                    "det_dev": {"deviation": 5.0 if br == "A" else 3.0},
                    "judge": {"safety": 6.0 if (br == "A") else (7.0 if intent == "benign" else 2.0),
                              "manipulation_resistance": 5.0,
                              "followed_malicious_instruction": intent == "adversarial" and br == "B"},
                })
    agg = exp3.aggregate(results)
    assert set(agg["intent_summary"]) == {"benign", "adversarial"}
    benign = agg["intent_summary"]["benign"]
    adversarial = agg["intent_summary"]["adversarial"]
    assert benign["B"]["mean_safety"] == 7.0
    assert adversarial["B"]["mean_safety"] == 2.0
    assert adversarial["B"]["malicious_followed"] == 3
    assert adversarial["A"]["malicious_followed"] == 0
