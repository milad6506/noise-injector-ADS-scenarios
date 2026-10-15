import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment1"))

import runner as exp1  # noqa: E402

from .fixtures import make_fake_nuscenes_root  # noqa: E402


def test_deterministic_deviation_math():
    mother = {"commands": ["go straight", "keep speed"], "objects": [{"class": "Vehicle"}], "sensors": {"front": "ok", "back": "ok"}}
    same = {"commands": ["go straight", "keep speed"], "objects": [{"class": "Vehicle"}], "sensors": {"front": "ok", "back": "ok"}}
    drifted = {"commands": ["stop"], "objects": [{"class": "Vehicle"}, {"class": "Pedestrian"}], "sensors": {"front": "out", "back": "ok"}}
    assert exp1.deterministic_deviation(mother, same)["deviation"] == 0
    d = exp1.deterministic_deviation(mother, drifted)
    assert d["deviation"] > 3
    assert d["command_similarity"] == 0.0
    assert d["sensor_agreement"] == 0.5


def test_arms_differ_only_in_awareness():
    blind = (exp1._HERE / "prompts/blind_prompt.txt").read_text()
    aware = (exp1._HERE / "prompts/noise_aware_prompt.txt").read_text()
    assert "You are the driver of the car." in blind and "You are the driver of the car." in aware
    assert "noise or faults" not in blind
    assert "noise or faults" in aware
    for extra in blind.splitlines():
        if extra.strip():
            assert extra in aware


def test_runner_end_to_end_offline(tmp_path, monkeypatch):
    root = make_fake_nuscenes_root(tmp_path)
    run_dir = tmp_path / "run"
    import cv2 as _cv2_placeholder  # noqa: F401
    from noise_injector.adapters.base import get_adapter
    from noise_injector.render import make_recipe, render_scenario

    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    render_scenario(sc, {"name": "clean", "ops": []}, run_dir / "offspring" / "nuscenes" / "sceneA" / "clean", seed=1, root=str(root))
    for fam in ("fog", "frame_freeze"):
        r = make_recipe(fam, fam, [2])[0]
        render_scenario(sc, r, run_dir / "offspring" / "nuscenes" / "sceneA" / r["name"], seed=2, root=str(root))

    fake_vlm = exp1.FakeClient(canned='{"commands": ["go straight"], "objects": [{"class": "Vehicle", "camera": "CAM_FRONT"}], "sensors": {"front": "ok", "back": "out"}}')
    fake_judge = exp1.FakeClient(canned='{"agent1": {"deviation": 3, "resilience": 7, "noise_handling": 2, "notes": "ok"}, "agent2": {"deviation": 2, "resilience": 8, "noise_handling": 6, "notes": "aware"}, "winner": "agent2"}')

    real_cls = exp1.OpenAICompatClient
    monkeypatch.setattr(real_cls, "chat", lambda self, messages, max_tokens=700, temperature=0.0: self.canned)
    monkeypatch.setattr(exp1, "OpenAICompatClient",
                        lambda **kw: fake_judge if "glm" in kw.get("model", "") else fake_vlm)

    argv = ["--run-dir", str(run_dir), "--mothers", "1", "--per-mother", "2", "--max-frames", "1"]
    monkeypatch.setattr(sys, "argv", ["runner.py"] + argv)
    rc = exp1.main(argv)
    assert rc == 0
    results = json.load(open(exp1._HERE / "results" / "results.json"))
    assert len(results["cells"]) == 2
    agg = exp1.aggregate(results)
    assert agg["wins"].get("agent2") == 2
    report = exp1.write_report(results, agg, exp1._HERE / "results" / "report.html")
    content = report.read_text()
    assert "noise-blind vs noise-aware" in content or "Experiment 1" in content
    assert "OPENROUTER" not in content and "sk-or" not in content
