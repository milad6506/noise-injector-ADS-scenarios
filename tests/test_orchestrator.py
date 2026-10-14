import json

import numpy as np
import pytest

from noise_injector.llm import FakeLLMBackend
from noise_injector.orchestrator import Orchestrator, fallback_policy, load_experiment, propose_policy
from noise_injector.scoring import perception_metrics, resilience_summary
from noise_injector.targets import BaseTarget, make_target

from .fixtures import make_fake_nuscenes_root

_ENGINE_POLICY = json.dumps({
    "ops": [{"family": "fog", "severity": 2, "pattern": "persistent"},
            {"family": "awgn", "severity": 1, "pattern": "persistent"}],
    "rationale": "probe weather family first",
})


class DecayTarget(BaseTarget):
    """Fake target: score decays with injected severity — deterministic, offline."""

    def __init__(self, name="fake-vlm", role="perception", decay=0.12):
        self.name = name
        self.role = role
        self.decay = decay
        self.evaluated: list[str] = []

    def evaluate(self, offspring_dir, annotations=None):
        self.evaluated.append(str(offspring_dir))
        name = offspring_dir.name
        if name == "clean":
            score = 1.0
        else:
            sev = int(name.rsplit("s", 1)[-1])
            score = round(max(0.0, 1.0 - self.decay * sev), 3)
        return {"score": score, "frames": 4}


@pytest.fixture
def exp_cfg(tmp_path, monkeypatch):
    root = make_fake_nuscenes_root(tmp_path)
    cfg = {
        "name": "test-run",
        "engine": {"base_url": "http://localhost:0/v1", "model": "fake-engine"},
        "dataset": {"type": "nuscenes", "root": str(root)},
        "scenarios": "1",
        "rounds": 2,
        "severity_cap": 5,
        "budget": {"ops_per_round": 2},
        "targets": [{"name": "fake-target", "type": "vlm", "base_url": "http://x", "model": "m", "role": "perception"}],
    }
    return cfg, tmp_path


def test_fallback_policy_rotation():
    pol = fallback_policy(0, ["fog", "awgn", "jpeg"], 2, 5)
    assert [o["family"] for o in pol["ops"]] == ["fog", "awgn"]
    pol2 = fallback_policy(1, ["fog", "awgn", "jpeg"], 2, 5)
    assert [o["family"] for o in pol2["ops"]] == ["jpeg", "fog"]
    assert pol2["ops"][0]["severity"] == 1


def test_propose_policy_validates():
    backend = FakeLLMBackend(canned=_ENGINE_POLICY)
    catalog = [{"family": "fog", "layer": "A", "pattern": "persistent", "description": ""}]
    pol = propose_policy(backend, catalog, [], 2, 1, 3)
    assert pol["ops"][0]["family"] == "fog"
    assert pol["rationale"] == "probe weather family first"
    backend_bad = FakeLLMBackend(canned="no json here")
    with pytest.raises(ValueError):
        propose_policy(backend_bad, catalog, [], 2, 1, 3)


def test_orchestrator_rounds(exp_cfg):
    cfg, tmp_path = exp_cfg
    out = tmp_path / "run"
    orch = Orchestrator(cfg, out)
    fake = DecayTarget(name="fake-target")

    orig_make_target = make_target
    import noise_injector.orchestrator as orch_mod

    monkey_patch = lambda t, judge_backend=None: fake
    orch_mod.make_target = monkey_patch
    try:
        result = orch.run()
    finally:
        orch_mod.make_target = orig_make_target

    assert len(orch.round_log) == 3
    assert orch.round_log[0]["round"] == 0
    scores = {r["score"] for r in orch.results}
    assert 1.0 in scores
    summary = result["summary"]["per_target"]["fake-target"]
    assert summary["clean_score"] == 1.0
    assert 0 < summary["mean_retention"] < 1
    assert summary["min_retention"] <= summary["mean_retention"]
    assert (out / "resilience_report.html").exists()
    assert (out / "state.json").exists()


def test_orchestrator_resume(exp_cfg):
    cfg, tmp_path = exp_cfg
    out = tmp_path / "run"
    orch = Orchestrator(cfg, out)
    fake = DecayTarget()
    import noise_injector.orchestrator as orch_mod

    orig = orch_mod.make_target
    orch_mod.make_target = lambda t, judge_backend=None: fake
    try:
        orch.run()
        n_first = len(fake.evaluated)
        orch2 = Orchestrator(cfg, out)
        orch2.run()
        assert len(fake.evaluated) == n_first
    finally:
        orch_mod.make_target = orig


def test_engine_managed_rounds_recorded(exp_cfg):
    cfg, tmp_path = exp_cfg
    out = tmp_path / "run"
    orch = Orchestrator(cfg, out)
    orch.exp["engine"]["__fake__"] = _ENGINE_POLICY
    fake = DecayTarget()
    import noise_injector.orchestrator as orch_mod

    orig = orch_mod.make_target
    orch_mod.make_target = lambda t, judge_backend=None: fake

    class PolicyBackend(FakeLLMBackend):
        def chat(self, system, user, temperature=0.0, max_tokens=300):
            if "fault-injection rounds" in system:
                return _ENGINE_POLICY
            return super().chat(system, user, temperature, max_tokens)

    orig_backend = orch_mod.LLMBackend
    orch_mod.LLMBackend = PolicyBackend
    try:
        result = orch.run()
    finally:
        orch_mod.make_target = orig
        orch_mod.LLMBackend = orig_backend
    engine_rounds = [rl for rl in orch.round_log if rl.get("source") == "engine"]
    assert engine_rounds, "engine policy was not used in any round"
    assert any(o["family"] == "fog" for rl in engine_rounds for o in rl["ops"])


def test_perception_metrics_basics():
    gt = [{"Vehicle", "Pedestrian"}, {"Vehicle"}]
    pred = [{"Vehicle"}, {"Pedestrian"}]
    m = perception_metrics(gt, pred)
    assert abs(m["presence_recall"] - 1 / 3) < 0.01
    assert m["presence_fp_rate"] == 0.5


def test_resilience_summary_math():
    rows = [
        {"round": 0, "family": None, "target": "t", "score": 1.0},
        {"round": 1, "family": "fog", "severity": 2, "target": "t", "score": 0.8},
        {"round": 1, "family": "fog", "severity": 4, "target": "t", "score": 0.4},
    ]
    s = resilience_summary(rows)
    assert s["per_target"]["t"]["mean_retention"] == 0.6
    assert s["per_target"]["t"]["min_retention"] == 0.4
    assert s["retentions"][0]["retention"] == 0.8


def test_make_target_types():
    t = make_target({"name": "v", "type": "vlm", "base_url": "http://x", "model": "m", "role": "driving"})
    assert t.role == "driving"
    with pytest.raises(KeyError):
        make_target({"type": "nope"})
