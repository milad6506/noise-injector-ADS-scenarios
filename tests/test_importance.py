import json

import numpy as np
import pytest

from noise_injector.cli import main
from noise_injector.importance import camera_priorities, camera_weights, detect_maneuver
from noise_injector.render import make_recipe, render_scenario
from noise_injector.temporal import activation_mask, derive_seed

from .fixtures import make_fake_nuscenes_root


def test_maneuver_detection():
    assert detect_maneuver("The ego vehicle is backing out of the parking space.") == "backing"
    assert detect_maneuver("The ego proceeds through the intersection.") == "forward"
    assert detect_maneuver("The ego vehicle is turning left at the light.") == "turning_left"
    assert detect_maneuver("Ego turns right onto the side street.") == "turning_right"
    assert detect_maneuver("") == "unknown"


def test_camera_priorities_forward_and_backing():
    cams = ["CAM_BACK", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    fwd = camera_priorities("The ego drives straight ahead.", cams)
    assert fwd[0] == "CAM_FRONT"
    back = camera_priorities("The ego vehicle is backing into a parking spot.", cams)
    assert back[0] == "CAM_BACK"
    left = camera_priorities("Ego is turning left.", cams)
    assert left[0] == "CAM_FRONT_LEFT"


def test_camera_weights_normalized():
    w = camera_weights("The ego proceeds forward.", ["CAM_FRONT", "CAM_BACK"])
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["CAM_FRONT"] > w["CAM_BACK"]


def test_flaky_camera_importance_weighted(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    from noise_injector.adapters.base import get_adapter

    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    assert sc.description == "Ego drives straight."
    from collections import Counter

    draws = Counter()
    for seed in range(40):
        recipe = {"name": f"drop{seed}", "ops": [
            {"family": "camera_dropout", "severity": 2, "pattern": "flaky_camera", "params_override": {"mode": "missing"}},
        ]}
        out = tmp_path / f"o{seed}"
        m = render_scenario(sc, recipe, out, seed=derive_seed(1000 + seed, "sceneA", f"drop{seed}"), root=str(root))
        ev = m["injected_events"][0]
        for t in ev["targets"]:
            if t["cameras"]:
                draws[t["cameras"][0]] += 1
    front = sum(v for k, v in draws.items() if "FRONT" in k)
    back = sum(v for k, v in draws.items() if "BACK" in k)
    assert len(draws) == 6
    assert front > back * 2


def test_per_camera_batch_limit(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    out = tmp_path / "ts"
    rc = main([
        "generate", "--dataset", "nuscenes", "--data-root", str(root),
        "--samples", "2", "--family", "camera_dropout", "--pattern", "camera_scoped",
        "--per-camera", "--batch-limit", "3", "--severities", "1", "--out", str(out),
    ])
    assert rc == 0
    scenea = {p.name for p in (out / "offspring" / "nuscenes" / "sceneA").iterdir() if p.name != "clean"}
    sceneb = {p.name for p in (out / "offspring" / "nuscenes" / "sceneB").iterdir() if p.name != "clean"}
    assert len(scenea) + len(sceneb) == 3
    assert scenea == {
        "camera_dropout_camera_scoped_CAM_FRONT_s1",
        "camera_dropout_camera_scoped_CAM_FRONT_LEFT_s1",
        "camera_dropout_camera_scoped_CAM_FRONT_RIGHT_s1",
    }
    assert sceneb == set()


def test_force_camera_respected():
    cams = ["CAM_A", "CAM_B"]
    recipe_op = {"family": "camera_dropout", "severity": 1, "pattern": "camera_scoped", "params_override": {"mode": "black"}, "force_camera": "CAM_B"}
    from noise_injector.spec import family
    assert family("camera_dropout").default_pattern == "camera_scoped"


def test_flaky_camera_mask_intermittent():
    rng = np.random.default_rng(5)
    m = activation_mask("flaky_camera", 20, rng, severity=3, camera_index=2)
    assert m["camera"] == 2
    on = sum(m["frames"])
    assert 1 <= on < 20
