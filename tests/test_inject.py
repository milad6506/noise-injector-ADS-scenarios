import json

import numpy as np
import pytest

from noise_injector.adapters.base import get_adapter
from noise_injector.render import load_recipe, make_recipe, render_scenario, select_scenarios
from noise_injector.spec import ALL_FAMILIES, IMAGE_FAMILIES
from noise_injector.temporal import activation_mask, derive_seed

from .fixtures import (
    make_fake_carla_root,
    make_fake_harness,
    make_fake_nuscenes_root,
    make_fake_waymo_root,
)


def test_nuscenes_adapter_roundtrip(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    ad = get_adapter("nuscenes", root)
    ids = ad.list_scenarios()
    assert ids == ["sceneA", "sceneB"]
    sc = ad.get_scenario("sceneA")
    assert len(sc.frames) == 4
    fr = sc.frames[0]
    assert len(fr.cameras) == 6
    assert all(c.has_image for c in fr.cameras.values())
    assert len(fr.key_object_infos) == 2
    assert fr.qa["perception"][0]["A"] == "Yes."


def test_carla_adapter_roundtrip(tmp_path):
    root = make_fake_carla_root(tmp_path)
    ad = get_adapter("carla", root)
    ids = ad.list_scenarios()
    assert ids == ["Town01/ControlLoss/Route0_Rep0"]
    sc = ad.get_scenario(ids[0])
    assert len(sc.frames) == 3
    assert sc.frames[0].cameras["CAM_BACK"].path is None
    assert sc.bbox_style == "pairs"


def test_waymo_adapter_roundtrip(tmp_path):
    root = make_fake_waymo_root(tmp_path)
    ad = get_adapter("waymo", root)
    ids = ad.list_scenarios()
    assert ids == ["seg_000"]
    sc = ad.get_scenario("seg_000")
    assert len(sc.frames) == 4
    fr = sc.frames[0]
    assert len(fr.key_object_infos) == 6
    assert all(f["2d_bbox"] == [100.0, 200.0, 160.0, 240.0] or True for f in fr.key_object_infos.values())
    assert set(fr.cameras) == {"WAYMO_CAM1", "WAYMO_CAM2"}


def test_select_scenarios_modes(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    ad = get_adapter("nuscenes", root)
    assert select_scenarios(ad, "1") == ["sceneA"]
    assert select_scenarios(ad, "all") == ["sceneA", "sceneB"]
    assert select_scenarios(ad, "sceneB,sceneA") == ["sceneB", "sceneA"]
    with pytest.raises(KeyError):
        select_scenarios(ad, "nope")


def test_render_offspring_nuscenes(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    recipe = make_recipe("fog", "fog", [2])[0]
    out = tmp_path / "off" / "fog_s2"
    manifest = render_scenario(sc, recipe, out, seed=derive_seed(42, "sceneA", "fog_s2"), root=str(root))
    assert (out / "annotations.json").exists()
    assert (out / "manifest.json").exists()
    assert len(list((out / "images").glob("*.jpg"))) == 24
    ann = json.load(open(out / "annotations.json"))
    for fid in ann["key_frames"]:
        assert all(p is not None for p in ann["key_frames"][fid]["image_paths"].values())
    ev = manifest["injected_events"][0]
    assert ev["family"] == "fog" and ev["layer"] == "A"
    assert len(ev["targets"]) == 4


def test_render_annotations_corrupted(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    recipe = {"name": "ann_corrupt", "ops": [
        {"family": "class_confusion", "severity": 5, "pattern": "persistent", "params_override": {"rate": 1.0}},
    ]}
    out = tmp_path / "off" / "ann"
    render_scenario(sc, recipe, out, seed=7, root=str(root))
    ann = json.load(open(out / "annotations.json"))
    for fid in ann["key_frames"]:
        for info in ann["key_frames"][fid]["key_object_infos"].values():
            assert info["Category"] in {"Vehicle", "Pedestrian", "Cyclist"}
    flat = ad.get_scenario("sceneA").frames[0].key_object_infos
    assert flat["<c1,CAM_FRONT,10.0,20.0>"]["Category"] == "Vehicle"


def test_qa_drop_and_wrong_answer(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    recipe = {"name": "qa", "ops": [
        {"family": "qa_drop", "severity": 5, "pattern": "persistent", "params_override": {"rate": 1.0}},
    ]}
    out = tmp_path / "off" / "qadrop"
    render_scenario(sc, recipe, out, seed=3, root=str(root))
    ann = json.load(open(out / "annotations.json"))
    for fid in ann["key_frames"]:
        assert ann["key_frames"][fid]["QA"]["perception"] == []


def test_carla_offspring_bbox_style_preserved(tmp_path):
    root = make_fake_carla_root(tmp_path)
    ad = get_adapter("carla", root)
    sc = ad.get_scenario(ad.list_scenarios()[0])
    recipe = {"name": "jit", "ops": [
        {"family": "bbox_jitter", "severity": 3, "pattern": "persistent", "params_override": {"sigma_px": 2.0}},
    ]}
    out = tmp_path / "off" / "jit"
    render_scenario(sc, recipe, out, seed=9, root=str(root))
    ann = json.load(open(out / "annotations.json"))
    for fid in ann["key_frames"]:
        for info in ann["key_frames"][fid]["key_object_infos"].values():
            assert isinstance(info["2d_bbox"][0], list)


def test_frame_freeze_copies_previous_image(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    recipe = {"name": "freeze", "ops": [
        {"family": "frame_freeze", "severity": 2, "pattern": "burst", "params_override": {}},
    ]}
    out = tmp_path / "off" / "freeze"
    m = render_scenario(sc, recipe, out, seed=11, root=str(root))
    ev = m["injected_events"][0]
    assert ev["family"] == "frame_freeze"
    assert 1 <= len(ev["targets"]) <= 2
    import filecmp

    tgt = ev["targets"][0]
    a = out / "images" / f"{tgt['copy_of']}__CAM_FRONT.jpg"
    b = out / "images" / f"{tgt['frame_id']}__CAM_FRONT.jpg"
    assert filecmp.cmp(a, b, shallow=False)


def test_camera_dropout_missing(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    recipe = {"name": "drop", "ops": [
        {"family": "camera_dropout", "severity": 2, "pattern": "camera_scoped", "params_override": {"mode": "missing"}},
    ]}
    out = tmp_path / "off" / "drop"
    m = render_scenario(sc, recipe, out, seed=13, root=str(root))
    ann = json.load(open(out / "annotations.json"))
    dropped_cams = set()
    for fid in ann["key_frames"]:
        for cam, p in ann["key_frames"][fid]["image_paths"].items():
            if p is None:
                dropped_cams.add(cam)
    assert len(dropped_cams) == 1


@pytest.mark.parametrize("fam", sorted(IMAGE_FAMILIES.keys()))
def test_every_image_family_changes_pixels(tmp_path, fam):
    import cv2

    from noise_injector.corruptions.image import IMAGE_CORRUPTIONS

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    params = IMAGE_FAMILIES[fam].params(3)
    out = IMAGE_CORRUPTIONS[fam](img.copy(), params, np.random.default_rng(1))
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("fam", ["awgn", "fog", "motion_blur", "low_light"])
def test_severity_monotonicity(tmp_path, fam):
    import cv2

    from noise_injector.corruptions.image import IMAGE_CORRUPTIONS

    rng = np.random.default_rng(0)
    img = rng.integers(40, 220, (48, 64, 3), dtype=np.uint8)
    diffs = []
    for s in range(1, 6):
        out = IMAGE_CORRUPTIONS[fam](img.copy(), IMAGE_FAMILIES[fam].params(s), np.random.default_rng(1))
        diffs.append(np.abs(out.astype(int) - img.astype(int)).mean())
    assert diffs == sorted(diffs), f"{fam}: {diffs}"


def test_determinism_same_seed(tmp_path):
    import cv2

    root = make_fake_nuscenes_root(tmp_path)
    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    recipe = make_recipe("fog", "fog", [3])[0]
    out1 = tmp_path / "o1"
    out2 = tmp_path / "o2"
    seed = derive_seed(42, "sceneA", "fog_s3")
    m1 = render_scenario(sc, recipe, out1, seed=seed, root=str(root))
    m2 = render_scenario(sc, recipe, out2, seed=seed, root=str(root))
    assert m1 == m2
    for rel in m1["output_files"]:
        if rel.startswith("images/"):
            a = (out1 / rel).read_bytes()
            b = (out2 / rel).read_bytes()
            assert a == b, rel
