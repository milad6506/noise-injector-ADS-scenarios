import pytest

from noise_injector import __version__
from noise_injector.cli import main


def test_version():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "noise_injector.cli", "--version"], capture_output=True, text=True)
    assert __version__ in out.stdout + out.stderr


def test_list_families():
    rc = main(["list-families"])
    assert rc == 0


def test_inject_family_sweep(tmp_path, capsys):
    from .fixtures import make_fake_nuscenes_root

    root = make_fake_nuscenes_root(tmp_path)
    out = tmp_path / "offspring"
    rc = main([
        "inject", "--type", "nuscenes", "--root", str(root),
        "--family", "fog", "--severities", "1-2", "--scenarios", "1", "--out", str(out),
    ])
    assert rc == 0
    assert (out / "nuscenes" / "sceneA" / "fog_s1" / "manifest.json").exists()
    assert (out / "nuscenes" / "sceneA" / "fog_s2" / "manifest.json").exists()


def test_inject_recipe_file(tmp_path):
    from .fixtures import make_fake_carla_root

    root = make_fake_carla_root(tmp_path)
    recipe = tmp_path / "r.yaml"
    recipe.write_text("name: test_mix\nops:\n  - family: rain\n    severity: 2\n  - family: bbox_jitter\n    severity: 1\n")
    out = tmp_path / "off"
    rc = main([
        "inject", "--type", "carla", "--root", str(root),
        "--recipe", str(recipe), "--scenarios", "all", "--out", str(out),
    ])
    assert rc == 0
    m = __import__("json").load(open(out / "carla" / "Town01/ControlLoss/Route0_Rep0" / "test_mix" / "manifest.json"))
    assert len(m["recipe"]["ops"]) == 2


def test_inject_waymo(tmp_path):
    from .fixtures import make_fake_waymo_root

    root = make_fake_waymo_root(tmp_path)
    out = tmp_path / "off"
    rc = main([
        "inject", "--type", "waymo", "--root", str(root),
        "--family", "missed_detections", "--severities", "3", "--scenarios", "1", "--out", str(out),
    ])
    assert rc == 0
    assert (out / "waymo" / "seg_000" / "missed_detections_s3" / "manifest.json").exists()


def test_harness_cli(tmp_path):
    from .fixtures import make_fake_harness

    root = make_fake_harness(tmp_path)
    out = tmp_path / "off"
    rc = main(["harness", "--root", str(root), "--op", "vocab_drop", "--strength", "2", "--out", str(out)])
    assert rc == 0
    assert (out / "manifest.json").exists()


def test_inject_bad_family_fails():
    with pytest.raises(SystemExit):
        main(["inject", "--type", "nuscenes", "--root", "/tmp/definitely-missing", "--family", "nope", "--out", "/tmp/x"])
