import json
import subprocess
import zipfile

import numpy as np
import pytest

from noise_injector.fetch import download, fetch_link
from noise_injector.report import collect, tree_html, write_report
from noise_injector.render import make_recipe, render_scenario
from noise_injector.adapters.base import get_adapter

from .fixtures import make_fake_carla_root, make_fake_harness, make_fake_nuscenes_root


def make_zip(tmp_path):
    import cv2

    pkg = tmp_path / "pkgsrc"
    (pkg / "scenarios").mkdir(parents=True)
    cv2.imwrite(str(pkg / "scenarios" / "a.jpg"), np.zeros((10, 10, 3), np.uint8))
    with open(pkg / "scenarios" / "meta.json", "w") as f:
        json.dump({"name": "pkg"}, f)
    with open(pkg / "scenarios" / "bulk.bin", "wb") as f:
        f.write(np.random.default_rng(0).integers(0, 255, 64 * 1024, dtype=np.uint8).tobytes())
    arch = tmp_path / "package.zip"
    with zipfile.ZipFile(arch, "w") as z:
        for p in pkg.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(pkg))
    return arch


def test_download_from_file_url(tmp_path):
    arch = make_zip(tmp_path)
    dest = tmp_path / "out" / "package.zip"
    download(arch.as_uri(), dest, 0.1)
    assert dest.exists() and dest.stat().st_size > 0


def test_fetch_link_extracts_zip(tmp_path):
    arch = make_zip(tmp_path)
    out = tmp_path / "fetched"
    fetch_link(arch.as_uri(), out, 0.1)
    assert (out / "link" / "package" / "scenarios" / "meta.json").exists()


def test_fetch_link_size_guard(tmp_path):
    arch = make_zip(tmp_path)
    out = tmp_path / "fetched"
    with pytest.raises(ValueError, match="max-gb"):
        download(arch.as_uri(), out / "x.zip", 0.000001)


def test_carla_images_only_fallback(tmp_path):
    import cv2

    root = tmp_path / "carla_root"
    rgb = root / "PDM_Lite_Carla_LB2" / "Town02" / "data" / "ControlLoss" / "Route0_Rep0" / "rgb"
    rgb.mkdir(parents=True)
    for t in range(3):
        cv2.imwrite(str(rgb / f"{t:04d}.jpg"), np.zeros((20, 30, 3), np.uint8))
    ad = get_adapter("carla", root)
    ids = ad.list_scenarios()
    assert ids == ["Town02/ControlLoss/Route0_Rep0"]
    sc = ad.get_scenario(ids[0])
    assert len(sc.frames) == 3
    assert sc.frames[0].key_object_infos == {}
    assert sc.metadata["images_only"]


def test_harness_from_git_link(tmp_path):
    harness_src = make_fake_harness(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=harness_src)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], cwd=harness_src)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=harness_src)
    link = f"file://{harness_src}/.git"

    out = tmp_path / "harness_out"
    from noise_injector.fetch import fetch_harness_link

    fetched = fetch_harness_link(link, tmp_path / "fetched")
    assert (fetched / "config.json").exists()
    assert (fetched / "prompts" / "simple_prompt.txt").exists()


def test_report_tree_view(tmp_path):
    root = make_fake_nuscenes_root(tmp_path)
    ad = get_adapter("nuscenes", root)
    sc = ad.get_scenario("sceneA")
    off = tmp_path / "offspring"
    for fam in ("fog", "awgn"):
        for sev in (1, 2):
            recipe = make_recipe(fam, fam, [sev])[0]
            render_scenario(sc, recipe, off / fam + f"_s{sev}" if False else off / f"{fam}_s{sev}",
                            seed=1, root=str(root))
    runs = collect(off)
    assert len(runs) == 4
    html = tree_html(runs)
    assert "root-node" in html and "sceneA" in html
    assert html.count("child leaf") == 4
    report = write_report(off, tmp_path / "report.html")
    content = report.read_text()
    assert "Scenario tree" in content
    assert "/Users" not in content
