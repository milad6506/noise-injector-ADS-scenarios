import json

import numpy as np
import pytest
import yaml

from noise_injector.adapters.base import get_adapter
from noise_injector.render import load_recipe, make_recipe, render_scenario, select_scenarios
from noise_injector.spec import ALL_FAMILIES, HARNESS_OPS, IMAGE_FAMILIES
from noise_injector.temporal import activation_mask, derive_seed


def make_fake_nuscenes_root(tmp_path):
    import cv2

    root = tmp_path / "drivelm"
    (root / "nuscenes" / "samples" / "CAM_FRONT").mkdir(parents=True)
    (root / "nuscenes" / "samples" / "CAM_BACK").mkdir(parents=True)
    rng = np.random.default_rng(0)
    frames = {}
    for t in range(4):
        fid = f"f{t}"
        for cam in ("CAM_FRONT", "CAM_BACK"):
            img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "nuscenes" / "samples" / cam / f"img_{t}_{cam}.jpg"), img)
        frames[fid] = {
            "key_object_infos": {
                "<c1,CAM_FRONT,10.0,20.0>": {"Category": "Vehicle", "Status": "Moving",
                                              "Visual_description": "white van", "2d_bbox": [5, 5, 30, 30]},
                "<c2,CAM_BACK,5.0,5.0>": {"Category": "Pedestrian", "Status": "Stationary",
                                           "Visual_description": "person", "2d_bbox": [10, 10, 20, 40]},
            },
            "QA": {
                "perception": [{"Q": "Is there a vehicle? <c1,CAM_FRONT,10.0,20.0>", "A": "Yes.", "C": None}],
                "planning": [{"Q": "What should the ego do?", "A": "Decelerate gradually without braking.", "C": None}],
            },
            "image_paths": {
                "CAM_FRONT": f"../nuscenes/samples/CAM_FRONT/img_{t}_CAM_FRONT.jpg",
                "CAM_BACK": f"../nuscenes/samples/CAM_BACK/img_{t}_CAM_BACK.jpg",
            },
        }
    scenes = {
        "sceneA": {"scene_description": "Ego drives straight.", "key_frames": frames},
        "sceneB": {"scene_description": "Ego turns left.", "key_frames": frames},
    }
    with open(root / "v1_1_train_nus.json", "w") as f:
        json.dump(scenes, f)
    return root


def make_fake_carla_root(tmp_path):
    import cv2

    root = tmp_path / "carla_root"
    route = root / "drivelm_carla_vqas" / "Town01" / "ControlLoss" / "Route0_Rep0"
    rgb = root / "PDM_Lite_Carla_LB2" / "Town01" / "data" / "ControlLoss" / "Route0_Rep0" / "rgb"
    route.mkdir(parents=True)
    rgb.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for t in range(3):
        fid = f"{t:04d}"
        cv2.imwrite(str(rgb / f"{fid}.jpg"), rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
        with open(route / f"{fid}.json", "w") as f:
            json.dump(
                {
                    "key_object_infos": {
                        "<c1,CAM_FRONT,30.0,25.0>": {"Category": "Vehicle", "Status": None,
                                                      "Visual_description": "red car",
                                                      "2d_bbox": [[10, 10], [40, 40]]},
                    },
                    "QA": {"perception": [{"Q": "What is this? <c1,CAM_FRONT,30.0,25.0>", "A": "A red car.", "C": None}]},
                    "image_paths": {
                        "CAM_FRONT": f"PDM_Lite_Carla_LB2/Town01/data/ControlLoss/Route0_Rep0/rgb/{fid}.jpg",
                        "CAM_BACK": None,
                    },
                },
                f,
            )
    return root


def make_fake_waymo_root(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "waymo_root"
    (root / "camera_box").mkdir(parents=True)
    rows = []
    for cam in (1, 2):
        for ts in (100, 200, 300, 400):
            for oid_i in range(3):
                rows.append({
                    "index": f"seg1;{ts};{oid_i}",
                    "key.segment_context_name": "seg1",
                    "key.frame_timestamp_micros": ts,
                    "key.camera_name": cam,
                    "key.camera_object_id": f"obj-{oid_i}-{cam}",
                    "[CameraBoxComponent].box.center.x": 100.0 + oid_i * 50,
                    "[CameraBoxComponent].box.center.y": 200.0,
                    "[CameraBoxComponent].box.size.x": 60.0,
                    "[CameraBoxComponent].box.size.y": 40.0,
                    "[CameraBoxComponent].type": 1,
                    "[CameraBoxComponent].difficulty_level.detection": None,
                    "[CameraBoxComponent].difficulty_level.tracking": None,
                })
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "camera_box" / "seg_000.parquet")
    return root


def make_fake_harness(tmp_path):
    root = tmp_path / "harness"
    (root / "prompts").mkdir(parents=True)
    (root / "prompts" / "simple_prompt.txt").write_text(
        "You are a driving agent.\n\nRULES: Stop for school buses. Yield to pedestrians.\n\nOUTPUT: JSON."
    )
    (root / "cot_prompt.txt").write_text("Think step by step.\n\n1. Perceive.\n\n2. Predict.\n\n3. Plan.")
    (root / "config.json").write_text(
        json.dumps({
            "arms": [{"key": "C", "detector_vocab": ["car", "pedestrian", "bus"], "detector_conf": 0.25, "max_tokens": 1600}],
            "generation": {"temperature": 0.2, "max_tokens": 4096},
            "images": {"max_px": 768, "jpeg_quality": 85},
        })
    )
    (root / "runs").mkdir()
    (root / "runs" / "x.json").write_text("{}")
    (root / "model.pt").write_bytes(b"\x00")
    return root
