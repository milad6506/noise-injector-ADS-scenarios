"""DriveLM-nuScenes adapter: v1_1_train_nus.json + nuscenes/samples/{CAM}/."""

from __future__ import annotations

import json
from pathlib import Path

from .base import Adapter, CameraImage, Frame, Scenario

_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class NuScenesAdapter(Adapter):
    name = "nuscenes"

    def __init__(self, root: Path, json_name: str = "v1_1_train_nus.json", **kwargs):
        super().__init__(root)
        self.json_path = self.root / json_name
        if not self.json_path.exists():
            raise FileNotFoundError(f"DriveLM-nuScenes JSON not found: {self.json_path}")
        with open(self.json_path) as f:
            self.data = json.load(f)
        self.samples_dir = self.root / "nuscenes" / "samples"

    def list_scenarios(self) -> list[str]:
        return list(self.data.keys())

    def get_scenario(self, scenario_id: str) -> Scenario:
        if scenario_id not in self.data:
            raise KeyError(f"scene not found: {scenario_id}")
        sc = self.data[scenario_id]
        frames = []
        for fid, fr in sc.get("key_frames", {}).items():
            cameras = {}
            for cam, rel in fr.get("image_paths", {}).items():
                if not isinstance(rel, str):
                    cameras[cam] = CameraImage(None, cam)
                    continue
                path = self.samples_dir / cam / Path(rel).name
                cameras[cam] = CameraImage(path, cam)
            frame = Frame(
                frame_id=fid,
                cameras=cameras,
                key_object_infos=fr.get("key_object_infos", {}),
                qa=fr.get("QA", {}),
                metadata={"scene_index": fid},
            )
            frames.append(frame)
        return Scenario(
            scenario_id=scenario_id,
            source=self.name,
            description=sc.get("scene_description"),
            frames=frames,
            bbox_style="flat",
            metadata={"json": self.json_path.name},
        )
