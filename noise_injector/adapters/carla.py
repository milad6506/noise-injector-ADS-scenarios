"""DriveLM-CARLA adapter: drivelm_carla_vqas/{Town}/{Scenario}/{Route}/frame.json files."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .base import Adapter, CameraImage, Frame, Scenario

_VQAS_DIR = "drivelm_carla_vqas"


class CarlaAdapter(Adapter):
    name = "carla"

    def __init__(self, root: Path, vqas_dir: str = _VQAS_DIR, **kwargs):
        super().__init__(root)
        self.vqas = self.root / vqas_dir
        if not self.vqas.exists():
            raise FileNotFoundError(f"CARLA VQAs directory not found: {self.vqas}")

    def list_scenarios(self) -> list[str]:
        out = []
        for town in sorted(self.vqas.iterdir()):
            if not town.is_dir():
                continue
            for scen in sorted(town.iterdir()):
                if scen.is_dir():
                    for route in sorted(scen.iterdir()):
                        if route.is_dir() and any(route.glob("*.json")):
                            out.append(f"{town.name}/{scen.name}/{route.name}")
        return out

    def get_scenario(self, scenario_id: str) -> Scenario:
        route_dir = self.vqas / scenario_id
        if not route_dir.is_dir():
            raise KeyError(f"scenario not found: {scenario_id}")

        def frame_num(p: Path):
            m = re.search(r"(\d+)", p.stem)
            return int(m.group(1)) if m else 0

        files = sorted(route_dir.glob("*.json"), key=frame_num)
        frames = []
        for fp in files:
            with open(fp) as f:
                d = json.load(f)
            cameras = {}
            for cam, rel in d.get("image_paths", {}).items():
                if isinstance(rel, str):
                    cameras[cam] = CameraImage(self.root / rel, cam)
                else:
                    cameras[cam] = CameraImage(None, cam)
            frames.append(
                Frame(
                    frame_id=fp.stem,
                    cameras=cameras,
                    key_object_infos=d.get("key_object_infos", {}),
                    qa=d.get("QA", {}),
                    metadata={"file": fp.name},
                )
            )
        return Scenario(
            scenario_id=scenario_id,
            source=self.name,
            description=None,
            frames=frames,
            bbox_style="pairs",
            metadata={"route_dir": str(route_dir.relative_to(self.root))},
        )
