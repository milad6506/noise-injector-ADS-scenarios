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
        if not self.vqas.exists() and not self._pdm_dir():
            raise FileNotFoundError(
                f"CARLA layout not found under {self.root}: need {vqas_dir}/ and/or PDM_Lite_Carla_LB2/"
            )

    def list_scenarios(self) -> list[str]:
        if not self.vqas.exists():
            return self._list_scenarios_images_only()
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

    def _pdm_dir(self) -> Path | None:
        pdm = self.root / "PDM_Lite_Carla_LB2"
        return pdm if pdm.exists() else None

    def _iter_rgb_routes(self, pdm: Path):
        for rgb_dir in sorted(pdm.rglob("rgb")):
            if not any(rgb_dir.glob("*.jpg")):
                continue
            rel = rgb_dir.relative_to(pdm)
            parts = [p for p in rel.parts if p not in ("data", "rgb")]
            if parts:
                yield "/".join(parts), rgb_dir

    def _list_scenarios_images_only(self) -> list[str]:
        pdm = self._pdm_dir()
        if not pdm:
            raise FileNotFoundError(
                f"no {self.vqas} and no PDM_Lite_Carla_LB2/ images under {self.root}"
            )
        return [sid for sid, _ in self._iter_rgb_routes(pdm)]

    def get_scenario(self, scenario_id: str) -> Scenario:
        route_dir = self.vqas / scenario_id
        if not route_dir.is_dir():
            return self._get_scenario_images_only(scenario_id)

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

    def _get_scenario_images_only(self, scenario_id: str) -> Scenario:
        pdm = self._pdm_dir()
        if not pdm:
            raise KeyError(f"scenario not found: {scenario_id}")
        for sid, rgb_dir in self._iter_rgb_routes(pdm):
            if sid == scenario_id:
                route_dir = rgb_dir.parent
                rgb = rgb_dir
                break
        else:
            raise KeyError(f"scenario not found (no rgb dir): {scenario_id}")
        if not rgb.is_dir():
            raise KeyError(f"scenario not found (no rgb dir): {scenario_id}")
        frames = []
        for img in sorted(rgb.glob("*.jpg"), key=lambda p: p.stem):
            frames.append(
                Frame(
                    frame_id=img.stem,
                    cameras={"CAM_FRONT": CameraImage(img, "CAM_FRONT")},
                    key_object_infos={},
                    qa={},
                    metadata={"source": "images-only"},
                )
            )
        return Scenario(
            scenario_id=scenario_id,
            source=self.name,
            description="CARLA images-only scenario (QA annotations not fetched)",
            frames=frames,
            bbox_style="pairs",
            metadata={"route_dir": str(route_dir.relative_to(self.root)), "images_only": True},
        )
