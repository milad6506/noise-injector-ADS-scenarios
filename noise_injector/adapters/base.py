"""Scenario model shared by all adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CameraImage:
    path: Optional[Path]
    camera: str

    @property
    def has_image(self) -> bool:
        return self.path is not None and Path(self.path).exists()


@dataclass
class Frame:
    frame_id: str
    cameras: dict[str, CameraImage]
    key_object_infos: dict
    qa: dict
    metadata: dict = field(default_factory=dict)


@dataclass
class Scenario:
    scenario_id: str
    source: str
    description: Optional[str]
    frames: list[Frame]
    bbox_style: str = "flat"
    metadata: dict = field(default_factory=dict)


def get_adapter(name: str, root: Path, **kwargs) -> "Adapter":
    if name == "nuscenes":
        from .nuscenes import NuScenesAdapter

        return NuScenesAdapter(root, **kwargs)
    if name == "carla":
        from .carla import CarlaAdapter

        return CarlaAdapter(root, **kwargs)
    if name == "waymo":
        from .waymo import WaymoAdapter

        return WaymoAdapter(root, **kwargs)
    raise KeyError(f"unknown adapter type: {name} (choose nuscenes|carla|waymo)")


class Adapter:
    name = "base"

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root not found: {self.root}")

    def list_scenarios(self) -> list[str]:
        raise NotImplementedError

    def get_scenario(self, scenario_id: str) -> Scenario:
        raise NotImplementedError
