"""Waymo adapter: camera_box parquet segments (Waymo Open Dataset v2.x).

Builds annotation-only scenarios (2D boxes, no images) — image corruption ops
are skipped automatically; annotation + pattern + QA layers apply.
"""

from __future__ import annotations

from pathlib import Path

from .base import Adapter, CameraImage, Frame, Scenario
from ..spec import WAYMO_TYPE_MAP

_DEFAULT_BOX_DIR = "camera_box"


class WaymoAdapter(Adapter):
    name = "waymo"

    def __init__(self, root: Path, box_dir: str = _DEFAULT_BOX_DIR, max_frames: int = 60, **kwargs):
        super().__init__(root)
        self.box_dir = self.root / box_dir if (self.root / box_dir).is_dir() else self.root
        self.parquet_files = sorted(self.box_dir.glob("*.parquet"))
        if not self.parquet_files:
            raise FileNotFoundError(f"no camera_box parquet files under: {self.box_dir}")
        self.max_frames = max_frames
        try:
            import pyarrow.parquet as pq
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Waymo adapter requires pyarrow + pandas: pip install 'noise-injector[waymo]'"
            ) from e
        self._pd = pd
        self._pq = pq

    def list_scenarios(self) -> list[str]:
        return [p.stem for p in self.parquet_files]

    def get_scenario(self, scenario_id: str) -> Scenario:
        path = self.box_dir / f"{scenario_id}.parquet"
        if not path.exists():
            raise KeyError(f"segment not found: {scenario_id}")
        df = self._pd.read_parquet(path)
        cam_col = "key.camera_name"
        ts_col = "key.frame_timestamp_micros"
        seg_col = "key.segment_context_name"
        oid_col = "key.camera_object_id"
        cols = {
            "cx": "[CameraBoxComponent].box.center.x",
            "cy": "[CameraBoxComponent].box.center.y",
            "sx": "[CameraBoxComponent].box.size.x",
            "sy": "[CameraBoxComponent].box.size.y",
            "type": "[CameraBoxComponent].type",
        }
        segments = sorted(df[seg_col].unique()) if seg_col in df else [scenario_id]
        seg = segments[0]

        ts_values = sorted(df[ts_col].unique())
        if self.max_frames and len(ts_values) > self.max_frames:
            ts_values = ts_values[: self.max_frames]

        frames = []
        for ts in ts_values:
            sub = df[df[ts_col] == ts]
            koi = {}
            cams_present = set()
            for _, row in sub.iterrows():
                cam = int(row[cam_col])
                cams_present.add(cam)
                oid = str(row[oid_col])
                cx, cy = float(row[cols["cx"]]), float(row[cols["cy"]])
                sx, sy = float(row[cols["sx"]]), float(row[cols["sy"]])
                bbox = [cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2]
                box_type = int(row[cols["type"]]) if row[cols["type"]] == row[cols["type"]] else 0
                koi[f"<c1,CAM{cam},{oid[:8]}>"] = {
                    "Category": WAYMO_TYPE_MAP.get(box_type, "Unknown"),
                    "Status": None,
                    "Visual_description": "",
                    "2d_bbox": bbox,
                }
            cameras = {f"WAYMO_CAM{c}": CameraImage(None, f"WAYMO_CAM{c}") for c in sorted(cams_present)}
            frames.append(
                Frame(
                    frame_id=str(ts),
                    cameras=cameras,
                    key_object_infos=koi,
                    qa={},
                    metadata={"timestamp_micros": int(ts), "camera_names": sorted(cams_present)},
                )
            )
        return Scenario(
            scenario_id=scenario_id,
            source=self.name,
            description=f"Waymo segment {seg} (camera_box annotations, no imagery)",
            frames=frames,
            bbox_style="flat",
            metadata={"segment": seg, "n_rows": int(len(df)), "images": False},
        )
