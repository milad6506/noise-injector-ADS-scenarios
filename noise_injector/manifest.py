"""Manifest construction and serialization for offspring scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from . import __version__


def build_manifest(
    scenario_id: str,
    adapter_name: str,
    root: str,
    recipe_dict: dict,
    master_seed: int,
    injected_events: list[dict],
    output_files: list[str],
) -> dict:
    return {
        "schema_version": 1,
        "tool": "noise-injector",
        "tool_version": __version__,
        "parent": {
            "scenario_id": scenario_id,
            "adapter": adapter_name,
            "root": str(root),
        },
        "recipe": recipe_dict,
        "master_seed": master_seed,
        "injected_events": injected_events,
        "output_files": output_files,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
