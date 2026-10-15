"""Batch presets: expand a preset YAML into the full (family, severity, pattern, camera) work list."""

from __future__ import annotations

from pathlib import Path

import yaml

from .importance import camera_priorities
from .spec import ALL_FAMILIES, PATTERNS


def load_preset(path: str | Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "entries" not in raw:
        raise ValueError(f"preset must be a mapping with 'entries': {path}")
    return raw


def expand_preset(preset: dict, scenarios: list[dict]) -> list[dict]:
    """scenarios: [{'id':..., 'description':..., 'cameras': [...]}] in user batch order.
    Returns work items: [{'sid','sc','family','severity','pattern','force_camera'(opt),'child_name'}]."""
    work = []
    for entry in preset["entries"]:
        fam = entry["family"]
        if fam not in ALL_FAMILIES:
            raise ValueError(f"preset entry: unknown family {fam}")
        pattern = entry.get("pattern") or ALL_FAMILIES[fam].default_pattern
        if pattern not in PATTERNS:
            raise ValueError(f"preset entry {fam}: bad pattern {pattern}")
        severities = entry["severities"]
        if isinstance(severities, int):
            severities = [severities]
        per_camera = bool(entry.get("per_camera")) or pattern in ("camera_scoped", "flaky_camera")
        if per_camera:
            for sc in scenarios:
                cams = sorted(sc["cameras"])
                ordered = camera_priorities(sc["description"], cams) if entry.get("importance", True) else cams
                for cam in ordered:
                    for s in severities:
                        work.append({
                            "sid": sc["id"], "sc": sc, "family": fam, "severity": s,
                            "pattern": pattern, "force_camera": cam,
                            "child_name": f"{fam}_{pattern}_{cam}_s{s}",
                        })
        else:
            for sc in scenarios:
                for s in severities:
                    name = f"{fam}_{pattern}_s{s}" if entry.get("pattern") else f"{fam}_s{s}"
                    work.append({
                        "sid": sc["id"], "sc": sc, "family": fam, "severity": s,
                        "pattern": pattern, "force_camera": None, "child_name": name,
                    })
    return work
