"""Temporal pattern engine: deterministic per-frame (and camera) activation masks."""

from __future__ import annotations

import numpy as np

from .spec import PATTERNS


def activation_mask(
    pattern: str,
    n_frames: int,
    rng: np.random.Generator,
    severity: int = 1,
    n_cameras: int = 1,
) -> dict:
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern: {pattern} (choose from {PATTERNS})")

    if pattern == "persistent":
        return {"frames": [True] * n_frames, "camera": None}

    if pattern == "transient":
        idx = int(rng.integers(0, max(1, n_frames)))
        return {"frames": [i == idx for i in range(n_frames)], "camera": None}

    if pattern == "burst":
        k = max(1, min(n_frames, severity))
        start = int(rng.integers(0, max(1, n_frames - k + 1)))
        return {"frames": [start <= i < start + k for i in range(n_frames)], "camera": None}

    if pattern == "intermittent":
        p_stay_on = 0.55 + 0.08 * severity
        p_on = 0.25 + 0.12 * severity
        mask = []
        on = bool(rng.random() < p_on)
        for _ in range(n_frames):
            if on:
                mask.append(True)
                if rng.random() > p_stay_on:
                    on = False
            else:
                mask.append(False)
                if rng.random() < p_on * 0.5:
                    on = True
        return {"frames": mask, "camera": None}

    if pattern == "camera_scoped":
        cam = int(rng.integers(0, max(1, n_cameras)))
        return {"frames": [True] * n_frames, "camera": cam}

    raise ValueError(f"unhandled pattern: {pattern}")


def derive_seed(master_seed: int, *parts: str) -> int:
    import hashlib

    h = hashlib.sha256(f"{master_seed}:{':'.join(parts)}".encode()).digest()
    return int.from_bytes(h[:8], "little")
