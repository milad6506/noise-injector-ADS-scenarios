"""Importance sampling over cameras: which views matter for this scenario's maneuver.

Driven by the DriveLM scene description: a vehicle moving forward has its front
views as primary evidence; backing favors rear views; turns favor the turn side.
Used to (a) pick fault-target cameras with importance-weighted probability and
(b) order per-camera offspring creation so important views are generated first
when the user caps the batch size.
"""

from __future__ import annotations

import re

MANEUVER_PATTERNS = {
    "backing": [
        r"\bback(s|ing|ed)?\b", r"\breverse\b", r"\bbacking up\b", r"\bpark(ing|ed)?\b",
        r"\bmaneuver(s|ing)?\b.*\b(park|back|reverse)\b", r"\bexit(ing)? (a |the )?parking\b",
        r"\bpull(ing)? (out|back)\b",
    ],
    "turning_left": [r"\bturn(s|ing|ed)? left\b", r"\bleft turn\b", r"\bleft onto\b", r"\bveer(s|ing)? left\b"],
    "turning_right": [r"\bturn(s|ing|ed)? right\b", r"\bright turn\b", r"\bright onto\b", r"\bveer(s|ing)? right\b"],
    "forward": [
        r"\bproceed(s|ing)?\b", r"\bgo(es)? (straight|forward|ahead)\b", r"\bgoing straight\b",
        r"\bdriv(e|es|ing|en)\b.*\b(straight|forward|ahead|along)\b", r"\bcontinu(e|es|ing)\b",
        r"\bmove(s|ing)? (forward|ahead)\b", r"\btravel(s|ing)?\b", r"\bapproach(es|ing)?\b",
        r"\bdrive(s)? through\b",
    ],
}

CAMERA_WEIGHTS = {
    "forward": {"CAM_FRONT": 1.0, "CAM_FRONT_LEFT": 0.8, "CAM_FRONT_RIGHT": 0.8,
                "CAM_BACK": 0.3, "CAM_BACK_LEFT": 0.3, "CAM_BACK_RIGHT": 0.3},
    "backing": {"CAM_BACK": 1.0, "CAM_BACK_LEFT": 0.85, "CAM_BACK_RIGHT": 0.85,
                "CAM_FRONT": 0.35, "CAM_FRONT_LEFT": 0.25, "CAM_FRONT_RIGHT": 0.25},
    "turning_left": {"CAM_FRONT_LEFT": 1.0, "CAM_FRONT": 0.9, "CAM_BACK_LEFT": 0.6,
                     "CAM_FRONT_RIGHT": 0.4, "CAM_BACK": 0.4, "CAM_BACK_RIGHT": 0.3},
    "turning_right": {"CAM_FRONT_RIGHT": 1.0, "CAM_FRONT": 0.9, "CAM_BACK_RIGHT": 0.6,
                      "CAM_FRONT_LEFT": 0.4, "CAM_BACK": 0.4, "CAM_BACK_LEFT": 0.3},
    "unknown": {"CAM_FRONT": 0.7, "CAM_FRONT_LEFT": 0.5, "CAM_FRONT_RIGHT": 0.5,
                "CAM_BACK": 0.5, "CAM_BACK_LEFT": 0.4, "CAM_BACK_RIGHT": 0.4},
}


def detect_maneuver(description: str | None) -> str:
    low = (description or "").lower()
    if not low:
        return "unknown"
    for maneuver in ("backing", "turning_left", "turning_right"):
        if any(re.search(p, low) for p in MANEUVER_PATTERNS[maneuver]):
            return maneuver
    for p in MANEUVER_PATTERNS["forward"]:
        if re.search(p, low):
            return "forward"
    return "unknown"


def camera_priorities(description: str | None, cameras: list[str]) -> list[str]:
    maneuver = detect_maneuver(description)
    weights = CAMERA_WEIGHTS[maneuver]
    return sorted(
        cameras,
        key=lambda c: (-(weights.get(c, 0.4)), c),
    )


def camera_weights(description: str | None, cameras: list[str]) -> dict[str, float]:
    maneuver = detect_maneuver(description)
    weights = CAMERA_WEIGHTS[maneuver]
    picked = {c: weights.get(c, 0.4) for c in cameras}
    total = sum(picked.values()) or 1.0
    return {c: w / total for c, w in picked.items()}


def maneuver_of(description: str | None) -> str:
    return detect_maneuver(description)
