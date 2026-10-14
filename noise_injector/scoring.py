"""Scoring: objective perception metrics, judge plumbing, resilience aggregation."""

from __future__ import annotations

import numpy as np

_GT_ALIASES = {
    "Vehicle": {"Vehicle", "vehicle", "car", "Car", "truck", "bus", "van", "ConstructionVehicle"},
    "Pedestrian": {"Pedestrian", "pedestrian", "person", "Human.pedestrian.adult"},
    "Cyclist": {"Cyclist", "cyclist", "bicycle", "Bicycle", "motorcycle"},
    "TrafficSign": {"TrafficSign", "TrafficSignElement", "traffic light", "TrafficLight",
                    "stop sign", "sign", "Barrier", "traffic cone"},
}


def _canon(cls: str) -> str:
    for canon, aliases in _GT_ALIASES.items():
        if cls in aliases:
            return canon
    return "Unknown"


def perception_metrics(gt_sets: list[set], pred_sets: list[set], iou_threshold: float = 0.5) -> dict:
    """Presence-based metrics per frame (camera-agnostic class sets)."""
    n = len(gt_sets)
    if n == 0:
        return {"presence_recall": 0.0, "presence_fp_rate": 0.0, "class_accuracy": 0.0}
    hits = 0
    gt_total = 0
    fp_frames = 0
    correct_class = 0
    total_pred = 0
    for gt, pred in zip(gt_sets, pred_sets):
        gt_c = {_canon(c) for c in gt if _canon(c) != "Unknown"}
        pred_c = {_canon(c) for c in pred if _canon(c) != "Unknown"}
        gt_total += len(gt_c)
        hits += len(gt_c & pred_c)
        if pred_c - gt_c:
            fp_frames += 1
        total_pred += len(pred_c)
        correct_class += len(gt_c & pred_c)
    return {
        "presence_recall": round(hits / max(1, gt_total), 3),
        "presence_fp_rate": round(fp_frames / n, 3),
        "class_accuracy": round(correct_class / max(1, total_pred), 3),
    }


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def resilience_summary(results: list[dict]) -> dict:
    """results: [{'round': int, 'family': str|None(clean), 'severity': int, 'target': str, 'score': float}]"""
    by_target: dict[str, dict[str, float]] = {}
    clean_scores: dict[str, float] = {}
    for r in results:
        t = r["target"]
        if r.get("family") is None:
            clean_scores[t] = r["score"]
    retentions = []
    for r in results:
        if r.get("family") is None:
            continue
        t = r["target"]
        base = clean_scores.get(t)
        if not base:
            continue
        ret = round(r["score"] / base, 3) if base else None
        by_target.setdefault(t, []).append(ret if ret is not None else 0.0)
        retentions.append({"target": t, "family": r["family"], "severity": r["severity"],
                           "clean": base, "noisy": r["score"], "retention": ret})
    summary = {}
    for t, rets in by_target.items():
        valid = [x for x in rets if x is not None]
        summary[t] = {
            "clean_score": clean_scores.get(t, 0.0),
            "mean_retention": round(float(np.mean(valid)), 3) if valid else None,
            "min_retention": round(float(np.min(valid)), 3) if valid else None,
        }
    return {"per_target": summary, "retentions": retentions}
