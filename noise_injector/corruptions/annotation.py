"""Layer B: annotation corruption families (operating on key_object_infos dicts)."""

from __future__ import annotations

import re

import numpy as np

from ..spec import CONFUSION_SETS

_OBJ_KEY_RE = re.compile(r"^<c(\d+)")


def _all_objects(koi: dict):
    return list(koi.items())


def _bbox_of(info: dict):
    b = info.get("2d_bbox")
    if b is None:
        return None
    if isinstance(b[0], (list, tuple)):
        return [b[0][0], b[0][1], b[1][0], b[1][1]]
    return list(b[:4])


def _set_bbox(info: dict, flat, style: str):
    if style == "pairs":
        info["2d_bbox"] = [[flat[0], flat[1]], [flat[2], flat[3]]]
    else:
        info["2d_bbox"] = flat


def _bbox_style(sample_bbox) -> str:
    return "pairs" if isinstance(sample_bbox[0], (list, tuple)) else "flat"


def missed_detections(koi: dict, params, rng):
    rate = params["rate"]
    dropped = []
    for key in list(koi.keys()):
        if rng.random() < rate:
            del koi[key]
            dropped.append(key)
    return dropped


def phantom_objects(koi: dict, params, rng):
    rate = params["rate"]
    n_existing = len(koi)
    n_add = max(0, int(round(rate * n_existing + (1 if rng.random() < rate else 0))))
    if n_existing == 0 and rng.random() >= rate:
        n_add = 0
    added = []
    style = "flat"
    if n_existing:
        style = _bbox_style(next(iter(koi.values()))["2d_bbox"])
    for i in range(n_add):
        if n_existing:
            src = koi[next(iter(koi))]
            b = _bbox_of(src)
        else:
            b = [40, 40, 120, 140]
        jittered = [
            max(0.0, b[0] + rng.normal(0, 30)),
            max(0.0, b[1] + rng.normal(0, 30)),
            max(0.0, b[2] + rng.normal(0, 30)),
            max(0.0, b[3] + rng.normal(0, 30)),
        ]
        if jittered[2] < jittered[0] + 5:
            jittered[2] = jittered[0] + 5
        if jittered[3] < jittered[1] + 5:
            jittered[3] = jittered[1] + 5
        cats = list(CONFUSION_SETS.keys())
        cat = cats[int(rng.integers(0, len(cats)))]
        new_key = f"<p{i},{cat}>"
        koi[new_key] = {
            "Category": cat,
            "Status": None,
            "Visual_description": "phantom object",
            "2d_bbox": [[jittered[0], jittered[1]], [jittered[2], jittered[3]]]
            if style == "pairs"
            else jittered,
        }
        added.append(new_key)
    return added


def bbox_jitter(koi: dict, params, rng):
    sigma = params["sigma_px"]
    touched = []
    for key, info in koi.items():
        b = _bbox_of(info)
        if b is None:
            continue
        style = _bbox_style(info["2d_bbox"])
        flat = [max(0.0, v + rng.normal(0, sigma)) for v in b]
        if flat[2] < flat[0] + 2:
            flat[2] = flat[0] + 2
        if flat[3] < flat[1] + 2:
            flat[3] = flat[1] + 2
        _set_bbox(info, flat, style)
        touched.append(key)
    return touched


def class_confusion(koi: dict, params, rng):
    rate = params["rate"]
    swapped = []
    for key, info in koi.items():
        cat = info.get("Category")
        options = CONFUSION_SETS.get(cat)
        if not options or rng.random() >= rate:
            continue
        info["Category"] = options[int(rng.integers(0, len(options)))]
        swapped.append(f"{key}:{cat}->{info['Category']}")
    return swapped


def status_flip(koi: dict, params, rng):
    rate = params["rate"]
    flipped = []
    flip_map = {"Moving": "Stationary", "Stationary": "Moving"}
    for key, info in koi.items():
        st = info.get("Status")
        if st in flip_map and rng.random() < rate:
            info["Status"] = flip_map[st]
            flipped.append(key)
    return flipped


def max_det_truncation(koi: dict, params, rng):
    cap = params["cap"]
    keys = list(koi.keys())
    if len(keys) <= cap:
        return []
    removed = []
    order = list(rng.permutation(len(keys)))
    keep = set(order[:cap])
    for i in range(len(keys)):
        if i not in keep:
            del koi[keys[i]]
            removed.append(keys[i])
    return removed


ANNOTATION_CORRUPTIONS = {
    "missed_detections": missed_detections,
    "phantom_objects": phantom_objects,
    "bbox_jitter": bbox_jitter,
    "class_confusion": class_confusion,
    "status_flip": status_flip,
    "max_det_truncation": max_det_truncation,
}


def apply_annotation_op(name: str, koi: dict, params: dict, rng: np.random.Generator):
    fn = ANNOTATION_CORRUPTIONS.get(name)
    if fn is None:
        raise KeyError(f"unknown annotation family: {name}")
    return fn(koi, params, rng)
