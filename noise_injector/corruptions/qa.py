"""Layer C: QA corruption families (operating on QA category dicts)."""

from __future__ import annotations

import re

import numpy as np

from ..spec import DRIVE_TOKEN_SWAPS

_CREF_RE = re.compile(r"<c(\d+)[,.]")

_CANNED_WRONG = [
    "No.",
    "Yes.",
    "The road is clear.",
    "A vehicle is on the left.",
    "There is a pedestrian crossing.",
    "The traffic light is red.",
    "The ego vehicle should accelerate.",
    "The ego vehicle should turn left.",
]


def _iter_qa(qa: dict):
    for cat, items in qa.items():
        if isinstance(items, list):
            for i, item in enumerate(items):
                yield cat, i, item


def qa_wrong_answer(qa: dict, params, rng):
    rate = params["rate"]
    changed = []
    pool = []
    for cat, i, item in _iter_qa(qa):
        if isinstance(item.get("A"), str):
            pool.append(item["A"])
    for cat, i, item in _iter_qa(qa):
        if not isinstance(item.get("A"), str) or rng.random() >= rate:
            continue
        choices = item.get("C")
        old = item["A"]
        if isinstance(choices, list) and len(choices) > 1:
            others = [c for c in choices if c != old]
            item["A"] = others[int(rng.integers(0, len(others)))]
        else:
            candidates = [a for a in _CANNED_WRONG + pool if a != old]
            if not candidates:
                continue
            item["A"] = candidates[int(rng.integers(0, len(candidates)))]
        changed.append({"cat": cat, "idx": i, "old": old, "new": item["A"]})
    return changed


def qa_answer_drift(qa: dict, params, rng):
    rate = params["rate"]
    changed = []
    for cat, i, item in _iter_qa(qa):
        ans = item.get("A")
        if not isinstance(ans, str) or rng.random() >= rate:
            continue
        words = ans.split()
        hit = False
        for wi, w in enumerate(words):
            low = w.lower().strip(".,")
            if low in DRIVE_TOKEN_SWAPS and not hit:
                words[wi] = w.replace(low, DRIVE_TOKEN_SWAPS[low])
                hit = True
        if hit:
            new = " ".join(words)
            changed.append({"cat": cat, "idx": i, "old": ans, "new": new})
            item["A"] = new
    return changed


def qa_category_misassign(qa: dict, params, rng):
    rate = params["rate"]
    cats = [c for c, items in qa.items() if isinstance(items, list)]
    if len(cats) < 2:
        return []
    moved = []
    for cat in cats:
        items = qa[cat]
        if not isinstance(items, list) or not items:
            continue
        moved_out = []
        kept = []
        for item in items:
            if rng.random() < rate:
                moved_out.append(item)
            else:
                kept.append(item)
        if not moved_out:
            continue
        qa[cat] = kept
        others = [c for c in cats if c != cat]
        for item in moved_out:
            if not others:
                kept.append(item)
                continue
            target = others[int(rng.integers(0, len(others)))]
            qa[target].append(item)
            moved.append({"from": cat, "to": target, "q": item.get("Q", "")[:60]})
    return moved


def qa_drop(qa: dict, params, rng):
    rate = params["rate"]
    dropped = []
    for cat in list(qa.keys()):
        items = qa[cat]
        if not isinstance(items, list):
            continue
        keep = []
        for i, item in enumerate(items):
            if rng.random() < rate:
                dropped.append({"cat": cat, "q": item.get("Q", "")[:60]})
            else:
                keep.append(item)
        qa[cat] = keep
    return dropped


def qa_duplicate(qa: dict, params, rng):
    rate = params["rate"]
    dup = []
    for cat in list(qa.keys()):
        items = qa[cat]
        if not isinstance(items, list):
            continue
        extras = []
        for item in items:
            if rng.random() < rate:
                extras.append(dict(item))
                dup.append({"cat": cat, "q": item.get("Q", "")[:60]})
        items.extend(extras)
    return dup


def qa_crossref_break(qa: dict, params, rng):
    rate = params["rate"]
    broken = []
    for cat, i, item in _iter_qa(qa):
        q = item.get("Q")
        if not isinstance(q, str) or rng.random() >= rate:
            continue
        new_q, n = _CREF_RE.subn(lambda m: f"<c{int(m.group(1)) + 17}", q)
        if n:
            item["Q"] = new_q
            broken.append({"cat": cat, "idx": i, "n_broken": n})
    return broken


QA_CORRUPTIONS = {
    "qa_wrong_answer": qa_wrong_answer,
    "qa_answer_drift": qa_answer_drift,
    "qa_category_misassign": qa_category_misassign,
    "qa_drop": qa_drop,
    "qa_duplicate": qa_duplicate,
    "qa_crossref_break": qa_crossref_break,
}


def apply_qa_op(name: str, qa: dict, params: dict, rng: np.random.Generator):
    fn = QA_CORRUPTIONS.get(name)
    if fn is None:
        raise KeyError(f"unknown QA family: {name}")
    return fn(qa, params, rng)
