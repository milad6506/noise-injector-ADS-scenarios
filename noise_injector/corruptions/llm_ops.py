"""LLM-powered corruption ops (bring-your-own OpenAI-compatible backend)."""

from __future__ import annotations

import numpy as np

from ..llm import LLMBackend, LLMCache, cached_chat

QA_SYSTEM = (
    "You are corrupting autonomous-driving QA ground-truth data to build robustness benchmarks. "
    "Return ONLY the rewritten answer text, nothing else."
)

DESC_SYSTEM = (
    "You are corrupting object descriptions in driving datasets for robustness benchmarks. "
    "Return ONLY the rewritten description, nothing else."
)


def llm_qa_rewrite(qa: dict, params, rng, backend: LLMBackend, cache: LLMCache | None) -> list:
    rate = params["rate"]
    style = params.get("style", "plausible_wrong")
    changed = []
    for cat in list(qa.keys()):
        items = qa[cat]
        if not isinstance(items, list):
            continue
        for i, item in enumerate(items):
            if not isinstance(item.get("A"), str) or rng.random() >= rate:
                continue
            old = item["A"]
            task = {
                "plausible_wrong": "rewrite the answer so it is plausible but factually wrong for this scene (change the object, state, or action). Keep it to one short sentence.",
                "vague": "rewrite the answer to be vague and generic, losing all scene-specific details. One short sentence.",
                "hallucinated": "rewrite the answer adding one concrete object or traffic element that is NOT in the scene. One short sentence.",
            }[style]
            user = f"Question: {item.get('Q', '')}\nOriginal answer: {old}\nTask: {task}"
            new = cached_chat(backend, cache, QA_SYSTEM, user, max_tokens=80)
            new = new.strip().strip('"')
            if new and new != old:
                item["A"] = new
                changed.append({"cat": cat, "idx": i, "old": old, "new": new})
    return changed


def llm_description_drift(koi: dict, params, rng, backend: LLMBackend, cache: LLMCache | None) -> list:
    rate = params["rate"]
    changed = []
    for key, info in koi.items():
        desc = info.get("Visual_description")
        if not isinstance(desc, str) or not desc or rng.random() >= rate:
            continue
        user = (
            f"Object description: {desc}\n"
            "Task: rewrite it with exactly one subtly wrong attribute (wrong color, size, count, or type). "
            "Keep the same length."
        )
        new = cached_chat(backend, cache, DESC_SYSTEM, user, max_tokens=60)
        new = new.strip().strip('"')
        if new and new != desc:
            info["Visual_description"] = new
            changed.append({"object": key, "old": desc, "new": new})
    return changed


def apply_llm_op(name: str, payload, params, rng, backend, cache):
    if backend is None:
        raise ValueError(
            f"family '{name}' requires an LLM backend: pass --llm-base-url and --llm-model "
            "(API key via env var; see README 'LLM backend')"
        )
    if name == "llm_qa_rewrite":
        return llm_qa_rewrite(payload, params, rng, backend, cache)
    if name == "llm_description_drift":
        return llm_description_drift(payload, params, rng, backend, cache)
    raise KeyError(f"unknown LLM op: {name}")
