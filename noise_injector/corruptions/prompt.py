"""Layer F: prompt-file corruption ops (no model execution)."""

from __future__ import annotations

import random

from ..spec import PROMPT_INJECTION_POOL


def prompt_truncation(text: str, strength: int, rng) -> str:
    frac = [0.9, 0.75, 0.6, 0.45, 0.3][max(0, min(4, strength - 1))]
    return text[: max(1, int(len(text) * frac))]


def prompt_section_delete(text: str, strength: int, rng) -> str:
    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        return text
    n_drop = min(len(paras) - 1, max(1, strength))
    rng2 = random.Random(int(rng.integers(0, 2**31)))
    idxs = sorted(rng2.sample(range(len(paras)), n_drop), reverse=True)
    for i in idxs:
        del paras[i]
    return "\n\n".join(paras)


def prompt_instruction_injection(text: str, strength: int, rng) -> str:
    rng2 = random.Random(int(rng.integers(0, 2**31)))
    n = max(1, min(3, strength))
    lines = [rng2.choice(PROMPT_INJECTION_POOL) for _ in range(n)]
    return text + "\n\n" + "\n".join(lines)


def prompt_garble(text: str, strength: int, rng) -> str:
    rate = [0.02, 0.05, 0.09, 0.14, 0.2][max(0, min(4, strength - 1))]
    chars = list(text)
    for i in range(len(chars)):
        if rng.random() < rate and chars[i].isalnum():
            chars[i] = chr(ord("a") + int(rng.integers(0, 26)))
    return "".join(chars)


def prompt_format_break(text: str, strength: int, rng) -> str:
    out = text
    for marker in ("```", "{", "}", "#"):
        if marker in out:
            out = out.replace(marker, "")
    lines = out.split("\n")
    keep = [ln for i, ln in enumerate(lines) if i % max(1, strength + 1) != strength]
    return "\n".join(keep)


PROMPT_CORRUPTIONS = {
    "prompt_truncation": prompt_truncation,
    "prompt_section_delete": prompt_section_delete,
    "prompt_instruction_injection": prompt_instruction_injection,
    "prompt_garble": prompt_garble,
    "prompt_format_break": prompt_format_break,
}


def apply_prompt_op(name: str, text: str, strength: int, rng):
    fn = PROMPT_CORRUPTIONS.get(name)
    if fn is None:
        raise KeyError(f"unknown prompt op: {name}")
    return fn(text, strength, rng)
