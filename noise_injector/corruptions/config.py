"""Layer F: harness config.json corruption ops (no model execution)."""

from __future__ import annotations

import json
import random

import numpy as np

from ..spec import (
    CONF_THETA_DRIFT,
    JPEG_Q_TABLE,
    MAX_PX_TABLE,
    MAX_TOKENS_TABLE,
    VOCAB_SIBLINGS,
)


def _misspell(s: str, rng) -> str:
    if len(s) < 2:
        return s
    i = int(rng.integers(0, len(s) - 1))
    return s[:i] + s[i + 1 :]


def vocab_drop(cfg: dict, strength: int, rng):
    n = max(1, strength)
    changed = []
    for arm in cfg.get("arms", []):
        vocab = arm.get("detector_vocab")
        if not isinstance(vocab, list) or len(vocab) <= 1:
            continue
        drop = min(n, len(vocab) - 1)
        for idx in sorted(rng.choice(len(vocab), size=drop, replace=False), reverse=True):
            changed.append(f"{arm.get('key')}:-{vocab[int(idx)]}")
            del vocab[int(idx)]
    return changed


def vocab_misspell(cfg: dict, strength: int, rng):
    changed = []
    n = max(1, strength)
    for arm in cfg.get("arms", []):
        vocab = arm.get("detector_vocab")
        if not isinstance(vocab, list):
            continue
        for i in range(min(n, len(vocab))):
            old = vocab[i]
            vocab[i] = _misspell(old, rng)
            changed.append(f"{arm.get('key')}:{old}->{vocab[i]}")
    return changed


def vocab_swap(cfg: dict, strength: int, rng):
    changed = []
    n = max(1, strength)
    for arm in cfg.get("arms", []):
        vocab = arm.get("detector_vocab")
        if not isinstance(vocab, list):
            continue
        for i in range(min(n, len(vocab))):
            old = vocab[i]
            new = VOCAB_SIBLINGS.get(old)
            if new:
                vocab[i] = new
                changed.append(f"{arm.get('key')}:{old}->{new}")
    return changed


def confidence_drift(cfg: dict, strength: int, rng):
    changed = []
    drift = CONF_THETA_DRIFT[max(0, min(4, strength - 1))]
    for arm in cfg.get("arms", []):
        if "detector_conf" in arm:
            old = arm["detector_conf"]
            arm["detector_conf"] = round(min(0.99, old + drift), 3)
            changed.append(f"{arm.get('key')}:conf {old}->{arm['detector_conf']}")
    return changed


def max_tokens_starve(cfg: dict, strength: int, rng):
    changed = []
    frac = MAX_TOKENS_TABLE[max(0, min(4, strength - 1))]
    gen = cfg.get("generation", {})
    for key in ("max_tokens", "judge_max_tokens"):
        if key in gen:
            old = gen[key]
            new = max(16, int(old * frac))
            gen[key] = new
            changed.append(f"generation.{key}: {old}->{new}")
    for arm in cfg.get("arms", []):
        if "max_tokens" in arm:
            old = arm["max_tokens"]
            arm["max_tokens"] = max(16, int(old * frac))
            changed.append(f"{arm.get('key')}:max_tokens {old}->{arm['max_tokens']}")
    return changed


def quality_drift(cfg: dict, strength: int, rng):
    changed = []
    i = max(0, min(4, strength - 1))
    img = cfg.get("images", {})
    if "max_px" in img:
        old = img["max_px"]
        img["max_px"] = MAX_PX_TABLE[i]
        changed.append(f"images.max_px: {old}->{img['max_px']}")
    if "jpeg_quality" in img:
        old = img["jpeg_quality"]
        img["jpeg_quality"] = JPEG_Q_TABLE[i]
        changed.append(f"images.jpeg_quality: {old}->{img['jpeg_quality']}")
    return changed


def temperature_drift(cfg: dict, strength: int, rng):
    changed = []
    bump = 0.2 * max(1, strength)
    gen = cfg.get("generation", {})
    if "temperature" in gen:
        old = gen["temperature"]
        gen["temperature"] = round(min(2.0, old + bump), 3)
        changed.append(f"generation.temperature: {old}->{gen['temperature']}")
    return changed


CONFIG_CORRUPTIONS = {
    "vocab_drop": vocab_drop,
    "vocab_misspell": vocab_misspell,
    "vocab_swap": vocab_swap,
    "confidence_drift": confidence_drift,
    "max_tokens_starve": max_tokens_starve,
    "quality_drift": quality_drift,
    "temperature_drift": temperature_drift,
}


def apply_config_op(name: str, cfg: dict, strength: int, rng: np.random.Generator):
    fn = CONFIG_CORRUPTIONS.get(name)
    if fn is None:
        raise KeyError(f"unknown config op: {name}")
    return fn(cfg, strength, rng)


def dump_cfg(cfg: dict) -> str:
    return json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
