"""Layer F: agent-harness corruption (prompts + config) — artifact-only, never executes models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import yaml

from .corruptions.config import apply_config_op, dump_cfg
from .corruptions.prompt import apply_prompt_op
from .manifest import build_manifest, write_manifest
from .spec import HARNESS_CONFIG_OPS, HARNESS_PROMPT_OPS
from .temporal import derive_seed

_EXCLUDE_DIRS = {".venv", "venv", "__pycache__", "runs", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
_EXCLUDE_SUFFIXES = {".pt", ".pth", ".gguf", ".bin", ".onnx", ".log"}


def _is_prompt_file(p: Path) -> bool:
    return p.suffix.lower() in {".txt", ".md"}


def _is_config_file(p: Path) -> bool:
    return p.name in {"config.json"} or p.suffix.lower() in {".yaml", ".yml"}


def discover_files(root: Path, include_yaml_prompts: bool = False):
    prompts, configs = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.suffix.lower() in _EXCLUDE_SUFFIXES:
            continue
        if _is_prompt_file(p):
            prompts.append(rel)
        elif _is_config_file(p):
            configs.append(rel)
    return prompts, configs


def render_harness(root: Path, recipe: dict, out_dir: Path, seed: int) -> dict:
    root = Path(root).expanduser().resolve()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts, configs = discover_files(root)
    events = []
    copied = []
    for rel in prompts + configs:
        src = root / rel
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied.append(str(rel))

    prompt_state: dict[str, str] = {}
    config_state: dict[str, dict] = {}

    for op in recipe["ops"]:
        op_name = op["family"]
        strength = int(op.get("severity", 3))
        targets = op.get("targets")
        if op_name in HARNESS_PROMPT_OPS:
            candidates = [rel for rel in prompts if not targets or str(rel) in targets]
            if not candidates:
                continue
            rng = np.random.default_rng(derive_seed(seed, op_name, "prompt"))
            for rel in candidates:
                key = str(rel)
                if key not in prompt_state:
                    prompt_state[key] = (out_dir / rel).read_text()
                text = prompt_state[key]
                new_text = apply_prompt_op(op_name, text, strength, rng)
                if new_text != text:
                    (out_dir / rel).write_text(new_text)
                    prompt_state[key] = new_text
                    events.append(
                        {"family": op_name, "layer": "F", "file": key, "strength": strength,
                         "detail": {"chars_before": len(text), "chars_after": len(new_text)}}
                    )
        elif op_name in HARNESS_CONFIG_OPS:
            if targets:
                candidates = [rel for rel in configs if str(rel) in targets]
            else:
                candidates = [rel for rel in configs if rel.suffix == ".json"]
            if not candidates:
                continue
            for rel in candidates:
                key = str(rel)
                if key not in config_state:
                    with open(out_dir / rel) as f:
                        config_state[key] = json.load(f)
                cfg = config_state[key]
                rng = np.random.default_rng(derive_seed(seed, op_name, "config", key))
                detail = apply_config_op(op_name, cfg, strength, rng)
                if detail:
                    (out_dir / rel).write_text(dump_cfg(cfg))
                    events.append(
                        {"family": op_name, "layer": "F", "file": key, "strength": strength, "detail": detail}
                    )
        else:
            raise KeyError(f"unknown harness op: {op_name}")

    manifest = build_manifest(
        root.name, "harness", str(root), {"name": recipe.get("name", "harness"), "ops": recipe["ops"]},
        seed, events, copied + ["manifest.json"],
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return manifest
