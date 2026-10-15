"""Render pipeline: scenario × recipe → offspring directory."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .adapters.base import Adapter, Scenario
from .corruptions.annotation import apply_annotation_op
from .corruptions.image import IMAGE_CORRUPTIONS
from .corruptions.llm_ops import apply_llm_op
from .corruptions.qa import apply_qa_op
from .llm import LLMBackend, LLMCache
from .manifest import build_manifest, write_manifest
from .spec import ALL_FAMILIES, HARNESS_OPS, IMAGE_FAMILIES, LAYER_OF, LLM_FAMILIES, STRUCTURAL_FAMILIES, family
from .temporal import activation_mask, derive_seed


def load_recipe(path: str | Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "ops" not in raw:
        raise ValueError(f"recipe file must be a mapping with 'ops': {path}")
    name = raw.get("name", Path(path).stem)
    ops = []
    for i, op in enumerate(raw["ops"]):
        fam = op.get("family")
        if fam not in ALL_FAMILIES and fam not in HARNESS_OPS:
            raise ValueError(f"op[{i}]: unknown family: {fam}")
        ops.append(
            {
                "family": fam,
                "severity": int(op.get("severity", 3)),
                "pattern": op.get("pattern", family(fam).default_pattern),
                "params_override": op.get("params") or {},
            }
        )
    return {"name": name, "ops": ops}


def make_recipe(name: str, family_name: str, severities: list[int]) -> list[dict]:
    out = []
    for s in severities:
        out.append(
            {
                "name": f"{family_name}_s{s}",
                "ops": [
                    {
                        "family": family_name,
                        "severity": s,
                        "pattern": family(family_name).default_pattern,
                        "params_override": {},
                    }
                ],
            }
        )
    return out


def select_scenarios(adapter: Adapter, selector: str) -> list[str]:
    ids = adapter.list_scenarios()
    if selector == "all":
        return ids
    sel = selector.strip()
    if sel.isdigit():
        return ids[: int(sel)]
    wanted = [s.strip() for s in sel.split(",") if s.strip()]
    missing = [w for w in wanted if w not in ids]
    if missing:
        raise KeyError(f"scenarios not found: {missing}")
    return wanted


def _apply_image_op(op, sc: Scenario, seed: int) -> dict:
    fam = op["family"]
    famdef = family(fam)
    params = {**famdef.params(op["severity"]), **op["params_override"]}
    pattern = op["pattern"]
    cam_names = sorted({c for fr in sc.frames for c in fr.cameras})
    rng = np.random.default_rng(derive_seed(seed, fam, "image"))
    mask = activation_mask(pattern, len(sc.frames), rng, op["severity"], len(cam_names))
    cam_idx = mask["camera"]
    targets = []
    for i, fr in enumerate(sc.frames):
        if not mask["frames"][i]:
            continue
        if cam_idx is not None:
            cams = [cam_names[cam_idx]] if cam_names[cam_idx] in fr.cameras else []
        else:
            cams = sorted(fr.cameras)
        if fam == "camera_dropout":
            mode = params["mode"]
            for c in cams:
                fr.metadata.setdefault("dropout", {})[c] = mode
            targets.append({"frame_id": fr.frame_id, "cameras": cams, "detail": {"mode": mode}})
            continue
        if fam == "camera_desync":
            idx = sc.frames.index(fr)
            if idx == 0:
                continue
            src = sc.frames[idx - 1]
            for c in cams:
                fr.cameras[c] = copy.copy(src.cameras[c])
            targets.append({"frame_id": fr.frame_id, "cameras": cams, "detail": {"stale_from": src.frame_id}})
            continue
        for c in cams:
            fr.metadata.setdefault("pixel_ops", {}).setdefault(c, []).append(
                {"family": fam, "params": params}
            )
        targets.append({"frame_id": fr.frame_id, "cameras": cams})
    return {"family": fam, "layer": "A", "pattern": pattern, "severity": op["severity"], "params": params, "targets": targets}


def _apply_frame_freeze(op, sc: Scenario, seed: int) -> dict:
    famdef = STRUCTURAL_FAMILIES["frame_freeze"]
    params = {**famdef.params(op["severity"]), **op["params_override"]}
    rng = np.random.default_rng(derive_seed(seed, "frame_freeze", "frames"))
    pattern = op["pattern"]
    mask = activation_mask(pattern, len(sc.frames), rng, op["severity"])
    targets = []
    for i, fr in enumerate(sc.frames):
        if not mask["frames"][i] or i == 0:
            continue
        fr.metadata["frozen_from"] = sc.frames[i - 1].frame_id
        targets.append({"frame_id": fr.frame_id, "copy_of": sc.frames[i - 1].frame_id})
    return {"family": "frame_freeze", "layer": "A", "pattern": pattern, "severity": op["severity"], "params": params, "targets": targets}


def _apply_annotation_op(op, sc: Scenario, seed: int) -> dict:
    fam = op["family"]
    famdef = family(fam)
    params = {**famdef.params(op["severity"]), **op["params_override"]}
    rng = np.random.default_rng(derive_seed(seed, fam, "annotation"))
    mask = activation_mask(op["pattern"], len(sc.frames), rng, op["severity"])
    targets = []
    for i, fr in enumerate(sc.frames):
        if not mask["frames"][i] or not fr.key_object_infos:
            continue
        detail = apply_annotation_op(fam, fr.key_object_infos, params, rng)
        if detail:
            targets.append({"frame_id": fr.frame_id, "detail": detail})
    return {"family": fam, "layer": "B", "pattern": op["pattern"], "severity": op["severity"], "params": params, "targets": targets}


def _apply_qa_op(op, sc: Scenario, seed: int) -> dict:
    fam = op["family"]
    famdef = family(fam)
    params = {**famdef.params(op["severity"]), **op["params_override"]}
    rng = np.random.default_rng(derive_seed(seed, fam, "qa"))
    mask = activation_mask(op["pattern"], len(sc.frames), rng, op["severity"])
    targets = []
    for i, fr in enumerate(sc.frames):
        if not mask["frames"][i] or not fr.qa:
            continue
        detail = apply_qa_op(fam, fr.qa, params, rng)
        if detail:
            targets.append({"frame_id": fr.frame_id, "detail": detail})
    return {"family": fam, "layer": "C", "pattern": op["pattern"], "severity": op["severity"], "params": params, "targets": targets}


def _apply_llm_op(op, sc: Scenario, seed: int, backend: LLMBackend, cache: LLMCache | None) -> dict:
    fam = op["family"]
    famdef = family(fam)
    params = {**famdef.params(op["severity"]), **op["params_override"]}
    rng = np.random.default_rng(derive_seed(seed, fam, "llm"))
    mask = activation_mask(op["pattern"], len(sc.frames), rng, op["severity"])
    layer = LLM_FAMILIES[fam].layer
    targets = []
    for i, fr in enumerate(sc.frames):
        if not mask["frames"][i]:
            continue
        if fam == "llm_qa_rewrite" and fr.qa:
            detail = apply_llm_op(fam, fr.qa, params, rng, backend, cache)
        elif fam == "llm_description_drift" and fr.key_object_infos:
            detail = apply_llm_op(fam, fr.key_object_infos, params, rng, backend, cache)
        else:
            continue
        if detail:
            targets.append({"frame_id": fr.frame_id, "detail": detail})
    return {
        "family": fam,
        "layer": layer,
        "pattern": op["pattern"],
        "params": params,
        "severity": op["severity"],
        "backend": f"{backend.model}" if backend else None,
        "targets": targets,
    }


def render_scenario(sc: Scenario, recipe: dict, out_dir: Path, seed: int, root: str = "", backend: LLMBackend | None = None, cache: LLMCache | None = None) -> dict:
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    sc = copy.deepcopy(sc)
    events = []
    for op in recipe["ops"]:
        fam = op["family"]
        if fam == "frame_freeze":
            events.append(_apply_frame_freeze(op, sc, seed))
        elif fam in STRUCTURAL_FAMILIES:
            events.append(_apply_image_op(op, sc, seed))
        elif fam in IMAGE_FAMILIES:
            events.append(_apply_image_op(op, sc, seed))
        elif fam in LLM_FAMILIES:
            events.append(_apply_llm_op(op, sc, seed, backend, cache))
        elif LAYER_OF[fam] == "B":
            events.append(_apply_annotation_op(op, sc, seed))
        elif LAYER_OF[fam] == "C":
            events.append(_apply_qa_op(op, sc, seed))

    output_files = []
    image_paths_by_frame = {}
    rendered_by_frame = {}
    for fr in sc.frames:
        frozen_from = fr.metadata.get("frozen_from")
        dropout = fr.metadata.get("dropout", {})
        rendered_for_frame = {}
        for cam_name, cam_img in sorted(fr.cameras.items()):
            rel = f"{fr.frame_id}__{cam_name}.jpg"
            dest = images_dir / rel
            cam_drop = dropout.get(cam_name)
            if cam_drop == "missing":
                image_paths_by_frame[(fr.frame_id, cam_name)] = None
                continue
            if frozen_from and cam_name in rendered_by_frame.get(frozen_from, {}):
                src_path = rendered_by_frame[frozen_from][cam_name]
                if src_path is None:
                    image_paths_by_frame[(fr.frame_id, cam_name)] = None
                    continue
                shutil.copyfile(src_path, dest)
            elif cam_img.path is None:
                image_paths_by_frame[(fr.frame_id, cam_name)] = None
                continue
            elif cam_drop == "black":
                src = cv2.imread(str(cam_img.path))
                h, w = src.shape[:2] if src is not None else (720, 1280)
                cv2.imwrite(str(dest), np.zeros((h, w, 3), np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                img = cv2.imread(str(cam_img.path))
                if img is None:
                    image_paths_by_frame[(fr.frame_id, cam_name)] = None
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for pix in fr.metadata.get("pixel_ops", {}).get(cam_name, []):
                    fn = IMAGE_CORRUPTIONS[pix["family"]]
                    rng = np.random.default_rng(derive_seed(seed, pix["family"], fr.frame_id, cam_name))
                    img = fn(img, pix["params"], rng)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_paths_by_frame[(fr.frame_id, cam_name)] = f"images/{rel}"
            rendered_for_frame[cam_name] = str(dest)
            output_files.append(f"images/{rel}")
        rendered_by_frame[fr.frame_id] = rendered_for_frame

    key_frames = {}
    for fr in sc.frames:
        key_frames[fr.frame_id] = {
            "key_object_infos": fr.key_object_infos,
            "QA": fr.qa,
            "image_paths": {c: image_paths_by_frame.get((fr.frame_id, c)) for c in fr.cameras},
        }
    annotations = {
        "scenario_id": sc.scenario_id,
        "source": sc.source,
        "scene_description": sc.description,
        "bbox_style": sc.bbox_style,
        "key_frames": key_frames,
    }
    ann_path = out_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=1)
    output_files.append("annotations.json")

    recipe_dict = {"name": recipe["name"], "ops": recipe["ops"]}
    manifest = build_manifest(
        sc.scenario_id,
        sc.source,
        root or str(sc.metadata.get("root", sc.source)),
        recipe_dict,
        seed,
        events,
        output_files,
    )
    write_manifest(out_dir / "manifest.json", manifest)
    return manifest
