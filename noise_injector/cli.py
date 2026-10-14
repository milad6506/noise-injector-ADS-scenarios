"""CLI: noise-injector inject|harness|list-families|preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import __version__
from .adapters.base import get_adapter
from .corruptions.image import IMAGE_CORRUPTIONS
from .harness import render_harness
from .render import load_recipe, make_recipe, render_scenario, select_scenarios
from .report import write_report
from .spec import ALL_FAMILIES, HARNESS_OPS, SEVERITY_MAX, list_families
from .temporal import derive_seed


def _parse_severities(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


def cmd_inject(args) -> int:
    if args.family is not None and args.family not in ALL_FAMILIES:
        print(f"error: unknown family '{args.family}'; use list-families", file=sys.stderr)
        raise SystemExit(2)
    adapter = get_adapter(args.type, Path(args.root), max_frames=args.max_frames)
    if args.recipe:
        recipes = [load_recipe(args.recipe)]
    elif args.family:
        severities = _parse_severities(args.severities)
        recipes = make_recipe(args.family, args.family, severities)
    else:
        print("error: provide --recipe FILE or --family NAME", file=sys.stderr)
        return 2

    backend = cache = None
    needs_llm = any(op["family"].startswith("llm_") for r in recipes for op in r["ops"])
    if needs_llm:
        from .llm import make_backend

        out_root = Path(args.out)
        backend, cache = make_backend(
            args.llm_base_url, args.llm_model, args.llm_api_key_env,
            cache_path=(out_root / ".llm_cache.json") if not args.no_llm_cache else None,
        )
        print(f"LLM backend: {args.llm_model} @ {args.llm_base_url} (key from ${args.llm_api_key_env})")

    scenario_sel = args.scenarios or "1"
    ids = select_scenarios(adapter, scenario_sel)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    n_offspring = 0
    for sid in ids:
        sc = adapter.get_scenario(sid)
        for recipe in recipes:
            seed = derive_seed(args.seed, sid, recipe["name"])
            dest = out_root / adapter.name / sid / recipe["name"]
            render_scenario(sc, recipe, dest, seed, root=str(adapter.root), backend=backend, cache=cache)
            n_offspring += 1
            print(f"rendered: {dest}")
    print(f"done: {n_offspring} offspring scenario(s) -> {out_root}")
    return 0


def cmd_harness(args) -> int:
    if args.recipe:
        recipe = load_recipe(args.recipe)
    elif args.op:
        if args.op not in HARNESS_OPS:
            print(f"error: unknown harness op '{args.op}'; use list-families", file=sys.stderr)
            return 2
        recipe = {"name": f"{args.op}_f{args.strength}", "ops": [{"family": args.op, "severity": args.strength, "pattern": "single_shot", "params_override": {}}]}
    else:
        print("error: provide --recipe FILE or --op NAME", file=sys.stderr)
        return 2
    manifest = render_harness(Path(args.root), recipe, Path(args.out), args.seed)
    print(f"done: corrupted harness copy -> {args.out} ({len(manifest['injected_events'])} events)")
    return 0


def cmd_list_families(_args) -> int:
    layers = list_families()
    for layer in sorted(layers):
        print(f"\n== Layer {layer} ==")
        for entry in sorted(layers[layer], key=lambda e: e["family"]):
            print(f"  {entry['family']:<24} [{entry['pattern']:<13}] {entry['description']}")
    return 0


def cmd_preview(args) -> int:
    adapter = get_adapter(args.type, Path(args.root), max_frames=args.max_frames)
    sc = adapter.get_scenario(adapter.list_scenarios()[0] if not args.scenario else args.scenario)
    frame = None
    for fr in sc.frames:
        for cam_img in fr.cameras.values():
            if cam_img.has_image:
                frame, camera = fr, cam_img.camera
                break
        if frame:
            break
    if frame is None:
        print("error: scenario has no images to preview (annotation-only adapter)", file=sys.stderr)
        return 2
    if args.family not in IMAGE_CORRUPTIONS:
        print("error: --family must be an image family for preview", file=sys.stderr)
        return 2

    severities = _parse_severities(args.severities)
    from .spec import family as famdef

    base = cv2.cvtColor(cv2.imread(str(frame.cameras[camera].path)), cv2.COLOR_BGR2RGB)
    tiles = [_label_tile(base, "original")]
    for s in severities:
        params = famdef(args.family).params(s)
        rng = np.random.default_rng(derive_seed(args.seed, args.family, frame.frame_id, camera, f"prev{s}"))
        img = IMAGE_CORRUPTIONS[args.family](base.copy(), params, rng)
        tiles.append(_label_tile(img, f"{args.family} S{s}"))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    grid = np.hstack(tiles)
    Image.fromarray(grid).save(args.out)
    print(f"preview -> {args.out}")
    return 0


def _label_tile(img: np.ndarray, label: str) -> np.ndarray:
    tile = img.copy()
    bar = np.zeros((26, tile.shape[1], 3), np.uint8)
    pil = Image.fromarray(np.vstack([tile, bar]))
    draw = ImageDraw.Draw(pil)
    draw.text((6, tile.shape[0] + 6), label, fill=(255, 255, 0))
    return np.asarray(pil)


def cmd_report(args) -> int:
    path = write_report(args.offspring, args.out, args.previews)
    print(f"report -> {path}")
    return 0


def cmd_run(args) -> int:
    from .orchestrator import Orchestrator, load_experiment

    exp = load_experiment(args.experiment)
    orch = Orchestrator(exp, Path(args.out))
    result = orch.run()
    print("resilience per target:")
    for t, v in result["summary"]["per_target"].items():
        print(f"  {t}: clean {v['clean_score']} · mean retention {v['mean_retention']} · min {v['min_retention']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="noise-injector", description="Deterministic noise/error injection for ADS scenario datasets and agent harnesses.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inject", help="create offspring scenarios from a scenario dataset")
    pi.add_argument("--type", required=True, choices=["nuscenes", "carla", "waymo"])
    pi.add_argument("--root", required=True, help="dataset root (DriveLM dir, waymo_sample dir, ...)")
    pi.add_argument("--recipe", help="recipe YAML path")
    pi.add_argument("--family", help=f"corruption family (one of: {', '.join(sorted(ALL_FAMILIES))})")
    pi.add_argument("--severities", default="1-5", help="severity: single int or range like 1-5")
    pi.add_argument("--scenarios", default="1", help="count N | 'all' | comma-separated ids")
    pi.add_argument("--out", required=True, help="offspring output root")
    pi.add_argument("--seed", type=int, default=42)
    pi.add_argument("--max-frames", type=int, default=60, help="cap frames per scenario (waymo adapter)")
    pi.add_argument("--llm-base-url", default=None, help="OpenAI-compatible endpoint for LLM ops (e.g. https://api.openai.com/v1)")
    pi.add_argument("--llm-model", default=None, help="model name for LLM ops")
    pi.add_argument("--llm-api-key-env", default="NOISE_INJECTOR_API_KEY", help="env var holding the API key (never pass the key itself)")
    pi.add_argument("--no-llm-cache", action="store_true", help="disable LLM response caching")
    pi.set_defaults(fn=cmd_inject)

    ph = sub.add_parser("harness", help="corrupt an agent harness copy (prompts/config)")
    ph.add_argument("--root", required=True, help="harness directory (agent code + prompts + config)")
    ph.add_argument("--recipe", help="recipe YAML path")
    ph.add_argument("--op", help=f"harness op (one of: {', '.join(sorted(HARNESS_OPS))})")
    ph.add_argument("--strength", type=int, default=3)
    ph.add_argument("--out", required=True)
    ph.add_argument("--seed", type=int, default=42)
    ph.set_defaults(fn=cmd_harness)

    pl = sub.add_parser("list-families", help="print all families by layer")
    pl.set_defaults(fn=cmd_list_families)

    pv = sub.add_parser("preview", help="render original vs severity grid for one image family")
    pv.add_argument("--type", required=True, choices=["nuscenes", "carla"])
    pv.add_argument("--root", required=True)
    pv.add_argument("--family", required=True)
    pv.add_argument("--severities", default="1-5")
    pv.add_argument("--scenario", default=None)
    pv.add_argument("--out", required=True, help="output PNG path")
    pv.add_argument("--seed", type=int, default=42)
    pv.add_argument("--max-frames", type=int, default=60)
    pv.set_defaults(fn=cmd_preview)

    pr = sub.add_parser("report", help="build a self-contained HTML report from an offspring tree")
    pr.add_argument("--offspring", required=True, help="offspring root to scan for manifests")
    pr.add_argument("--out", required=True, help="output HTML path")
    pr.add_argument("--previews", default=None, help="directory of preview PNGs to embed")
    pr.set_defaults(fn=cmd_report)

    po = sub.add_parser("run", help="run a full resilience experiment (LLM-managed rounds)")
    po.add_argument("--experiment", required=True, help="experiment YAML (engine + dataset + targets + rounds)")
    po.add_argument("--out", required=True, help="run output directory (resumable)")
    po.set_defaults(fn=cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
