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
    if args.report:
        write_report(out_root, args.report, args.previews)
        print(f"report -> {args.report}")
    print(f"done: {n_offspring} offspring scenario(s) -> {out_root}")
    return 0


def cmd_harness(args) -> int:
    from .fetch import fetch_harness_link

    if args.recipe:
        recipe = load_recipe(args.recipe)
    elif args.op:
        if args.op not in HARNESS_OPS:
            print(f"error: unknown harness op '{args.op}'; use list-families", file=sys.stderr)
            return 2
        recipe = {"name": f"{args.op}_f{args.strength}", "ops": [{"family": args.op, "severity": args.strength, "pattern": "single_shot", "params_override": {}}]}
    else:
        print("error: provide --recipe FILE or --op NAME (or --link to fetch a harness first)", file=sys.stderr)
        return 2
    root = Path(args.root) if args.root else None
    if args.link:
        fetched = fetch_harness_link(args.link, Path(args.fetched_dir or (Path(args.out).parent / "fetched")))
        print(f"harness fetched from link -> {fetched}")
        root = Path(fetched)
    if root is None:
        print("error: provide --root <harness-dir> or --link <git/repo/archive URL>", file=sys.stderr)
        return 2
    manifest = render_harness(root, recipe, Path(args.out), args.seed)
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


def cmd_fetch(args) -> int:
    from .fetch import FETCHERS, fetch_link, fetch_harness_link

    if args.link:
        if args.kind == "harness":
            root = fetch_harness_link(args.link, Path(args.out))
            print(f"harness fetched -> {root}")
        else:
            fetch_link(args.link, Path(args.out), args.max_gb)
        return 0
    if args.dataset == "lyft":
        from .fetch import fetch_lyft

        try:
            fetch_lyft(args.samples, Path(args.out))
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    fn = FETCHERS.get(args.dataset)
    if fn is None:
        print(f"error: unknown dataset '{args.dataset}' (waymo | drivelm | carla | lyft) — "
              f"or use --link <url> for any package", file=sys.stderr)
        return 2
    data_dir = Path(args.out) / "data" / args.dataset
    if args.dataset == "waymo":
        fn(args.samples, Path(args.out) / "data" / "waymo", max_gb=args.max_gb)
    elif args.dataset == "drivelm":
        from .fetch import fetch_drivelm

        fetch_drivelm(args.split, data_dir, args.max_gb, args.with_images)
    else:
        fn(args.samples, data_dir, scenario=args.scenario, max_gb=args.max_gb)
    return 0


def cmd_generate(args) -> int:
    from .fetch import fetch_lyft

    out_root = Path(args.out)
    data_dir = Path(args.data_root).expanduser() if args.data_root else out_root / "data" / args.dataset
    adapter_type = {"drivelm": "nuscenes", "carla": "carla", "waymo": "waymo"}.get(args.dataset, args.dataset)
    from .fetch import fetch_lyft

    need_fetch = args.fetch_if_missing and not data_dir.exists()
    if need_fetch:
        if args.dataset == "lyft":
            try:
                fetch_lyft(args.samples, data_dir)
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
        print(f"[fetch] {args.dataset} x{args.samples} -> {data_dir}")
        if args.dataset == "waymo":
            from .fetch import fetch_waymo

            fetch_waymo(args.samples, data_dir, args.max_gb)
        elif args.dataset == "drivelm":
            from .fetch import fetch_drivelm

            fetch_drivelm(args.split, data_dir, args.max_gb, args.with_images)
        else:
            from .fetch import fetch_carla

            fetch_carla(args.samples, args.scenario, data_dir, args.max_gb)
    elif args.dataset != "lyft":
        print(f"[fetch] using cached data at {data_dir}")

    adapter = get_adapter(adapter_type, data_dir, max_frames=args.max_frames)
    severities = _parse_severities(args.severities)
    ids = select_scenarios(adapter, str(args.samples))
    from .importance import camera_priorities, maneuver_of

    if args.per_camera and args.pattern not in ("camera_scoped", "flaky_camera"):
        print("note: --per-camera applies to camera-scoped patterns (camera_scoped, flaky_camera); ignoring it", file=sys.stderr)

    if args.per_camera and args.pattern in ("camera_scoped", "flaky_camera"):
        scen_by_id = {}
        work = []
        for sid in ids:
            sc = adapter.get_scenario(sid)
            scen_by_id[sid] = sc
            cams = sorted(sc.frames[0].cameras)
            for cam in camera_priorities(sc.description, cams):
                for s in severities:
                    work.append((sid, cam, s))
        if args.batch_limit:
            work = work[: args.batch_limit]
        print(f"[inject] {len(ids)} parent(s) x {args.family} per-camera (importance-ordered) x severities {severities}"
              f" — batch limit {args.batch_limit or 'none'} → {len(work)} children")
        for sid in ids:
            sc = scen_by_id[sid]
            clean_dest = out_root / "offspring" / adapter.name / sid / "clean"
            if not (clean_dest / "annotations.json").exists():
                render_scenario(sc, {"name": "clean", "ops": []}, clean_dest,
                                derive_seed(args.seed, sid, "clean"), root=str(adapter.root))
                print(f"    rendered: {clean_dest} (baseline)")
        for sid, cam, s in work:
            sc = scen_by_id[sid]
            recipe = make_recipe(args.family, args.family, [s])[0]
            recipe["ops"][0]["pattern"] = args.pattern
            recipe["ops"][0]["force_camera"] = cam
            recipe["name"] = f"{args.family}_{args.pattern}_{cam}_s{s}"
            dest = out_root / "offspring" / adapter.name / sid / recipe["name"]
            render_scenario(sc, recipe, dest, derive_seed(args.seed, sid, recipe["name"]),
                            root=str(adapter.root))
            print(f"    rendered: {dest}")
    else:
        print(f"[inject] {len(ids)} parent(s) x {args.family} severities {severities}"
              + (f" pattern {args.pattern}" if args.pattern else ""))
        for sid in ids:
            sc = adapter.get_scenario(sid)
            clean_dest = out_root / "offspring" / adapter.name / sid / "clean"
            if not (clean_dest / "annotations.json").exists():
                render_scenario(sc, {"name": "clean", "ops": []}, clean_dest,
                                derive_seed(args.seed, sid, "clean"), root=str(adapter.root))
                print(f"    rendered: {clean_dest} (baseline)")
            for s in severities:
                recipe = make_recipe(args.family, args.family, [s])[0]
                if args.pattern:
                    recipe["ops"][0]["pattern"] = args.pattern
                    recipe["name"] = f"{args.family}_{args.pattern}_s{s}"
                dest = out_root / "offspring" / adapter.name / sid / recipe["name"]
                render_scenario(sc, recipe, dest, derive_seed(args.seed, sid, recipe["name"]),
                                root=str(adapter.root))
                print(f"    rendered: {dest}")

    report_path = out_root / "report.html"
    write_report(out_root / "offspring", report_path, args.previews)
    print(f"done: test scenarios + report -> {report_path}")
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
    pi.add_argument("--report", default=None, help="auto-generate HTML report at this path after rendering")
    pi.add_argument("--previews", default=None, help="preview PNG dir to embed (with --report)")
    pi.add_argument("--seed", type=int, default=42)
    pi.add_argument("--max-frames", type=int, default=60, help="cap frames per scenario (waymo adapter)")
    pi.add_argument("--llm-base-url", default=None, help="OpenAI-compatible endpoint for LLM ops (e.g. https://api.openai.com/v1)")
    pi.add_argument("--llm-model", default=None, help="model name for LLM ops")
    pi.add_argument("--llm-api-key-env", default="NOISE_INJECTOR_API_KEY", help="env var holding the API key (never pass the key itself)")
    pi.add_argument("--no-llm-cache", action="store_true", help="disable LLM response caching")
    pi.set_defaults(fn=cmd_inject)

    ph = sub.add_parser("harness", help="corrupt an agent harness copy (prompts/config) — local dir or from a link")
    ph.add_argument("--root", default=None, help="harness directory (agent code + prompts + config)")
    ph.add_argument("--link", default=None, help="fetch harness first: git URL / github repo / archive URL")
    ph.add_argument("--fetched-dir", default=None, help="where to place the fetched harness clone")
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

    pf = sub.add_parser("fetch", help="download scenario data (waymo|drivelm|carla|lyft) or any package/harness by link")
    pf.add_argument("--dataset", default=None, help="waymo | drivelm | carla | lyft")
    pf.add_argument("--samples", type=int, default=3, help="number of scenarios/segments to download")
    pf.add_argument("--split", default="val", help="drivelm split: val | train")
    pf.add_argument("--scenario", default=None, help="carla scenario filter, e.g. Town02/ControlLoss")
    pf.add_argument("--with-images", action="store_true", help="drivelm: also fetch the image archive (large)")
    pf.add_argument("--link", default=None, help="direct URL: scenario package (zip/tar/parquet/json) or harness (git/repo/archive)")
    pf.add_argument("--kind", default="data", choices=["data", "harness"], help="what --link provides")
    pf.add_argument("--max-gb", type=float, default=8.0, help="download size guard")
    pf.add_argument("--out", required=True, help="data output directory")
    pf.set_defaults(fn=cmd_fetch)

    pg = sub.add_parser("generate", help="one-call flow: fetch (if needed) → inject → HTML report with scenario tree")
    pg.add_argument("--dataset", required=True, help="waymo | drivelm | carla | lyft")
    pg.add_argument("--samples", type=int, default=3, help="scenarios to fetch/sample")
    pg.add_argument("--scenario", default=None, help="carla scenario filter, e.g. Town02/ControlLoss")
    pg.add_argument("--split", default="val", help="drivelm split: val | train")
    pg.add_argument("--with-images", action="store_true", help="drivelm: also fetch image archive (large)")
    pg.add_argument("--family", required=True, help="noise family (see list-families)")
    pg.add_argument("--severities", default="1-3", help="severity: int or range like 1-3")
    pg.add_argument("--max-gb", type=float, default=8.0)
    pg.add_argument("--max-frames", type=int, default=60)
    pg.add_argument("--fetch-if-missing", dest="fetch_if_missing", action="store_true", default=True)
    pg.add_argument("--no-fetch-if-missing", dest="fetch_if_missing", action="store_false")
    pg.add_argument("--previews", default=None, help="preview PNG dir to embed in the report")
    pg.add_argument("--pattern", default=None, help=f"override the family's temporal pattern (one of: {', '.join(['persistent', 'intermittent', 'burst', 'transient', 'camera_scoped', 'flaky_camera'])})")
    pg.add_argument("--per-camera", action="store_true", help="camera-scoped patterns: one child per camera, importance-ordered (front first for forward scenes, rear first for backing)")
    pg.add_argument("--batch-limit", type=int, default=None, help="cap total perturbed children; important (maneuver-relevant) cameras are created first")
    pg.add_argument("--data-root", default=None, help="use an existing local dataset root (skips fetching)")
    pg.add_argument("--seed", type=int, default=42)
    pg.add_argument("--out", required=True, help="testset output directory (report.html lands here)")
    pg.set_defaults(fn=cmd_generate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
