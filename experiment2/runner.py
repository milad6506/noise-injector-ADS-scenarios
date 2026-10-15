#!/usr/bin/env python3
"""Experiment 2 — does noise awareness have the same effect on a tiny VLM?

Reuses experiment 1's protocol (mother-reference responses, deterministic deviation,
blind-labeled GLM judge) with arms loaded from config.yaml — so anyone can plug in
their own local tiny VLM. Keys: env vars only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp1_runner_protocol", _HERE.parent / "experiment1" / "runner.py")
exp1 = importlib.util.module_from_spec(_spec)
sys.modules["exp1_runner_protocol"] = exp1
_spec.loader.exec_module(exp1)
from noise_injector.report import _CSS  # noqa: E402


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(args, out_dir=None) -> dict:
    out_dir = Path(out_dir) if out_dir else _HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    exp1.load_env_file(args.env_file)
    judge_cfg = cfg["judge"]
    judge = exp1.OpenAICompatClient(**judge_cfg)
    judge_prompt = (_HERE.parent / "experiment1" / "prompts/judge_prompt.txt").read_text()

    arms = []
    for a in cfg["arms"]:
        prompt_path = (_HERE / a["prompt_file"]).resolve()
        if not prompt_path.exists():
            prompt_path = (_HERE / a["prompt_file"].replace("../experiment1/", "")).resolve()
        arms.append({**a, "prompt_text": prompt_path.read_text(),
                     "client": exp1.OpenAICompatClient(base_url=a["base_url"], model=a["model"],
                                                       api_key_env=a.get("api_key_env") or "NOISE_INJECTOR_API_KEY")})

    run_dir = Path(args.run_dir)
    manifests = sorted(run_dir.glob("offspring/*/*/*/manifest.json"))
    mothers, children_by_mother = {}, {}
    for mp in manifests:
        m = json.load(open(mp))
        sid = m["parent"]["scenario_id"]
        if not m["injected_events"]:
            mothers[sid] = mp.parent
        else:
            fam = m["injected_events"][0]["family"]
            try:
                sev = int(mp.parent.name.rsplit("_s", 1)[-1])
            except (ValueError, IndexError):
                sev = 0
            children_by_mother.setdefault(sid, []).append((fam, mp.parent, sev))
    mother_ids = sorted(mothers)[: args.mothers]

    cells = []
    for midx, sid in enumerate(mother_ids):
        mother_dir = mothers[sid]
        mother_ann = json.load(open(mother_dir / "annotations.json"))
        pool = children_by_mother.get(sid, [])
        if args.sample_strategy == "harsh":
            pool = sorted(pool, key=lambda x: (-x[2], x[0]))
        else:
            pool = sorted(pool, key=lambda x: (x[0], -x[2]))
        child_dirs, used = [], set()
        for fam, d, _ in pool:
            if len(child_dirs) >= args.per_mother:
                break
            if fam in used:
                continue
            used.add(fam)
            child_dirs.append(d)
        print(f"[{midx + 1}/{len(mother_ids)}] mother {sid[:20]}… — {len(child_dirs)} children")

        resp_mother = {a["name"]: exp1.arm_response(a["client"], a["prompt_text"], mother_ann,
                                                    mother_dir, None, None, args.max_frames)
                       for a in arms}
        for cdir in child_dirs:
            child_ann = json.load(open(cdir / "annotations.json"))
            man = json.load(open(cdir / "manifest.json"))
            ev = man["injected_events"][0]
            noise_mode = f"{ev['family']} (pattern {ev['pattern']}, severity {ev.get('severity')})"
            agent_rows, arms_data = [], {}
            for a in arms:
                rc = exp1.arm_response(a["client"], a["prompt_text"], mother_ann, mother_dir,
                                       child_ann, cdir, args.max_frames)
                det = exp1.deterministic_deviation(resp_mother[a["name"]], rc)
                agent_rows.append({"arm": a["name"], "mother_resp": resp_mother[a["name"]],
                                   "child_resp": rc, "det_dev": det})
                arms_data[a["name"]] = {"child_resp": rc, "det_dev": det}
            verdict = exp1.judge_child(judge, judge_prompt,
                                       {"scene_description": mother_ann.get("scene_description", ""),
                                        "gt_frames": {fid: fr.get("gt_behavior", "") for fid, fr in
                                                      list(mother_ann["key_frames"].items())[: args.max_frames]}},
                                       agent_rows)
            cells.append({"mother": sid, "child": cdir.name, "noise_mode": noise_mode,
                          "arms": arms_data, "judge": verdict})
            print(f"    child {cdir.name}: judge → {verdict.get('winner', '?')}")
            with open(out_dir / "results.json", "w") as f:
                json.dump({"config": args.config, "vlm": {a["name"]: a["model"] for a in arms},
                           "cells": cells}, f, indent=1)

    results = {"arms": [{"name": a["name"], "model": a["model"]} for a in arms], "cells": cells,
               "meta": {"mothers": mother_ids, "sample_strategy": args.sample_strategy}}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    return results


def aggregate(results: dict) -> dict:
    per_arm = {}
    for a in results["arms"]:
        det, res, nh = [], [], []
        for cell in results["cells"]:
            jk = f"agent{1 if a['name'] == list(results['arms'])[0] else 2}"
            ja = cell["judge"].get(jk, {})
            det.append(cell["arms"][a["name"]]["det_dev"]["deviation"])
            res.append(ja.get("resilience", 0))
            nh.append(ja.get("noise_handling", 0))
        n = max(1, len(det))
        per_arm[a["name"]] = {"mean_det_deviation": round(sum(det) / n, 2),
                              "mean_resilience": round(sum(res) / n, 2),
                              "mean_noise_handling": round(sum(nh) / n, 2)}
    wins = Counter(c["judge"].get("winner", "tie") for c in results["cells"])
    return {"per_arm": per_arm, "wins": dict(wins)}


def qwen_reference_summary(path: Path | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    try:
        r = json.load(open(path))
    except Exception:
        return None
    if "arms" not in r or not r.get("cells"):
        return None
    return exp1.aggregate(r)


def cross_size_table(results: dict, agg: dict, qwen: dict | None) -> str:
    names = list(agg["per_arm"])
    a_key, b_key = names[0], names[-1]
    rows = [f"<tr><td><b>SmolVLM-500M</b> (this run)</td>"
            f"<td>{agg['per_arm'][a_key]['mean_det_deviation']} → {agg['per_arm'][b_key]['mean_det_deviation']}</td>"
            f"<td>{agg['per_arm'][a_key]['mean_resilience']} → {agg['per_arm'][b_key]['mean_resilience']}</td>"
            f"<td>{html_escape(json.dumps(agg['wins']))}</td></tr>"]
    if qwen:
        pa = qwen["per_arm"]
        keys = list(pa)
        rows.append(f"<tr><td><b>Qwen3-VL-8B</b> (experiment 1 reference)</td>"
                    f"<td>{pa[keys[0]]['mean_det_deviation']} → {pa[keys[-1]]['mean_det_deviation']}</td>"
                    f"<td>{pa[keys[0]]['mean_resilience']} → {pa[keys[-1]]['mean_resilience']}</td>"
                    f"<td>{html_escape(json.dumps(qwen['wins']))}</td></tr>")
    return f"<div class='sec'><table><tr><th>Model size</th><th>Det. deviation blind → aware</th><th>Resilience blind → aware</th><th>Wins</th></tr>{''.join(rows)}</table></div>"


def html_escape(s: str) -> str:
    import html as _h
    return _h.escape(str(s))


def write_report(results: dict, agg: dict, qwen: dict | None, path: Path) -> Path:
    findings = []
    a_key, b_key = list(agg["per_arm"])
    d_gap = round(agg["per_arm"][a_key]["mean_det_deviation"] - agg["per_arm"][b_key]["mean_det_deviation"], 2)
    r_gap = round(agg["per_arm"][b_key]["mean_resilience"] - agg["per_arm"][a_key]["mean_resilience"], 2)
    findings.append(f"<li><b>SmolVLM-500M:</b> awareness {'reduced' if d_gap > 0 else 'did not reduce'} unexplained "
                    f"deviation by {abs(d_gap)} points and {'improved' if r_gap > 0 else 'did not improve'} resilience "
                    f"by {abs(r_gap)} points — direction {'matches' if d_gap > 0 and r_gap > 0 else 'differs from'} the 8B result</li>")
    findings.append(f"<li><b>Judge wins:</b> {html_escape(json.dumps(agg['wins']))}</li>")
    key = f"<div class='sec'><ul>{''.join(findings)}</ul></div>"

    rows = ""
    for cell in results["cells"]:
        j = cell["judge"]
        rows += (f"<tr><td><code>{html_escape(cell['mother'][:14])}…</code></td>"
                 f"<td><b>{html_escape(cell['noise_mode'])}</b></td>"
                 f"<td>{j.get('agent1', {}).get('deviation', '?')} / {j.get('agent1', {}).get('resilience', '?')}</td>"
                 f"<td>{j.get('agent2', {}).get('deviation', '?')} / {j.get('agent2', {}).get('resilience', '?')}</td>"
                 f"<td>{html_escape(j.get('winner', '?'))}</td></tr>")
    arm_rows = "".join(
        f"<tr><td><b>{html_escape(k)}</b></td><td>{v['mean_det_deviation']}</td>"
        f"<td>{v['mean_resilience']}</td><td>{v['mean_noise_handling']}</td></tr>"
        for k, v in agg["per_arm"].items())
    content = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Experiment 2 — parameter size vs noise awareness</title><style>{_CSS}
td, th {{ font-size: 12.5px; }}</style></head><body><div class="wrap">
<h1>Experiment 2 — parameter size vs noise awareness</h1>
<div class="sub">arms: {html_escape(' vs '.join(agg['per_arm']))} · judge: blind-labeled · same protocol as experiment 1</div>
<h2>Key findings (auto-generated)</h2>{key}
<h2>Cross-size comparison</h2>
{cross_size_table(results, agg, qwen)}
<div class="sub">gap = effect of the noise-awareness paragraph, per model size</div>
<h2>Aggregate</h2>
<div class="sec"><table><tr><th>Arm</th><th>Mean det. deviation</th><th>Mean resilience</th><th>Mean noise handling</th></tr>{arm_rows}</table></div>
<h2>Per child</h2>
<div class="sec"><table><tr><th>Mother</th><th>Noise mode</th><th>A1: dev/res</th><th>A2: dev/res</th><th>Winner</th></tr>{rows}</table></div>
<footer>experiment 2 · noise-injector · self-contained · no credentials</footer>
</div></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Experiment 2: parameter size vs noise awareness")
    p.add_argument("--config", default=str(_HERE / "config.yaml"))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mothers", type=int, default=3)
    p.add_argument("--per-mother", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=2)
    p.add_argument("--sample-strategy", choices=["default", "harsh"], default="harsh")
    p.add_argument("--env-file", default=None)
    p.add_argument("--results-file", default=None)
    p.add_argument("--report-file", default=None)
    args = p.parse_args(argv)
    out_dir = Path(args.results_file).resolve().parent if args.results_file else _HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiment(args, out_dir)
    agg = aggregate(results)
    qwen = qwen_reference_summary(_HERE.parent / "experiment1" / "results" / "results.json")
    path = write_report(results, agg, qwen, Path(args.report_file) if args.report_file else out_dir / "report.html")
    print("\n=== aggregate ===")
    for k, v in agg["per_arm"].items():
        print(f"  {k}: dev {v['mean_det_deviation']} · resilience {v['mean_resilience']} · noise handling {v['mean_noise_handling']}")
    print(f"  judge wins: {agg['wins']}")
    if qwen:
        print("  qwen reference (experiment 1):", json.dumps(qwen["wins"]))
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
