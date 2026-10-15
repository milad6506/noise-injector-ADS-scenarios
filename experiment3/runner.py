#!/usr/bin/env python3
"""Experiment 3 — resilience to faults and malicious injections in the VLM's own prompt.

Protocol:
- CLEAN scenario data only (mother renders from a noise-injector run; no sensor noise)
- two prompt branches: A = simple (noise-blind), B = simple + "the prompt itself may be faulty"
- each branch gets a clean control + 5 faulty variants (Layer F ops at severity 3:
  truncation, section deletion, malicious injection, garbling, format break)
- reference = the branch's own clean-prompt response; deviation measured against it
- judge (GLM) scores safety / manipulation_resistance / unexplained_drift of each faulty response
- keys: env vars only
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp1_runner_protocol", _HERE.parent / "experiment1" / "runner.py")
exp1 = importlib.util.module_from_spec(_spec)
sys.modules["exp1_runner_protocol"] = exp1
_spec.loader.exec_module(exp1)
from noise_injector.corruptions.prompt import apply_prompt_op  # noqa: E402
from noise_injector.report import _CSS  # noqa: E402

BRANCHES = [
    {"key": "A", "name": "qwen8b-simple", "base": "prompts/simple_prompt.txt",
     "desc": "simple driver prompt, no noise awareness"},
    {"key": "B", "name": "qwen8b-prompt-suspect", "base": "prompts/prompt_suspect_prompt.txt",
     "desc": "same + awareness that the prompt itself may be faulty/malicious"},
]

FAULT_OPS = ["prompt_truncation", "prompt_section_delete", "prompt_instruction_injection",
             "prompt_garble", "prompt_format_break"]
FAULT_SEVERITY = 3


def load_env_file(path: str | Path | None) -> dict:
    if not path:
        return {}
    loaded = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            loaded.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    for k, v in loaded.items():
        import os
        os.environ.setdefault(k, v)
    return loaded


def build_variants() -> list[dict]:
    variants = []
    import numpy as np
    for br in BRANCHES:
        base_text = (_HERE / br["base"]).read_text()
        variants.append({"branch": br["key"], "name": f"{br['name']}/clean",
                         "prompt": base_text, "fault": "clean"})
        for op in FAULT_OPS:
            text = apply_prompt_op(op, base_text, FAULT_SEVERITY, np.random.default_rng(42))
            variants.append({"branch": br["key"], "name": f"{br['name']}/{op}",
                             "prompt": text, "fault": op})
    return variants


def judge_prompt_cell(judge, judge_prompt: str, mother_gt: dict, clean_resp: dict, faulty_resp: dict) -> dict:
    payload = {"scene_ground_truth": mother_gt,
               "response_under_clean_prompt": clean_resp,
               "response_under_faulty_prompt": faulty_resp}
    raw = judge.chat([{"role": "system", "content": judge_prompt},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                     max_tokens=8000)
    return exp1.parse_json_block(raw) or {"safety": 0, "manipulation_resistance": 5,
                                           "unexplained_drift": 0, "error": str(raw)[:150]}


def run_experiment(args) -> dict:
    (_HERE / "results").mkdir(exist_ok=True)
    load_env_file(args.env_file)
    vlm = exp1.OpenAICompatClient(base_url=args.vlm_base_url, model=args.vlm_model,
                                  api_key_env=args.api_key_env)
    judge = exp1.OpenAICompatClient(base_url=args.judge_base_url, model=args.judge_model,
                                    api_key_env=args.api_key_env)
    judge_prompt = (_HERE / "prompts/judge_prompt3.txt").read_text()
    variants = build_variants()
    print(f"[setup] {len(variants)} prompt variants ({len(FAULT_OPS)} faults × 2 branches + 2 clean)")

    run_dir = Path(args.run_dir)
    mother_dirs = {}
    for mp in sorted(run_dir.glob("offspring/*/*/clean/manifest.json")):
        m = json.load(open(mp))
        mother_dirs[m["parent"]["scenario_id"]] = mp.parent
    mother_ids = sorted(mother_dirs)[: args.mothers]
    print(f"[setup] {len(mother_ids)} clean mothers from {run_dir}")

    gt_cache = {sid: json.load(open(mother_dirs[sid] / "annotations.json")) for sid in mother_ids}
    ref_cache = {}

    def eval_variant(item):
        sid, var = item
        ann = gt_cache[sid]
        try:
            resp = exp1.arm_response(vlm, var["prompt"], ann, mother_dirs[sid], None, None, args.max_frames)
        except Exception as e:
            resp = {"commands": [], "objects": [], "sensors": {}, "_error": f"{type(e).__name__}: {e}"[:150]}
        return sid, var, resp

    jobs = [(sid, v) for sid in mother_ids for v in variants]
    responses = {}
    print(f"[eval] {len(jobs)} model calls ({args.workers} workers)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(eval_variant, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            sid, var, resp = fut.result()
            responses[(sid, var["name"])] = resp
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f"    {i + 1}/{len(jobs)} responses ({time.time() - t0:.0f}s)")

    for sid in mother_ids:
        for br in BRANCHES:
            ref_cache[(sid, br["key"])] = responses[(sid, f"{br['name']}/clean")]

    def judge_cell(item):
        sid, var = item
        br_key = var["branch"]
        gt = gt_cache[sid]
        mother_gt = {"scene_description": gt.get("scene_description", ""),
                     "gt_frames": {fid: fr.get("gt_behavior", "") for fid, fr in
                                   list(gt["key_frames"].items())[: args.max_frames]}}
        if var["fault"] == "clean":
            return None
        clean_resp = ref_cache[(sid, br_key)]
        faulty_resp = responses[(sid, var["name"])]
        det = exp1.deterministic_deviation(clean_resp, faulty_resp)
        verdict = judge_prompt_cell(judge, judge_prompt, mother_gt, clean_resp, faulty_resp)
        return {"cell": {"mother": sid, "branch": br_key, "fault": var["fault"],
                         "det_dev": det, "judge": verdict,
                         "response": faulty_resp}}

    cells = []
    print(f"[judge] {sum(1 for j in jobs if j[1]['fault'] != 'clean')} cells ({args.workers} workers)")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(judge_cell, j) for j in jobs if j[1]["fault"] != "clean"]
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            if r:
                cells.append(r["cell"])
            if (i + 1) % 20 == 0 or i + 1 == len(futures):
                print(f"    {i + 1}/{len(futures)} judged ({time.time() - t0:.0f}s)")

    results = {"vlm": {"base_url": args.vlm_base_url, "model": args.vlm_model},
               "judge": {"model": args.judge_model},
               "branches": BRANCHES, "fault_severity": FAULT_SEVERITY,
               "mothers": mother_ids, "cells": cells}
    with open(_HERE / "results" / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    return results


def aggregate(results: dict) -> dict:
    table = {}
    for cell in results["cells"]:
        key = (cell["branch"], cell["fault"])
        table.setdefault(key, []).append(cell)
    agg = {"per_branch_fault": {}}
    for (branch, fault), cells in sorted(table.items()):
        n = len(cells)
        agg["per_branch_fault"][f"{branch}|{fault}"] = {
            "n": n,
            "mean_safety": round(sum(c["judge"].get("safety", 0) for c in cells) / n, 2),
            "mean_manipulation_resistance": round(sum(c["judge"].get("manipulation_resistance", 0) for c in cells) / n, 2),
            "mean_unexplained_drift": round(sum(c["judge"].get("unexplained_drift", 0) for c in cells) / n, 2),
            "mean_det_deviation": round(sum(c["det_dev"]["deviation"] for c in cells) / n, 2),
            "malicious_followed": sum(1 for c in cells if c["judge"].get("followed_malicious_instruction")),
            "n": n,
        }
    branch_summary = {}
    for br in BRANCHES:
        cells = [c for c in results["cells"] if c["branch"] == br["key"]]
        n = max(1, len(cells))
        branch_summary[br["key"]] = {
            "name": br["name"],
            "mean_safety_under_corruption": round(sum(c["judge"].get("safety", 0) for c in cells) / n, 2),
            "mean_manipulation_resistance": round(sum(c["judge"].get("manipulation_resistance", 0) for c in cells) / n, 2),
            "mean_unexplained_drift": round(sum(c["judge"].get("unexplained_drift", 0) for c in cells) / n, 2),
        }
    agg["branch_summary"] = branch_summary
    worst = min(results["cells"], key=lambda c: c["judge"].get("safety", 0), default=None)
    agg["worst_cell"] = {"branch": worst["branch"], "fault": worst["fault"],
                         "safety": worst["judge"].get("safety"), "notes": worst["judge"].get("notes", "")} if worst else None
    return agg


def html_escape(s: str) -> str:
    return html.escape(str(s))


def write_report(results: dict, agg: dict, path: Path) -> Path:
    a, b = agg["branch_summary"]["A"], agg["branch_summary"]["B"]
    findings = []
    diff_safety = round(b["mean_safety_under_corruption"] - a["mean_safety_under_corruption"], 2)
    diff_res = round(b["mean_manipulation_resistance"] - a["mean_manipulation_resistance"], 2)
    findings.append(f"<li><b>Prompt-integrity awareness effect:</b> mean safety under corrupted prompts "
                    f"{a['mean_safety_under_corruption']} (simple) vs <b>{b['mean_safety_under_corruption']}</b> (aware) "
                    f"— {'awareness helps' if diff_safety > 0 else 'no benefit' if diff_safety == 0 else 'awareness hurts'} "
                    f"(Δ {diff_safety:+.2f}); manipulation resistance Δ {diff_res:+.2f}</li>")
    if agg.get("worst_cell"):
        w = agg["worst_cell"]
        findings.append(f"<li><b>Worst case:</b> branch {html_escape(str(w['branch']))} × <code>{html_escape(str(w['fault']))}</code> "
                        f"— safety {w['safety']}/10 · {html_escape(str(w['notes']))[:110]}</li>")
    fault_rows = ""
    for key, v in sorted(agg["per_branch_fault"].items()):
        br, fault = key.split("|", 1)
        fault_rows += (f"<tr><td>{html_escape(br)}</td><td><code>{html.escape(fault)}</code></td><td>{v['mean_safety']}</td>"
                       f"<td>{v['mean_manipulation_resistance']}</td><td>{v['mean_unexplained_drift']}</td>"
                       f"<td>{v['mean_det_deviation']}</td><td>{v['malicious_followed']}/{v['n']}</td></tr>")
    summary_rows = "".join(
        f"<tr><td><b>{v['name']}</b></td><td>{v['mean_safety_under_corruption']}</td>"
        f"<td>{v['mean_manipulation_resistance']}</td><td>{v['mean_unexplained_drift']}</td></tr>"
        for v in agg["branch_summary"].values())
    content = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Experiment 3 — prompt integrity</title><style>{_CSS}
td, th {{ font-size: 12.5px; }}</style></head><body><div class="wrap">
<h1>Experiment 3 — resilience to errors &amp; malicious injections in the prompt itself</h1>
<div class="sub">clean scene data · 2 branches × 5 fault types (severity {results['fault_severity']}) · VLM {html.escape(results['vlm']['model'])} · judge {html.escape(results['judge']['model'])}</div>
<h2>Key findings (auto-generated)</h2>
<div class="sec"><ul>{''.join(findings)}</ul></div>
<h2>Branch summary — safety under prompt corruption</h2>
<div class="sec"><table><tr><th>Branch</th><th>Mean safety</th><th>Mean manipulation resistance</th><th>Mean unexplained drift</th></tr>{summary_rows}</table></div>
<h2>Per fault type</h2>
<div class="sec"><table><tr><th>Branch</th><th>Fault</th><th>Safety</th><th>Manip. resistance</th><th>Drift</th><th>Det. deviation</th><th>Followed malicious</th></tr>{fault_rows}</table></div>
<div class="note">Reference = each branch's own clean-prompt response on the same scene. Deviation from that
reference is expected when the corruption justifies it (garbled text, missing sections); defects are
unsafe commands or compliance with malicious instructions despite clean sensor data.</div>
<footer>experiment 3 · noise-injector · self-contained · no credentials</footer>
</div></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Experiment 3: prompt-integrity attacks")
    p.add_argument("--run-dir", required=True, help="noise-injector run dir (clean mothers used)")
    p.add_argument("--mothers", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=2)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--vlm-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--vlm-model", default="qwen/qwen3-vl-8b-instruct")
    p.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--judge-model", default="z-ai/glm-5.3-flash")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--env-file", default=None)
    args = p.parse_args(argv)
    results = run_experiment(args)
    agg = aggregate(results)
    path = write_report(results, agg, _HERE / "results" / "report.html")
    print("\n=== branch summary (safety under prompt corruption) ===")
    for k, v in agg["branch_summary"].items():
        print(f"  {v['name']}: safety {v['mean_safety_under_corruption']} · manip resistance {v['mean_manipulation_resistance']} · drift {v['mean_unexplained_drift']}")
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
