#!/usr/bin/env python3
"""Experiment 1 — resilience of a driving VLM to injected noise: noise-blind vs noise-aware.

Protocol:
- data comes from a noise-injector run directory (mothers = clean renders, children = perturbed)
- two arms use the SAME VLM (Qwen3-VL-8B) with the SAME minimal driver prompt, except ONE is
  told the input line may carry noise/faults
- each arm produces, per scenario: commands + objects + sensor states — for the mother sequence
  and for each child sequence (mother always shown as reference)
- deviation (deterministic): distance between an arm's own mother-response and child-response
- judge (GLM via OpenRouter): sees the mother ground truth + both arms' responses, blind-labeled,
  and scores deviation/resilience/noise_handling per arm
- keys come from env vars only (loaded via --env-file); nothing secret ever enters the repo
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from noise_injector.report import _CSS  # noqa: E402

_HERE = Path(__file__).resolve().parent


def load_env_file(path: str | Path | None) -> dict:
    if not path:
        return {}
    loaded = {}
    p = Path(path).expanduser()
    if not p.exists():
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        loaded[k] = v
        os.environ.setdefault(k, v)
    return loaded


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENROUTER_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env

    def chat(self, messages: list, max_tokens: int = 700, temperature: float = 0.0) -> str:
        for attempt in range(5):
            try:
                return self._chat_once(messages, max_tokens, temperature)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < 4:
                    wait = 15.0 * (attempt + 1)
                    print(f"    [rate-limit {e.code}] waiting {wait:.0f}s (attempt {attempt + 1}/5)")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("unreachable")

    def _chat_once(self, messages: list, max_tokens: int, temperature: float) -> str:
        key = os.environ.get(self.api_key_env)
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
        if "error" in data:
            raise RuntimeError(f"endpoint error: {str(data['error'])[:150]}")
        msg = data["choices"][0].get("message", {})
        content = msg.get("content")
        if not content:
            content = msg.get("reasoning") or ""
        return content or ""


class FakeClient(OpenAICompatClient):
    def __init__(self, canned: str = "{}", **kw):
        super().__init__(base_url="http://localhost:0", model="fake", **kw)
        self.canned = canned
        self.calls = []

    def chat(self, messages, max_tokens=700, temperature=0.0):
        self.calls.append(messages)
        return self.canned


def parse_json_block(text: str) -> dict | None:
    if not text:
        return None
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _data_url(path: Path, width: int = 300, quality: int = 60) -> str | None:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


def sequence_parts(run_dir: Path, ann: dict, max_frames: int, tag: str) -> list:
    parts = [{"type": "text", "text": f"=== {tag} SEQUENCE ==="}]
    for fid in list(ann["key_frames"])[:max_frames]:
        fr = ann["key_frames"][fid]
        parts.append({"type": "text", "text": f"frame {fid} — six surround views:"})
        for cam in sorted(fr.get("image_paths") or {}):
            rel = (fr.get("image_paths") or {}).get(cam)
            if rel and (run_dir / rel).exists():
                du = _data_url(run_dir / rel)
                if du:
                    parts.append({"type": "text", "text": f"{cam}:"})
                    parts.append({"type": "image_url", "image_url": {"url": du}})
        gt = fr.get("gt_behavior")
        if gt:
            parts.append({"type": "text", "text": f"annotated ego behavior at this frame: {gt}"})
    return parts


def arm_response(client, prompt_text: str, mother: dict, mother_dir: Path,
                 child: dict | None, child_dir: Path | None, max_frames: int) -> dict:
    parts = sequence_parts(mother_dir, mother, max_frames, "MOTHER (reference — same drive, healthy sensors)")
    tag = "CHILD (the drive you must act on now)"
    if child is None:
        parts = sequence_parts(mother_dir, mother, max_frames, "CURRENT SEQUENCE (this is the mother drive itself)")
    else:
        parts += sequence_parts(child_dir, child, max_frames, tag)
    parts.append({"type": "text", "text": "Produce the required JSON for the CURRENT sequence."})
    raw = client.chat([{"role": "system", "content": prompt_text},
                       {"role": "user", "content": parts}])
    obj = parse_json_block(raw) or {"commands": [], "objects": [], "sensors": {}, "raw": raw[:200]}
    obj["_raw_len"] = len(raw)
    return obj


def deterministic_deviation(mother_resp: dict, child_resp: dict) -> dict:
    def jaccard(a: list, b: list) -> float:
        sa = {s.strip().lower() for s in a if isinstance(s, str)}
        sb = {s.strip().lower() for s in b if isinstance(s, str)}
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(1, len(sa | sb))

    def multiset_sim(a: list, b: list) -> float:
        ca = Counter(str(o.get("class", "?")).lower() for o in a if isinstance(o, dict))
        cb = Counter(str(o.get("class", "?")).lower() for o in b if isinstance(o, dict))
        inter = sum((ca & cb).values())
        return inter / max(1, sum(ca.values()), sum(cb.values()))

    cmd_sim = jaccard(mother_resp.get("commands", []), child_resp.get("commands", []))
    obj_sim = multiset_sim(mother_resp.get("objects", []), child_resp.get("objects", []))
    sm, sc_ = mother_resp.get("sensors", {}), child_resp.get("sensors", {})
    keys = set(sm) | set(sc_)
    sensor_agree = (sum(1 for k in keys if sm.get(k) == sc_.get(k)) / len(keys)) if keys else 1.0
    deviation = round(10 * (1 - (0.5 * cmd_sim + 0.3 * obj_sim + 0.2 * sensor_agree)), 2)
    return {"deviation": deviation, "command_similarity": round(cmd_sim, 3),
            "object_similarity": round(obj_sim, 3), "sensor_agreement": round(sensor_agree, 3)}


def judge_child(judge: OpenAICompatClient, judge_prompt: str, mother_gt: dict, agent_rows: list[dict]) -> dict:
    payload = {
        "mother": mother_gt,
        "agents": [{"agent": f"Agent {i + 1}",
                    "response_to_mother": r["mother_resp"],
                    "response_to_child": r["child_resp"],
                    "deterministic_deviation": r["det_dev"]} for i, r in enumerate(agent_rows)],
    }
    raw = judge.chat([{"role": "system", "content": judge_prompt},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                     max_tokens=8000)
    obj = parse_json_block(raw) or {"error": raw[:200], "winner": "tie"}
    return obj


ARMS = [
    {"key": "A", "name": "qwen8b-blind", "prompt_file": "prompts/blind_prompt.txt",
     "desc": "minimal driver prompt, no noise awareness"},
    {"key": "B", "name": "qwen8b-noise-aware", "prompt_file": "prompts/noise_aware_prompt.txt",
     "desc": "same prompt + awareness that the input line may carry noise/faults"},
]


def run_experiment(args) -> dict:
    load_env_file(args.env_file)
    vlm_cfg = {"base_url": args.vlm_base_url, "model": args.vlm_model, "api_key_env": args.api_key_env}
    judge_cfg = {"base_url": args.judge_base_url, "model": args.judge_model, "api_key_env": args.api_key_env}
    vlm = OpenAICompatClient(**vlm_cfg)
    judge = OpenAICompatClient(**judge_cfg)
    prompts = {a["key"]: (_HERE / a["prompt_file"]).read_text() for a in ARMS}
    judge_prompt = (_HERE / "prompts/judge_prompt.txt").read_text()

    run_dir = Path(args.run_dir)
    manifests = sorted(run_dir.glob("offspring/*/*/*/manifest.json"))
    mothers = {}
    children_by_mother: dict[str, list] = {}
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
    results = {"arms": ARMS, "vlm": vlm_cfg, "judge": judge_cfg, "cells": [], "meta": {
        "run_dir": str(run_dir), "max_frames": args.max_frames,
        "mothers": mother_ids, "per_mother": args.per_mother}}
    out_dir = _HERE / "results"
    out_dir.mkdir(exist_ok=True)

    for midx, sid in enumerate(mother_ids):
        mother_dir = mothers[sid]
        mother_ann = json.load(open(mother_dir / "annotations.json"))
        child_dirs = []
        used_fams = set()
        pool = children_by_mother.get(sid, [])
        if getattr(args, "sample_strategy", "default") == "harsh":
            pool = sorted(pool, key=lambda x: (-x[2], x[0]))
        else:
            pool = sorted(pool, key=lambda x: (x[0], -x[2]))
        for fam, d, _sev in pool:
            if len(child_dirs) >= args.per_mother:
                break
            if fam in used_fams and len(child_dirs) + len(
                    set(f for f, _, _ in pool if f not in used_fams)) > args.per_mother:
                continue
            used_fams.add(fam)
            child_dirs.append(d)
        print(f"[{midx + 1}/{len(mother_ids)}] mother {sid[:20]}… — {len(child_dirs)} children")

        resp_mother = {a["key"]: arm_response(vlm, prompts[a["key"]], mother_ann, mother_dir,
                                              None, None, args.max_frames) for a in ARMS}
        for cdir in child_dirs:
            child_ann = json.load(open(cdir / "annotations.json"))
            man = json.load(open(cdir / "manifest.json"))
            ev = man["injected_events"][0]
            noise_mode = f"{ev['family']} (pattern {ev['pattern']}, severity {ev.get('severity')})"
            agent_rows = []
            cell = {"mother": sid, "child": cdir.name, "noise_mode": noise_mode, "arms": {}}
            for a in ARMS:
                rc = arm_response(vlm, prompts[a["key"]], mother_ann, mother_dir,
                                  child_ann, cdir, args.max_frames)
                det = deterministic_deviation(resp_mother[a["key"]], rc)
                agent_rows.append({"arm": a["key"], "mother_resp": resp_mother[a["key"]],
                                   "child_resp": rc, "det_dev": det})
                cell["arms"][a["key"]] = {"child_resp": rc, "det_dev": det}
            verdict = judge_child(judge, judge_prompt,
                                  {"scene_description": mother_ann.get("scene_description", ""),
                                   "gt_frames": {fid: fr.get("gt_behavior", "") for fid, fr in
                                                 list(mother_ann["key_frames"].items())[: args.max_frames]}},
                                  agent_rows)
            cell["judge"] = verdict
            cell["judge_usage_ok"] = not verdict.get("error")
            results["cells"].append(cell)
            print(f"    child {cdir.name}: judge → {verdict.get('winner', '?')}")
            with open(out_dir / "results.json", "w") as f:
                json.dump(results, f, indent=1)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    return results


def aggregate(results: dict) -> dict:
    per_arm = {}
    for a in results["arms"]:
        det, res, nh = [], [], []
        for cell in results["cells"]:
            ja = cell["judge"].get(f"agent{1 if a['key'] == 'A' else 2}", {})
            det.append(cell["arms"][a["key"]]["det_dev"]["deviation"])
            res.append(ja.get("resilience", 0))
            nh.append(ja.get("noise_handling", 0))
        n = max(1, len(det))
        per_arm[a["key"]] = {
            "name": a["name"], "desc": a["desc"],
            "mean_det_deviation": round(sum(det) / n, 2),
            "mean_resilience": round(sum(res) / n, 2),
            "mean_noise_handling": round(sum(nh) / n, 2),
        }
    wins = Counter(c["judge"].get("winner", "tie") for c in results["cells"])

    per_mode = {}
    modes = {}
    for cell in results["cells"]:
        modes.setdefault(cell["noise_mode"], []).append(cell)
    for mode, cells in modes.items():
        entry = {"n": len(cells), "wins": dict(Counter(c["judge"].get("winner", "tie") for c in cells))}
        for a in results["arms"]:
            jk = f"agent{1 if a['key'] == 'A' else 2}"
            entry[f"res_{a['key']}"] = round(sum(c["judge"].get(jk, {}).get("resilience", 0) for c in cells) / len(cells), 2)
            entry[f"det_{a['key']}"] = round(sum(c["arms"][a["key"]]["det_dev"]["deviation"] for c in cells) / len(cells), 2)
        per_mode[mode] = entry

    worst = max(results["cells"],
                key=lambda c: c["arms"]["A"]["det_dev"]["deviation"] + c["arms"]["B"]["det_dev"]["deviation"],
                default=None)
    best = max(results["cells"],
               key=lambda c: min(c["judge"].get("agent1", {}).get("resilience", 0),
                                 c["judge"].get("agent2", {}).get("resilience", 0)),
               default=None)
    return {"per_arm": per_arm, "wins": dict(wins), "per_mode": per_mode, "worst_cell": worst, "best_cell": best}


def write_report(results: dict, agg: dict, path: Path) -> Path:
    rows = ""
    for cell in results["cells"]:
        j = cell["judge"]
        a1 = j.get("agent1", {})
        a2 = j.get("agent2", {})
        w = j.get("winner", "tie")
        rows += f"""<tr>
          <td><code>{html.escape(cell['mother'][:16])}…</code></td>
          <td><b>{html.escape(cell['noise_mode'])}</b></td>
          <td>{a1.get('deviation', '?')} / {a1.get('resilience', '?')} / {a1.get('noise_handling', '?')}</td>
          <td>{a2.get('deviation', '?')} / {a2.get('resilience', '?')} / {a2.get('noise_handling', '?')}</td>
          <td style="color: var(--{'ok' if w == 'agent2' else ('warn' if w == 'tie' else 'bad')})">{html.escape(w)}</td>
          <td class="sub">{html.escape(str(a1.get('notes', ''))[:60])} · {html.escape(str(a2.get('notes', ''))[:60])}</td>
        </tr>"""
    arm_rows = "".join(
        f"<tr><td><b>{v['name']}</b><div class='sub'>{html.escape(v['desc'])}</div></td>"
        f"<td>{v['mean_det_deviation']}</td><td>{v['mean_resilience']}</td>"
        f"<td>{v['mean_noise_handling']}</td></tr>"
        for v in agg["per_arm"].values())

    findings = []
    a, b = agg["per_arm"]["A"], agg["per_arm"]["B"]
    wins = agg["wins"]
    blind_key = max(wins, key=wins.get) if wins else "tie"
    findings.append(f"<li><b>Arm comparison:</b> deviation {a['mean_det_deviation']} (blind) vs "
                    f"{b['mean_det_deviation']} (aware) · resilience {a['mean_resilience']} vs "
                    f"{b['mean_resilience']} · judge wins {html.escape(json.dumps(wins))} — "
                    f"{'noise awareness pays off' if b['mean_resilience'] > a['mean_resilience'] and blind_key == 'B' else 'no clear advantage of awareness in this sample'}</li>")
    if agg.get("worst_cell"):
        wc = agg["worst_cell"]
        worst_mode = wc["noise_mode"]
        findings.append(f"<li><b>Hardest noise mode:</b> <code>{html.escape(worst_mode)}</code> — combined deviation "
                        f"{wc['arms']['A']['det_dev']['deviation']} (blind) / {wc['arms']['B']['det_dev']['deviation']} (aware)</li>")
    mode_rows = ""
    for mode, entry in sorted(agg["per_mode"].items()):
        findings.append(f"<li><b>{html.escape(mode)}</b>: resilience blind {entry['res_A']} vs aware {entry['res_B']}"
                        f" · deterministic deviation {entry['det_A']} vs {entry['det_B']} · wins {html.escape(json.dumps(entry['wins']))}</li>")
        mode_rows += (f"<tr><td>{html.escape(mode)}</td><td>{entry['n']}</td>"
                      f"<td>{entry['det_A']} vs {entry['det_B']}</td>"
                      f"<td>{entry['res_A']} vs {entry['res_B']}</td>"
                      f"<td>{html.escape(json.dumps(entry['wins']))}</td></tr>")
    key_findings = f"<div class='sec'><ul>{''.join(findings)}</ul></div>"

    content = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Experiment 1 — noise-blind vs noise-aware VLM</title><style>{_CSS}
td, th {{ font-size: 12.5px; }}</style></head><body><div class="wrap">
<h1>Experiment 1 — resilience of a driving VLM to injected noise</h1>
<div class="sub">arms: Qwen3-VL-8B blind vs noise-aware (identical otherwise) · judge: {html.escape(results['judge']['model'])} ·
data: noise-injector run ({len(results['cells'])} cells)</div>
<h2>Key findings (auto-generated)</h2>
{key_findings}
<h2>Aggregate</h2>
<div class="sec"><table><tr><th>Arm</th><th>Mean deviation (det.)</th><th>Mean resilience (judge)</th><th>Mean noise handling</th></tr>{arm_rows}</table>
<div class="sub">judge winners: {html.escape(json.dumps(agg['wins']))}</div></div>
<h2>Per noise mode</h2>
<div class="sec"><table><tr><th>Noise mode</th><th>n</th><th>Det. deviation (blind vs aware)</th><th>Resilience (blind vs aware)</th><th>Wins</th></tr>{mode_rows}</table></div>
<h2>Per child</h2>
<div class="sec"><table><tr><th>Mother</th><th>Noise mode</th><th>A1: dev/res/nh</th><th>A2: dev/res/nh</th><th>Winner</th><th>Notes</th></tr>{rows}</table></div>
<div class="note">Protocol: both arms see the mother sequence as reference; deviation = drift of a child response from
the arm's own mother response (expected when it reflects a real injected fault; defective when unexplained).
The judge is blind to which arm is noise-aware.</div>
<footer>experiment 1 · noise-injector · self-contained · no credentials</footer>
</div></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Experiment 1: noise-blind vs noise-aware driving VLM")
    p.add_argument("--run-dir", required=True, help="noise-injector run dir (offspring/ with mothers)")
    p.add_argument("--mothers", type=int, default=3)
    p.add_argument("--per-mother", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=2)
    p.add_argument("--sample-strategy", choices=["default", "harsh"], default="default",
                   help="harsh: sample the highest-severity children (direct severe-noise test)")
    p.add_argument("--vlm-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--vlm-model", default="qwen/qwen3-vl-8b-instruct")
    p.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--judge-model", default="z-ai/glm-5.3-flash")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--env-file", default=None, help="optional .env file with the API key (never committed)")
    args = p.parse_args(argv)
    results = run_experiment(args)
    agg = aggregate(results)
    path = write_report(results, agg, _HERE / "results" / "report.html")
    print("\n=== aggregate ===")
    for k, v in agg["per_arm"].items():
        print(f"  {v['name']}: dev {v['mean_det_deviation']} · resilience {v['mean_resilience']} · noise handling {v['mean_noise_handling']}")
    print(f"  judge winners: {agg['wins']}")
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
