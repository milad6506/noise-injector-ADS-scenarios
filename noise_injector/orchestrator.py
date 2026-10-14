"""Round-based resilience orchestrator.

The user provides: an LLM engine (manages the process), targets (training datasets
+ the target model deployed on the device), and a round budget. The engine proposes
each round's perturbations from the family catalog given the results so far; the
injector renders them; targets are evaluated; resilience is aggregated and reported.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .adapters.base import get_adapter
from .llm import LLMBackend
from .render import make_recipe, render_scenario
from .report import _CSS
from .scoring import resilience_summary
from .spec import ALL_FAMILIES, PATTERNS, family, list_families
from .targets import make_target
from .temporal import derive_seed

_POLICY_SYSTEM = (
    "You manage fault-injection rounds for testing the resilience of autonomous-driving models. "
    "You will receive: a catalog of available corruption families, the results of previous rounds, "
    "and the remaining budget. Propose the NEXT round as a small set of perturbations that maximizes "
    "what we learn about the target models' resilience: probe untested families, escalate severities "
    "where models still cope, and stay within the ops budget. Respond with ONLY JSON: "
    '{"ops": [{"family": "<name>", "severity": <1-5>, "pattern": "<persistent|intermittent|burst|transient|camera_scoped>"}], '
    '"rationale": "<max 40 words>"}'
)


def load_experiment(path: str | Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    for key in ("engine", "targets", "dataset"):
        if key not in raw:
            raise ValueError(f"experiment file missing key: {key}")
    return raw


def fallback_policy(round_no: int, fam_names: list[str], ops_per_round: int, severity_cap: int) -> dict:
    ops = []
    for i in range(min(ops_per_round, len(fam_names))):
        fam = fam_names[(round_no * ops_per_round + i) % len(fam_names)]
        sev = min(severity_cap, 1 + (round_no * ops_per_round + i) // len(fam_names))
        ops.append({"family": fam, "severity": sev, "pattern": family(fam).default_pattern})
    return {"ops": ops, "rationale": "fallback: deterministic rotation sweep (no engine proposal)"}


def propose_policy(backend: LLMBackend, catalog: list[dict], results: list[dict],
                   ops_per_round: int, round_no: int, rounds_left: int) -> dict:
    recent = [{k: r[k] for k in ("family", "severity", "target", "score") if k in r}
              for r in results[-12:]]
    user = json.dumps({
        "families": catalog,
        "previous_results": recent,
        "ops_budget_this_round": ops_per_round,
        "round": round_no,
        "rounds_left": rounds_left,
    }, ensure_ascii=False)
    out = backend.chat(_POLICY_SYSTEM, user, temperature=0.0, max_tokens=500)
    start = out.find("{")
    if start < 0:
        raise ValueError(f"engine returned no JSON: {out[:120]}")
    obj, _ = json.JSONDecoder().raw_decode(out[start:])
    valid_ops = []
    for op in obj.get("ops", [])[:ops_per_round]:
        if not isinstance(op, dict) or op.get("family") not in ALL_FAMILIES:
            continue
        try:
            sev = int(np.clip(int(op.get("severity", 3)), 1, 5))
        except (TypeError, ValueError):
            sev = 3
        pattern = op.get("pattern")
        valid_ops.append({
            "family": op["family"],
            "severity": sev,
            "pattern": pattern if pattern in PATTERNS else family(op["family"]).default_pattern,
        })
    if not valid_ops:
        raise ValueError(f"engine proposed no valid ops: {out[:200]}")
    return {"ops": valid_ops, "rationale": str(obj.get("rationale", ""))[:200]}


class Orchestrator:
    def __init__(self, exp: dict, out_dir: Path):
        self.exp = exp
        self.out_dir = Path(out_dir)
        self.name = exp.get("name", "experiment")
        self.rounds = int(exp.get("rounds", 3))
        self.ops_per_round = int(exp.get("budget", {}).get("ops_per_round", 2))
        self.severity_cap = int(exp.get("severity_cap", 5))
        self.scenario_selector = str(exp.get("scenarios", "1"))
        allowed = exp.get("families")
        fam_names = [n for n, f in ALL_FAMILIES.items() if f.layer in ("A", "B", "C")]
        if allowed:
            fam_names = [n for n in fam_names if n in set(allowed)]
        self.fam_names = fam_names
        self.state_path = self.out_dir / "state.json"
        self.results: list[dict] = []
        self.round_log: list[dict] = []

    def _catalog(self) -> list[dict]:
        fams = list_families()
        return [
            {"family": e["family"], "layer": layer, "pattern": e["pattern"], "description": e["description"]}
            for layer in ("A", "B", "C")
            for e in fams[layer]
            if e["family"] in self.fam_names
        ]

    def _load_state(self) -> None:
        if self.state_path.exists():
            with open(self.state_path) as f:
                st = json.load(f)
            self.results = st.get("results", [])
            self.round_log = st.get("round_log", [])

    def _save_state(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"results": self.results, "round_log": self.round_log}, f, indent=1)
        tmp.replace(self.state_path)

    def _select_scenarios(self, adapter) -> list[str]:
        all_ids = adapter.list_scenarios()
        sel = self.scenario_selector
        if sel == "all":
            return all_ids
        if sel.isdigit():
            return all_ids[: int(sel)]
        wanted = [s.strip() for s in sel.split(",") if s.strip()]
        missing = [w for w in wanted if w not in all_ids]
        if missing:
            raise KeyError(f"scenarios not found: {missing}")
        return wanted

    def _augment_gt_file(self, dest: Path, parent: dict) -> None:
        ann_path = dest / "annotations.json"
        ann = json.load(open(ann_path))
        for fid, fr in ann.get("key_frames", {}).items():
            src = parent["key_frames"].get(fid, {})
            qa = src.get("QA", {})
            behavior = [i.get("A", "") for i in qa.get("behavior", []) if isinstance(i, dict)]
            planning = [i.get("A", "") for i in qa.get("planning", []) if isinstance(i, dict)]
            fr["gt_behavior"] = " ".join(behavior) or " ".join(planning)
        with open(ann_path, "w") as f:
            json.dump(ann, f, ensure_ascii=False, indent=1)

    def evaluate_offspring(self, offspring_dir: Path, targets: list, family_name: str | None,
                           severity: int | None) -> list[dict]:
        rows = []
        for t in targets:
            started = time.time()
            try:
                metrics = t.evaluate(offspring_dir, annotations=None)
                err = None
            except Exception as e:
                metrics = {"score": 0.0}
                err = f"{type(e).__name__}: {e}"[:200]
            rows.append({
                "round": None,
                "family": family_name,
                "severity": severity,
                "target": t.name,
                "role": t.role,
                "score": float(metrics.get("score", 0.0)),
                "metrics": {k: v for k, v in metrics.items() if k != "per_frame"},
                "eval_s": round(time.time() - started, 1),
                "error": err,
            })
            print(f"    [{t.name}] {'FAILED ' + err if err else 'score ' + str(rows[-1]['score'])}")
        return rows

    def run(self) -> dict:
        engine_cfg = self.exp["engine"]
        backend = LLMBackend(engine_cfg["base_url"], engine_cfg["model"],
                             engine_cfg.get("api_key_env", "NOISE_INJECTOR_API_KEY"))
        targets = [make_target(t, judge_backend=backend) for t in self.exp["targets"]]

        ds_cfg = self.exp["dataset"]
        adapter = get_adapter(ds_cfg["type"], Path(ds_cfg["root"]), max_frames=ds_cfg.get("max_frames", 60))
        ids = self._select_scenarios(adapter)

        self._load_state()
        done_rounds = {rl["round"] for rl in self.round_log}
        catalog = self._catalog()

        if 0 not in done_rounds:
            print("[round 0] clean baseline")
            clean_rows = []
            for sid in ids:
                sc = adapter.get_scenario(sid)
                dest = self.out_dir / "offspring" / sid / "clean"
                render_scenario(sc, {"name": "clean", "ops": []}, dest,
                                derive_seed(42, sid, "clean"), root=str(adapter.root))
                parent = {"key_frames": {f.frame_id: {"QA": f.qa} for f in sc.frames}}
                self._augment_gt_file(dest, parent)
                clean_rows += self.evaluate_offspring(dest, targets, None, None)
            for r in clean_rows:
                r["round"] = 0
            self.results = clean_rows
            self.round_log.append({"round": 0, "ops": [], "rationale": "clean baseline",
                                   "source": "none", "rows": clean_rows})
            self._save_state()

        for round_no in range(1, self.rounds + 1):
            if round_no in done_rounds:
                continue
            try:
                policy = propose_policy(backend, catalog, self.results, self.ops_per_round,
                                        round_no, self.rounds + 1 - round_no)
                policy_source = "engine"
            except Exception as e:
                print(f"    engine proposal failed ({str(e)[:120]}) — using fallback sweep")
                policy = fallback_policy(round_no - 1, self.fam_names, self.ops_per_round, self.severity_cap)
                policy_source = "fallback"
            print(f"[round {round_no}] ({policy_source}) ops: {[(o['family'], o['severity']) for o in policy['ops']]}")
            rows = []
            for sid in ids:
                sc = adapter.get_scenario(sid)
                parent = {"key_frames": {f.frame_id: {"QA": f.qa} for f in sc.frames}}
                for op in policy["ops"]:
                    recipe = make_recipe(op["family"], op["family"], [op["severity"]])[0]
                    recipe["ops"][0]["pattern"] = op["pattern"]
                    dest = self.out_dir / "offspring" / sid / f"{op['family']}_s{op['severity']}"
                    render_scenario(sc, recipe, dest,
                                    derive_seed(42, sid, op["family"], op["severity"], round_no),
                                    root=str(adapter.root))
                    self._augment_gt_file(dest, parent)
                    rows += self.evaluate_offspring(dest, targets, op["family"], op["severity"])
            for r in rows:
                r["round"] = round_no
            self.results += rows
            self.round_log.append({"round": round_no, "ops": policy["ops"], "rationale": policy["rationale"],
                                   "source": policy_source, "rows": rows})
            self._save_state()

        summary = resilience_summary(self.results)
        report_path = self.build_report(summary, engine_cfg)
        print(f"done: {len(self.results)} evaluations over {self.rounds + 1} rounds -> {report_path}")
        return {"summary": summary, "report": str(report_path), "results": self.results}

    def build_report(self, summary: dict, engine_cfg: dict) -> Path:
        rows_html = ""
        for rl in self.round_log:
            ops = ", ".join(f"<code>{o['family']}</code>S{o['severity']}" for o in rl["ops"]) or "<span class='sub'>clean</span>"
            per_target = {}
            for r in rl["rows"]:
                per_target.setdefault(r["target"], []).append(r["score"])
            tgt = " · ".join(f"{t}: {sum(s) / len(s):.2f}" for t, s in per_target.items())
            rat = html.escape(str(rl.get("rationale", ""))[:160])
            rows_html += f"<tr><td>{rl['round']}</td><td>{ops}</td><td>{tgt}</td><td class='sub'>{rat}</td></tr>"

        ret_rows = ""
        for r in summary["retentions"]:
            ret = r["retention"]
            color = "var(--ok)" if (ret or 0) >= 0.8 else ("var(--warn)" if (ret or 0) >= 0.5 else "var(--bad)")
            ret_rows += (f"<tr><td>{html.escape(r['target'])}</td><td><code>{html.escape(r['family'])}</code>"
                         f"</td><td>{r['severity']}</td><td>{r['clean']}</td><td>{r['noisy']}</td>"
                         f"<td style='color:{color}'>{ret}</td></tr>")

        sum_rows = "".join(
            f"<tr><td>{html.escape(t)}</td><td>{v['clean_score']}</td>"
            f"<td>{v['mean_retention']}</td><td>{v['min_retention']}</td></tr>"
            for t, v in summary["per_target"].items()
        )

        content = f"""<!doctype html><html><head><meta charset="utf-8">
<title>resilience run — {html.escape(self.name)}</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>Resilience run — {html.escape(self.name)}</h1>
<div class="sub">engine: <code>{html.escape(engine_cfg['model'])}</code> manages the rounds · dataset: <code>{html.escape(self.exp['dataset']['type'])}</code> · dataset paths redacted</div>
<h2>Final resilience summary</h2>
<div class="sec"><table><tr><th>Target</th><th>Clean score</th><th>Mean retention</th><th>Min retention</th></tr>{sum_rows}</table>
<div class="sub">retention = noisy score / clean score · resilience = mean retention across all injected offspring</div></div>
<h2>Retention per (family, severity)</h2>
<div class="sec"><table><tr><th>Target</th><th>Family</th><th>Sev</th><th>Clean</th><th>Noisy</th><th>Retention</th></tr>{ret_rows}</table></div>
<h2>Rounds</h2>
<div class="sec"><table><tr><th>Round</th><th>Ops</th><th>Mean scores</th><th>Engine rationale</th></tr>{rows_html}</table></div>
<footer>noise-injector orchestrator · self-contained report · no paths, no credentials</footer>
</div></body></html>"""
        path = self.out_dir / "resilience_report.html"
        path.write_text(content)
        return path
