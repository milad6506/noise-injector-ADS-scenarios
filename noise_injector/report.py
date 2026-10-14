"""Self-contained HTML report generator for offspring scenario trees."""

from __future__ import annotations

import base64
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from . import __version__
from .spec import list_families

_CSS = """
:root { --bg:#0f1115; --card:#171a21; --line:#262b36; --text:#d7dae0; --mut:#8b93a5; --acc:#4f8cff; --ok:#2fbf71; --warn:#e0a526; --bad:#e05252; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.55 -apple-system, 'Segoe UI', Roboto, sans-serif; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 28px 20px 60px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 34px 0 10px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 14px; margin: 18px 0 6px; color: var(--acc); }
.sub { color: var(--mut); font-size: 12.5px; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin:18px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 18px; min-width:130px; }
.card b { display:block; font-size:22px; } .card span { color:var(--mut); font-size:11.5px; text-transform:uppercase; letter-spacing:.06em; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 12.5px; }
th { text-align:left; color:var(--mut); font-weight:600; padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
td { padding:6px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
.tag { display:inline-block; padding:1px 8px; border-radius:99px; font-size:11px; margin-right:4px; border:1px solid var(--line); background:var(--card); }
.tag.A{border-color:#4f8cff;color:#4f8cff}.tag.B{border-color:#2fbf71;color:#2fbf71}.tag.C{border-color:#e0a526;color:#e0a526}.tag.F{border-color:#c06cff;color:#c06cff}.tag.D,.tag.E{border-color:#e05252;color:#e05252}
.sec { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 18px; margin:14px 0; }
.imgs { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
.imgs figure { margin:0; } .imgs img { width:300px; border-radius:6px; border:1px solid var(--line); display:block; }
.imgs figcaption { color:var(--mut); font-size:10.5px; padding-top:3px; }
code { background:#0d0f13; padding:1px 6px; border-radius:4px; font-size:12px; color:#9ecbff; }
.note { border-left:3px solid var(--acc); background:var(--card); padding:10px 14px; border-radius:0 8px 8px 0; margin:12px 0; font-size:13px; }
.ok { color: var(--ok); } .warn { color: var(--warn); }
.grid2 { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 760px){ .grid2{grid-template-columns:1fr} }
.tree { list-style:none; margin:10px 0; padding-left:0; font-size:13px; }
.tree ul { list-style:none; margin:0; padding-left:26px; position:relative; }
.tree ul::before { content:''; border-left:1px solid var(--line); position:absolute; left:10px; top:0; bottom:8px; }
.tree li { position:relative; padding:3px 0 3px 12px; }
.tree li::before { content:''; border-top:1px solid var(--line); position:absolute; left:10px; top:14px; width:10px; }
.tree .node { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.tree .root-node { font-weight:700; }
.tree .leaf { color:var(--text); }
.tree .meta { color:var(--mut); font-size:11px; }
.tree .child { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:2px 10px; }
.tree .badges { display:inline-flex; gap:4px; align-items:center; }
.tree .ev { color:var(--warn); font-size:11px; }
footer { color:var(--mut); font-size:11.5px; margin-top:40px; }
"""


def _b64_image(path: Path, width: int = 300, quality: int = 78) -> str | None:
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / w)))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()
    except Exception:
        return None


def _fmt_event(e: dict) -> str:
    fam = html.escape(str(e.get("family")))
    layer = html.escape(str(e.get("layer")))
    pat = html.escape(str(e.get("pattern", "")))
    n = sum(
        len(t.get("detail", [])) if isinstance(t.get("detail"), list) else len(t.get("cameras", []))
        for t in e.get("targets", [])
    ) or len(e.get("targets", []))
    backend = f" · <span class='sub'>{html.escape(e.get('backend') or '')}</span>" if e.get("backend") else ""
    return f"<span class='tag {layer}'>{layer}</span> <code>{fam}</code> <span class='sub'>{pat} · ~{n} targets{backend}</span>"


def _scrub(text: str) -> str:
    if "/" in text and (text.startswith("/") or text.startswith("~") or "Users/" in text):
        return text.rsplit("/", 1)[-1]
    return text


def collect(offspring_root: Path) -> list[dict]:
    found = []
    for mp in sorted(Path(offspring_root).rglob("manifest.json")):
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        found.append({"manifest": m, "dir": mp.parent, "rel": mp.parent.relative_to(offspring_root)})
    return found


def tree_html(runs: list[dict]) -> str:
    """Parent scenario → children offspring hierarchy."""
    by_adapter: dict[str, dict[str, list[dict]]] = {}
    for r in runs:
        m = r["manifest"]
        adapter = m["parent"]["adapter"]
        sid = _scrub(str(m["parent"]["scenario_id"]))
        by_adapter.setdefault(adapter, {}).setdefault(sid, []).append(r)

    def node_badges(m: dict) -> str:
        evs = m.get("injected_events", [])
        badges = []
        for e in evs:
            layer = e.get("layer", "")
            fam = html.escape(str(e.get("family")))
            sev = e.get("params", {}).get("severity") or e.get("strength") or ""
            badges.append(f"<span class='tag {layer}'>{layer}</span><code>{fam}</code>"
                          f"<span class='ev'>S{sev}</span>")
        n = sum(len(t.get("detail", [])) if isinstance(t.get("detail"), list) else len(t.get("targets", [])) for e in evs for t in e.get("targets", []))
        return (" ".join(badges) +
                f" <span class='meta'>{len(evs)} op(s) · ~{n} targets · {len(m.get('output_files', []))} files</span>")

    parts = ["<ul class='tree'>"]
    for adapter, scen in sorted(by_adapter.items()):
        parts.append(f"<li><div class='node root-node'>📦 {html.escape(adapter)}</div><ul>")
        for sid, children in sorted(scen.items()):
            parts.append(f"<li><div class='node parent'>scene <code>{html.escape(sid)}</code>"
                         f"<span class='meta'>→ {len(children)} child scenario(s)</span></div><ul>")
            for r in sorted(children, key=lambda x: str(x["rel"])):
                parts.append(f"<li><div class='node child leaf'>{html.escape(str(r['rel']))}"
                             f"<span class='badges'>{node_badges(r['manifest'])}</span></div></li>")
            parts.append("</ul></li>")
        parts.append("</ul></li>")
    parts.append("</ul>")
    return "".join(parts)


def build_report(offspring_root: Path, out_path: Path, previews_dir: Path | None = None, max_imgs_per: int = 4) -> str:
    runs = collect(offspring_root)
    if not runs:
        raise SystemExit(f"no manifests found under {offspring_root}")

    n_fams = Counter()
    layers = Counter()
    adapters = Counter()
    per_adapter: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        m = r["manifest"]
        adapters[m["parent"]["adapter"]] += 1
        for e in m.get("injected_events", []):
            n_fams[e["family"]] += 1
            layers[e["layer"]] += 1
        per_adapter[m["parent"]["adapter"]].append(r)

    cards = f"""
    <div class="cards">
      <div class="card"><b>{len(runs)}</b><span>offspring scenarios</span></div>
      <div class="card"><b>{len(adapters)}</b><span>dataset sources</span></div>
      <div class="card"><b>{sum(layers.values())}</b><span>injected events</span></div>
      <div class="card"><b>{len(n_fams)}</b><span>families used</span></div>
      <div class="card"><b>{' '.join(sorted(layers))}</b><span>layers touched</span></div>
    </div>"""

    legend_rows = "".join(
        f"<tr><td>{layer}</td><td>{html.escape(e['family'])}</td><td class='sub'>{html.escape(e['description'])}</td><td class='sub'>{html.escape(e['pattern'])}</td></tr>"
        for layer in sorted(list_families())
        for e in sorted(list_families()[layer], key=lambda x: x["family"])
    )

    sections = []
    for adapter_name in sorted(per_adapter):
        rows = []
        for r in per_adapter[adapter_name]:
            r["manifest"]["parent"]["scenario_id"] = _scrub(str(r["manifest"]["parent"]["scenario_id"]))
            m = r["manifest"]
            evs = m.get("injected_events", [])
            ev_html = "<br>".join(_fmt_event(e) for e in evs) or "<span class='sub'>none</span>"
            imgs = []
            if (r["dir"] / "images").exists():
                for f in sorted((r["dir"] / "images").glob("*.jpg"))[:max_imgs_per]:
                    b64 = _b64_image(f)
                    if b64:
                        imgs.append(
                            f"<figure><img src='{b64}'><figcaption>{html.escape(f.stem)}</figcaption></figure>"
                        )
            img_html = f"<div class='imgs'>{''.join(imgs)}</div>" if imgs else ""
            determinism = f"seed <code>{m['master_seed']}</code>"
            harness_note = (
                "<div class='note'>harness copy — original directory untouched</div>"
                if adapter_name == "harness"
                else ""
            )
            rows.append(
                f"""
                <div class="sec">
                  <h3>{html.escape(str(r['rel']))}</h3>
                  <div class="sub">parent: <code>{html.escape(_scrub(str(m['parent']['scenario_id'])))}</code> · {determinism}</div>
                  {harness_note}
                  <div style="margin:8px 0">{ev_html}</div>
                  {img_html}
                </div>"""
            )
        sections.append(f"<h2>Dataset source: {html.escape(adapter_name)}</h2>{''.join(rows)}")

    previews = ""
    if previews_dir and Path(previews_dir).exists():
        figs = []
        for p in sorted(Path(previews_dir).glob("*.png")):
            b64 = _b64_image(p, width=960, quality=80)
            if b64:
                figs.append(f"<figure><img src='{b64}' style='width:100%'><figcaption>{html.escape(p.name)} — original → severity 5</figcaption></figure>")
        if figs:
            previews = f"<h2>Severity progressions (visual)</h2><div class='sec'>{''.join(figs)}</div>"

    tree = f"<h2>Scenario tree — parent → children</h2><div class='sec'>{tree_html(runs)}</div>"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>noise-injector report</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>noise-injector — injection report</h1>
<div class="sub">v{__version__} · deterministic fault injection · manifests redact dataset paths by default</div>
{cards}
{tree}
<div class="note">Every offspring records a <code>manifest.json</code> (recipe, seeds, per-frame injected events).
Seeds derive from <code>sha256(scenario ‖ recipe)</code> — identical inputs reproduce byte-identical offspring,
so all models under test see exactly the same corrupted inputs.</div>
{previews}
{''.join(sections)}
<h2>Family / layer legend</h2>
<div class="sec"><table><tr><th>Layer</th><th>Family</th><th>Description</th><th>Default pattern</th></tr>{legend_rows}</table></div>
<h2>LLM-powered ops (bring your own API)</h2>
<div class="note">Layer B/C corruptions can be executed by an LLM you connect at run time — any OpenAI-compatible
endpoint. The key is read from an environment variable (<code>NOISE_INJECTOR_API_KEY</code> by default), never
from the command line, and is never written to code or manifests. Responses are cached on disk, so re-renders
stay deterministic.<br><br>
<code>export NOISE_INJECTOR_API_KEY=...&nbsp; noise-injector inject --type nuscenes --root &lt;dataset&gt; --family llm_qa_rewrite --severities 3 --llm-base-url https://api.openai.com/v1 --llm-model gpt-4o-mini --out offspring/</code></div>
<footer>generated by noise-injector v{__version__} · all manifests path-redacted · no credentials included</footer>
</div></body></html>"""


def write_report(offspring_root: str | Path, out_path: str | Path, previews_dir: str | Path | None = None) -> Path:
    content = build_report(Path(offspring_root), Path(out_path), Path(previews_dir) if previews_dir else None)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(content)
    return Path(out_path)
