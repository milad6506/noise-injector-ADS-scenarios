"""Interactive HTML report: parent scenario tree + playable frame scrubber per scenario."""

from __future__ import annotations

import base64
import html
import json
from collections import defaultdict
from pathlib import Path

import cv2

from . import __version__
from .spec import family as family_def
from .spec import list_families

_CSS = """
:root { --bg:#0e1014; --card:#161a22; --line:#262b36; --text:#d7dae0; --mut:#8b93a5; --acc:#4f8cff; --ok:#2fbf71; --warn:#e0a526; --bad:#e05252; --pur:#c06cff; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.55 -apple-system,'Segoe UI',Roboto,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:26px 18px 70px; }
h1 { font-size:21px; margin:0 0 2px; }
h2 { font-size:17px; margin:26px 0 8px; border-bottom:1px solid var(--line); padding-bottom:6px; }
.sub { color:var(--mut); font-size:12.5px; }
.stats { display:flex; gap:10px; flex-wrap:wrap; margin:16px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px 14px; font-size:12.5px; }
.stat b { font-size:17px; margin-right:5px; }
.tree { list-style:none; margin:12px 0; padding:0; }
.tree > li { margin:14px 0; }
.node-head { display:flex; align-items:center; gap:10px; background:var(--card); border:1px solid var(--line);
             border-radius:12px; padding:12px 16px; cursor:pointer; user-select:none; }
.node-head:hover { border-color:var(--acc); }
.caret { color:var(--acc); font-size:16px; width:16px; transition:transform .15s; display:inline-block; }
.open > .node-head .caret { transform:rotate(90deg); }
.node-title { font-weight:700; }
.node-desc { color:var(--mut); font-size:12px; margin-top:2px; width:100%; }
.node-body { display:none; padding:6px 8px 12px 26px; }
.open > .node-body { display:block; }
.child-box { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 16px; margin:10px 0; }
.mother-box { background:linear-gradient(180deg,#1a2030,#161a22); border:1px solid #3350a0; border-radius:12px;
              padding:14px 18px; margin:6px 0 16px; }
.mother-tag { display:inline-block; font-size:10.5px; letter-spacing:.08em; text-transform:uppercase;
              color:#9ecbff; border:1px solid #3350a0; border-radius:99px; padding:1px 10px; margin-bottom:6px; }
.scene-desc { color:var(--text); font-size:13px; margin:4px 0 2px; }
.mode-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:0 0 2px; }
.mode-name { font-weight:700; font-size:13.5px; color:#9ecbff; font-family:ui-monospace,Menlo,monospace; }
.chip { font-size:10.5px; padding:1px 9px; border-radius:99px; border:1px solid var(--line); color:var(--mut); }
.children-wrap { margin-top:6px; }
.children-title { font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--mut); margin:14px 0 4px; }
.child-head { display:flex; align-items:baseline; gap:9px; flex-wrap:wrap; }
.child-title { font-weight:600; font-size:13.5px; }
.badge { padding:1px 9px; border-radius:99px; font-size:11px; font-weight:600; border:1px solid var(--line); }
.badge.A{color:#4f8cff;border-color:#4f8cff}.badge.B{color:#2fbf71;border-color:#2fbf71}
.badge.C{color:#e0a526;border-color:#e0a526}.badge.F{color:#c06cff;border-color:#c06cff}
.badge.clean{color:var(--ok);border-color:var(--ok)}
.dose { color:var(--mut); font-size:11.5px; }
.player { margin-top:10px; }
.viewport { background:#20242c; border:1px solid var(--line); border-radius:10px; min-height:300px; display:flex;
            align-items:center; justify-content:center; overflow:hidden; }
.viewport img, .viewport canvas { max-width:100%; max-height:100%; display:none; }
.viewport img.active, .viewport canvas.active { display:block; }
.controls { display:flex; align-items:center; gap:10px; margin-top:8px; }
.playbtn { background:var(--acc); border:none; color:#fff; border-radius:8px; width:34px; height:28px;
           font-size:13px; cursor:pointer; flex:none; }
.playbtn:hover { filter:brightness(1.15); }
input[type=range] { flex:1; accent-color: var(--acc); height:5px; }
.flabel { color:var(--mut); font-size:11.5px; min-width:150px; text-align:right; white-space:nowrap; }
.cams { display:inline-flex; gap:5px; margin-left:8px; }
.camchip { font-size:10.5px; padding:1px 8px; border-radius:99px; border:1px solid var(--line);
           background:transparent; color:var(--mut); cursor:pointer; }
.camchip.active { color:var(--acc); border-color:var(--acc); }
footer { color:var(--mut); font-size:11.5px; margin-top:44px; }
code { background:#0d0f13; padding:1px 6px; border-radius:4px; font-size:11.5px; color:#9ecbff; }
"""


def _b64_image(path: Path, width: int = 480, quality: int = 62) -> str | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode() if ok else None


def _dose_summary(ev: dict) -> str:
    n = 0
    for t in ev.get("targets", []):
        d = t.get("detail")
        n += len(d) if isinstance(d, list) else len(t.get("cameras", []))
    return f"~{n} target(s) across {len(ev.get('targets', []))} frame(s)"


def _child_description(m: dict, sev_hint: str = "") -> str:
    parts = []
    for ev in m.get("injected_events", []):
        fam = ev.get("family", "")
        try:
            desc = family_def(fam).description
        except KeyError:
            desc = ""
        sev = ev.get("severity") or (ev.get("params") or {}).get("severity") or ev.get("strength") or sev_hint
        parts.append(f"{html.escape(desc)} · pattern {html.escape(str(ev.get('pattern','')))} · severity {sev} — {_dose_summary(ev)}")
    return " | ".join(parts) or "clean copy of the parent scenario (baseline)"


def _frames_from_annotations(offspring_dir: Path, ann: dict) -> tuple[str, list]:
    has_images = any(
        rel and (offspring_dir / rel).exists()
        for fr in ann.get("key_frames", {}).values()
        for rel in (fr.get("image_paths") or {}).values()
    )
    if has_images:
        frames = []
        cams_all = []
        for fid, fr in ann["key_frames"].items():
            cams = {}
            for cam, rel in (fr.get("image_paths") or {}).items():
                if rel and (offspring_dir / rel).exists():
                    b64 = _b64_image(offspring_dir / rel)
                    if b64:
                        cams[cam] = b64
            if cams:
                frames.append({"id": fid, "cameras": cams})
                cams_all += list(cams)
        return "image", frames, sorted(set(cams_all))
    frames = []
    dims: dict[str, list[float]] = defaultdict(lambda: [1.0, 1.0])
    for fid, fr in ann.get("key_frames", {}).items():
        boxes = []
        for key, info in (fr.get("key_object_infos") or {}).items():
            b = info.get("2d_bbox")
            if not b:
                continue
            if isinstance(b[0], (list, tuple)):
                b = [b[0][0], b[0][1], b[1][0], b[1][1]]
            cam = key.split(",")[1].strip() if "," in key else "CAM"
            cam_name = cam.replace("CAM", "CAM ")
            boxes.append({"c": info.get("Category", "Unknown"), "cam": cam_name,
                          "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]})
            dims[cam_name][0] = max(dims[cam_name][0], b[2])
            dims[cam_name][1] = max(dims[cam_name][1], b[3])
        frames.append({"id": fid, "boxes": boxes})
    return "boxes", frames, {k: [round(v[0] + 20), round(v[1] + 20)] for k, v in dims.items()}


def _scenario_payload(scenario_key: str, offspring_dir: Path, label: str, badge: str, desc: str, veil: float = 0.0, weather: dict | None = None) -> dict:
    with open(offspring_dir / "annotations.json") as f:
        ann = json.load(f)
    kind, frames, extra = _frames_from_annotations(offspring_dir, ann)
    payload = {"label": label, "badge": badge, "desc": desc, "kind": kind, "frames": frames,
               "veil": veil, "scene_desc": ann.get("scene_description") or ""}
    if weather:
        payload["weather"] = weather
    if kind == "boxes":
        payload["dims"] = extra
    else:
        payload["cams"] = extra
    return payload


def collect(offspring_root: Path) -> list[dict]:
    found = []
    for mp in sorted(Path(offspring_root).rglob("manifest.json")):
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        found.append({"manifest": m, "dir": mp.parent, "rel": mp.parent.relative_to(offspring_root)})
    return found


def build_report(offspring_root: Path, out_path: Path, previews_dir: Path | None = None, max_frames_per: int = 60) -> str:
    runs = collect(Path(offspring_root))
    if not runs:
        raise SystemExit(f"no manifests found under {offspring_root}")

    by_adapter: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for r in runs:
        m = r["manifest"]
        sid = str(m["parent"]["scenario_id"])
        if "/" in sid and sid.startswith(("/", "~")) or "Users/" in sid:
            sid = sid.rsplit("/", 1)[-1]
        clean = not m.get("injected_events")
        key = "clean (parent baseline)" if clean else str(r["rel"]).split("/")[-1]
        by_adapter.setdefault(m["parent"]["adapter"], {}).setdefault(sid, {})  # parent key
        by_adapter[m["parent"]["adapter"]][sid].setdefault("children", []).append(
            {"key": key, "m": m, "dir": r["dir"]}
        )

    scenario_data = {}
    tree_items = []
    n_children = 0
    for adapter, scen in sorted(by_adapter.items()):
        items = []
        for sid, bundle in sorted(scen.items()):
            kids = bundle["children"]
            parent_dir = next((k["dir"] for k in kids if not k["m"].get("injected_events")), None)
            if parent_dir is None:
                parent_payload = None
            else:
                parent_payload = _scenario_payload(f"{adapter}:{sid}:parent", parent_dir,
                                                   "clean parent", "clean",
                                                   "the untouched scenario — baseline for comparison")
                scenario_data[f"{adapter}:{sid}:parent"] = parent_payload
            child_items = []
            for k in sorted(kids, key=lambda x: x["key"]):
                if not k["m"].get("injected_events"):
                    continue
                n_children += 1
                skey = f"{adapter}:{sid}:{k['key']}"
                veil = 0.0
                weather = {}
                for ev in k["m"]["injected_events"]:
                    if ev.get("family") == "fog":
                        veil = float((ev.get("params") or {}).get("amount", 0.0))
                    if ev.get("family") == "rain":
                        weather = {"rain": int((ev.get("params") or {}).get("n_streaks", 0)),
                                   "darken": float((ev.get("params") or {}).get("darken", 0.0))}
                layer = str(k["m"]["injected_events"][0].get("layer", ""))
                ev0 = k["m"]["injected_events"][0]
                fam = ev0.get("family", "")
                sev = ev0.get("severity") or (k["key"].rsplit("_s", 1)[-1] if "_s" in k["key"] else "")
                pat = ev0.get("pattern", "")
                scenario_data[skey] = _scenario_payload(skey, k["dir"], html.escape(k["key"]),
                                                        layer,
                                                        _child_description(k["m"], str(sev)), veil=veil,
                                                        weather=weather or None)
                child_items.append(f"""
    <div class="child-box">
      <div class="mode-row">
        <span class="badge {layer}">{layer}</span>
        <span class="mode-name">{html.escape(fam)}</span>
        <span class="chip">severity {sev}</span>
        <span class="chip">pattern: {html.escape(str(pat))}</span>
      </div>
      <div class="dose">{scenario_data[skey]['desc']}</div>
      {player_html(skey)}
    </div>""")
            mother_meta = ""
            if parent_payload:
                mother_meta = (f"<div class='sub'>{len(parent_payload['frames'])} frames · "
                               f"{html.escape(parent_payload['kind'])} view · seed-deterministic</div>")
            items.append(f"""
  <li class="node" id="node-{html.escape(adapter)}-{html.escape(sid)}">
    <div class="node-head" onclick="toggleNode('{html.escape(adapter)}|{html.escape(sid)}')">
      <span class="caret">▶</span>
      <div style="flex:1">
        <div class="node-title">{html.escape(sid)}</div>
        <div class="node-desc">{len(child_items)} perturbed scenario(s) generated from this mother</div>
      </div>
    </div>
    <div class="node-body">
    {'<div class="mother-box"><span class="mother-tag">mother scenario — original</span>' +
      ('<div class="scene-desc">' + html.escape(str(parent_payload['scene_desc'])) + '</div>' if parent_payload['scene_desc'] else '') +
      mother_meta + player_html(f"{adapter}:{sid}:parent") + '</div>' if parent_payload else ''}
    {'<div class="children-title">perturbed children — noise mode per child</div>' if child_items else ''}
    {''.join(child_items)}
    </div>
  </li>""")
        tree_items.append(f"<h2>{html.escape(adapter)}</h2><ul class='tree'>{''.join(items)}</ul>")

    stats = (f"<div class='stats'><div class='stat'><b>{len(by_adapter)}</b>sources</div>"
             f"<div class='stat'><b>{sum(len(s) for s in by_adapter.values())}</b>parent scenarios</div>"
             f"<div class='stat'><b>{n_children}</b>perturbed children</div></div>")

    body = f"""<div class="wrap">
<h1>noise-injector — generated test scenarios</h1>
<div class="sub">v{__version__} · click a parent scenario to expand its generated children · every scenario is playable (scrub first → last frame)</div>
{stats}
{''.join(tree_items)}
<footer>generated by noise-injector v{__version__} · self-contained · no paths, no credentials · seed-deterministic offspring</footer>
</div>"""

    script = _js()
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>noise-injector — test scenarios</title><style>{_CSS}</style></head><body>
{body}
<script>const DATA = {json.dumps(scenario_data)};{script}</script></body></html>"""


def player_html(key: str) -> str:
    import hashlib

    dom = "p" + hashlib.md5(key.encode()).hexdigest()[:10]
    return f"""
    <div class="player" data-key="{key}">
      <div class="viewport" id="vp-{dom}"></div>
      <div class="controls">
        <button class="playbtn" id="btn-{dom}" onclick="togglePlay('{key}','{dom}')">▶</button>
        <input type="range" id="sl-{dom}" min="0" max="0" value="0" step="1"
               oninput="onScrub('{key}','{dom}')" style="flex:1">
        <span class="flabel" id="lb-{dom}">frame 0/0</span>
      </div>
    </div>"""


def _js() -> str:
    return """
const players = {};
let timer = null;
function toggleNode(key){
  const li = document.getElementById('node-' + key.replace('|', '-'));
  if (li) {
    li.classList.toggle('open');
    li.dataset.init = '1';
  }
}
function initPlayer(key){
  const d = DATA[key];
  if (!d) return;
  for (const pl of document.querySelectorAll('.player[data-key="'+key+'"]')) {
    const m = pl.querySelector('input[type=range]');
    if (!m) continue;
    const domId = m.id.slice(3);
    m.max = d.frames.length - 1;
    m.value = 0;
    showFrame(key, domId, 0);
  }
}
function showFrame(key, dom, i){
  const d = DATA[key];
  const vp = document.getElementById('vp-'+dom);
  if (!vp) return;
  vp.innerHTML = '';
  if (d.kind === 'image') {
    const fr = d.frames[i];
    let cam = players[key] && players[key].cam && fr.cameras[players[key].cam] ? players[key].cam : Object.keys(fr.cameras)[0];
    const img = document.createElement('img');
    img.src = fr.cameras[cam]; img.className = 'active';
    vp.appendChild(img);
    if (Object.keys(fr.cameras).length > 1) {
      const chips = document.createElement('div'); chips.className = 'cams';
      for (const c of Object.keys(fr.cameras)) {
        const b = document.createElement('button');
        b.className = 'camchip' + (c === cam ? ' active' : '');
        b.textContent = c.replace('CAM_', '');
        b.onclick = () => { players[key] = {cam: c}; showFrame(key, dom, i); };
        chips.appendChild(b);
      }
      vp.style.position = 'relative';
      chips.style.cssText = 'position:absolute;top:6px;left:6px;';
      vp.appendChild(chips);
    }
  } else {
    const fr = d.frames[i];
    const canvas = document.createElement('canvas');
    canvas.className = 'active';
    canvas.width = 860; canvas.height = 280;
    const ctx = canvas.getContext('2d');
    const horizon = canvas.height * 0.42;
    const sky = ctx.createLinearGradient(0, 0, 0, horizon);
    sky.addColorStop(0, '#e8edf4'); sky.addColorStop(1, '#c3ccd8');
    ctx.fillStyle = sky; ctx.fillRect(0, 0, canvas.width, horizon);
    const road = ctx.createLinearGradient(0, horizon, 0, canvas.height);
    road.addColorStop(0, '#9aa3ad'); road.addColorStop(1, '#6f7883');
    ctx.fillStyle = road; ctx.fillRect(0, horizon, canvas.width, canvas.height - horizon);
    if (d.weather && d.weather.rain) {
      const n = Math.min(120, Math.round(d.weather.rain / 4));
      ctx.strokeStyle = 'rgba(190,205,225,0.5)'; ctx.lineWidth = 1;
      for (let s = 0; s < n; s++) {
        const h = (Math.sin(s * 12.9898 + fr.id.length * 7.7) * 43758.5453);
        const fx = (h - Math.floor(h));
        const h2 = (Math.sin(s * 78.233 + fr.id.length * 3.3) * 12345.6789);
        const fy = (h2 - Math.floor(h2));
        const x = fx * canvas.width, y = fy * canvas.height;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 4, y + 14); ctx.stroke();
      }
      ctx.fillStyle = 'rgba(10,14,26,' + d.weather.darken + ')';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.setLineDash([14, 12]); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(canvas.width / 2 - 30, canvas.height); ctx.lineTo(canvas.width / 2 + 4, horizon); ctx.stroke();
    ctx.setLineDash([]);
    const colors = {Vehicle:'#1c5fd8', Pedestrian:'#1e7d47', Cyclist:'#b57d13', Unknown:'#5c6570'};
    const camsPresent = [...new Set(fr.boxes.map(b=>b.cam))].sort();
    const panelW = canvas.width / Math.max(1, Math.min(camsPresent.length, 3));
    camsPresent.forEach((cam, idx) => {
      const col = idx % 3, row = Math.floor(idx / 3);
      const ox = col * panelW + 6, oy = row * 6;
      const dim = d.dims[cam] || [900, 600];
      const pw = panelW - 14, ph = canvas.height - oy - 10;
      const sw = pw / dim[0], sh = ph / dim[1];
      ctx.strokeStyle = 'rgba(90,100,115,0.5)'; ctx.lineWidth = 1;
      ctx.strokeRect(ox, oy, pw, ph);
      ctx.font = '10px sans-serif'; ctx.fillStyle = '#5c6570';
      ctx.fillText(cam, ox + 5, oy + 12);
      for (const b of fr.boxes) {
        if (b.cam !== cam) continue;
        const depth = Math.max(0, Math.min(1, (b.y2 / dim[1]) - 0.35) / 0.65);
        const alpha = (0.25 + 0.75 * depth) * (1 - (d.veil || 0) * 0.55);
        ctx.strokeStyle = colors[b.c] || '#5c6570';
        ctx.globalAlpha = alpha; ctx.lineWidth = 2.5;
        ctx.strokeRect(ox + b.x1 * sw, oy + b.y1 * sh, (b.x2 - b.x1) * sw, (b.y2 - b.y1) * sh);
        ctx.fillStyle = colors[b.c] || '#5c6570';
        ctx.fillText(b.c, ox + b.x1 * sw, oy + b.y1 * sh - 3);
      }
      ctx.globalAlpha = 1;
    });
    if (d.veil > 0) {
      const fogGrad = ctx.createLinearGradient(0, 0, 0, canvas.height);
      fogGrad.addColorStop(0, 'rgba(233,236,240,' + Math.min(1, d.veil * 1.05) + ')');
      fogGrad.addColorStop(0.42, 'rgba(228,231,236,' + (d.veil * 0.95) + ')');
      fogGrad.addColorStop(1, 'rgba(225,228,233,' + (d.veil * 0.35) + ')');
      ctx.fillStyle = fogGrad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = 'bold 12px sans-serif'; ctx.fillStyle = 'rgba(50,60,80,0.8)';
      ctx.fillText('fog severity veil ' + Math.round(d.veil * 100) + '%', 10, canvas.height - 10);
    }
    vp.appendChild(canvas);
  }
  const sl = document.getElementById('sl-'+dom);
  const lb = document.getElementById('lb-'+dom);
  if (lb) lb.textContent = 'frame ' + (i + 1) + '/' + d.frames.length + (i === 0 ? ' (first)' : (i === d.frames.length - 1 ? ' (last)' : ''));
  players[key] = Object.assign(players[key] || {}, {frame: i});
}
function onScrub(key, dom){
  const i = +document.getElementById('sl-'+dom).value;
  stopPlay(key, dom);
  showFrame(key, dom, i);
}
function togglePlay(key, dom){
  if (timer) { stopPlay(key, dom); return; }
  const btn = document.getElementById('btn-'+dom);
  btn.textContent = '❚❚';
  const sl = document.getElementById('sl-'+dom);
  timer = setInterval(() => {
    let i = (+sl.value + 1) % (+sl.max + 1);
    sl.value = i;
    showFrame(key, dom, i);
  }, 420);
}
function stopPlay(key, dom){
  if (timer) { clearInterval(timer); timer = null; }
  const btn = document.getElementById('btn-'+dom);
  if (btn) btn.textContent = '▶';
}
window.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tree > li.node').forEach(li => {
    li.classList.add('open');
    li.dataset.init = '1';
  });
  document.querySelectorAll('.player').forEach(pl => initPlayer(pl.dataset.key));
});
"""


def write_report(offspring_root: str | Path, out_path: str | Path, previews_dir: str | Path | None = None) -> Path:
    content = build_report(Path(offspring_root), Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(content)
    return Path(out_path)
