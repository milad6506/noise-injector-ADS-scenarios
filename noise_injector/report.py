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
.child-head { display:flex; align-items:baseline; gap:9px; flex-wrap:wrap; }
.child-title { font-weight:600; font-size:13.5px; }
.badge { padding:1px 9px; border-radius:99px; font-size:11px; font-weight:600; border:1px solid var(--line); }
.badge.A{color:#4f8cff;border-color:#4f8cff}.badge.B{color:#2fbf71;border-color:#2fbf71}
.badge.C{color:#e0a526;border-color:#e0a526}.badge.F{color:#c06cff;border-color:#c06cff}
.badge.clean{color:var(--ok);border-color:var(--ok)}
.dose { color:var(--mut); font-size:11.5px; }
.player { margin-top:10px; }
.viewport { background:#0a0c10; border:1px solid var(--line); border-radius:10px; height:320px; display:flex;
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


def _child_description(m: dict) -> str:
    parts = []
    for ev in m.get("injected_events", []):
        fam = ev.get("family", "")
        try:
            desc = family_def(fam).description
        except KeyError:
            desc = ""
        sev = (ev.get("params") or {}).get("severity") or ev.get("strength") or ""
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


def _scenario_payload(scenario_key: str, offspring_dir: Path, label: str, badge: str, desc: str) -> dict:
    with open(offspring_dir / "annotations.json") as f:
        ann = json.load(f)
    kind, frames, extra = _frames_from_annotations(offspring_dir, ann)
    payload = {"label": label, "badge": badge, "desc": desc, "kind": kind, "frames": frames}
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
                scenario_data[skey] = _scenario_payload(skey, k["dir"], html.escape(k["key"]),
                                                        str(k["m"]["injected_events"][0].get("layer", "")),
                                                        _child_description(k["m"]))
                child_items.append(f"""
    <div class="child-box">
      <div class="child-head"><span class="badge {scenario_data[skey]['badge']}">{scenario_data[skey]['badge']}</span>
        <span class="child-title">{scenario_data[skey]['label']}</span>
        <span class="dose">{scenario_data[skey]['desc']}</span></div>
      {player_html(skey)}
    </div>""")
            items.append(f"""
  <li class="node" id="node-{html.escape(adapter)}-{html.escape(sid)}">
    <div class="node-head" onclick="toggleNode('{html.escape(adapter)}|{html.escape(sid)}')">
      <span class="caret">▶</span>
      <div style="flex:1">
        <div class="node-title">{html.escape(sid)}</div>
        <div class="node-desc">{len(kids) - (1 if parent_payload else 0)} perturbed scenario(s) generated from this parent
        {' · <span class="badge clean">clean</span> baseline included' if parent_payload else ''}</div>
      </div>
    </div>
    <div class="node-body">
    {'<div class="child-box">' + player_html(f"{adapter}:{sid}:parent") + '</div>' if parent_payload else ''}
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
    const key_dims = {};
    for (const b of fr.boxes) { key_dims[b.cam] = d.dims[b.cam] || [800, 500]; }
    const canvas = document.createElement('canvas');
    canvas.className = 'active';
    const dims = Object.values(key_dims)[0] || [800, 500];
    const H = Math.min(300, dims[1] * (280 / dims[0]) + 20);
    canvas.width = Math.min(860, dims[0] * (280 / dims[1]) + 20);
    canvas.height = Math.max(120, H);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0a0c10'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    const colors = {Vehicle:'#4f8cff', Pedestrian:'#2fbf71', Cyclist:'#e0a526', Unknown:'#8b93a5'};
    const camsPresent = [...new Set(fr.boxes.map(b=>b.cam))];
    let idx = 0;
    for (const cam of camsPresent) {
      const dim = d.dims[cam] || [900, 600];
      const ox = idx * 10, oy = idx * 8;
      const sw = (canvas.width - ox - 10) / dim[0], sh = (canvas.height - oy - 10) / dim[1];
      ctx.font = '10px sans-serif'; ctx.fillStyle = '#8b93a5';
      ctx.fillText(cam, ox + 4, oy + 12);
      for (const b of fr.boxes) {
        if (b.cam !== cam) continue;
        ctx.strokeStyle = colors[b.c] || '#8b93a5';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(ox + b.x1 * sw, oy + b.y1 * sh, (b.x2 - b.x1) * sw, (b.y2 - b.y1) * sh);
        ctx.fillStyle = colors[b.c] || '#8b93a5';
        ctx.fillText(b.c, ox + b.x1 * sw, oy + b.y1 * sh - 3);
      }
      idx++;
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
