"""Dataset fetchers: download scenario packages by name (waymo/drivelm/carla/lyft) or link.

Every fetcher writes a self-contained data directory that the matching adapter consumes.
Large downloads are guarded by --max-gb; gated Hugging Face repos require HF_TOKEN (env).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

WAYMO_BUCKET = "gs://waymo_open_dataset_v_2_0_1/training/camera_box"
DRIVELM_REPO = "OpenDriveLab/DriveLM"
PDM_LITE_REPO = "autonomousvision/PDM_Lite_Carla_LB2"


def _progress_hook(count, block, total):
    if total > 0:
        done = count * block / total
        sys.stdout.write(f"\r    downloading: {done * 100:5.1f}%")
        sys.stdout.flush()


def download(url: str, dest: Path, max_gb: float = 8.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60.0) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            if total > max_gb * 1024**3:
                raise ValueError(f"{url} is {total / 1024**3:.1f} GB — exceeds --max-gb {max_gb}")
            copied = 0
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > max_gb * 1024**3:
                    raise ValueError(f"download exceeded --max-gb {max_gb}: {url}")
                f.write(chunk)
                if total:
                    sys.stdout.write(f"\r    downloading: {copied / total * 100:5.1f}% ({copied / 1048576:.0f} MB)")
                    sys.stdout.flush()
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()
    tmp.replace(dest)
    return dest


def _extract(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    elif archive.suffix == ".parquet":
        shutil.copyfile(archive, dest_dir / archive.name)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if not archive.name.endswith(".tar") else "r:"
        with tarfile.open(archive, mode) as t:
            t.extractall(dest_dir)
    elif archive.suffix == ".json":
        shutil.copyfile(archive, dest_dir / archive.name)
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")


def fetch_waymo(samples: int, out: Path, max_gb: float = 1.0) -> Path:
    out = Path(out)
    box_dir = out / "camera_box"
    box_dir.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(["gsutil", "ls", WAYMO_BUCKET], capture_output=True, text=True, check=True)
    files = sorted(l.strip() for l in listing.stdout.splitlines() if l.strip().endswith(".parquet"))
    if not files:
        raise RuntimeError("no parquet segments listed; check gsutil auth (gcloud auth login)")
    picked = files[:samples]
    for gcs in picked:
        dest = box_dir / Path(gcs).name
        if dest.exists():
            print(f"    cached: {dest.name}")
            continue
        print(f"    fetching {Path(gcs).name}")
        subprocess.run(["gsutil", "-q", "cp", gcs, str(dest)], check=True)
    print(f"    waymo: {len(picked)} camera_box segment(s) -> {box_dir}")
    return out


def _hf_url(repo: str, file: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{file}"


def fetch_drivelm(split: str, out: Path, max_gb: float = 8.0, with_images: bool = False) -> Path:
    out = Path(out)
    token = _hf_token_hint()
    fname = "v1_1_val_nus_q_only.json" if split == "val" else "v1_1_train_nus.json"
    print(f"    downloading {fname} (gated repo: uses $HF_TOKEN)")
    download(_hf_url(DRIVELM_REPO, fname), out / fname, max_gb)
    if with_images:
        img_zip = out / ("drivelm_nus_imgs_val.zip" if split == "val" else "drivelm_nus_imgs_train.zip")
        if not img_zip.exists():
            print("    downloading image archive (large!)")
            download(_hf_url(DRIVELM_REPO, img_zip.name), img_zip, max_gb)
            _extract(img_zip, out)
    print(f"    drivelm: annotations ready -> {out}")
    return out


def _hf_token_hint() -> str | None:
    import os

    return os.environ.get("HF_TOKEN")


def _hf_json_list(repo: str) -> list[str]:
    import json

    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=30) as r:
        d = json.loads(r.read().decode())
    return [s.get("rfilename", "") for s in d.get("siblings", [])]


def fetch_carla(samples: int, scenario: str, out: Path, max_gb: float = 8.0) -> Path:
    out = Path(out)
    files = _hf_json_list(PDM_LITE_REPO)
    data_files = [f for f in files if f.endswith(".zip") and "/data/" in f]
    if scenario:
        parts = [p for p in scenario.split("/") if p]

        def matches(f: str) -> bool:
            segs = f.split("/")
            return all(any(seg.startswith(p) for seg in segs) for p in parts)

        picked = [f for f in data_files if matches(f)][:samples]
    else:
        picked = data_files[:samples]
    if not picked:
        raise ValueError(f"scenario '{scenario}' not found in {PDM_LITE_REPO} (got {len(data_files)} data zips)")
    token = _hf_token_hint()
    qa_zip = out / "drivelm_carla_vqas.zip"
    if not qa_zip.exists():
        try:
            print("    downloading CARLA QA annotations (OpenDriveLab/DriveLM)")
            download(_hf_url(DRIVELM_REPO, "drivelm_carla_vqas.zip"), qa_zip, max_gb)
            _extract(qa_zip, out)
        except Exception as e:
            print(f"    WARN: QA annotations unavailable ({str(e)[:80]}) — set HF_TOKEN for the gated repo; "
                  f"continuing with image-only scenarios")
    pdm_root = out / "PDM_Lite_Carla_LB2"
    pdm_root.mkdir(parents=True, exist_ok=True)
    for f in picked:
        name = f.split("/")[-1]
        town = f.split("/")[0]
        dest = out / "zips" / f"{town}_{name}"
        if not dest.exists():
            print(f"    downloading scene images: {f}")
            download(_hf_url(PDM_LITE_REPO, f), dest, max_gb)
        _extract(dest, pdm_root)
    print(f"    carla: {len(picked)} scene package(s) -> {out}")
    return out


def fetch_link(url: str, out: Path, max_gb: float = 8.0) -> Path:
    out = Path(out)
    name = url.rstrip("/").split("?")[0].split("/")[-1] or "package"
    dest = out / "link" / name
    if dest.exists():
        print(f"    cached: {dest}")
        return dest
    print(f"    downloading {url}")
    download(url, dest, max_gb)
    if dest.suffix in {".zip", ".parquet", ".json"} or dest.name.endswith((".tar.gz", ".tgz", ".tar")):
        _extract(dest, out / "link" / dest.stem)
    print(f"    link package -> {dest}")
    return dest


def fetch_harness_link(url: str, out: Path) -> Path:
    """Clone/download a harness from a git URL or archive link; returns the harness root to corrupt."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "harness"
    if dest.exists():
        return dest
    if url.endswith(".git") or "git@" in url:
        print(f"    shallow-cloning {url}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True,
                       capture_output=True, text=True)
    else:
        base = url.rstrip("/")
        if "github.com" in base and not base.endswith((".zip", ".tar.gz")):
            parts = base.split("github.com/")[-1].strip("/")
            base = f"https://github.com/{parts}/archive/refs/heads/main.tar.gz"
        dest_arch = out / "harness_archive.tar.gz"
        download(base, dest_arch, 2.0)
        _extract(dest_arch, out)
        extracted = [p for p in out.iterdir() if p.is_dir() and p.name.startswith(("main", "harness-"))]
        if extracted:
            extracted[0].rename(dest)
        else:
            dest = out
    shutil.rmtree(dest / ".git", ignore_errors=True)
    print(f"    harness fetched -> {dest}")
    return dest


def fetch_lyft(samples: int, out: Path) -> None:
    raise RuntimeError(
        "Lyft Level 5 open data is no longer publicly hosted. "
        "Provide a direct link to a scenario package instead: "
        "noise-injector fetch --link <url-to-zip|parquet|json> --out <dir>"
    )


FETCHERS = {
    "waymo": lambda samples, out, **kw: fetch_waymo(samples, out, max_gb=kw.get("max_gb", 1.0)),
    "drivelm": lambda samples, out, **kw: fetch_drivelm(kw.get("split", "val"), out,
                                                        max_gb=kw.get("max_gb", 8.0),
                                                        with_images=kw.get("with_images", False)),
    "carla": lambda samples, out, **kw: fetch_carla(samples, kw.get("scenario", ""), out,
                                                    max_gb=kw.get("max_gb", 8.0)),
    "lyft": lambda samples, out, **kw: fetch_lyft(samples, out),
}


def data_root_for(dataset: str, out: Path) -> Path:
    if dataset == "drivelm":
        return Path(out) / "data" / "drivelm"
    return Path(out) / "data" / dataset


def adapter_root(dataset: str, data_dir: Path) -> Path:
    return Path(data_dir)


def adapter_type(dataset: str) -> str:
    return {"drivelm": "nuscenes", "carla": "carla", "waymo": "waymo"}.get(dataset, dataset)
