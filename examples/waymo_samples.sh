#!/usr/bin/env bash
# Fetch Waymo camera_box segments (requires gsutil + gcloud auth), inject annotation noise, build the tree report.
set -euo pipefail

SAMPLES="${1:-3}"
OUT="${2:-testset-waymo}"

noise-injector generate \
  --dataset waymo \
  --samples "$SAMPLES" \
  --family missed_detections \
  --severities 1-3 \
  --out "$OUT"

echo "report: $OUT/report.html"
