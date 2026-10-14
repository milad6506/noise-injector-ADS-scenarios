#!/usr/bin/env bash
# Fetch a public DriveLM-CARLA scene package (no token needed), inject fog, build the tree report.
set -euo pipefail

SCENARIO="${1:-ControlLoss}"
OUT="${2:-testset-carla}"

noise-injector generate \
  --dataset carla \
  --scenario "$SCENARIO" \
  --samples 1 \
  --family fog \
  --severities 1-3 \
  --max-gb 1.0 \
  --out "$OUT"

echo "report: $OUT/report.html"
