#!/usr/bin/env bash
# Fetch an agent harness from a git/repo link and corrupt a copy of its prompts/config. Originals untouched.
set -euo pipefail

LINK="${1:-https://github.com/milad6506/noise-injector-ADS-scenarios}"
OP="${2:-prompt_garble}"
OUT="${3:-corrupted-harness-copy}"

noise-injector harness --link "$LINK" --op "$OP" --strength 2 --out "$OUT"

echo "corrupted harness copy: $OUT (see $OUT/manifest.json)"
