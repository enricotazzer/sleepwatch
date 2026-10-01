#!/usr/bin/env bash
# Reproduce every Phase 2 staging experiment (about 3 hours on an Apple M3, CPU only).
# Results go to results/<name>/<timestamp>/ and one row per run to results/index.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
for name in hgb_main hgb_shifted hgb_dreem_labels gru_main gru_shifted; do
  echo "=== $name $(date -u +%FT%TZ)"
  uv run sleepwatch train "configs/staging/$name.yaml"
done
echo "=== done $(date -u +%FT%TZ)"
