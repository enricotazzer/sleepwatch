#!/usr/bin/env bash
# Reproduce every Phase 2 staging experiment on CPU. Results go to results/<name>/<timestamp>/ and
# one row per run to results/index.csv. The GRU trains its five folds in parallel, one per core
# (results don't depend on --jobs); boosting already uses every core, so its folds run in turn.
set -euo pipefail
cd "$(dirname "$0")/.."
run() {
  echo "=== $1 $(date -u +%FT%TZ)"
  uv run sleepwatch train "configs/staging/$1.yaml" "${@:2}"
}
for name in gru_main gru_shifted; do run "$name" --jobs 5; done
for name in hgb_main hgb_shifted hgb_dreem_labels; do run "$name"; done
echo "=== done $(date -u +%FT%TZ)"
