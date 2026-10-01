#!/usr/bin/env bash
# Reproduce every staging experiment (Phases 2 and 2b) on CPU. Results go to
# results/<name>/<timestamp>/ and one row per run to results/index.csv. GRU experiments train their
# five folds in parallel, one per core (results don't depend on --jobs); boosting already uses
# every core, so its folds run in turn.
set -euo pipefail
cd "$(dirname "$0")/.."
run() {
  echo "=== $1 $(date -u +%FT%TZ)"
  uv run sleepwatch train "configs/staging/$1.yaml" "${@:2}"
}
for name in gru_personal gru_main gru_shifted; do run "$name" --jobs 5; done
for name in hgb_expanding hgb_main hgb_shifted hgb_dreem_labels; do run "$name"; done
echo "=== done $(date -u +%FT%TZ)"
