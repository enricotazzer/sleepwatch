#!/usr/bin/env bash
# Reproduce the Phase 3 anomaly experiment on CPU (reads raw nights from data/raw), then the
# Phase 3b rescoring with the heart-rate artifact screen (no raw data or GRU training needed).
# Results go to results/<name>/<timestamp>/ and one row each to results/anomaly_index.csv.
# caffeinate keeps the Mac from idling to sleep (a closed lid on battery still sleeps).
# --rescore-only skips Phase 3 and rescores the latest complete anomaly_main run.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--rescore-only" ]]; then
    echo "=== anomaly_main $(date -u +%FT%TZ)"
    caffeinate -is uv run sleepwatch anomaly run configs/anomaly/main.yaml --jobs 5
fi
base=$(uv run python -c "import pandas as pd; i = pd.read_csv('results/anomaly_index.csv'); \
r = i[(i['name'] == 'anomaly_main') & ~i['partial']].iloc[-1]; print(f'results/{r[\"name\"]}/{r.run}')")
echo "=== anomaly_screened from $base $(date -u +%FT%TZ)"
caffeinate -is uv run sleepwatch anomaly rescore configs/anomaly/screened.yaml --from-run "$base"
echo "=== done $(date -u +%FT%TZ)"
