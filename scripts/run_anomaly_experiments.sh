#!/usr/bin/env bash
# Reproduce the Phase 3 anomaly experiment on CPU (reads raw nights from data/raw). Results go to
# results/anomaly_main/<timestamp>/ and one row to results/anomaly_index.csv. caffeinate keeps the
# Mac from idling to sleep while it runs (a closed lid on battery still sleeps).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== anomaly_main $(date -u +%FT%TZ)"
caffeinate -is uv run sleepwatch anomaly run configs/anomaly/main.yaml --jobs 5
echo "=== done $(date -u +%FT%TZ)"
