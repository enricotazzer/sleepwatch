#!/usr/bin/env bash
# Phase 4 test run (docs/decisions.md 49): conversations with simulated people, then the judge,
# then metrics. Needs Ollama with the models in configs/agent/test.yaml, the T7 attached (models
# live there), AC power and the lid open. Takes several hours.
# Results go to results/agent_test/<timestamp>/ and one row to results/agent_index.csv.
# After an interruption: scripts/run_agent_eval.sh --resume results/agent_test/<timestamp>
# AGENT_CONFIG=configs/agent/dev.yaml runs the dev set instead.
#
# If an attempt fails, the run is resumed when the failed attempt saved new results, or after
# waiting (up to 2 h) for the T7 and Ollama's models to be available again: the T7 has dropped
# out mid-run, and Ollama then answers 400. An attempt that saved nothing while the models were
# available stops the script, so a real error is not retried in a loop.
set -uo pipefail
cd "$(dirname "$0")/.."
config=${AGENT_CONFIG:-configs/agent/test.yaml}
name=$(sed -n 's/^name: *//p' "$config" | tr -d "\"'")
judge=$(sed -n 's/^judge: *//p' "$config" | tr -d "\"'")
judge=${judge:-qwen3.5:9b}

models_ok() {
    [ -d /Volumes/T7/ollama-models ] && curl -sf localhost:11434/api/tags | grep -q "\"$judge\""
}
saved() {  # lines written so far in the run
    cat "$1"/conversations.jsonl "$1"/judgements.jsonl "$1"/judge_probes.jsonl 2>/dev/null | wc -l
}

echo "=== $name $(date -u +%FT%TZ)"
args=("$@")
before=-1
for attempt in 1 2 3 4 5 6 7 8; do
    if caffeinate -is uv run sleepwatch agent eval "$config" ${args[@]+"${args[@]}"}; then
        echo "=== done $(date -u +%FT%TZ)"
        exit 0
    fi
    if [ "${args[0]:-}" = "--resume" ]; then
        run=${args[1]%/}
    else
        run=$(ls -td "results/$name"/*/ 2>/dev/null | head -1)
        run=${run%/}
    fi
    if [ -z "$run" ]; then
        echo "=== failed before a run was created"
        exit 1
    fi
    now=$(saved "$run")
    if models_ok && [ "$now" -le "$before" ]; then
        echo "=== attempt $attempt failed with no new results and the models available: stopping"
        exit 1
    fi
    before=$now
    echo "=== attempt $attempt failed at $(date -u +%FT%TZ) ($now lines saved); waiting for the models"
    for _ in $(seq 1 720); do
        models_ok && break
        sleep 10
    done
    if ! models_ok; then
        echo "=== models still unavailable after 2 h: stopping"
        exit 1
    fi
    args=(--resume "$run")
done
echo "=== too many attempts: stopping"
exit 1
