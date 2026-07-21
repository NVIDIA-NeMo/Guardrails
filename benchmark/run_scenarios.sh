#!/usr/bin/env bash
# Run the four IORails vs LLMRails benchmark scenarios (Tim Gasser's tech blog plan).
#
# Prerequisites:
#   1. Main LLM NIM running on port 8000
#   2. Nemotron 3.5 Content Safety NIM running on port 8001
#   3. Guardrails server running on port 9000 (see Procfile.llmrails / Procfile.iorails)
#   4. aiperf venv active (or PYTHONPATH set to repo root + aiperf venv on PATH)
#
# Usage:
#   ./benchmark/run_scenarios.sh llmrails   # Run all 4 scenarios against LLMRails server
#   ./benchmark/run_scenarios.sh iorails    # Run all 4 scenarios against IORails server
#
# The ENGINE arg is used only to label the output subdirectory — it does NOT start or
# stop the server. Restart the Guardrails server between llmrails and iorails runs:
#   LLMRails: nemoguardrails server --config ... --port 9000
#   IORails:  NEMO_GUARDRAILS_IORAILS_ENGINE=1 nemoguardrails server --config ... --port 9000

set -euo pipefail

ENGINE=${1:-llmrails}
if [[ "$ENGINE" != "llmrails" && "$ENGINE" != "iorails" ]]; then
    echo "Usage: $0 [llmrails|iorails] [scenario ...]" >&2
    echo "  scenarios: dialog rag code_gen agent (default: all four)" >&2
    exit 1
fi
shift

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/benchmark/aiperf/configs"

ALL_SCENARIOS=(dialog rag code_gen agent)
if [[ $# -gt 0 ]]; then
    SCENARIOS=("$@")
else
    SCENARIOS=("${ALL_SCENARIOS[@]}")
fi
PYTHON=${AIPERF_PYTHON:-/ephemeral/venv-aiperf/bin/python}

echo "=== IORails Tech Blog Benchmark Suite ==="
echo "Engine : $ENGINE"
echo "Configs: $CONFIG_DIR"
echo ""

FAILED=()

for SCENARIO in "${SCENARIOS[@]}"; do
    CONFIG="$CONFIG_DIR/scenario_${SCENARIO}_sweep.yaml"
    echo "--- Scenario: $SCENARIO ---"
    echo "Config: $CONFIG"

    # Inject engine label into batch_name so results land in a labelled subdirectory.
    # We use sed to produce a temp config with batch_name = <engine>_scenario_<name>.
    TEMP_CONFIG=$(mktemp /tmp/aiperf_XXXXXX.yaml)
    sed "s/^batch_name: scenario_${SCENARIO}/batch_name: ${ENGINE}_scenario_${SCENARIO}/" \
        "$CONFIG" > "$TEMP_CONFIG"

    if PYTHONPATH="$REPO_ROOT" "$PYTHON" -m benchmark.aiperf \
            --config-file "$TEMP_CONFIG"; then
        echo "✓ $SCENARIO complete"
    else
        echo "✗ $SCENARIO FAILED"
        FAILED+=("$SCENARIO")
    fi

    rm -f "$TEMP_CONFIG"
    echo ""
done

echo "=== Summary ==="
echo "Engine   : $ENGINE"
echo "Scenarios: ${#SCENARIOS[@]}"
echo "Failed   : ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed scenarios: ${FAILED[*]}"
    exit 1
fi
echo "All scenarios complete."
