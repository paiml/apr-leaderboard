#!/usr/bin/env bash
# eval-sweep.sh — Run eval across multiple prompt strategies sequentially
#
# Runs eval-pass-at-k.sh for each strategy, waiting for completion between runs.
# Designed for long-running CPU evals where GPU is occupied.
#
# Usage:
#   ./scripts/eval-sweep.sh BENCHMARK MODEL [STRATEGIES] [MAX_TOKENS]
#
# Examples:
#   ./scripts/eval-sweep.sh humaneval checkpoints/model.apr
#   ./scripts/eval-sweep.sh humaneval checkpoints/model.apr "standard scot few-shot cgo"
#   ./scripts/eval-sweep.sh humaneval checkpoints/model.apr "standard scot" 512

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: eval-sweep.sh BENCHMARK MODEL [STRATEGIES] [MAX_TOKENS]"
    echo ""
    echo "Strategies: standard scot few-shot cgo (space-separated, quoted)"
    exit 1
fi

BENCHMARK="$1"
MODEL="$2"
STRATEGIES="${3:-standard scot few-shot cgo}"
MAX_TOKENS="${4:-512}"
RESULTS_DIR="results"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Strategy Sweep ==="
echo "Benchmark:   ${BENCHMARK}"
echo "Model:       ${MODEL}"
echo "Strategies:  ${STRATEGIES}"
echo "Max tokens:  ${MAX_TOKENS}"
echo ""

SWEEP_START="$(date +%s)"
COMPLETED=0
TOTAL="$(echo "$STRATEGIES" | wc -w)"

for strategy in $STRATEGIES; do
    COMPLETED=$((COMPLETED + 1))
    echo "──────────────────────────────────────────────────"
    echo "[${COMPLETED}/${TOTAL}] Running strategy: ${strategy}"
    echo "──────────────────────────────────────────────────"
    echo ""

    START="$(date +%s)"
    # Batch mode auto-detected by eval script (APR_BATCH_MODE inherited from env)
    "${SCRIPT_DIR}/eval-pass-at-k.sh" "$BENCHMARK" "$MODEL" "$RESULTS_DIR" "$MAX_TOKENS" 0.0 1 "$strategy"
    ELAPSED=$(( $(date +%s) - START ))

    echo ""
    echo "Strategy ${strategy} completed in ${ELAPSED}s"
    echo ""
done

TOTAL_ELAPSED=$(( $(date +%s) - SWEEP_START ))
echo "=== Sweep Complete ==="
echo "Total time: ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 ))m)"
echo ""

# Show comparison
echo "=== Strategy Comparison ==="
"${SCRIPT_DIR}/results-history.sh" "$RESULTS_DIR" "$BENCHMARK"
