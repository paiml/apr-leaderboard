#!/usr/bin/env bash
# results-history.sh — View and compare evaluation results
#
# Reads result JSON files from the results directory and displays
# a formatted comparison table. Supports filtering by benchmark
# and model.
#
# Usage:
#   ./scripts/results-history.sh [RESULTS_DIR] [BENCHMARK] [MODEL_FILTER]
#
# Examples:
#   ./scripts/results-history.sh                              # all results
#   ./scripts/results-history.sh results humaneval            # humaneval only
#   ./scripts/results-history.sh results "" qwen              # qwen models only

set -euo pipefail

RESULTS_DIR="${1:-results}"
BENCHMARK_FILTER="${2:-}"
MODEL_FILTER="${3:-}"

command -v jq >/dev/null 2>&1 || { echo "ERROR: jq not found"; exit 1; }

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "No results directory found: ${RESULTS_DIR}"
    exit 0
fi

RESULT_FILES=("$RESULTS_DIR"/*.json)
if [[ ! -f "${RESULT_FILES[0]:-}" ]]; then
    echo "No result files found in ${RESULTS_DIR}/"
    exit 0
fi

echo "=== Evaluation Results ==="
echo ""
printf "%-12s %-30s %-10s %8s %6s %6s %6s %12s\n" \
    "BENCHMARK" "MODEL" "STRATEGY" "pass@1" "PASS" "TOTAL" "ERRS" "DATE"
printf "%s\n" "$(printf '%.0s-' {1..100})"

for f in "${RESULT_FILES[@]}"; do
    [[ -f "$f" ]] || continue

    BENCH="$(jq -r '.benchmark // "?"' < "$f" 2>/dev/null)"
    MODEL="$(jq -r '.model // "?"' < "$f" 2>/dev/null)"
    PASS1="$(jq -r '.results.pass_at_1 // "?"' < "$f" 2>/dev/null)"
    PASSED="$(jq -r '.results.passed // "?"' < "$f" 2>/dev/null)"
    TOTAL="$(jq -r '.results.completed // "?"' < "$f" 2>/dev/null)"
    ERRS="$(jq -r '.results.errors // "?"' < "$f" 2>/dev/null)"
    STRATEGY="$(jq -r '.config.prompt_strategy // "standard"' < "$f" 2>/dev/null)"
    TIMESTAMP="$(jq -r '.timestamp // "?"' < "$f" 2>/dev/null | cut -dT -f1)"

    # Apply filters
    if [[ -n "$BENCHMARK_FILTER" && "$BENCH" != "$BENCHMARK_FILTER" ]]; then
        continue
    fi
    if [[ -n "$MODEL_FILTER" && "$MODEL" != *"$MODEL_FILTER"* ]]; then
        continue
    fi

    # Truncate model path for display
    MODEL_SHORT="$(basename "$MODEL" .apr)"
    if (( ${#MODEL_SHORT} > 28 )); then
        MODEL_SHORT="${MODEL_SHORT:0:25}..."
    fi

    printf "%-12s %-30s %-10s %7s%% %6s %6s %6s %12s\n" \
        "$BENCH" "$MODEL_SHORT" "$STRATEGY" "$PASS1" "$PASSED" "$TOTAL" "$ERRS" "$TIMESTAMP"
done

echo ""

# Show best result per benchmark
echo "=== Best Results ==="
echo ""
for bench in humaneval mbpp bigcodebench; do
    BEST="$(for f in "${RESULT_FILES[@]}"; do
        [[ -f "$f" ]] || continue
        jq -r "select(.benchmark == \"${bench}\") | \"\(.results.pass_at_1)\t\(.model)\"" < "$f" 2>/dev/null
    done | sort -t$'\t' -k1 -rn | head -1)"

    if [[ -n "$BEST" ]]; then
        SCORE="$(cut -f1 <<< "$BEST")"
        MDL="$(basename "$(cut -f2 <<< "$BEST")" .apr)"
        printf "  %-16s %s%%  (%s)\n" "$bench:" "$SCORE" "$MDL"
    fi
done

# Show throughput results separately
THROUGHPUT_BEST="$(for f in "${RESULT_FILES[@]}"; do
    [[ -f "$f" ]] || continue
    jq -r "select(.benchmark == \"throughput\") | \"\(.results.tokens_per_second)\t\(.results.time_to_first_token_ms)\t\(.backend // \"?\")\t\(.model)\"" < "$f" 2>/dev/null
done | sort -t$'\t' -k1 -rn | head -1)"

if [[ -n "$THROUGHPUT_BEST" ]]; then
    TPS="$(cut -f1 <<< "$THROUGHPUT_BEST")"
    TTFT="$(cut -f2 <<< "$THROUGHPUT_BEST")"
    BACKEND="$(cut -f3 <<< "$THROUGHPUT_BEST")"
    MDL="$(basename "$(cut -f4 <<< "$THROUGHPUT_BEST")" .apr)"
    printf "  %-16s %s tok/s, %sms TTFT [%s]  (%s)\n" "throughput:" "$TPS" "$TTFT" "$BACKEND" "$MDL"
fi
echo ""
