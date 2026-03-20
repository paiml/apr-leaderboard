#!/usr/bin/env bash
# leaderboard-summary.sh --Generate markdown leaderboard from eval results
#
# Reads all result JSONs and outputs a ranked markdown table suitable
# for README.md or spec docs. Groups by benchmark, ranks by pass@1.
#
# Usage:
#   ./scripts/leaderboard-summary.sh [RESULTS_DIR]
#
# Output modes:
#   Default: markdown table to stdout
#   --json:  JSON array to stdout

set -euo pipefail

RESULTS_DIR="${1:-results}"
OUTPUT_JSON=0
if [[ "${2:-}" == "--json" ]]; then
    OUTPUT_JSON=1
fi

command -v jq >/dev/null 2>&1 || { echo "ERROR: jq not found"; exit 1; }

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "No results directory: ${RESULTS_DIR}" >&2
    exit 0
fi

RESULT_FILES=("$RESULTS_DIR"/*.json)
if [[ ! -f "${RESULT_FILES[0]:-}" ]]; then
    echo "No result files in ${RESULTS_DIR}/" >&2
    exit 0
fi

# Collect all results into a temp file for sorting
TMPFILE="$(mktemp)"
trap 'rm -f "$TMPFILE"' EXIT

for f in "${RESULT_FILES[@]}"; do
    [[ -f "$f" ]] || continue
    jq -c '{
        benchmark: (.benchmark // "?"),
        model: ((.model // "?") | split("/") | last | rtrimstr(".apr")),
        strategy: (.config.prompt_strategy // "standard"),
        pass_at_1: (.results.pass_at_1 // 0),
        passed: (.results.passed // 0),
        total: (.results.total // 0),
        errors: (.results.errors // 0),
        timestamp: (.timestamp // "?" | split("T")[0]),
        file: "'"$(basename "$f")"'"
    }' < "$f" >> "$TMPFILE" 2>/dev/null
done

if [[ "$OUTPUT_JSON" -eq 1 ]]; then
    jq -s 'sort_by(-.pass_at_1)' "$TMPFILE"
    exit 0
fi

# Group by benchmark, rank by pass@1 descending
echo "## HumanEval Leaderboard"
echo ""
echo "| Rank | Model | Strategy | pass@1 | Passed | Date |"
echo "|------|-------|----------|--------|--------|------|"

RANK=0
jq -r 'select(.benchmark == "humaneval")' "$TMPFILE" \
    | jq -s 'sort_by(-.pass_at_1)[]' \
    | jq -r '[.model, .strategy, .pass_at_1, .passed, .total, .timestamp] | @tsv' \
    | while IFS=$'\t' read -r model strategy pass1 passed total ts; do
        RANK=$((RANK + 1))
        printf "| %d | %s | %s | **%.2f%%** | %s/%s | %s |\n" \
            "$RANK" "$model" "$strategy" "$pass1" "$passed" "$total" "$ts"
    done

# Check for MBPP results
MBPP_COUNT="$(jq -r 'select(.benchmark == "mbpp")' "$TMPFILE" | wc -l)"
if (( MBPP_COUNT > 0 )); then
    echo ""
    echo "## MBPP Leaderboard"
    echo ""
    echo "| Rank | Model | Strategy | pass@1 | Passed | Date |"
    echo "|------|-------|----------|--------|--------|------|"

    RANK=0
    jq -r 'select(.benchmark == "mbpp")' "$TMPFILE" \
        | jq -s 'sort_by(-.pass_at_1)[]' \
        | jq -r '[.model, .strategy, .pass_at_1, .passed, .total, .timestamp] | @tsv' \
        | while IFS=$'\t' read -r model strategy pass1 passed total ts; do
            RANK=$((RANK + 1))
            printf "| %d | %s | %s | **%.2f%%** | %s/%s | %s |\n" \
                "$RANK" "$model" "$strategy" "$pass1" "$passed" "$total" "$ts"
        done
fi

echo ""
echo "_Generated $(date -Iseconds)_"
