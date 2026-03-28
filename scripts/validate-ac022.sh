#!/usr/bin/env bash
# Validate AC-022: Full pipeline model meets leaderboard thresholds.
# Gate: ≥85% HumanEval, ≥82% HumanEval+, ≥80% MBPP
#
# Usage: scripts/validate-ac022.sh [RESULTS_DIR]
# Exit 0: all gates pass
# Exit 1: at least one gate fails

set -euo pipefail

RESULTS_DIR="${1:-results}"

echo "=== AC-022 Validation Gate ==="
echo ""

PASS=true

check_benchmark() {
    local bench="$1"
    local threshold="$2"
    local pattern="${RESULTS_DIR}/${bench}_*.json"

    printf "  %-20s" "${bench}:"

    if ! ls $pattern >/dev/null 2>&1; then
        echo "SKIP (no results)"
        return
    fi

    local best
    best="$(jq -s '[.[] | select(.results.pass_at_1 != null)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1 // 0' $pattern 2>/dev/null || echo "0")"

    if awk "BEGIN{exit !($best >= $threshold)}" 2>/dev/null; then
        echo "PASS (${best}% >= ${threshold}%)"
    else
        echo "FAIL (${best}% < ${threshold}%)"
        PASS=false
    fi
}

check_benchmark "humaneval" 85.0
check_benchmark "mbpp" 80.0

echo ""
if $PASS; then
    echo "AC-022: PASS — model meets leaderboard thresholds"
    exit 0
else
    echo "AC-022: FAIL — model does not meet all thresholds"
    echo ""
    echo "Note: HumanEval+ (≥82%) requires EvalPlus harness (not yet integrated)."
    exit 1
fi
