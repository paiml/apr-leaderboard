#!/usr/bin/env bash
# compare-results.sh --Compare two eval result files at per-problem level
#
# Shows which problems were gained/lost between two runs.
# Requires results with "problems" array (eval-pass-at-k.sh with per-problem output).
#
# Usage:
#   ./scripts/compare-results.sh BASELINE.json NEW.json

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: compare-results.sh BASELINE.json NEW.json"
    exit 1
fi

BASELINE="$1"
NEW="$2"

command -v jq >/dev/null 2>&1 || { echo "ERROR: jq not found"; exit 1; }

for f in "$BASELINE" "$NEW"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

# Check for per-problem data
for f in "$BASELINE" "$NEW"; do
    if ! jq -e '.problems' < "$f" > /dev/null 2>&1; then
        echo "ERROR: $f missing 'problems' array (re-run eval with updated script)"
        exit 1
    fi
done

B_STRATEGY="$(jq -r '.config.prompt_strategy // "?"' < "$BASELINE")"
N_STRATEGY="$(jq -r '.config.prompt_strategy // "?"' < "$NEW")"
B_MODEL="$(basename "$(jq -r '.model' < "$BASELINE")" .apr)"
N_MODEL="$(basename "$(jq -r '.model' < "$NEW")" .apr)"
B_PASS1="$(jq -r '.results.pass_at_1' < "$BASELINE")"
N_PASS1="$(jq -r '.results.pass_at_1' < "$NEW")"
B_PASSED="$(jq -r '.results.passed' < "$BASELINE")"
N_PASSED="$(jq -r '.results.passed' < "$NEW")"
B_TOTAL="$(jq -r '.results.total' < "$BASELINE")"

echo "=== Result Comparison ==="
echo ""
printf "  Baseline: %s [%s] --%s%% (%s/%s)\n" "$B_MODEL" "$B_STRATEGY" "$B_PASS1" "$B_PASSED" "$B_TOTAL"
printf "  New:      %s [%s] --%s%% (%s/%s)\n" "$N_MODEL" "$N_STRATEGY" "$N_PASS1" "$N_PASSED" "$B_TOTAL"
echo ""

# Build task_id -> passed maps and compute delta
REGRESSIONS="$(jq -r --slurpfile new "$NEW" '
    [.problems[] | {(.task_id): .passed}] | add as $base |
    [$new[0].problems[] | {(.task_id): .passed}] | add as $new_map |
    [.problems[] | .task_id] | .[] |
    select($base[.] > 0 and ($new_map[.] // 0) == 0)
' < "$BASELINE" 2>/dev/null)"

GAINS="$(jq -r --slurpfile new "$NEW" '
    [.problems[] | {(.task_id): .passed}] | add as $base |
    [$new[0].problems[] | {(.task_id): .passed}] | add as $new_map |
    [.problems[] | .task_id] | .[] |
    select(($base[.] // 0) == 0 and ($new_map[.] // 0) > 0)
' < "$BASELINE" 2>/dev/null)"

REG_COUNT="$(echo "$REGRESSIONS" | grep -c . 2>/dev/null || echo 0)"
GAIN_COUNT="$(echo "$GAINS" | grep -c . 2>/dev/null || echo 0)"

if [[ -n "$REGRESSIONS" && "$REGRESSIONS" != "" ]]; then
    echo "Regressions (${REG_COUNT} problems baseline passed, new failed):"
    echo "$REGRESSIONS" | while read -r tid; do
        echo "  - $tid"
    done
    echo ""
fi

if [[ -n "$GAINS" && "$GAINS" != "" ]]; then
    echo "Gains (${GAIN_COUNT} problems baseline failed, new passed):"
    echo "$GAINS" | while read -r tid; do
        echo "  + $tid"
    done
    echo ""
fi

NET=$(( GAIN_COUNT - REG_COUNT ))
echo "Net: ${NET} (${GAIN_COUNT} gained, ${REG_COUNT} lost)"
