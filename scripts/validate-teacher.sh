#!/usr/bin/env bash
# Validate that a teacher model meets minimum quality for distillation.
# Per §12.2: "Never generate training data from a teacher whose inference
# quality hasn't been verified."
#
# Usage: scripts/validate-teacher.sh <CHECKPOINT> [BENCHMARK] [MIN_PASS_AT_1]
# Exit 0: teacher is valid
# Exit 1: teacher is too weak or not found

set -euo pipefail

CHECKPOINT="${1:?Usage: validate-teacher.sh <CHECKPOINT> [BENCHMARK] [MIN_PASS_AT_1]}"
BENCHMARK="${2:-humaneval}"
MIN_PASS_AT_1="${3:-0.60}"

echo "=== Teacher Quality Validation ==="
echo "  Model:     $CHECKPOINT"
echo "  Benchmark: $BENCHMARK"
echo "  Threshold: ${MIN_PASS_AT_1} ($(awk "BEGIN{printf \"%.0f\", $MIN_PASS_AT_1 * 100}")%)"
echo ""

# Check model exists
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "FAIL: Model not found: $CHECKPOINT"
    exit 1
fi

# Check if we already have eval results for this model
MODEL_NAME="$(basename "$CHECKPOINT" .apr)"
RESULTS_DIR="${RESULTS_DIR:-results}"

# Search existing results for this model
BEST_SCORE=0
BEST_FILE=""
for result_file in "$RESULTS_DIR"/${BENCHMARK}_*.json; do
    [[ -f "$result_file" ]] || continue

    model_in_result="$(jq -r '.model // ""' "$result_file" 2>/dev/null)"
    if echo "$model_in_result" | grep -qi "$MODEL_NAME"; then
        score="$(jq -r '.results.pass_at_1 // 0' "$result_file" 2>/dev/null)"
        if awk "BEGIN{exit !($score > $BEST_SCORE)}" 2>/dev/null; then
            BEST_SCORE="$score"
            BEST_FILE="$result_file"
        fi
    fi
done

if [[ -n "$BEST_FILE" ]]; then
    echo "Found existing result: $BEST_FILE"
    echo "  pass@1: ${BEST_SCORE}%"

    # Convert percentage to decimal for comparison
    SCORE_DEC="$(awk "BEGIN{printf \"%.4f\", $BEST_SCORE / 100}")"

    if awk "BEGIN{exit !($SCORE_DEC >= $MIN_PASS_AT_1)}" 2>/dev/null; then
        echo ""
        echo "PASS: Teacher qualifies (${BEST_SCORE}% >= $(awk "BEGIN{printf \"%.0f\", $MIN_PASS_AT_1 * 100}")%)"
        exit 0
    else
        echo ""
        echo "FAIL: Teacher too weak (${BEST_SCORE}% < $(awk "BEGIN{printf \"%.0f\", $MIN_PASS_AT_1 * 100}")%)"
        echo "  Distillation from a weak teacher will degrade the student."
        exit 1
    fi
fi

echo "No existing results found for '$MODEL_NAME' on $BENCHMARK."
echo ""
echo "To validate teacher quality, run evaluation first:"
echo "  make eval-${BENCHMARK} CHECKPOINT=$CHECKPOINT"
echo ""
echo "FAIL: Cannot verify teacher quality without eval results."
exit 1
