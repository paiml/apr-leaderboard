#!/usr/bin/env bash
# generate-preference-pairs.sh - Generate DPO preference pairs from N-sampling eval results
#
# PMAT-014: Converts N-sampling eval results into preference training data.
# For each problem where at least 1 sample passed and 1 sample failed,
# pairs correct (chosen) and incorrect (rejected) completions.
#
# Usage:
#   ./scripts/generate-preference-pairs.sh WORK_DIR OUTPUT_FILE
#
# Input:
#   WORK_DIR/problems/{idx}.json       - Benchmark problem definitions
#   WORK_DIR/completions/{idx}_s{n}.py - N completion samples per problem
#   WORK_DIR/task_results.tsv          - Per-problem n,c results (from eval)
#
# Output:
#   OUTPUT_FILE - JSONL with {"prompt": ..., "chosen": ..., "rejected": ...} lines
#
# Requires: N-sampling eval run first (NUM_SAMPLES > 1)

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: generate-preference-pairs.sh WORK_DIR OUTPUT_FILE"
    echo ""
    echo "Run an N-sampling eval first:"
    echo "  make eval-humaneval CHECKPOINT=m.apr NUM_SAMPLES=10 TEMPERATURE=0.8"
    echo ""
    echo "Then pass the work directory (printed during eval) and output file."
    exit 1
fi

WORK_DIR="$1"
OUTPUT_FILE="$2"

if [[ ! -d "$WORK_DIR" ]]; then
    echo "ERROR: Work directory not found: $WORK_DIR"
    exit 1
fi

if [[ ! -f "$WORK_DIR/task_results.tsv" ]]; then
    echo "ERROR: No task_results.tsv found. Run N-sampling eval first."
    exit 1
fi

echo "=== Preference Pair Generation (PMAT-014) ==="
echo "Work dir:   $WORK_DIR"
echo "Output:     $OUTPUT_FILE"

TOTAL_PAIRS=0
PROBLEMS_WITH_PAIRS=0
TOTAL_PROBLEMS=0

: > "$OUTPUT_FILE"

while IFS=$'\t' read -r task_id num_samples num_passed; do
    TOTAL_PROBLEMS=$((TOTAL_PROBLEMS + 1))

    # Need at least 1 pass and 1 fail to create a preference pair
    num_failed=$((num_samples - num_passed))
    if (( num_passed == 0 || num_failed == 0 )); then
        continue
    fi

    # Extract problem index from task_id (handles both "idx" and "HumanEval/idx" formats)
    problem_idx="${task_id##*/}"
    problem_idx="${problem_idx%%_s*}"

    problem_file="${WORK_DIR}/problems/${problem_idx}.json"
    if [[ ! -f "$problem_file" ]]; then
        continue
    fi

    prompt="$(jq -r '.instruct_prompt // .prompt // .text // .instruction // ""' < "$problem_file" 2>/dev/null)"
    if [[ -z "$prompt" || "$prompt" == "null" ]]; then
        continue
    fi

    # Collect passing and failing completions
    passing=()
    failing=()
    for sample in $(seq 0 $((num_samples - 1))); do
        completion_file="${WORK_DIR}/completions/${problem_idx}_s${sample}.py"
        if [[ ! -f "$completion_file" ]]; then
            # Try non-sample format for n=1
            completion_file="${WORK_DIR}/completions/${problem_idx}.py"
        fi
        if [[ ! -f "$completion_file" ]]; then continue; fi

        completion="$(cat "$completion_file")"
        if [[ "$completion" == "ERROR" || "$completion" == "SKIP" ]]; then continue; fi

        # Re-test to classify this specific sample
        test_file="$(mktemp)"
        entry_point="$(jq -r '.entry_point // ""' < "$problem_file" 2>/dev/null)"
        has_test="$(jq -r 'has("test")' < "$problem_file" 2>/dev/null)"

        if [[ "$has_test" == "true" ]]; then
            test_code="$(jq -r '.test' < "$problem_file" 2>/dev/null)"
            has_code_prompt="$(jq -r 'has("code_prompt")' < "$problem_file" 2>/dev/null)"
            {
                if [[ "$has_code_prompt" == "true" ]]; then
                    echo "$completion"
                else
                    echo "$prompt"
                    echo "$completion"
                fi
                echo ""
                echo "$test_code"
                if [[ -n "$entry_point" && "$entry_point" != "null" && "$has_code_prompt" != "true" ]]; then
                    echo "check(${entry_point})"
                fi
            } > "$test_file"
        else
            {
                echo "$prompt"
                echo "$completion"
            } > "$test_file"
        fi

        if timeout 10 python3 "$test_file" > /dev/null 2>&1; then
            passing+=("$completion")
        else
            failing+=("$completion")
        fi
        rm -f "$test_file"
    done

    # Generate all (chosen, rejected) pairs
    if (( ${#passing[@]} > 0 && ${#failing[@]} > 0 )); then
        PROBLEMS_WITH_PAIRS=$((PROBLEMS_WITH_PAIRS + 1))
        for chosen in "${passing[@]}"; do
            for rejected in "${failing[@]}"; do
                jq -nc --arg prompt "$prompt" --arg chosen "$chosen" --arg rejected "$rejected" \
                    '{prompt: $prompt, chosen: $chosen, rejected: $rejected}' >> "$OUTPUT_FILE"
                TOTAL_PAIRS=$((TOTAL_PAIRS + 1))
            done
        done
    fi

done < "$WORK_DIR/task_results.tsv"

echo ""
echo "=== Results ==="
echo "Problems processed:    $TOTAL_PROBLEMS"
echo "Problems with pairs:   $PROBLEMS_WITH_PAIRS"
echo "Total preference pairs: $TOTAL_PAIRS"
echo "Output:                $OUTPUT_FILE"

if (( TOTAL_PAIRS > 0 )); then
    echo ""
    echo "Use for DPO training:"
    echo "  apr finetune checkpoint.apr --method dpo --data $OUTPUT_FILE"
fi
