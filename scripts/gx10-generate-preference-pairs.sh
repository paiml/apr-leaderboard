#!/usr/bin/env bash
# gx10-generate-preference-pairs.sh — PMAT-014: Generate DPO preference pairs
#
# Runs N-sampling eval on HumanEval, then extracts preference pairs
# from borderline problems (where some samples pass and some fail).
#
# CPU batch mode: ~3h for HumanEval (164×10=1640 prompts)
# MBPP pairs generated from existing multi-strategy results.
#
# Usage:
#   ssh gx10 'cd ~/src/apr-leaderboard && nohup bash scripts/gx10-generate-preference-pairs.sh > logs/pmat-014.log 2>&1 &'

set -euo pipefail

CHECKPOINT="${1:-checkpoints/qwen2.5-coder-7b-instruct-q4k.apr}"
NUM_SAMPLES="${2:-10}"
TEMPERATURE="${3:-0.8}"

mkdir -p logs data

echo "=== PMAT-014: Preference Pair Generation ==="
echo "Date: $(date)"
echo "Model: $CHECKPOINT"
echo "Samples: $NUM_SAMPLES, Temperature: $TEMPERATURE"
echo ""

# Step 1: N-sampling HumanEval with preserved work dir
echo "--- Step 1: N-sampling HumanEval (${NUM_SAMPLES} samples × 164 problems) ---"
echo "Estimated: ~3h CPU batch, ~23h wgpu batch"
echo "Start: $(date)"

export APR_KEEP_WORKDIR=1
OUTPUT=$(bash scripts/eval-pass-at-k.sh humaneval "$CHECKPOINT" results 512 "$TEMPERATURE" "$NUM_SAMPLES" standard 1 2>&1)
echo "$OUTPUT"

# Extract work dir path from eval output
WORK_DIR_HE=$(echo "$OUTPUT" | grep "WORK_DIR preserved:" | sed 's/WORK_DIR preserved: //')
if [[ -z "$WORK_DIR_HE" || ! -d "$WORK_DIR_HE" ]]; then
    echo "ERROR: Could not find work dir from eval output"
    exit 1
fi
echo "Work dir: $WORK_DIR_HE"
echo "End: $(date)"
echo ""

# Step 2: Extract preference pairs
echo "--- Step 2: Extract HumanEval preference pairs ---"
bash scripts/generate-preference-pairs.sh "$WORK_DIR_HE" data/preference-pairs-humaneval.jsonl
echo ""

# Step 3: Combine
echo "--- Step 3: Combine preference pairs ---"
cat data/preference-pairs-humaneval.jsonl > data/preference-pairs.jsonl
PAIR_COUNT=$(wc -l < data/preference-pairs.jsonl)
echo "Total preference pairs: $PAIR_COUNT"

# Verify contract: FALSIFY-PREF-001 (>= 50 pairs)
if (( PAIR_COUNT >= 50 )); then
    echo "FALSIFY-PREF-001: PASS ($PAIR_COUNT >= 50)"
else
    echo "FALSIFY-PREF-001: FAIL ($PAIR_COUNT < 50)"
    echo "Consider increasing NUM_SAMPLES or adding MBPP N-sampling"
fi

# Clean up work dir
rm -rf "$WORK_DIR_HE"
echo ""

echo "=== PMAT-014 Complete ==="
echo "Date: $(date)"
echo "Output: data/preference-pairs.jsonl ($PAIR_COUNT pairs)"
echo ""
echo "Next steps:"
echo "  1. DPO training: apr finetune $CHECKPOINT --method qlora --data data/preference-pairs.jsonl --epochs 3"
echo "  2. Merge + quantize + eval"
