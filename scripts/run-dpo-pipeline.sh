#!/usr/bin/env bash
# run-dpo-pipeline.sh - End-to-end DPO alignment pipeline (PMAT-008)
#
# Runs after preference pairs are generated (PMAT-014):
# 1. DPO fine-tune on preference pairs
# 2. Merge DPO adapter into base model
# 3. Quantize to Q4K
# 4. Evaluate on HumanEval + MBPP
#
# Usage:
#   ./scripts/run-dpo-pipeline.sh [PREFERENCE_PAIRS] [BASE_MODEL]
#
# Defaults:
#   PREFERENCE_PAIRS=data/preference-pairs.jsonl
#   BASE_MODEL=checkpoints/qwen2.5-coder-7b-instruct-q4k.apr
#
# Contract: contracts/dpo-alignment.yaml v2.0
# Recipe: configs/recipes/recipe-l-dpo-alignment.yaml

set -euo pipefail

PAIRS="${1:-data/preference-pairs.jsonl}"
BASE="${2:-checkpoints/qwen2.5-coder-7b-instruct-q4k.apr}"
APR="${APR:-apr}"

ADAPTER="checkpoints/qwen2.5-coder-7b-dpo-adapter"
MERGED="checkpoints/qwen2.5-coder-7b-dpo-merged.apr"
Q4K="checkpoints/qwen2.5-coder-7b-dpo-q4k.apr"

echo "=== PMAT-008: DPO Alignment Pipeline ==="
echo "Preference pairs: $PAIRS"
echo "Base model:       $BASE"
echo "Adapter output:   $ADAPTER"
echo ""

# Validate inputs
if [[ ! -f "$PAIRS" ]]; then
    echo "ERROR: Preference pairs not found: $PAIRS"
    echo "Run: make generate-preference-pairs WORK_DIR=/tmp/... OUTPUT=$PAIRS"
    exit 1
fi
if [[ ! -f "$BASE" ]]; then
    echo "ERROR: Base model not found: $BASE"
    exit 1
fi

pair_count=$(wc -l < "$PAIRS")
echo "Preference pairs: $pair_count"
if (( pair_count < 50 )); then
    echo "WARNING: Only $pair_count pairs. DPO contract requires >= 50 valid pairs."
fi

# Step 1: DPO fine-tune
echo ""
echo "=== Step 1/4: DPO Fine-tune ==="
echo "Method: dpo, rank=16, lr=5e-5, epochs=3, beta=0.1"
$APR finetune "$BASE" \
    --method dpo \
    --data "$PAIRS" \
    --rank 16 \
    --learning-rate 5e-5 \
    --epochs 3 \
    -o "$ADAPTER" \
    --verbose

echo "Adapter saved: $ADAPTER"
ls -lh "$ADAPTER"

# Step 2: Merge adapter into base
echo ""
echo "=== Step 2/4: Merge Adapter ==="
$APR finetune "$BASE" \
    --merge \
    --adapter "$ADAPTER" \
    -o "$MERGED" \
    --verbose

echo "Merged model: $MERGED"
ls -lh "$MERGED"

# Step 3: Quantize to Q4K
echo ""
echo "=== Step 3/4: Quantize to Q4K ==="
$APR quantize "$MERGED" \
    --scheme q4k \
    -o "$Q4K" \
    --verbose

echo "Q4K model: $Q4K"
ls -lh "$Q4K"

# Step 4: Evaluate
echo ""
echo "=== Step 4/4: Evaluate ==="

echo "--- HumanEval ---"
bash scripts/eval-pass-at-k.sh humaneval "$Q4K" results 512 0.0 1 standard 1

echo ""
echo "--- MBPP ---"
bash scripts/eval-pass-at-k.sh mbpp "$Q4K" results 512 0.0 1 standard 1

echo ""
echo "=== PMAT-008 COMPLETE ==="
echo "Check results/ for eval JSONs."
echo "DPO contract gate: MBPP >= 78%, HumanEval >= 84%"
