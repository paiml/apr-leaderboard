#!/usr/bin/env bash
# gx10-eval-distilled.sh — Evaluate the PMAT-007 distilled model
#
# Runs on gx10 after merge completes. Chain:
#   1. Verify merged model (apr check + inference smoke test)
#   2. Quantize to Q4K
#   3. Evaluate on HumanEval
#   4. Evaluate on MBPP
#
# Usage:
#   ssh gx10 'cd ~/src/apr-leaderboard && bash scripts/gx10-eval-distilled.sh'

set -euo pipefail

APR="$HOME/data/targets/aprender/release/apr"
MERGED="checkpoints/qwen2.5-coder-7b-distilled-merged-v2.apr"
Q4K="checkpoints/qwen2.5-coder-7b-distilled-v2-q4k.apr"
TOKENIZER="checkpoints/qwen2.5-coder-7b-instruct-q4k.tokenizer.json"

echo "=== PMAT-007: Distilled Model Evaluation Pipeline ==="
echo "Date: $(date)"
echo ""

# Step 1: Verify merged model
echo "--- Step 1: Verify merged model ---"
if [[ ! -f "$MERGED" ]]; then
    echo "ERROR: Merged model not found: $MERGED"
    exit 1
fi
echo "Size: $(du -h "$MERGED" | cut -f1)"
"$APR" check "$MERGED" --json 2>&1 | tail -20
echo ""

# Smoke test inference
echo "--- Step 1b: Smoke test inference ---"
OUTPUT=$("$APR" run "$MERGED" "def fibonacci(n):" --max-tokens 64 2>&1)
if echo "$OUTPUT" | grep -q "ERROR\|error\|failed"; then
    echo "ERROR: Inference smoke test FAILED"
    echo "$OUTPUT"
    exit 1
fi
echo "Smoke test PASSED: $(echo "$OUTPUT" | head -3)"
echo ""

# Step 2: Quantize to Q4K
echo "--- Step 2: Quantize to Q4K ---"
if [[ -f "$Q4K" ]]; then
    echo "Q4K already exists: $(du -h "$Q4K" | cut -f1), skipping"
else
    "$APR" quantize "$MERGED" --scheme q4k -o "$Q4K" 2>&1 | tail -10
    # Copy tokenizer
    cp "$TOKENIZER" "checkpoints/qwen2.5-coder-7b-distilled-v2-q4k.tokenizer.json"
    echo "Q4K: $(du -h "$Q4K" | cut -f1)"
fi
echo ""

# Step 3: Evaluate on HumanEval
echo "--- Step 3: HumanEval Evaluation ---"
echo "Start: $(date)"
bash scripts/eval-pass-at-k.sh humaneval "$Q4K" results 512 0.0 1 standard 1
echo "End: $(date)"
echo ""

# Step 4: Evaluate on MBPP
echo "--- Step 4: MBPP Evaluation ---"
echo "Start: $(date)"
bash scripts/eval-pass-at-k.sh mbpp "$Q4K" results 512 0.0 1 standard 1
echo "End: $(date)"
echo ""

echo "=== PMAT-007 Evaluation Complete ==="
echo "Results in results/ directory"
ls -lt results/*.json | head -5
