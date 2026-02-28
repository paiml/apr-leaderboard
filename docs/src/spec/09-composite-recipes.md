# Composite Recipes

## 9.0 Step Zero: Establish Baseline (REQUIRED for all recipes)

Every recipe must begin by establishing the apr-native baseline for the model. This catches inference implementation gaps before optimization work begins.

```bash
# Import the target model
apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct -o baseline-instruct.apr

# Establish apr-native baseline on all target benchmarks
apr eval baseline-instruct.apr --task classify --data humaneval.jsonl --json > results/baseline.json

# Compare against HuggingFace reference scores
apr compare-hf baseline-instruct.apr --json > results/parity-baseline.json

# Gate: if apr baseline is >5% below HF reference, investigate inference bugs first
```

**Why this matters:** Qwen2.5-Coder-7B-Instruct scores ~84% pass@1 on HumanEval in the PyTorch/HF stack. If the apr-native baseline is significantly lower, no amount of optimization will close the gap — fix inference fidelity first. All "expected gain" numbers below are relative to the apr-native baseline, not absolute.

## 9.1 Recipe A: "The Distilled Expert" (Maximum Quality)

**Target:** Highest pass@1 regardless of model size. For 7B submissions.

```bash
# 1. Import
apr import hf://Qwen/Qwen2.5-Coder-32B -o teacher.apr
apr import hf://Qwen/Qwen2.5-Coder-7B -o student.apr

# 2. Distill 32B → 7B
apr distill teacher.apr \
    --student student.apr \
    --strategy progressive \
    --temperature 3.0 \
    --alpha 0.7 \
    --epochs 5 \
    --data code-corpus-100k.jsonl \
    -o distilled.apr

# 3. LoRA fine-tune on curated instruction data
apr finetune distilled.apr \
    --method lora \
    --rank 32 \
    --data code-instruct-curated.jsonl \
    --epochs 3 \
    --learning-rate 2e-4 \
    -o distilled-lora/

# 4. Merge adapter
apr finetune distilled.apr \
    --adapter distilled-lora/ \
    --merge \
    -o distilled-finetuned.apr

# 5. Eval
apr eval distilled-finetuned.apr --task classify --data humaneval.jsonl --json
```

**Expected:** +5-13% pass@1 over apr-native 7B base baseline. Target: match or exceed the instruct model's HF-reference score once inference parity is established.

## 9.2 Recipe B: "The Merge Alchemist" (Zero Training Compute)

**Target:** Best score achievable with NO GPU training at all. Pure weight manipulation.

```bash
# 1. Import distinct specialist variants (different fine-tunes, not base+instruct)
apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct -o instruct.apr
apr import hf://Qwen/Qwen2.5-Coder-7B -o base.apr

# Note: For best results, find community fine-tunes that specialize in
# different code domains (e.g., one tuned on Python, one on algorithms).
# Merging base+instruct rarely beats the instruct model alone.

# 2. TIES merge instruct variants (resolve sign conflicts between specialists)
apr merge instruct.apr variant-b.apr \
    --strategy ties \
    --base-model base.apr \
    --density 0.2 \
    -o ties-blend.apr

# 3. Prune: remove redundant attention heads (structured)
apr prune ties-blend.apr \
    --method structured \
    --target-ratio 0.15 \
    -o pruned.apr

# 4. Quantize for fast inference
apr quantize pruned.apr --scheme q4k -o submit-q4k.apr

# 5. Eval
apr eval submit-q4k.apr --task classify --data humaneval.jsonl --json
```

**Expected:** Within 1-3% of the best input specialist's pass@1, potentially exceeding it. Merging is not a guaranteed gain — always eval against the unmerged instruct model as control.

## 9.3 Recipe C: "The Full Pipeline" (Kitchen Sink)

**Target:** Absolute maximum. Every technique stacked.

```bash
#!/bin/bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-Coder-7B"
TEACHER="Qwen/Qwen2.5-Coder-32B"

echo "=== Phase 1: Import ==="
apr import "hf://${TEACHER}" -o teacher.apr
apr import "hf://${MODEL}" -o base.apr

echo "=== Phase 2: Distill (32B → 7B) ==="
apr distill teacher.apr \
    --student base.apr \
    --strategy progressive \
    --temperature 3.0 --alpha 0.7 --epochs 5 \
    --data code-corpus.jsonl \
    -o distilled.apr

echo "=== Phase 3: HPO Scout ==="
apr tune distilled.apr \
    --task classify \
    --data code-instruct.jsonl \
    --budget 20 --scout --strategy tpe --scheduler asha

echo "=== Phase 4: LoRA Fine-tune (using scout-optimal params) ==="
apr finetune distilled.apr \
    --method lora --rank 32 \
    --data code-instruct-50k.jsonl \
    --epochs 5 --learning-rate 2e-4 \
    -o finetuned-lora/

apr finetune distilled.apr \
    --adapter finetuned-lora/ --merge \
    -o finetuned.apr

echo "=== Phase 5: Train 2nd variant for merging ==="
apr finetune distilled.apr \
    --method lora --rank 16 \
    --data code-reasoning.jsonl \
    --epochs 3 --learning-rate 1e-4 \
    -o reasoning-lora/

apr finetune distilled.apr \
    --adapter reasoning-lora/ --merge \
    -o reasoning-variant.apr

echo "=== Phase 6: TIES Merge ==="
apr merge finetuned.apr reasoning-variant.apr \
    --strategy ties \
    --base-model distilled.apr \
    --density 0.2 \
    -o merged.apr

echo "=== Phase 7: Wanda Prune (20%) ==="
apr prune merged.apr \
    --method wanda --target-ratio 0.2 \
    --calibration calib-code.jsonl \
    -o pruned.apr

echo "=== Phase 8: Quantize ==="
apr quantize pruned.apr --scheme int4 -o final.apr

echo "=== Phase 9: Evaluate ==="
apr eval final.apr --task classify --data humaneval.jsonl --json
apr eval final.apr --task classify --data mbpp.jsonl --json
apr bench final.apr --verbose

echo "=== Phase 10: Compile to standalone binary ==="
apr compile final.apr -o apr-coder --release --strip --lto

echo "=== Done ==="
echo "Standalone binary: $(ls -lh apr-coder)"
```

**Expected:** +8-17% pass@1 over apr-native 7B base baseline. Should match or exceed the instruct model's HF-reference score.

## 9.4 Recipe D: "Sovereign Binary" (The Differentiator)

**Target:** Ship the model AS a Rust binary. No runtime, no Python, no Docker.

```bash
# Full pipeline → compiled binary
apr import hf://Qwen/Qwen2.5-Coder-1.5B -o small.apr
apr finetune small.apr --method qlora --rank 16 --data instruct.jsonl -o tuned.apr
apr prune tuned.apr --method magnitude --target-ratio 0.4 -o slim.apr
apr quantize slim.apr --scheme int4 -o tiny.apr

# Compile to standalone binary (no runtime deps)
apr compile tiny.apr \
    -o qwen-coder \
    --target x86_64-unknown-linux-musl \
    --release --strip --lto --quantize int4

# Result: single static binary, ~800MB (750MB weights + runtime), runs on any Linux
./qwen-coder "def fibonacci(n):"
```

**Size estimates:** 1.5B INT4 ≈ 800MB, 7B INT4 ≈ 4GB, 32B INT4 ≈ 17GB. Still dramatically smaller than Docker + Python + CUDA runtime images (typically 10-20GB for a 7B setup).

**This is the marketing win:** While competitors need `pip install transformers torch accelerate bitsandbytes`, we ship `./qwen-coder`.
