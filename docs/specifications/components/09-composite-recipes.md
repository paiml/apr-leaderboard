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

**Size estimates:** 1.5B INT4 ≈ 800MB, 7B INT4 ≈ 4GB, 32B INT4 ≈ 17GB. Still dramatically smaller than Docker + Python + GPU runtime images (typically 10-20GB for a 7B setup).

**This is the marketing win:** While competitors need `pip install transformers torch accelerate bitsandbytes`, we ship `./qwen-coder`.

## 9.5 Recipe E: "Instruct LoRA" (Proven Training Loop)

**Target:** Validate the full LoRA instruction-tuning loop on the existing 7B Q4K checkpoint using ground truth corpora. This is the foundation recipe — it proves the training pipeline works end-to-end before attempting more expensive QLoRA or distillation.

**Model:** Qwen2.5-Coder-7B-Instruct (Q4K, already imported)
**Data:** 15,494 instruction/response pairs from `make prep-data`
**VRAM:** ~28 GB (full-precision LoRA on Q4K base)

```bash
# 0. Prerequisites: checkpoint + data must exist
ls checkpoints/qwen2.5-coder-7b-instruct-q4k.apr  # 7.48 GiB
ls data/instruct-corpus.jsonl                       # 15,494 pairs

# 1. Baseline eval (pre-training score)
make eval-humaneval CHECKPOINT=checkpoints/qwen2.5-coder-7b-instruct-q4k.apr

# 2. LoRA instruction fine-tune
apr finetune checkpoints/qwen2.5-coder-7b-instruct-q4k.apr \
    --task instruct \
    --data data/instruct-corpus.jsonl \
    --model-size 7B \
    --rank 16 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --output checkpoints/qwen2.5-coder-7b-instruct-lora.apr \
    --verbose

# 3. Post-training eval
make eval-humaneval CHECKPOINT=checkpoints/qwen2.5-coder-7b-instruct-lora.apr

# 4. Compare pre/post
diff results/humaneval-pre.json results/humaneval-post.json
```

**Config:** `configs/recipes/recipe-e-instruct-finetune.yaml`

**Gate criteria:**
- Training loss must decrease monotonically (proves optimizer is working)
- Post-training pass@1 ≥ pre-training pass@1 (no regression)
- If post < pre, investigate overfitting (reduce epochs) or data quality

**Expected:** +3-8% pass@1 from instruction tuning on domain-specific corpora. The 15.5K corpus covers algorithms (depyler), HuggingFace patterns (hf-gtc), JAX numerics (jax-gtc), and vLLM inference (vllm-gtc).

**Status (2026-03-04):** Training pipeline fully implemented. InstructPipeline supports CPU and NF4 QLoRA GPU paths via wgpu (KAIZEN-064/065/068). CLI wired: `apr finetune --task instruct --method qlora --quantize-nf4`. Ready for 7B training run on any GPU.

## 9.6 Recipe F: "Qwen3 QLoRA" (Consumer GPU Path)

**Target:** QLoRA fine-tune Qwen3-8B on consumer GPUs (8-16 GB VRAM). This is the primary leaderboard submission path — it produces a competitive model using hardware most developers already own.

**Model:** Qwen3-8B (FP16, 16 GB)
**Data:** Same 15,494 instruction/response pairs
**VRAM:** ~4.5 GB (NF4-quantized base + FP16 LoRA adapters)

**Why Qwen3-8B over Qwen2.5-7B:** Qwen3 is a newer architecture with improved training data and reasoning capabilities. QLoRA on FP16 base (not pre-quantized Q4K) produces better adapters because the NF4 quantization is applied optimally during training, not inherited from a pre-quantized checkpoint.

**Why QLoRA over LoRA:** At 8B parameters, full-precision LoRA requires ~32 GB VRAM. QLoRA reduces this to ~4.5 GB by quantizing base weights to NF4 (4-bit NormalFloat) while keeping LoRA adapters in FP16. The 0.85x quality factor (vs full-precision LoRA) is offset by the ability to use higher rank (32 vs 16) within the same VRAM budget.

```bash
# 0. Import Qwen3-8B at FP16 (already done: 16 GB checkpoint)
make import MODEL=Qwen/Qwen3-8B QUANTIZE=fp16
ls checkpoints/qwen_qwen3-8b.apr  # 16 GB FP16

# 1. Prepare instruction data
make prep-data
wc -l data/instruct-corpus.jsonl  # 15,494 pairs

# 2. Baseline eval (pre-QLoRA)
make eval-humaneval CHECKPOINT=checkpoints/qwen_qwen3-8b.apr

# 3. QLoRA fine-tune (NF4 base + FP16 adapters)
apr finetune checkpoints/qwen_qwen3-8b.apr \
    --method qlora \
    --task instruct \
    --data data/instruct-corpus.jsonl \
    --model-size 8B \
    --rank 16 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --max-seq-len 512 \
    --vram 8 \
    --output checkpoints/qwen3-8b-qlora.apr \
    --verbose

# 4. Post-QLoRA eval
make eval-humaneval CHECKPOINT=checkpoints/qwen3-8b-qlora.apr
make eval-bigcodebench CHECKPOINT=checkpoints/qwen3-8b-qlora.apr

# 5. Optional: quantize merged model for faster inference
apr quantize checkpoints/qwen3-8b-qlora.apr \
    --scheme q4k \
    -o checkpoints/qwen3-8b-qlora-q4k.apr
```

**Config:** `configs/recipes/recipe-f-qwen3-qlora.yaml`

**VRAM budget breakdown (rank-16, batch-1, seq-512):**

| Component | Bytes | Notes |
|-----------|-------|-------|
| NF4 base weights | ~4.0 GB | 8B params × 4 bits |
| LoRA A matrices (28 layers × Q,V) | ~6.1 MB | 56 × rank × hidden_dim × 2 bytes |
| LoRA B matrices (28 layers × Q,V) | ~6.1 MB | 56 × hidden_dim × rank × 2 bytes |
| Optimizer states (AdamW) | ~24.4 MB | 2 × LoRA params × 4 bytes (m, v) |
| Activations + gradients | ~400 MB | Depends on seq_len and batch_size |
| **Total** | **~4.5 GB** | Fits 3x within 24 GB GPU |

**Training brick benchmarks (measured on Qwen2 7B, same architecture class):**

| Brick | Dimensions | Budget | Notes |
|-------|-----------|--------|-------|
| `lora_forward` | d_in=3584, rank=16 | 54µs actual (CPU) | Real matmul, not analytical |
| `optimizer` | 6.4M LoRA params | 50µs analytical | SIMD AdamW over LoRA params |
| `loss` | vocab=152064, seq=128 | 20µs analytical | Cross-entropy |
| `train_step` | 28 layers, rank-16 | 5000µs analytical | Composite fwd+bwd+optim |

**Gate criteria:**
- VRAM peak < 8 GB (AC-005: QLoRA uses <50% VRAM vs LoRA)
- Training loss decreases over 3 epochs
- Post-QLoRA pass@1 > pre-QLoRA pass@1 on HumanEval
- No NaN loss (Jidoka: training bricks check for NaN)

**Expected:** +5-12% pass@1 over apr-native baseline. QLoRA on Qwen3-8B with curated instruction data should approach the instruct model's HF-reference score.

**Status (2026-03-04): READY.** QLoRA instruct pipeline fully implemented with wgpu NF4 support (GPU-resident gradients, fused causal cross-entropy, LoRA backward GEMM). GPU-SHARE infrastructure (143 tests) enables multi-adapter concurrent training. CLI: `apr finetune --task instruct --method qlora --quantize-nf4`. Ready for full training run on 15K-sample instruct corpus. Runs on any GPU via wgpu (Vulkan/Metal/DX12).

### 9.6.1 Recipe E vs Recipe F Decision Matrix

| Factor | Recipe E (Instruct LoRA) | Recipe F (Qwen3 QLoRA) |
|--------|-------------------------|------------------------|
| Model | Qwen2.5-Coder-7B Q4K | Qwen3-8B FP16 |
| Method | LoRA (full precision) | QLoRA (NF4 base) |
| VRAM required | ~28 GB | ~4.5 GB |
| GPU required | 32+ GB GPU (any vendor) | Any 8+ GB GPU (any vendor via wgpu) |
| Training quality | Highest (no quantization noise) | ~0.85x (NF4 noise in backward pass) |
| Use case | Maximum quality, server GPU | Consumer GPU, rapid iteration |
| **Recommended for** | Final submission | Development + ablation |

**Strategy:** Use Recipe F for rapid iteration and hyperparameter search (fast, cheap). Once optimal hyperparameters are found, run Recipe E on a server GPU for the final submission model.

## 9.7 Recipe G: "wgpu Training Proof" (GPU Verification)

**Target:** Prove wgpu GPU training works end-to-end: import → QLoRA train → verify loss decrease.

**Model:** Qwen2.5-Coder-1.5B (smallest model, fastest iteration)

```bash
# Full proof: import → train → verify
make prove-wgpu
# Equivalent to: scripts/prove-wgpu.sh
```

**Stages:** import → finetune (QLoRA, 2 epochs, 200 samples) → verify (loss decrease)

**Result:** Verified — loss decreases over 2 epochs on wgpu (Vulkan/Metal/DX12). No CUDA toolkit required. See §22.14 and §23 for detailed findings.

## 9.8 Recipe H: "Reasoning Distillation" (32B → 7B)

**Target:** Transfer 32B teacher's 89.63% HumanEval score into 7B student while preserving fast inference.

**Teacher:** Qwen2.5-Coder-32B-Instruct Q4K_M (89.63% HumanEval, 294s/problem GPU)
**Student:** Qwen2.5-Coder-7B-Instruct Q4K (85.37% HumanEval, 112s/problem GPU)

```bash
# Prerequisites: both checkpoints must exist
ls checkpoints/qwen2.5-coder-32b-instruct-q4km.apr  # 19 GB
ls checkpoints/qwen2.5-coder-7b-instruct-q4k.apr     # 7.48 GB

# 1. Progressive distillation (high temperature for soft labels)
apr distill checkpoints/qwen2.5-coder-32b-instruct-q4km.apr \
    --student checkpoints/qwen2.5-coder-7b-instruct-q4k.apr \
    --strategy progressive \
    --temperature 4.0 \
    --alpha 0.8 \
    --epochs 3 \
    -o checkpoints/qwen-7b-distilled.apr

# 2. Evaluate distilled student
make eval-humaneval CHECKPOINT=checkpoints/qwen-7b-distilled.apr

# 3. Compare with baseline
make compare-results \
    BASE=results/humaneval_7b_standard.json \
    NEW=results/humaneval_7b_distilled.json
```

**Config:** `configs/recipes/recipe-h-32b-distill.yaml`

**Expected:** Close the 4.3pp gap (89.63% → 85.37%). Progressive distillation with temperature 4.0 provides soft probability distributions that transfer the teacher's reasoning patterns into the smaller student network.

**Why not just use 32B?** The 32B model runs at ~14 tok/s (294s/problem) vs 7B at ~85 tok/s (112s/problem). For production inference, 7B is 2.6x faster. Distillation aims to get 32B quality at 7B speed.
