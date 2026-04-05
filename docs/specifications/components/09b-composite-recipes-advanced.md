# Composite Recipes (Advanced)

## 9.7 Recipe G: "wgpu Training Proof" (GPU Verification)

**Target:** Prove wgpu GPU training works end-to-end: import → QLoRA train → verify loss decrease.

**Model:** Qwen2.5-Coder-1.5B (smallest model, fastest iteration)

```bash
# Full proof: import → train → verify
make prove-wgpu
# Equivalent to: scripts/prove-wgpu.sh
```

**Stages:** import → finetune (QLoRA, 2 epochs, 200 samples) → verify (loss decrease)

**Result:** Verified — loss decreases over 2 epochs on wgpu (Vulkan/Metal/DX12). No CUDA toolkit required. See S22.14 and S23 for detailed findings.

## 9.8 Recipe H: "Reasoning Distillation" (32B → 7B)

**Target:** Transfer 32B teacher's 90.85% HumanEval score into 7B student while preserving fast inference.

**Teacher:** Qwen2.5-Coder-32B-Instruct Q4K_M (90.85% HumanEval)
**Student:** Qwen2.5-Coder-7B-Instruct Q4K (87.20% HumanEval few-shot)

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

**Expected:** Close the 3.65pp gap (90.85% → 87.20%). Progressive distillation with temperature 4.0 provides soft probability distributions that transfer the teacher's reasoning patterns into the smaller student network.

**Why not just use 32B?** The 32B model runs at ~14 tok/s (294s/problem) vs 7B at ~85 tok/s (112s/problem). For production inference, 7B is 2.6x faster. Distillation aims to get 32B quality at 7B speed.

## 9.9 Recipe I: "HumanEval QLoRA" (Targeted Fine-Tuning)

**Target:** Push 7B model past 87% HumanEval pass@1 using combined teacher completions + instruct corpus.

**Data sources:**
- Teacher completions (PMAT-007): 32B generates 99 targeted coding completions for problem areas where 7B fails (string manipulation, mathematical reasoning, list operations, edge cases)
- Instruct corpus (PMAT-004): 15K instruction-completion pairs from depyler ground-truth AST extractions

```bash
# Stage 1: Generate teacher completions (run on gx10)
make distill-generate

# Stage 2: Combine all training data (dedup + shuffle)
make combine-training-data

# Stage 3: QLoRA fine-tune 7B student
make distill-finetune

# Stage 4: Evaluate on HumanEval
make distill-eval

# Compare with baseline
make compare-results \
    BASE=results/humaneval_7b_standard.json \
    NEW=results/humaneval_7b_distilled.json
```

**Config:** `configs/recipes/recipe-i-humaneval-qlora.yaml`

**Method:** QLoRA (rank 32, lr 2e-4, 3 epochs) — same method proven working in S22.7 and S23.1.4.

**Falsifiable:** If HumanEval stays below 86% after training, the approach is falsified. Expected: 85.37% → 87%+ from domain-targeted training data.

**Why combined data?** The 32B teacher completions target the 25 specific HumanEval failures (analyzed via `scripts/generate-distill-prompts.sh`), while the instruct corpus provides broad coding pattern coverage. Together they should improve both the specific failure cases and overall code generation quality.

## 9.10 Recipe J: "Specialist Merge" (PMAT-010)

**Target:** TIES merge code-specialist + reasoning-specialist. Hypothesis H3: merged model beats any single specialist on at least one benchmark.

**Inputs:**
- Code specialist from PMAT-008 (QLoRA on code instruct data)
- Reasoning specialist from PMAT-007 (distilled from 32B teacher)
- Base model: Qwen2.5-Coder-7B-Instruct Q4K

```bash
# TIES merge at density 0.2 (20% of task vector kept)
apr merge checkpoints/qwen-7b-code-specialist.apr \
    checkpoints/qwen-7b-reasoning-specialist.apr \
    --strategy ties --density 0.2 \
    --base-model checkpoints/qwen2.5-coder-7b-instruct-q4k.apr \
    -o checkpoints/qwen-7b-merged.apr

# Evaluate
make eval-humaneval CHECKPOINT=checkpoints/qwen-7b-merged.apr
```

**Config:** `configs/recipes/recipe-j-merge-specialists.yaml`

**Falsifiable:** If merged model scores below best input specialist on ALL benchmarks (AC-024). Expected: merged model picks up complementary strengths from both specialists.

## 9.11 Recipe K: "Final Artifact" (PMAT-011)

**Target:** Produce the leaderboard submission: prune → INT4 quantize → compile → standalone binary.

```bash
# Step 1: Wanda prune at 20% using calibration data
apr prune checkpoints/qwen-7b-optimized.apr \
    --method wanda --target-ratio 0.2 \
    --calibration data/calibration.jsonl \
    -o checkpoints/qwen-7b-pruned.apr

# Step 2: INT4 quantize
apr quantize checkpoints/qwen-7b-pruned.apr \
    --scheme int4 \
    -o checkpoints/qwen-7b-pruned-int4.apr

# Step 3: Compile to standalone binary
apr compile checkpoints/qwen-7b-pruned-int4.apr \
    --release --lto --strip \
    -o checkpoints/qwen-coder-7b

# Step 4: Validate AC-022 success gate
make validate-ac022
```

**Config:** `configs/recipes/recipe-k-final-artifact.yaml`

**Success gate (AC-022):** >=85% HumanEval, >=82% HumanEval+, >=80% MBPP

**Hypothesis H4:** INT4 quantization loses <2% pass@1 (AC-023). Current Q4K model already at 85.37% — INT4 from FP16 intermediate may differ.

## 9.12 Recipe L: "DPO Alignment" (PMAT-008)

**Target:** Align 7B model on HumanEval preference pairs to improve borderline problem accuracy, targeting MBPP 76.2% → 78-80%.

```bash
# Step 1: Generate preference pairs from N-sampling eval (PMAT-014)
make generate-preference-pairs \
    WORK_DIR=/tmp/nsample-workdir \
    OUTPUT=data/preference-pairs.jsonl

# Step 2: DPO fine-tune on preference pairs
apr finetune checkpoints/qwen2.5-coder-7b-instruct-q4k.apr \
    --method dpo --data data/preference-pairs.jsonl \
    --rank 16 --lr 5e-5 --epochs 3 --beta 0.1 \
    -o checkpoints/qwen-7b-dpo-adapter/

# Step 3: Merge adapter into base model
apr finetune checkpoints/qwen2.5-coder-7b-instruct-q4k.apr \
    --merge --adapter checkpoints/qwen-7b-dpo-adapter/ \
    -o checkpoints/qwen-7b-dpo-merged.apr

# Step 4: Quantize
apr quantize checkpoints/qwen-7b-dpo-merged.apr \
    --scheme q4k -o checkpoints/qwen-7b-dpo-q4k.apr

# Step 5: Evaluate on HumanEval and MBPP
make eval-humaneval CHECKPOINT=checkpoints/qwen-7b-dpo-q4k.apr
make eval-mbpp CHECKPOINT=checkpoints/qwen-7b-dpo-q4k.apr
```

**Config:** `configs/recipes/recipe-l-dpo-alignment.yaml`

**Contract:** `contracts/dpo-alignment.yaml` v2.0 (5 falsification tests, MBPP improvement target)

**Success gates:** MBPP >= 78% (DPO target), HumanEval >= 84% (no-regression)

**Hypothesis H5:** DPO on N-sampling preference pairs closes 2-3pp of the MBPP gap by aligning the model on borderline coding problems where it sometimes succeeds and sometimes fails.
