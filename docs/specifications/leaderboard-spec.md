# APR Leaderboard Specification

**Status:** DRAFT
**Date:** 2026-02-28
**Authors:** APR Team
**Oracle:** `batuta oracle` — entrenar (90% confidence for LoRA), realizar (85% for serving), trueno (80% for compute)

## 1. Thesis

The Python ML ecosystem requires 200+ dependencies, GPU-locked CUDA toolchains, and multi-GB Docker images to compete on HuggingFace leaderboards. We will demonstrate that a single-binary Rust pipeline — using only `apr` CLI commands — can match or exceed these results with zero Python, zero CUDA driver dependency, and 10x smaller deployment artifacts.

**Constraint:** Every optimization step must be expressible as an `apr` subcommand. No Python. No notebooks. No HuggingFace Transformers library. Pure sovereign stack.

## 2. Target Leaderboards

| Leaderboard | Primary Metric | Benchmarks | Why |
|-------------|---------------|------------|-----|
| BigCode Models | pass@1 | HumanEval, MBPP, MultiPL-E | Code generation is our strongest use case |
| EvalPlus | pass@1 | HumanEval+, MBPP+ | Rigorous test suites expose real quality |
| Open LLM Leaderboard v2 | aggregate | MMLU, ARC, HellaSwag, etc. | Broad visibility |

**Primary target:** Qwen2.5-Coder family (1.5B, 7B, 32B) — best open code models with permissive Apache-2.0 license.

## 3. The `apr` CLI Toolchain

Every technique maps to a single shell command. This is the differentiator — our competitors use 500-line Python scripts; we use one-liners.

### 3.1 Import (HF → APR)

```bash
# Import from HuggingFace Hub — auto-detects architecture
apr import hf://Qwen/Qwen2.5-Coder-7B -o qwen-7b.apr --arch qwen2

# Import with quantization on ingest
apr import hf://Qwen/Qwen2.5-Coder-32B -o qwen-32b-q8.apr --quantize int8

# Import GGUF with provenance enforcement
apr import qwen-7b.gguf -o qwen-7b.apr --enforce-provenance
```

### 3.2 Evaluate (Baseline)

```bash
# Perplexity baseline
apr eval qwen-7b.apr --dataset wikitext-2 --threshold 20.0

# Classification eval with custom data
apr eval qwen-7b.apr --task classify --data humaneval.jsonl --json
```

### 3.3 Full Optimization Pipeline (preview)

```bash
# The complete leaderboard recipe in 6 commands:
apr import hf://Qwen/Qwen2.5-Coder-7B -o base.apr
apr distill base.apr --student student.apr --strategy progressive --temperature 3.0
apr finetune student.apr --method qlora --rank 32 --data code-instruct.jsonl -o tuned.apr
apr prune tuned.apr --method wanda --target-ratio 0.3 --calibration calib.jsonl -o pruned.apr
apr merge pruned.apr variant-b.apr --strategy slerp -o final.apr
apr quantize final.apr --scheme int4 -o submit.apr
```

## 4. Technique Playbook

### 4.1 Knowledge Distillation

**Goal:** Transfer 32B teacher knowledge into a 7B student that scores within 5% of teacher on pass@1.

**apr command:** `apr distill`

| Strategy | When to Use | apr Flags |
|----------|-------------|-----------|
| Standard KL | Single teacher, simple transfer | `--strategy standard --temperature 3.0 --alpha 0.7` |
| Progressive | Layer-by-layer matching for deep models | `--strategy progressive --temperature 2.0` |
| Ensemble | Multiple teacher variants | `--strategy ensemble --temperature 4.0` |

**Leaderboard Recipe:**

```bash
# Step 1: Import teacher (32B) and student (7B)
apr import hf://Qwen/Qwen2.5-Coder-32B -o teacher-32b.apr
apr import hf://Qwen/Qwen2.5-Coder-7B -o student-7b.apr

# Step 2: Distill with progressive strategy (best for code)
apr distill teacher-32b.apr \
    --student student-7b.apr \
    --strategy progressive \
    --temperature 3.0 \
    --alpha 0.7 \
    --epochs 5 \
    --data code-corpus.jsonl \
    -o distilled-7b.apr

# Step 3: Evaluate improvement
apr eval distilled-7b.apr --task classify --data humaneval.jsonl --json
```

**Why progressive:** Code models have deep attention patterns where layer-by-layer MSE loss preserves syntactic structure better than logit-only KL divergence. The 32B Qwen-Coder's deeper attention heads encode bracket-matching, indentation-awareness, and type-inference patterns that standard KL loses.

**Expected gain:** +3-8% pass@1 over baseline student.

### 4.2 Model Merging

**Goal:** Combine fine-tuned variants to get best-of-all-worlds without additional training.

**apr command:** `apr merge`

| Strategy | Mechanism | Best For |
|----------|-----------|----------|
| `average` | Arithmetic mean of weights | Quick baseline, similar models |
| `weighted` | `--weights 0.7,0.3` | Known-better model dominates |
| `slerp` | Spherical interpolation | Smooth blending, preserves magnitude |
| `ties` | Trim, Elect Sign, merge (sparse) | Resolving conflicting task vectors |
| `dare` | Drop And REscale random weights | Preventing catastrophic interference |

**Leaderboard Recipe — The "Merge Tournament":**

```bash
# Train 3 specialists on different code domains
apr finetune base.apr --method lora --data python-instruct.jsonl -o python-expert.apr
apr finetune base.apr --method lora --data rust-instruct.jsonl -o rust-expert.apr
apr finetune base.apr --method lora --data typescript-instruct.jsonl -o ts-expert.apr

# Round 1: TIES merge the Python + Rust experts
apr merge python-expert.apr rust-expert.apr \
    --strategy ties \
    --base-model base.apr \
    --density 0.2 \
    -o round1.apr

# Round 2: SLERP blend with TypeScript expert
apr merge round1.apr ts-expert.apr \
    --strategy slerp \
    -o semifinal.apr

# Round 3: DARE to prevent interference, keep 10% random weights
apr merge semifinal.apr base.apr \
    --strategy dare \
    --drop-rate 0.9 \
    --base-model base.apr \
    -o merged-final.apr
```

**Why TIES + SLERP + DARE cascade:** Each merge strategy has failure modes. TIES handles sign conflicts but can over-prune. SLERP preserves weight norms but doesn't resolve conflicts. DARE prevents interference through random masking. The cascade uses each where it's strongest.

**Expected gain:** +2-5% pass@1 over best individual specialist. Free compute — no GPU needed.

### 4.3 Pruning

**Goal:** Remove 20-50% of weights with <2% quality loss, yielding faster inference for benchmarks.

**apr command:** `apr prune`

| Method | Mechanism | Quality Preservation |
|--------|-----------|---------------------|
| `magnitude` | Remove smallest weights | Baseline, simple |
| `structured` | Remove entire attention heads/FFN dims | Fastest inference speedup |
| `depth` | Remove entire layers | Dramatic size reduction |
| `width` | Reduce hidden dimensions | Balanced size/quality |
| `wanda` | Weights AND Activations (calibration-based) | Best quality at high sparsity |
| `sparsegpt` | One-shot, column-by-column | Gold standard, needs calibration |

**Leaderboard Recipe — Wanda Pruning:**

```bash
# Step 1: Generate calibration data from code corpus
# (128 samples of representative code)

# Step 2: Analyze pruning opportunities first
apr prune model.apr --analyze --verbose

# Step 3: Wanda prune at 30% sparsity (sweet spot for code models)
apr prune model.apr \
    --method wanda \
    --target-ratio 0.3 \
    --calibration calibration-code.jsonl \
    -o pruned-30.apr

# Step 4: Verify quality didn't degrade
apr eval pruned-30.apr --dataset wikitext-2 --threshold 22.0
```

**Why Wanda over magnitude:** Magnitude pruning treats all weights equally. Wanda scores weights by `|weight| * ||activation||`, preserving weights on high-activation paths. For code models, the attention heads responsible for bracket-matching and indentation have high activations — Wanda preserves them.

**Pruning budget by model size:**

| Model | Safe Ratio | Aggressive Ratio | Speed Gain |
|-------|-----------|-------------------|------------|
| 1.5B | 40% | 50% | 1.5-1.8x |
| 7B | 30% | 40% | 1.3-1.6x |
| 32B | 20% | 30% | 1.2-1.4x |

**Expected impact:** Neutral on pass@1 at safe ratio, but 1.3-1.8x faster inference = more attempts within time budget.

### 4.4 Fine-tuning (LoRA)

**Goal:** Adapt base model to code-specific instruction-following with minimal compute.

**apr command:** `apr finetune`

```bash
# Auto-select method based on available VRAM
apr finetune qwen-7b.apr --method auto --vram 24 --plan

# LoRA fine-tune (rank 16, good default for code)
apr finetune qwen-7b.apr \
    --method lora \
    --rank 16 \
    --data code-instruct-50k.jsonl \
    --epochs 3 \
    --learning-rate 2e-4 \
    -o qwen-7b-lora/

# Merge adapter back into base
apr finetune qwen-7b.apr \
    --adapter qwen-7b-lora/ \
    --merge \
    -o qwen-7b-finetuned.apr
```

**Key parameters for leaderboard performance:**

| Parameter | Code Models | General Models |
|-----------|------------|----------------|
| Rank | 16-32 | 8-16 |
| Alpha | 2x rank | 2x rank |
| LR | 1e-4 to 3e-4 | 1e-4 to 2e-4 |
| Epochs | 3-5 | 2-3 |
| Target modules | q_proj, v_proj | q_proj, v_proj |

**Expected gain:** +5-15% pass@1 with curated instruction data.

### 4.5 Fine-tuning (QLoRA)

**Goal:** Same as LoRA but on consumer GPUs (8-16GB VRAM).

**apr command:** `apr finetune --method qlora`

```bash
# Plan QLoRA configuration for 16GB VRAM
apr tune qwen-7b.apr --method qlora --vram 16 --plan

# QLoRA fine-tune (quantized base, full-precision adapters)
apr finetune qwen-7b.apr \
    --method qlora \
    --rank 32 \
    --vram 16 \
    --data code-instruct-50k.jsonl \
    --epochs 3 \
    --learning-rate 2e-4 \
    -o qwen-7b-qlora/

# Merge adapter
apr finetune qwen-7b.apr \
    --adapter qwen-7b-qlora/ \
    --merge \
    -o qwen-7b-qlora-merged.apr
```

**QLoRA vs LoRA tradeoff:**

| Aspect | LoRA | QLoRA |
|--------|------|-------|
| VRAM (7B) | ~28GB | ~12GB |
| VRAM (32B) | ~80GB | ~24GB |
| Quality loss | None | ~0.5% pass@1 |
| Training speed | Faster | ~20% slower |

**When to use QLoRA:** Always for 32B models. For 7B, use LoRA if you have 32GB+ VRAM.

### 4.6 Quantization (Post-Training)

**Goal:** Reduce model size for faster inference with minimal quality loss.

**apr command:** `apr quantize`

```bash
# Plan quantization impact
apr quantize model.apr --scheme int4 --plan

# Quantize to INT4 (best size/quality for leaderboard)
apr quantize model.apr --scheme int4 -o model-q4.apr

# Batch quantize to compare schemes
apr quantize model.apr --batch int8,int4,fp16,q4k

# Quantize with format conversion for submission
apr quantize model.apr --scheme int4 --format gguf -o model.gguf
```

### 4.7 Hyperparameter Optimization (HPO)

**Goal:** Find optimal LoRA/QLoRA hyperparameters automatically.

**apr command:** `apr tune`

```bash
# Scout phase: 1-epoch trials to narrow search space
apr tune qwen-7b.apr \
    --task classify \
    --data code-instruct-50k.jsonl \
    --budget 20 \
    --strategy tpe \
    --scheduler asha \
    --scout \
    --json

# Full HPO: warm-start from scout results
apr tune qwen-7b.apr \
    --task classify \
    --data code-instruct-50k.jsonl \
    --budget 10 \
    --from-scout scout-results/ \
    --max-epochs 20 \
    --time-limit 8h
```

## 5. Composite Recipes

### 5.1 Recipe A: "The Distilled Expert" (Maximum Quality)

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

**Expected:** 70-78% pass@1 on HumanEval (vs ~65% baseline 7B).

### 5.2 Recipe B: "The Merge Alchemist" (Zero Training Compute)

**Target:** Best score achievable with NO GPU training at all. Pure weight manipulation.

```bash
# 1. Import multiple Qwen-Coder variants
apr import hf://Qwen/Qwen2.5-Coder-7B -o base.apr
apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct -o instruct.apr

# 2. SLERP merge base + instruct (t=0.6 favoring instruct)
apr merge base.apr instruct.apr \
    --strategy slerp \
    --weights 0.4,0.6 \
    -o slerp-blend.apr

# 3. Prune: remove redundant attention heads (structured)
apr prune slerp-blend.apr \
    --method structured \
    --target-ratio 0.15 \
    -o slerp-pruned.apr

# 4. Quantize for fast inference
apr quantize slerp-pruned.apr --scheme q4k -o slerp-q4k.apr

# 5. Eval
apr eval slerp-q4k.apr --task classify --data humaneval.jsonl --json
```

**Expected:** 62-68% pass@1 on HumanEval — competitive with no training.

### 5.3 Recipe C: "The Full Pipeline" (Kitchen Sink)

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

**Expected:** 73-82% pass@1 on HumanEval.

### 5.4 Recipe D: "Sovereign Binary" (The Differentiator)

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

# Result: single static binary, ~500MB, runs on any Linux
./qwen-coder "def fibonacci(n):"
```

**This is the marketing win:** While competitors need `pip install transformers torch accelerate bitsandbytes`, we ship `./qwen-coder`.

## 6. Technique Interaction Matrix

Techniques are not independent. Order matters.

```
                      ┌─────────────────────────────────────────────┐
                      │          TECHNIQUE INTERACTION MATRIX       │
                      │                                             │
                      │  Before │ distill  merge  prune  finetune  │
                      │  After  │                                   │
                      │─────────┼────────────────────────────────── │
                      │ distill │   —      ✗bad   ✓ok    ✓✓best   │
                      │ merge   │  ✓ok      —     ✓ok    ✓✓best   │
                      │ prune   │  ✓ok     ✓ok     —     ✗bad     │
                      │finetune │ ✓✓best  ✓✓best  ✗bad    —       │
                      │quantize │  ✓ok     ✓ok    ✓ok    ✓ok      │
                      └─────────────────────────────────────────────┘

  Legend: Read as "row AFTER column"
    ✓✓best  = Optimal ordering (do column first, then row)
    ✓ok     = Works but not optimal
    ✗bad    = Harmful (degrades quality or wastes compute)
```

**Golden ordering:** distill → finetune → merge → prune → quantize

Rationale:
1. **Distill first** — Knowledge transfer works best on an unmodified student architecture
2. **Finetune second** — LoRA adapts the distilled weights to target benchmarks
3. **Merge third** — Combine fine-tuned variants while representations are still rich
4. **Prune fourth** — Remove redundancy AFTER merging (merged models have more redundancy)
5. **Quantize last** — Always final step; quantization is lossy and non-reversible

**Anti-patterns:**
- Prune → Finetune: LoRA can't recover pruned knowledge effectively
- Finetune → Distill: Overwrites the fine-tuned specialization
- Quantize → anything: Quality loss compounds with every subsequent operation

## 7. Competitive Advantage: Why `apr` Wins

| Aspect | Python Ecosystem | `apr` CLI |
|--------|-----------------|-----------|
| Dependencies | transformers, torch, accelerate, bitsandbytes, peft, trl, vllm | Single binary |
| Setup time | 30-60 min (CUDA, conda, pip conflicts) | 0 min (`cargo install apr-cli`) |
| Merge | 50-line Python script | `apr merge --strategy slerp` |
| Prune | 100+ lines, custom hooks | `apr prune --method wanda` |
| LoRA | peft + trl + custom training loop | `apr finetune --method lora` |
| Distill | Custom training loop, 200+ lines | `apr distill --strategy progressive` |
| Quantize | bitsandbytes or GPTQ, GPU required | `apr quantize --scheme int4` |
| Reproducibility | requirements.txt + CUDA version + random seeds | Deterministic Rust binary |
| Deployment | Docker + CUDA runtime + Python | `apr compile → single binary` |
| CI/CD | Complex, flaky GPU runners | `cargo test` on any machine |
| Auditability | Opaque Python state | `apr check` — 10-stage integrity pipeline |

## 8. Data Strategy

The model is only as good as the fine-tuning data. Key datasets for code leaderboards:

| Dataset | Size | Purpose | Source |
|---------|------|---------|--------|
| Code Instruct (curated) | 50K | Instruction-following for code | Self-curated from OSS repos |
| Code Reasoning | 20K | Chain-of-thought for complex problems | Synthetic from teacher model |
| Code Tests | 10K | Test-driven examples (input→test→code) | HumanEval/MBPP-style |
| Multilingual Code | 30K | Python/Rust/TS/Go/Java coverage | MultiPL-E format |
| Calibration | 128 | Wanda/SparseGPT calibration | Random code samples |

**Data preparation via apr CLI:**

```bash
# Generate synthetic training data from teacher
apr chat teacher.apr --system "Generate code instruction pairs" \
    --batch instructions.txt --json > code-instruct.jsonl

# Format validation
apr validate --data code-instruct.jsonl --format jsonl
```

## 9. Evaluation Protocol

Every recipe must be evaluated identically for fair comparison.

```bash
# Standard eval battery (run for every candidate model)
apr eval model.apr --task classify --data humaneval.jsonl --json > results/humaneval.json
apr eval model.apr --task classify --data mbpp.jsonl --json > results/mbpp.json
apr eval model.apr --dataset wikitext-2 --json > results/perplexity.json
apr bench model.apr --json > results/throughput.json

# Compare against HuggingFace reference
apr compare-hf model.apr --json > results/parity.json

# Full QA gate before submission
apr qa model.apr --verbose
apr check model.apr
```

## 10. Submission Flow

```bash
# 1. Generate HuggingFace model card
apr eval final.apr --generate-card

# 2. Export to HuggingFace-compatible format
apr export final.apr --format safetensors -o submission/

# 3. Publish to HuggingFace Hub
apr publish submission/ --repo paiml/qwen-coder-7b-apr --private

# 4. Submit to leaderboard (via HF evaluation queue)
# The leaderboard pulls from your HF repo and runs evaluation
```

## 11. Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| HumanEval pass@1 | 70% (7B) | 78% (7B) |
| MBPP pass@1 | 65% (7B) | 72% (7B) |
| Pipeline commands | <= 10 | <= 6 |
| Total binary size (compiled) | < 4GB | < 2GB |
| Wall-clock (import → submit) | < 24h | < 8h |
| Python dependencies | 0 | 0 |
| CUDA requirement | Optional | None |

## 12. Open Questions

1. **Calibration data quality:** How much does Wanda calibration data selection affect code model pruning? Need ablation study.
2. **Merge tournament depth:** Is 2-round merging sufficient or do 3+ rounds compound gains?
3. **Distillation data volume:** What's the minimum code corpus size for progressive distillation to outperform standard KL?
4. **HPO budget:** Is 20-trial TPE scout sufficient to find good LoRA hyperparameters for code?
5. **Quantization floor:** At what pass@1 threshold does INT4 quantization degrade code generation quality measurably?
6. **Cross-architecture distillation:** Can we distill Qwen-32B into a different architecture (e.g., smaller custom model)?
