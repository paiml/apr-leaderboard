# Technique Playbook

## 7.1 Knowledge Distillation

**Goal:** Transfer 32B teacher knowledge into a 7B student that scores within 5% of teacher on pass@1.

**apr command:** `apr distill`

| Strategy | When to Use | apr Flags |
|----------|-------------|-----------|
| Standard KL | Single teacher, simple transfer | `--strategy standard --temperature 3.0 --alpha 0.7` |
| Progressive | Curriculum learning, easy→hard examples | `--strategy progressive --temperature 2.0` |
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

**Why progressive:** In aprender, progressive distillation uses curriculum learning — training on progressively harder examples — not layer-by-layer MSE matching. This is critical because the 32B teacher and 7B student have different layer counts with no 1:1 correspondence. Curriculum learning lets the student first learn simple code patterns (variable assignment, basic loops) from the teacher's soft targets, then graduate to complex patterns (nested control flow, type inference). Standard KL trains on all difficulties simultaneously, overwhelming the smaller student.

**Expected gain:** +3-8% pass@1 over baseline student.

## 7.2 Model Merging

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

# Round 1: DARE merge Python + Rust (resolve task-vector interference)
apr merge python-expert.apr rust-expert.apr \
    --strategy dare \
    --drop-rate 0.3 \
    --base-model base.apr \
    -o round1.apr

# Round 2: TIES merge with TypeScript expert (resolve sign conflicts)
apr merge round1.apr ts-expert.apr \
    --strategy ties \
    --base-model base.apr \
    --density 0.2 \
    -o semifinal.apr

# Round 3: SLERP blend with base for stability (preserve weight norms)
apr merge semifinal.apr base.apr \
    --strategy slerp \
    --weights 0.85,0.15 \
    -o merged-final.apr
```

**Why DARE → TIES → SLERP cascade:** DARE first resolves task-vector interference between the two specialists at a conservative 30% drop rate (not 90% — high drop rates destroy blended knowledge). TIES then handles sign conflicts when adding the third specialist. SLERP finally smooths the merged result against the base model with mild interpolation (85/15) to preserve weight norms without diluting specialization.

**Expected gain:** +2-5% pass@1 over best individual specialist. Free compute — no GPU needed.

## 7.3 Pruning

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

**Pruning budget by model size (Wanda):**

| Model | Conservative | Moderate | Aggressive | Speed Gain (conservative) |
|-------|-------------|----------|------------|--------------------------|
| 1.5B | 20% | 30% | 40% | 1.2-1.3x |
| 7B | 20% | 25% | 35% | 1.2-1.4x |
| 32B | 15% | 20% | 30% | 1.1-1.3x |

**Expected impact:** Conservative ratio targets <1% pass@1 degradation. Moderate allows 1-3% degradation for meaningful speedup. Aggressive (>30% for small models) risks measurable quality loss — validate with eval before accepting. Smaller models have less redundancy; budget accordingly.

## 7.4 Fine-tuning (LoRA)

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

## 7.5 Fine-tuning (QLoRA)

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

**QLoRA vs LoRA tradeoff (at rank 16):**

| Aspect | LoRA (rank 16) | QLoRA (rank 16) | QLoRA (rank 32) |
|--------|---------------|-----------------|-----------------|
| VRAM (7B) | ~28GB | ~12GB | ~16GB |
| VRAM (32B) | ~80GB | ~24GB | ~32GB |
| Quality loss | None | Data-dependent | Data-dependent |
| Training speed | Fastest | ~20% slower | ~25% slower |

**VRAM depends on rank:** Higher LoRA rank = more adapter parameters = more memory for gradients and optimizer states. The numbers above assume batch size 1 with gradient accumulation; larger batch sizes increase VRAM proportionally.

**When to use QLoRA:** Always for 32B models. For 7B, use LoRA if you have 32GB+ VRAM. When targeting INT4 deployment, prefer QLoRA — it provides implicit quantization awareness.

## 7.6 Prompt Strategy (Zero-Cost Technique)

**Goal:** Maximize pass@1 without any model modification. Zero training cost, immediate results.

**eval command:** `make eval-humaneval PROMPT_STRATEGY=few-shot`

| Strategy | HumanEval 7B | MBPP 7B | When to Use |
|----------|-------------|---------|-------------|
| `few-shot` | **87.20%** (+1.83pp) | — | Best for HumanEval. Use simplest possible exemplar. |
| `standard` | 85.37% (baseline) | **76.20%** | Default. Sufficient for most benchmarks. |
| `scot` | 82.32% (-3.05pp) | — | Hurts ≤7B models. May help ≥32B. |
| `cgo` | *under evaluation* | — | Asks for helper functions. |

**Key findings from dogfooding (§22.21):**
1. **Simpler exemplars win.** Trivial `add(a,b)` (87.20%) > 3 concrete exemplars (85.98%). The exemplar's purpose is format priming, not domain knowledge.
2. **SCoT hurts small models.** Reasoning overhead consumes tokens and degrades output quality on 7B. Reserve for 32B+ where reasoning is more concise.
3. **MBPP needs test assertions.** Including `test_list` assertions in the prompt = +25.4pp (50.80% → 76.20%). Without them, the model guesses function signatures wrong.
4. **Benchmark-specific prompting matters.** HumanEval provides function signatures (model completes); MBPP provides prose (model writes from scratch). Different benchmarks need different prompt structures.

**Leaderboard recipe:** Use `few-shot` for HumanEval, `standard` with test assertions for MBPP. This costs zero compute and yields the highest known apr-native scores.

## 7.8 Quantization (Post-Training)

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

## 7.9 Hyperparameter Optimization (HPO)

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
