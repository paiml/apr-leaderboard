# Leaderboard-Winning Techniques

The techniques in §7 optimize the *model*. This section covers techniques that optimize *inference-time behavior* — how you extract the best score from a given model. These are the techniques that separate top-10 leaderboard entries from median ones.

## 1. Sampling Strategy Tuning

**Why it matters:** The difference between greedy decoding and tuned sampling can be 5-15% pass@1. Most leaderboards evaluate pass@1 with greedy decoding, but the sampling parameters used during generation dramatically affect output quality.

**apr command:** `apr run`, `apr chat`, `apr eval`

```bash
# Greedy (temperature=0, deterministic — standard for leaderboard eval)
apr eval model.apr --task classify --data humaneval.jsonl \
    --temperature 0.0 --json

# Tuned nucleus sampling (better for diverse code generation)
apr eval model.apr --task classify --data humaneval.jsonl \
    --temperature 0.2 --top_p 0.95 --json

# High-temperature diverse sampling for pass@k (k>1)
apr eval model.apr --task classify --data humaneval.jsonl \
    --temperature 0.8 --top_p 0.95 --json
```

**Leaderboard sweet spots:**

| Metric | Temperature | Top-P | Rationale |
|--------|-------------|-------|-----------|
| pass@1 | 0.0 (greedy) | 1.0 | Deterministic, reproducible |
| pass@1 (tuned) | 0.1-0.2 | 0.95 | Slight diversity avoids greedy traps |
| pass@10 | 0.6-0.8 | 0.95 | Diversity yields more distinct solutions |
| pass@100 | 0.8-1.0 | 0.95 | Maximum diversity |

## 2. N-Sampling with Best-of-N Selection (pass@k Maximization)

**Why it matters:** Generating N completions and selecting the best one (via self-consistency, test execution, or log-probability scoring) can boost effective pass@1 by 10-30% over single-shot generation. This is the single most impactful inference-time technique [8].

**apr command:** `apr eval --n-samples`

```bash
# Generate 20 completions per problem, compute pass@1 and pass@10
apr eval model.apr --task classify --data humaneval.jsonl \
    --n-samples 20 --temperature 0.8 --json

# Best-of-N with log-probability reranking
apr eval model.apr --task classify --data humaneval.jsonl \
    --n-samples 10 --rerank logprob --json

# Best-of-N with self-consistency (majority voting on output)
apr eval model.apr --task classify --data humaneval.jsonl \
    --n-samples 10 --rerank majority --json
```

**Implementation status:** `--n-samples` and `--rerank` flags need to be added to `apr eval`. The generation engine supports temperature/top-p/top-k sampling. Best-of-N requires: (1) batched generation of N completions, (2) a reranking strategy (log-prob, majority vote, or test execution).

**Expected gain:** +10-30% effective pass@1 with N=10-50 over single-shot greedy.

## 3. Structured Prompting (System Prompt + Few-Shot + SCoT)

**Why it matters:** Structured Chain-of-Thought (SCoT) prompting improves HumanEval pass@1 by up to 13.79% over vanilla prompting by asking the model to reason through sequential, branch, and loop structures before generating code [9].

**apr command:** `apr eval --prompt-strategy`, `apr chat --system`

```bash
# Standard prompt (baseline)
apr eval model.apr --task classify --data humaneval.jsonl \
    --prompt-strategy standard --json

# Structured Chain-of-Thought prompting
apr eval model.apr --task classify --data humaneval.jsonl \
    --prompt-strategy scot --json

# Few-shot with curated exemplars
apr eval model.apr --task classify --data humaneval.jsonl \
    --prompt-strategy few-shot --exemplars exemplars.jsonl --json

# Custom system prompt for code generation
apr eval model.apr --task classify --data humaneval.jsonl \
    --system "You are an expert Python programmer. Think step by step." --json
```

**Prompt strategies to implement:**

| Strategy | Description | Expected Impact |
|----------|-------------|-----------------|
| `standard` | Raw problem → code | Baseline |
| `scot` | Problem → structured reasoning → code | +5-14% pass@1 |
| `few-shot` | N exemplars + problem → code | +3-8% pass@1 |
| `cgo` | Chain of Grounded Objectives — goal-oriented decomposition | +5-10% pass@1 |

**Implementation status:** `--system` flag exists. `--prompt-strategy` and `--exemplars` need to be added.

## 4. Speculative Decoding (Inference Speedup)

**Why it matters:** Speculative decoding yields 2-3x faster inference on code models, which means more attempts within a time budget and faster evaluation iteration. Code is particularly amenable to speculation because syntax is predictable.

**apr command:** `apr run --speculative`, `apr cbtop --speculative`

```bash
# Self-speculative decoding (model as its own draft)
apr run model.apr --speculative --speculation-k 4 "def fibonacci(n):"

# Draft model speculative decoding (faster, slightly less accurate)
apr run model.apr --speculative --draft-model-path draft.apr --speculation-k 6 \
    "def fibonacci(n):"

# Benchmark speculative vs standard throughput
apr bench model.apr --speculative --speculation-k 4 --json
```

**Implementation status:** Speculative decoding EXISTS in aprender (`generate_speculative_with_draft`, `generate_speculative_cuda`). CLI flags `--speculative`, `--speculation-k`, `--draft-model-path` are available.

**Expected gain:** 2-3x throughput improvement for code generation tasks. No quality change (output distribution is mathematically identical).

## 5. Preference Optimization (DPO/ORPO)

**Why it matters:** DPO and ORPO align models to prefer correct, well-structured code over plausible but buggy code. ORPO eliminates the need for a reference model, making it simpler than RLHF. Models trained with preference optimization consistently score 3-8% higher on code benchmarks than SFT-only models [10][11].

**apr command:** `apr align` (proposed)

```bash
# Generate preference pairs from eval results
# (correct completions = chosen, incorrect = rejected)
apr eval model.apr --task classify --data humaneval.jsonl \
    --n-samples 20 --export-pairs preference-pairs.jsonl

# DPO alignment (requires reference model)
apr align model.apr \
    --method dpo \
    --data preference-pairs.jsonl \
    --beta 0.1 \
    --ref-model base.apr \
    -o aligned.apr

# ORPO alignment (no reference model needed, simpler)
apr align model.apr \
    --method orpo \
    --data preference-pairs.jsonl \
    --lambda 0.1 \
    -o aligned.apr
```

**Implementation status:** NOT YET IMPLEMENTED. DPO requires: (1) paired preference data, (2) reference model log-probs, (3) DPO loss function. ORPO is simpler — single model, odds ratio penalty on rejected responses. Both build on the existing LoRA training infrastructure in entrenar.

**Expected gain:** +3-8% pass@1 over SFT-only models.

## 6. Continued Pretraining (Domain Adaptation)

**Why it matters:** Continued pretraining on a large code corpus before instruction fine-tuning lets the model absorb domain-specific patterns (API usage, idioms, error handling) that instruction tuning alone can't teach. This is how CodeLlama was built from Llama 2 [12].

**apr command:** `apr finetune --method full`

```bash
# Continued pretraining on code corpus (full fine-tuning, not LoRA)
apr finetune model.apr \
    --method full \
    --data code-corpus-500k.jsonl \
    --epochs 1 \
    --learning-rate 5e-5 \
    --json \
    -o domain-adapted.apr

# Then LoRA instruction-tune on top
apr finetune domain-adapted.apr \
    --method lora \
    --rank 16 \
    --data code-instruct-50k.jsonl \
    --epochs 3 \
    -o final-lora/
```

**Implementation status:** `--method full` EXISTS in aprender's finetune command. The training loop in entrenar supports full-model gradient computation.

**Key consideration:** Continued pretraining requires significant compute (full model gradients, not just adapter). Budget accordingly.

## 7. Data Decontamination

**Why it matters:** If training data overlaps with benchmark test cases, scores are inflated and meaningless. Leaderboards actively detect and penalize contaminated submissions. Data decontamination is a hard requirement, not optional.

**apr command:** `apr validate --decontaminate` (proposed)

```bash
# Check training data for benchmark overlap
apr validate --data code-instruct.jsonl \
    --decontaminate \
    --benchmarks humaneval,mbpp,bigcodebench \
    --threshold 0.8 \
    --json

# Generate clean training set (remove overlapping samples)
apr validate --data code-instruct.jsonl \
    --decontaminate \
    --benchmarks humaneval,mbpp \
    --output clean-instruct.jsonl
```

**Implementation status:** NOT YET IMPLEMENTED. Requires: (1) benchmark dataset fingerprinting, (2) n-gram overlap detection against training data, (3) semantic similarity filtering for paraphrased problems.

**Falsification gate (AC-016):** Any submission MUST demonstrate <1% n-gram overlap between training data and evaluation benchmarks.

## 8. Test-Time Compute Scaling

**Why it matters:** Recent results show that spending more compute at inference time (generating more candidates, longer chain-of-thought, iterative refinement) scales performance more efficiently than model size for code tasks. This is the "scaling at test time" paradigm.

**apr command:** Composition of existing commands

```bash
# Strategy: Generate many → Execute → Filter → Rerank
# Step 1: Generate 50 diverse completions per problem
apr eval model.apr --task classify --data humaneval.jsonl \
    --n-samples 50 --temperature 0.8 --json > candidates.json

# Step 2: Execute all candidates in sandbox (EXTERNAL)
# → produces pass/fail per candidate

# Step 3: Among passing candidates, select by log-probability
# → highest log-prob passing candidate = submission

# Step 4: For failing problems, retry with SCoT prompting
apr eval model.apr --task classify --data failing-problems.jsonl \
    --n-samples 50 --prompt-strategy scot --temperature 0.6 --json
```

**Expected gain:** Diminishing returns, but N=50 with test-based filtering can reach pass@1 equivalent of pass@50, which is typically 15-25% higher than greedy pass@1.

## 9. Technique Stacking: The Winning Formula

Leaderboard winners stack techniques multiplicatively. The winning formula, in priority order:

```
1. Best base model selection (Qwen2.5-Coder-7B-Instruct)     — biggest impact
2. Continued pretraining on code corpus                        — +5-10%
3. Distillation from 32B teacher                               — +3-8%
4. LoRA/QLoRA instruction fine-tuning                          — +5-15%
5. DPO/ORPO preference alignment                               — +3-8%
6. Merge tournament with specialist variants                   — +2-5%
7. Structured prompting (SCoT)                                 — +5-14%
8. N-sampling with test-based reranking                        — +10-30% effective
9. Pruning + quantization for inference speed                  — neutral quality, faster
```

**Not all gains stack linearly.** Steps 2-4 compound well. Steps 5-6 have diminishing returns if 2-4 are strong. Steps 7-8 are inference-time and always apply on top of model-time gains.

**The full apr recipe:**

```bash
#!/bin/bash
set -euo pipefail

# === Model Optimization (one-time) ===
apr import hf://Qwen/Qwen2.5-Coder-32B -o teacher.apr
apr import hf://Qwen/Qwen2.5-Coder-7B -o base.apr

apr finetune base.apr --method full --data code-corpus-500k.jsonl --epochs 1 -o adapted.apr
apr distill teacher.apr --student adapted.apr --strategy progressive -o distilled.apr
apr finetune distilled.apr --method lora --rank 32 --data code-instruct-50k.jsonl -o lora/
apr finetune distilled.apr --adapter lora/ --merge -o finetuned.apr
# apr align finetuned.apr --method orpo --data preference-pairs.jsonl -o aligned.apr  # when implemented
apr merge finetuned.apr variant-b.apr --strategy ties --base-model distilled.apr -o merged.apr
apr prune merged.apr --method wanda --target-ratio 0.2 --calibration calib.jsonl -o pruned.apr
apr quantize pruned.apr --scheme int4 -o final.apr

# === Inference-Time Optimization (per evaluation) ===
apr eval final.apr --task classify --data humaneval.jsonl \
    --n-samples 50 --temperature 0.8 --prompt-strategy scot --json
```
