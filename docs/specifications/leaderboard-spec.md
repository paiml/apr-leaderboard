---
title: "APR Leaderboard Competition"
version: "1.0.0"
status: "Draft"
created: "2026-02-28"
updated: "2026-02-28"
issue_refs: ["GH-001"]
epic: "APR-LEADERBOARD"
---

# APR Leaderboard Specification

**Status:** DRAFT
**Date:** 2026-02-28
**Authors:** APR Team
**Oracle:** `batuta oracle` — entrenar (90% confidence for LoRA), realizar (85% for serving), trueno (80% for compute)

## 1. What This Repo Does

### 1.1 Purpose

**apr-leaderboard** is a pipeline harness that proves the [sovereign AI stack](https://github.com/paiml) — aprender, entrenar, trueno — can compete on HuggingFace code generation leaderboards (HumanEval, MBPP, BigCodeBench) without Python, without the HuggingFace Transformers library, and without an external CUDA toolkit.

It is **not** a model training framework. It is **not** a general ML toolkit. It is a thin orchestration layer (~1,400 lines of Rust) that wires the sovereign stack's existing capabilities into a reproducible, config-driven leaderboard pipeline:

```
apr import → apr distill → apr finetune → apr merge → apr prune → apr quantize → apr eval → apr submit
```

Every command above is provided by **aprender** (`apr` CLI). This repo provides the pipeline config, benchmark metadata, result persistence, and the spec that defines the strategy.

### 1.2 What It Proves

This repo exists to answer one falsifiable question:

> **Can a single Rust binary (`apr`) match Python-ecosystem HumanEval/MBPP scores for Qwen2.5-Coder-7B, with zero Python dependencies?**

If the answer is yes, it proves:
1. **aprender** can import, infer, and evaluate HuggingFace models via the `.apr` format
2. **entrenar** can fine-tune those models with LoRA/QLoRA using its own autograd engine
3. **trueno** can run transformer attention at competitive throughput via SIMD/PTX
4. The full distill → finetune → merge → prune → quantize pipeline works end-to-end in pure Rust
5. **provable-contracts** kernel verification (Kani bounded model checking) doesn't prevent competitive performance — correctness and speed coexist

If the answer is no, it identifies exactly where the sovereign stack falls short (inference parity gap, training convergence, quantization quality loss) via `apr compare-hf`.

### 1.3 How It Relates to aprender

```
┌──────────────────────────────────────────────────────────┐
│                    apr-leaderboard                        │
│                                                          │
│  Pipeline configs    Benchmark metadata    Result JSON    │
│  (TOML)              (10 benchmarks)       persistence   │
│                                                          │
│  ┌──────────────── calls ─────────────────────────────┐  │
│  │                                                     │  │
│  ▼                                                     │  │
│  ┌──────────────────────────────────────────────────┐  │  │
│  │              aprender (apr CLI)                   │  │  │
│  │                                                   │  │  │
│  │  import   distill   finetune   merge   prune     │  │  │
│  │  quantize  eval    bench    compile   chat       │  │  │
│  │  compare-hf  qa    check   publish   export      │  │  │
│  │                                                   │  │  │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐         │  │  │
│  │  │ entrenar│  │  trueno   │  │provable │         │  │  │
│  │  │ LoRA    │  │  SIMD/PTX │  │contracts│         │  │  │
│  │  │ QLoRA   │  │  AVX2     │  │ Kani    │         │  │  │
│  │  │ AdamW   │  │  wgpu GPU │  │ L1-L4   │         │  │  │
│  │  │ autograd│  │  Q4K/Q6K  │  │ proofs  │         │  │  │
│  │  └─────────┘  └──────────┘  └─────────┘         │  │  │
│  └──────────────────────────────────────────────────┘  │  │
│                                                        │  │
└────────────────────────────────────────────────────────┘  │
                                                            │
   pmat comply ◄───── quality gate ─────────────────────────┘
```

**apr-leaderboard does NOT reimplement aprender.** It calls `apr` subcommands. The relationship is:

| Layer | Repo | Responsibility |
|---|---|---|
| **Orchestration** | apr-leaderboard | Pipeline config, benchmark metadata, result tracking, strategy spec |
| **ML Operations** | aprender (apr CLI) | Model import, inference, eval, distillation, merging, pruning, quantization |
| **Training** | entrenar | LoRA/QLoRA, autograd, optimizers, gradient checkpointing |
| **Compute** | trueno | SIMD tensor ops, GPU kernels, quantized matmul |
| **Correctness** | provable-contracts | Kernel contracts, Kani proofs, falsification tests |
| **Quality** | pmat comply | Compliance checks, spec scoring, cross-crate consistency |

### 1.4 Current Implementation Status

The repo is a **working scaffold** — the pipeline structure is complete, but the backends call placeholder implementations:

| Module | What works today | What's scaffolded |
|---|---|---|
| **convert/** | APR v2 file creation with correct magic bytes and compression | Real SafeTensors download and tensor conversion (writes placeholder weights) |
| **eval/** | Result JSON persistence, history viewer, benchmark lookup | Actual inference and metrics (returns zeros) |
| **harness/** | All 10 benchmark definitions with metadata | Dataset loading, problem parsing |
| **finetune/** | LoRA config construction, AdamW parameter setup | Actual training loop (prints loss=0.000) |
| **submit/** | Submission JSON formatting, model_id validation | HF Hub API push (tells user to use huggingface-cli) |
| **pipeline** | Full orchestration: convert → finetune → eval → submit | Each step calls scaffolded backends |

**To reach production:** Replace each scaffold with the corresponding `apr` CLI call. The `apr` binary already implements all required operations — this repo needs to shell out to `apr import`, `apr eval`, etc. rather than reimplementing them inline.

### 1.5 How People Use It

**For leaderboard competitors:**

```bash
# 1. Install
cargo install --path .

# 2. Configure your model and benchmarks
cp configs/qwen-coder-7b.toml configs/my-model.toml
# Edit: model_id, quantization, benchmarks, finetune params

# 3. Run the full pipeline
apr-leaderboard pipeline --config configs/my-model.toml

# 4. Review results
apr-leaderboard history
apr-leaderboard history --model qwen

# 5. Submit to leaderboard
apr-leaderboard submit --results results/latest.json \
    --model paiml/qwen-coder-7b-apr --leaderboard bigcode
```

**For sovereign stack developers:**

This repo is an integration test for the sovereign stack. If `apr-leaderboard pipeline` produces competitive scores, the stack works. If it doesn't, `apr compare-hf` and the per-step eval results pinpoint the weak component.

```bash
# Run baseline parity check
apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct -o baseline.apr
apr compare-hf baseline.apr --json    # identifies inference parity gap
apr eval baseline.apr --dataset wikitext-2   # perplexity sanity check
apr bench baseline.apr --json                # throughput measurement

# If parity gap > 5%, investigate:
apr trace baseline.apr --verbose      # layer-by-layer analysis
apr diff baseline.apr reference.apr   # weight comparison
apr parity baseline.apr               # GPU/CPU divergence check
```

**For researchers:**

The spec (this document) is the experimental protocol. The recipes in §9 are reproducible experiments. The acceptance criteria in §18 are the pass/fail conditions. Run them, report results, falsify or validate the thesis.

## 2. Thesis

The Python ML ecosystem requires 200+ dependencies, GPU-locked CUDA toolchains, and multi-GB Docker images to compete on HuggingFace leaderboards. We will demonstrate that a single-binary Rust pipeline — using only `apr` CLI commands — can match or exceed these results with zero Python, zero external CUDA toolkit (trueno generates PTX natively), and 10x smaller deployment artifacts.

**Constraint:** Every optimization step must be expressible as an `apr` subcommand. No Python. No notebooks. No HuggingFace Transformers library. Pure sovereign stack.

**Compute reality:** GPU hardware is recommended for training-phase techniques (distillation, fine-tuning). Inference-only techniques (merging, quantization) and small-model inference (≤7B quantized) run on CPU via trueno SIMD (AVX2/NEON). The "zero CUDA" claim means no `nvcc`, no `libcudart`, no CUDA toolkit install — trueno's PTX backend generates GPU kernels in pure Rust.

## 3. Target Leaderboards & Competitive Thresholds

| Leaderboard | Primary Metric | Benchmarks | Why |
|-------------|---------------|------------|-----|
| EvalPlus | pass@1 | HumanEval+, MBPP+ | Rigorous test suites (80x/35x more tests than originals) expose real quality — the gold standard |
| BigCodeBench | pass@1 | 1,140 practical tasks | Tests library usage, I/O, and dependencies — not yet saturated (GPT-4o scores ~61%) |
| LiveCodeBench | pass@1 | 1,055 fresh competitive problems | Continuously refreshed from LeetCode/CodeForces — contamination-resistant |
| BigCode Models | pass@1 | HumanEval, MBPP, MultiPL-E | Code generation visibility — our primary use case |

### 3.1 Competitive Score Thresholds (2025-2026)

HumanEval is approaching saturation (SOTA 92.7%). BigCodeBench and LiveCodeBench differentiate more meaningfully.

| Benchmark | Not Competitive | Entry | Strong | SOTA (Open) |
|-----------|-----------------|-------|--------|-------------|
| HumanEval (pass@1) | <60% | 60-75% | 75-85% | **85-93%** |
| HumanEval+ (pass@1) | <70% | 70-80% | 80-85% | **85-89%** |
| MBPP (pass@1) | <70% | 70-80% | 80-85% | **85-91%** |
| BigCodeBench-Full (pass@1) | <30% | 30-40% | 40-50% | **50%+** |
| LiveCodeBench (pass@1) | <20% | 20-40% | 40-60% | **60%+** |

### 3.2 The Landscape: Who Holds the Crown

**32B class — current SOTA:**

| Model | HumanEval | HE+ | MBPP | LiveCode | License |
|-------|-----------|------|------|----------|---------|
| Qwen2.5-Coder-32B-Instruct | **92.7%** | **87.2%** | **90.2%** | 31.4% | Apache-2.0 |
| OCR-Nemotron-32B | — | — | — | **61.8%** | Apache-2.0 |
| R1-Distill-Qwen-32B | — | — | — | 58.1% | MIT |
| DeepSeek-Coder-V2 (236B MoE) | 85.4% | 82.3% | — | — | Restricted |
| Codestral 25.01 (22B) | 86.6% | — | 91.2% | — | Restricted |

**7B class — current SOTA:**

| Model | HumanEval | HE+ | MBPP | LiveCode | License |
|-------|-----------|------|------|----------|---------|
| Qwen2.5-Coder-7B-Instruct | **88.4%** | **84.1%** | **83.5%** | 18.2% | Apache-2.0 |
| OCR-Nemotron-7B | — | — | — | **51.3%** | Apache-2.0 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 81.1% | — | — | — | Restricted |
| Phi-4 (14B) | 82.6% | — | — | — | MIT |

**Critical gap:** Qwen2.5-Coder dominates standard benchmarks (HumanEval, MBPP) but falls behind on LiveCodeBench. The gap is reasoning: OCR-Nemotron-32B (distilled from DeepSeek-R1) nearly doubles Qwen's LiveCodeBench score. This is the improvement vector.

## 4. Model Selection & Improvement Strategy

### 4.1 WHAT Models We Will Improve

We select models based on three criteria: (1) competitive baseline scores, (2) permissive licensing (Apache-2.0 or MIT), (3) architecture support in aprender.

**Primary targets (Tier 1 — submit to leaderboards):**

| Model | Size | Why This Model | Baseline HE | Target HE | Strategy |
|-------|------|----------------|-------------|-----------|----------|
| Qwen2.5-Coder-7B-Instruct | 7B | Best 7B code model. Apache-2.0. Beats CodeLlama-70B. | 88.4% | **90%+** | Distill + LoRA + DPO |
| Qwen2.5-Coder-32B-Instruct | 32B | Best open code model overall. Matches GPT-4o. | 92.7% | **94%+** | DPO + merge + speculative |
| Qwen2.5-Coder-7B (base) | 7B | Distillation target. Prove 32B→7B transfer works. | ~65% | **85%+** | Full pipeline (Recipe C) |

**Secondary targets (Tier 2 — prove stack generality):**

| Model | Size | Why This Model | Strategy |
|-------|------|----------------|----------|
| OCR-Nemotron-7B | 7B | Best 7B for LiveCodeBench (51.3%). Reasoning distilled. | Import + eval parity check |
| Phi-4 | 14B | Strong at 14B. Different architecture than Qwen. | Import + merge with Qwen variants |
| DeepSeek-R1-Distill-Qwen-7B | 7B | Reasoning-enhanced Qwen. Merge candidate. | Merge with Qwen2.5-Coder-7B |

**Stretch target (Tier 3 — marketing win):**

| Model | Size | Why This Model | Strategy |
|-------|------|----------------|----------|
| Qwen2.5-Coder-1.5B | 1.5B | Smallest competitive code model. `apr compile` → single binary demo. | LoRA + quantize + compile |

### 4.2 WHY We Will Improve Them

**The falsifiable claim:** A single Rust binary can produce models that score in the "Strong" tier or above on every target benchmark.

Five specific improvement hypotheses, each falsifiable:

**H1: Reasoning distillation closes the LiveCodeBench gap.**
- Qwen2.5-Coder-7B scores 18.2% on LiveCodeBench. OCR-Nemotron-7B (reasoning-distilled) scores 51.3%. Distilling from a reasoning teacher should lift LiveCodeBench by 2-3x without hurting HumanEval.
- *Falsified if:* LiveCodeBench stays below 30% after distillation.

**H2: DPO with execution feedback pushes HumanEval+ past 87%.**
- Current Qwen2.5-Coder-7B scores 84.1% on HumanEval+. The 84→87% gap is alignment, not capability. DPO using (correct_code, incorrect_code) pairs from execution feedback should close it.
- *Falsified if:* HumanEval+ stays below 86% after DPO.

**H3: Merge specialists beat any single model.**
- Merging a code-instruct specialist with a code-reasoning specialist (via TIES on the same Qwen2.5 backbone) should exceed either specialist alone.
- *Falsified if:* Merged model scores below the best input specialist on all benchmarks.

**H4: Quantization to INT4 loses <2% pass@1.**
- Conservative quantization (INT4 with calibration) should preserve almost all accuracy for code generation.
- *Falsified if:* INT4 model drops more than 2% pass@1 vs FP16 on HumanEval.

**H5: The full pipeline (distill→finetune→merge→prune→quantize) compounds gains.**
- Each technique contributes independently. Stacked in the golden ordering (§10), they should compound.
- *Falsified if:* Full pipeline scores lower than the best single-technique result.

### 4.3 HOW We Will Improve Each Model

#### 4.3.1 Qwen2.5-Coder-7B: "The Complete Proof" (Primary Target)

This is the model that proves the thesis. Every technique applied, every claim validated.

```
Phase 1: Baseline
  apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct → baseline.apr
  apr eval baseline.apr → establish apr-native HumanEval/MBPP scores
  apr compare-hf baseline.apr → measure parity gap

Phase 2: Reasoning Distillation (H1)
  apr import hf://Qwen/Qwen2.5-Coder-32B-Instruct → teacher.apr
  apr distill teacher.apr --student base.apr --strategy progressive
  → Expected: +5-13% on HumanEval, +15-30% on LiveCodeBench

Phase 3: LoRA Fine-tuning on Curated Code Data
  apr finetune distilled.apr --method qlora --rank 32 --data code-instruct.jsonl
  → Expected: +3-5% from domain-specific tuning

Phase 4: DPO Alignment (H2)
  apr align distilled-tuned.apr --method dpo --data preference-pairs.jsonl
  → Expected: +2-4% on HumanEval+ from execution-feedback alignment

Phase 5: Merge with Reasoning Variant (H3)
  apr merge code-specialist.apr reasoning-specialist.apr --strategy ties
  → Expected: best-of-both-worlds across benchmarks

Phase 6: Prune + Quantize (H4)
  apr prune merged.apr --method wanda --target-ratio 0.2
  apr quantize pruned.apr --scheme int4
  → Expected: <2% pass@1 loss, 4x smaller, 2x faster inference

Phase 7: Compile & Ship
  apr compile final.apr -o qwen-coder-7b --release --lto
  → Standalone binary, zero runtime deps
```

**Success gate:** Final model achieves ≥85% HumanEval, ≥82% HumanEval+, ≥80% MBPP, all via `apr` commands only.

#### 4.3.2 Qwen2.5-Coder-32B: "The Crown" (Maximum Score)

The 32B model is already at 92.7% HumanEval. The goal is to push past the ceiling using techniques that benefit from the model's existing strength.

```
Phase 1: Baseline + parity verification
Phase 2: DPO with execution feedback (primary lever)
Phase 3: Merge with reasoning variant (R1-Distill-Qwen-32B)
Phase 4: Speculative decoding for faster eval iteration
Phase 5: N-sampling (N=50) + reranking for maximum pass@1
```

**Success gate:** ≥94% HumanEval, ≥88% HumanEval+, ≥45% BigCodeBench.

#### 4.3.3 Qwen2.5-Coder-1.5B: "The Sovereign Binary" (Marketing Win)

```
Phase 1: Import + baseline
Phase 2: LoRA fine-tune on curated instruction data
Phase 3: INT4 quantize
Phase 4: apr compile → single static binary (~800MB)
Phase 5: Ship as downloadable executable
```

**Success gate:** ≥60% HumanEval in a standalone binary with zero dependencies. The demo: `./qwen-coder "def fibonacci(n):"` just works.

### 4.4 What Happens When Improvement Fails

Each hypothesis above has a falsification criterion. When falsified:

1. **Diagnose with five-whys:** `apr diagnose model.apr --method five-whys` identifies root cause (inference bug? data quality? technique misconfigured?)
2. **Compare against HF reference:** `apr compare-hf model.apr` — if parity gap is >5%, fix inference first, don't optimize on a broken baseline
3. **Ablation:** Remove the last technique applied and re-evaluate. If removal improves score, the technique was destructive in this combination.
4. **Escalate to next tier:** If a technique fundamentally doesn't work at world-class level, the tooling must improve (see §5 Sovereign Tooling Map)

## 5. Sovereign Tooling Map: World-Class or Wire It In

Every leaderboard-winning technique maps to a sovereign stack component. When a component doesn't support a technique at world-class level, we don't skip it — we find or build the capability and wire it into `apr` CLI commands.

### 5.1 Tooling Coverage Matrix

| Technique | Required Capability | Sovereign Component | Status | Gap Action |
|-----------|-------------------|-------------------|--------|------------|
| Import HF models | SafeTensors/GGUF → .apr | **aprender** 0.27 | ✅ Complete | `apr import` — 14+ architectures supported |
| Inference (decode) | Transformer forward pass | **realizar** 0.8 | ✅ Complete | `apr run` — 8-21% faster than llama.cpp |
| Inference (serve) | HTTP API, batching, streaming | **realizar** 0.8 | ✅ Complete | `apr serve` — OpenAI-compatible, PagedAttention |
| LoRA/QLoRA training | Low-rank adaptation, autograd | **entrenar** 0.7 | ✅ Complete | `apr finetune` — AdamW, cosine LR, checkpointing |
| Knowledge distillation | KL-divergence, progressive | **entrenar** 0.7 | ✅ Complete | `apr distill` — standard, progressive, ensemble |
| Model merging | SLERP, TIES, DARE | **aprender** 0.27 | ✅ Complete | `apr merge` — 5 strategies |
| Pruning | Wanda, SparseGPT, structured | **aprender** 0.27 | ✅ Complete | `apr prune` — 6 methods |
| Quantization | INT4, INT8, Q4K, Q6K | **aprender** 0.27 | ✅ Complete | `apr quantize` — 4 formats |
| SIMD tensor ops | AVX2, AVX-512, NEON matmul | **trueno** 0.16 | ✅ Complete | 6% faster than NumPy at 256×256 |
| GPU compute | PTX generation, wgpu | **trueno** 0.16 | ✅ Complete | Pure Rust, no nvcc |
| Speculative decoding | Draft model + verification | **realizar** 0.8 | ✅ Complete | `apr run --speculative` |
| KV cache management | PagedAttention, CoW | **realizar** 0.8 | ✅ Complete | vLLM-style paged KV |
| Data loading | Parquet, JSONL, Arrow, HF Hub | **alimentar** 0.2 | ✅ Complete | Zero-copy Arrow RecordBatches |
| Data quality | Null/outlier/drift detection | **alimentar** 0.2 | ✅ Complete | 100-point quality scoring |
| Data decontamination | N-gram overlap detection | **alimentar** 0.2 | ⚠️ Partial | Doctest extraction exists; need benchmark-specific decontamination |
| HPO | TPE, Hyperband, ASHA | **entrenar** 0.7 | ✅ Complete | `apr tune --strategy tpe` |
| Compile to binary | Model + runtime → executable | **aprender** 0.27 | ✅ Complete | `apr compile` |
| Correctness proofs | Kani bounded model checking | **provable-contracts** | ✅ Complete | 262 proof obligations |
| Quality gates | Compliance enforcement | **pmat** | ✅ Complete | 30+ automated checks |
| **DPO/ORPO alignment** | Preference optimization | **entrenar** 0.7 | ❌ **Missing** | **Must build** (see §5.2) |
| **Execution sandbox** | Run generated code safely | — | ❌ **Missing** | **External harness** (see §5.3) |
| **N-sampling + rerank** | Batched generation, voting | **aprender** 0.27 | ⚠️ Partial | Generation works; reranking logic needed |
| **Prompt templates** | SCoT, few-shot strategies | **aprender** 0.27 | ✅ **Ready** | `--prompt-strategy` implemented in apr-leaderboard (5 strategies); upstream `--system` available |
| **Synthetic data gen** | Teacher → training corpus | **alimentar** 0.2 + **aprender** | ⚠️ Partial | Generation via `apr chat --batch`; curation pipeline needed |
| **Continued pretraining** | Full-weight code corpus training | **entrenar** 0.7 | ⚠️ Partial | Full finetune works; needs large-corpus streaming |
| **Flash Attention** | Online softmax, tiled attention | **trueno** 0.16 | 🔧 In Progress | Phase 12 planned; tiling infra ready |

### 5.2 Gap 1: DPO/ORPO Preference Optimization (CRITICAL)

**Why world-class:** DPO is the single most impactful post-training technique for leaderboards. Merged + DPO models "completely dominate" HF leaderboard rankings. Without DPO, we compete with one hand tied.

**Current state:** entrenar has the training infrastructure (autograd, AdamW, LoRA) but no DPO loss function or preference pair data loader.

**Wire-in plan:**

```
Component: entrenar
  Add: src/dpo/mod.rs — DPO loss (β-scaled log-ratio of policy vs reference)
  Add: src/dpo/data.rs — preference pair loader (chosen/rejected format)
  Add: src/dpo/orpo.rs — ORPO variant (no reference model needed)

Component: aprender (apr-cli)
  Add: `apr align` subcommand
    apr align model.apr --method dpo \
      --reference base.apr \
      --data preference-pairs.jsonl \
      --beta 0.1 --epochs 3 \
      -o aligned.apr

    apr align model.apr --method orpo \
      --data preference-pairs.jsonl \
      --lambda 0.1 --epochs 3 \
      -o aligned.apr

Component: alimentar
  Add: Preference pair generation from execution feedback
    alimentar generate-preferences \
      --model model.apr \
      --problems humaneval.jsonl \
      --n-samples 10 \
      --judge execution \
      -o preference-pairs.jsonl

Component: Ground truth corpus
  Use: hf-ground-truth-corpus, algorithm-competition-corpus
    → Source of verified correct/incorrect code pairs for DPO training
```

**Acceptance criterion:** `apr align --method dpo` produces a model with ≥2% higher HumanEval+ than the input model after 3 epochs.

### 5.3 Gap 2: Code Execution Sandbox (CRITICAL)

**Why world-class:** HumanEval and MBPP require executing generated code against test cases. Without execution, we can't compute pass@k — we can only measure perplexity, which doesn't correlate well with code correctness.

**Current state:** aprender has no sandboxed code execution. Generated completions must be evaluated externally.

**Wire-in plan (two options):**

```
Option A: External EvalPlus harness (short-term, pragmatic)
  apr eval model.apr --data humaneval.jsonl --n-samples 10 \
    --output-completions completions/ --json
  # Then externally: evalplus.evaluate --samples completions/
  # This is what everyone does — even Google and Meta use external harnesses

Option B: WASM sandbox (long-term, sovereign)
  Component: realizar or new crate
  Add: Embedded WASM runtime (wasmtime) for safe code execution
    apr eval model.apr --data humaneval.jsonl \
      --sandbox wasm --timeout 10s --json
  Advantage: Fully sovereign, no Python dependency even for eval
  Risk: Python test cases require Python-in-WASM (CPython compiled to WASM)
```

**Decision:** Option A for v1.0 (get on the leaderboard), Option B as stretch goal. Neither compromises the "zero Python" claim for the model pipeline — eval is a separate concern.

### 5.4 Gap 3: N-Sampling + Reranking Pipeline

**Why world-class:** Generating N=10-50 completions and selecting the best one boosts effective pass@1 by 10-30%. This is the single most impactful inference-time technique.

**Current state:** aprender can generate multiple completions via temperature sampling. Missing: batched generation, reranking logic, majority voting.

**Wire-in plan:**

```
Component: aprender (apr-cli)
  Extend: `apr eval --n-samples N --rerank strategy`
    Strategies: logprob (sum of log-probabilities), majority (output voting),
                execution (run and pick passing code — requires sandbox)

Component: realizar
  Already supports: batched generation, concurrent requests
  Need: expose batch generation for N completions per prompt efficiently

Component: alimentar
  Add: Result aggregation and voting logic for N-sample outputs
```

### 5.5 Gap 4: Synthetic Training Data Pipeline

**Why world-class:** Qwen2.5-Coder, Phi-4, and NVIDIA OCR-Nemotron all credit large-scale synthetic data as core to their success. Without high-quality synthetic training data, fine-tuning is limited to existing datasets.

**Current state:** `apr chat --batch` can generate completions. alimentar handles data loading and quality scoring. Ground-truth corpora exist (hf-ground-truth-corpus, algorithm-competition-corpus). Missing: end-to-end curation pipeline.

**Wire-in plan:**

```
Component: alimentar
  CLI pipeline:
    # 1. Generate raw synthetic code from teacher
    apr chat teacher.apr --batch problems.txt --n-samples 5 \
      --temperature 0.8 --json > raw-synthetic.jsonl

    # 2. Quality-filter with alimentar
    alimentar quality raw-synthetic.jsonl --min-score 80 \
      -o filtered-synthetic.jsonl

    # 3. Decontaminate against eval benchmarks
    alimentar drift raw-synthetic.jsonl \
      --reference humaneval.jsonl mbpp.jsonl \
      --overlap-threshold 0.01 \
      -o clean-synthetic.jsonl

    # 4. Balance and split
    alimentar convert clean-synthetic.jsonl \
      -o training-data.parquet

Component: Ground truth corpora
  hf-ground-truth-corpus → HuggingFace API patterns, transformer implementations
  algorithm-competition-corpus → Algorithm problems with verified solutions
  → Both feed into fine-tuning data mix
```

### 5.6 Gap 5: Prompt Strategy Engine

**Why world-class:** SCoT prompting improves HumanEval pass@1 by up to 13.79%. Few-shot exemplars add 3-8%. The prompt template matters as much as the model weights.

**Current state:** `--prompt-strategy` is implemented in `apr-leaderboard eval` with 5 built-in strategies. The upstream `apr chat --system` and `apr run --chat` provide raw system prompt support.

**Implemented in apr-leaderboard:**

```bash
# All 5 strategies work via apr-leaderboard eval:
apr-leaderboard eval --model m.apr --benchmark humaneval --prompt-strategy standard
apr-leaderboard eval --model m.apr --benchmark humaneval --prompt-strategy scot
apr-leaderboard eval --model m.apr --benchmark humaneval --prompt-strategy few-shot
apr-leaderboard eval --model m.apr --benchmark humaneval --prompt-strategy cgo
apr-leaderboard eval --model m.apr --benchmark humaneval --prompt-strategy reflexion
```

**Built-in strategies (with aliases):**

| Strategy | Aliases | Description |
|---|---|---|
| `standard` | `default` | Raw problem → code (baseline) |
| `scot` | `structured-cot` | Structured chain-of-thought → code (+5-14%) |
| `few-shot` | `fewshot` | N exemplars + problem → code (+3-8%) |
| `cgo` | `code-gen-opt` | Chain of grounded objectives → code (+5-10%) |
| `reflexion` | `reflect` | Generate → test → reflect → regenerate (multi-turn) |

**Remaining wire-in for upstream apr:**

```
Component: realizar
  Already supports: chat templates (ChatML, LLaMA2, Mistral, Phi, Alpaca)
  Need: expose template composition for eval pipeline
```

### 5.7 Sovereign Stack Version Requirements

All gap closures must use published crates from crates.io. No git dependencies.

| Crate | Current | Required For Gaps | Minimum Version |
|-------|---------|-------------------|-----------------|
| aprender | 0.27.2 | `apr align`, `--n-samples --rerank` | **0.28** |
| entrenar | 0.7.5 | DPO loss, preference pair loader, ORPO | **0.8** |
| trueno | 0.16.1 | Flash attention (Phase 12) | **0.17** |
| realizar | 0.8.0 | Batch N-sampling, prompt template composition | **0.9** |
| alimentar | 0.2.6 | Decontamination pipeline, preference pair generation, quality filtering | **0.3** |
| provable-contracts | 0.1 | DPO kernel contracts | **0.2** |

### 5.8 The Decision Rule

When we find a gap:

1. **Can an existing sovereign crate do it?** → Wire it in via `apr` CLI. No new crates.
2. **Does a sovereign crate need a new module?** → Add it to that crate, publish to crates.io, bump apr-leaderboard's dependency.
3. **Is it fundamentally outside the stack's scope?** → Use an external tool (e.g., EvalPlus for code execution) and document the boundary explicitly.
4. **Is it a research problem with no clear solution?** → Add to §21 Open Questions. Don't block the pipeline.

**Hard rule:** We never add a Python dependency. We never add a C/C++ FFI dependency. If the sovereign stack can't do it in pure Rust, we either build it or scope it out with an explicit boundary.

### 5.9 Parity Check: Ludwig Feature Coverage

Ludwig (ludwig.ai) is the state-of-the-art declarative ML framework. Every feature Ludwig ships, the sovereign stack must match or exceed — in pure Rust, with zero Python. This is the parity bar.

#### 5.9.1 Feature-by-Feature Parity Matrix

**Training & Fine-tuning:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Full fine-tuning | PyTorch, trainable=true | **entrenar** `apr finetune --method full` | ✅ Parity |
| LoRA adapters | PEFT library, configurable rank/dropout/targets | **entrenar** `apr finetune --method lora` | ✅ Parity |
| QLoRA (4-bit base + LoRA) | bitsandbytes + PEFT | **entrenar** `apr finetune --method qlora` | ✅ Parity |
| AdaLoRA (dynamic rank allocation) | PEFT AdaLoRA | **entrenar** — not yet | ❌ **Gap** |
| IA3 (inhibiting/amplifying activations) | PEFT IA3 | **entrenar** — not yet | ❌ **Gap** |
| DoRA (weight-decomposed LoRA) | PEFT DoRA variant | **entrenar** — not yet | ❌ **Gap** |
| NEFTune (embedding noise) | noise injection during fine-tune | **entrenar** — not yet | ❌ **Gap** |
| Gradient accumulation | PyTorch native | **entrenar** gradient accumulation | ✅ Parity |
| Mixed precision (fp16/bf16) | PyTorch AMP | **entrenar** GradScaler, bf16/fp16 | ✅ Parity |
| Early stopping | callback-based | **entrenar** EarlyStopping callback | ✅ Parity |
| Checkpointing | periodic save | **entrenar** CheckpointCallback | ✅ Parity |
| Learning rate warmup + cosine decay | scheduler | **entrenar** WarmupCosineDecayLR | ✅ Parity |

**Optimizers:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| AdamW | PyTorch AdamW | **entrenar** AdamW (SIMD-accelerated) | ✅ Exceeds |
| Adam | PyTorch Adam | **entrenar** Adam | ✅ Parity |
| SGD with momentum | PyTorch SGD | **entrenar** SGD with momentum | ✅ Parity |
| 8-bit optimizers | bitsandbytes 8-bit Adam | — not yet | ❌ **Gap** |
| Paged optimizers | bitsandbytes paged | — not yet | ❌ **Gap** |

**Distributed Training:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Multi-GPU DDP | PyTorch DDP via Ray | — not yet (single-GPU) | ❌ **Gap** |
| DeepSpeed ZeRO | Microsoft DeepSpeed | — not yet | ❌ **Gap** |
| Multi-node training | Ray cluster | — not yet | ❌ **Gap** |
| Automatic batch size selection | binary search on GPU OOM | **aprender** `--vram` planning | ⚠️ Partial |

**Quantization:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| 4-bit quantization (nf4/fp4) | bitsandbytes | **aprender** INT4, Q4K | ✅ Parity |
| 8-bit quantization | bitsandbytes | **aprender** INT8, Q8_0 | ✅ Parity |
| Double quantization | bitsandbytes nested | — not yet | ⚠️ Partial |
| GPTQ | auto-gptq | — not yet | ❌ **Gap** |
| AWQ | autoawq | — not yet | ❌ **Gap** |

**Inference & Generation:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Greedy decoding | HF generate | **realizar** greedy | ✅ Parity |
| Temperature sampling | HF generate | **realizar** temperature | ✅ Parity |
| Top-k sampling | HF generate | **realizar** top-k | ✅ Parity |
| Nucleus (top-p) sampling | HF generate | **realizar** top-p | ✅ Parity |
| Beam search | HF generate | **aprender** num_beams | ✅ Parity |
| Contrastive search | HF generate | — not yet | ❌ **Gap** |
| Diverse beam search | HF generate | — not yet | ❌ **Gap** |
| Repetition penalty | HF generate | **aprender** repetition_penalty | ✅ Parity |
| Speculative decoding | not supported | **realizar** speculative | ✅ **Exceeds** |
| Streaming generation | not documented | **realizar** SSE streaming | ✅ **Exceeds** |
| OpenAI-compatible API | not supported | **realizar** /v1/chat/completions | ✅ **Exceeds** |
| PagedAttention KV cache | not supported | **realizar** paged KV | ✅ **Exceeds** |
| Continuous batching | not supported | **realizar** batch scheduling | ✅ **Exceeds** |

**Serving & Deployment:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| REST API serving | `ludwig serve` (Flask) | **realizar** `apr serve` (Axum) | ✅ Parity |
| Docker containers | prebuilt images | — user-provided | ⚠️ Partial |
| TorchScript export | PyTorch jit.trace | — not applicable (native binary) | N/A |
| Triton Inference Server | export format | — not applicable | N/A |
| HuggingFace Hub upload | `ludwig upload` | **aprender** `apr publish` | ✅ Parity |
| Compile to standalone binary | not supported | **aprender** `apr compile` | ✅ **Exceeds** |
| ONNX/CoreML/OpenVINO export | not supported | **aprender** `apr export` | ✅ **Exceeds** |

**Data Processing:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| CSV/JSON/Parquet/HDF5 loading | pandas | **alimentar** Arrow-native | ✅ Exceeds (zero-copy) |
| Auto preprocessing per feature type | Ludwig preprocessors | **alimentar** transforms | ✅ Parity |
| Train/val/test splitting | Ludwig split | **alimentar** DatasetSplit (stratified) | ✅ Parity |
| Larger-than-memory datasets | Ray datasets | **alimentar** MmapDataset, streaming | ✅ Parity |
| Data quality scoring | not built-in | **alimentar** 100-point quality scoring | ✅ **Exceeds** |
| Drift detection | not built-in | **alimentar** KS/Chi-sq/PSI/JSD | ✅ **Exceeds** |
| Imbalance detection + resampling | not built-in | **alimentar** SMOTE, oversample | ✅ **Exceeds** |

**Hyperparameter Optimization:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Random search | Ray Tune | **entrenar** RandomSearch | ✅ Parity |
| Grid search | Ray Tune | **entrenar** GridSearch | ✅ Parity |
| Bayesian (TPE) | Ray Tune Optuna | **entrenar** TPEOptimizer | ✅ Parity |
| ASHA scheduler | Ray Tune ASHA | **entrenar** HyperbandScheduler | ✅ Parity |
| Distributed HPO | Ray cluster | — not yet (local only) | ❌ **Gap** |

**Model Architecture:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| ECD (Encoder-Combiner-Decoder) | Ludwig native | — different architecture | N/A (not needed) |
| GBM (LightGBM) | LightGBM wrapper | — not in scope | N/A |
| LLM causal models | HF Transformers | **aprender** + **realizar** | ✅ Parity |
| Multi-modal (text+image+audio) | ECD combiner | — LLM-only for leaderboard | N/A (future) |
| Multi-task learning | multiple output heads | — not yet | ⚠️ Partial |
| Custom PyTorch modules | register API | — Rust modules via entrenar | ✅ Parity |

**Experiment Tracking:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| TensorBoard | callback | — not yet | ❌ **Gap** |
| Weights & Biases | callback | — not yet | ❌ **Gap** |
| MLflow | callback | — not yet | ❌ **Gap** |
| Comet ML | callback | — not yet | ❌ **Gap** |
| Built-in TUI monitoring | not supported | **entrenar** monitor + TUI | ✅ **Exceeds** |
| Prometheus metrics | not supported | **realizar** /metrics | ✅ **Exceeds** |

**Explainability & Visualization:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Feature importance | built-in | **entrenar** ExplainabilityCallback | ✅ Parity |
| Learning curves | matplotlib | **entrenar** MonitorCallback | ⚠️ Partial |
| Confusion matrices | built-in | **entrenar** eval metrics | ⚠️ Partial |
| Model architecture visualization | built-in | **aprender** `apr tree`, `apr flow` | ✅ Parity |

**Correctness & Quality (sovereign stack advantages):**

| Feature | Ludwig | Sovereign Stack | Advantage |
|---|---|---|---|
| Provable kernel correctness | none | **provable-contracts** Kani L4 | ✅ **Unique** |
| 262 proof obligations | none | **provable-contracts** | ✅ **Unique** |
| Compliance enforcement | none | **pmat comply** 30+ checks | ✅ **Unique** |
| Deterministic builds | pip/conda chaos | Cargo.lock | ✅ **Unique** |
| Pure Rust PTX generation | requires nvcc | **trueno** pure Rust | ✅ **Unique** |
| Format-agnostic conversion | not supported | **aprender** `apr rosetta` | ✅ **Unique** |
| Model diff/forensics | not supported | **aprender** `apr diff`, `apr hex` | ✅ **Unique** |
| 10-stage integrity check | not supported | **aprender** `apr check` | ✅ **Unique** |

#### 5.9.2 Summary: Where We Exceed, Where We Must Close Gaps

**We exceed Ludwig in 15+ areas:** speculative decoding, PagedAttention, continuous batching, streaming API, OpenAI-compatible serving, compile-to-binary, multi-format export (ONNX/CoreML/OpenVINO), data quality scoring, drift detection, imbalance detection, Prometheus metrics, TUI monitoring, provable contracts, deterministic builds, format forensics.

**We have parity in 25+ areas:** LoRA, QLoRA, full fine-tuning, AdamW/Adam/SGD, gradient accumulation, mixed precision, early stopping, checkpointing, LR scheduling, all sampling strategies, beam search, REST serving, HF upload, data loading, preprocessing, train/val/test splits, HPO (grid/random/TPE/ASHA), feature importance.

**Gaps to close (11 items):**

| Gap | Priority | Wire-in Target |
|---|---|---|
| AdaLoRA (dynamic rank) | Medium | **entrenar** 0.8 |
| IA3 adapter | Low | **entrenar** 0.8 |
| DoRA (weight-decomposed LoRA) | Medium | **entrenar** 0.8 |
| NEFTune (embedding noise) | Low | **entrenar** 0.8 |
| 8-bit optimizers | Low | **entrenar** 0.8 |
| Contrastive search decoding | Low | **aprender** 0.28 |
| Diverse beam search | Low | **aprender** 0.28 |
| Multi-GPU DDP | High | **entrenar** 0.9 |
| DeepSpeed ZeRO | Medium | **entrenar** 0.9 |
| GPTQ quantization | Medium | **aprender** 0.28 |
| Experiment tracking (W&B/MLflow) | Medium | **entrenar** 0.8 callbacks |

**Out of scope (not needed for leaderboard):** ECD architecture, GBM/LightGBM, multi-modal (text+image+audio), Triton export, TorchScript. These serve Ludwig's "general ML framework" positioning. We are a purpose-built leaderboard pipeline, not a general framework.

## 6. CLI Toolchain

Two CLIs work together: **`apr`** (upstream aprender — ML operations) and **`apr-leaderboard`** (this repo — orchestration). Every technique maps to a single shell command. Our competitors use 500-line Python scripts; we use one-liners.

### 6.1 The `apr` CLI (aprender)

The upstream `apr` binary provides all ML operations. `apr-leaderboard` calls these under the hood.

#### 6.1.1 Import (HF → APR)

```bash
# Import from HuggingFace Hub — auto-detects architecture
apr import hf://Qwen/Qwen2.5-Coder-7B -o qwen-7b.apr --arch qwen2

# Import with quantization on ingest
apr import hf://Qwen/Qwen2.5-Coder-32B -o qwen-32b-q8.apr --quantize int8

# Import GGUF with provenance enforcement
apr import qwen-7b.gguf -o qwen-7b.apr --enforce-provenance
```

#### 6.1.2 Evaluate (Baseline)

```bash
# Perplexity baseline
apr eval qwen-7b.apr --dataset wikitext-2 --threshold 20.0

# Classification eval with custom data
apr eval qwen-7b.apr --task classify --data humaneval.jsonl --json
```

#### 6.1.3 Full Optimization Pipeline (preview)

```bash
# The complete leaderboard recipe in 6 commands (follows golden ordering §10):
apr import hf://Qwen/Qwen2.5-Coder-7B -o base.apr
apr distill teacher.apr --student base.apr --strategy progressive --temperature 3.0 -o distilled.apr
apr finetune distilled.apr --method qlora --rank 32 --data code-instruct.jsonl -o tuned.apr
apr merge tuned.apr variant-b.apr --strategy slerp -o merged.apr
apr prune merged.apr --method wanda --target-ratio 0.2 --calibration calib.jsonl -o pruned.apr
apr quantize pruned.apr --scheme int4 -o submit.apr
```

### 6.2 The `apr-leaderboard` CLI (this repo)

The orchestration layer that drives the pipeline. Each subcommand maps to one or more upstream `apr` operations.

| Subcommand | Maps to | Description |
|---|---|---|
| `convert` | `apr import` | Download HF model → `.apr` format |
| `eval` | `apr eval` | Run benchmark suite with pass@k metrics |
| `finetune` | `apr finetune` (entrenar) | LoRA/QLoRA fine-tuning |
| `distill` | `apr distill` | Knowledge distillation (teacher → student) |
| `merge` | `apr merge` | Model merging (SLERP, TIES, DARE, linear) |
| `prune` | `apr prune` | Structured/unstructured pruning |
| `quantize` | `apr quantize` | Post-training quantization |
| `compare` | `apr compare-hf` | Parity check against HF reference |
| `submit` | — | Format + push results to HF leaderboard |
| `benchmarks` | — | List available benchmark suites |
| `history` | — | Show past evaluation results |
| `pipeline` | all of the above | Config-driven end-to-end pipeline |

#### 6.2.1 Convert

```bash
# Convert a HuggingFace model to .apr format
apr-leaderboard convert --model-id Qwen/Qwen2.5-Coder-7B

# With custom output and quantization
apr-leaderboard convert --model-id Qwen/Qwen2.5-Coder-7B --output models/ --quantization int8
```

#### 6.2.2 Eval

```bash
# Run HumanEval with defaults (standard prompt, 1 sample)
apr-leaderboard eval --model models/qwen-7b.apr --benchmark humaneval

# Full benchmark with structured CoT and best-of-20 selection
apr-leaderboard eval --model models/qwen-7b.apr --benchmark humaneval \
    --samples 0 --prompt-strategy scot --n-samples 20

# Subset evaluation on BigCodeBench
apr-leaderboard eval --model models/qwen-7b.apr --benchmark bigcodebench \
    --samples 100 --output results/
```

**Prompt strategies** (§8.3):

| Strategy | Flag value | Description |
|---|---|---|
| Standard | `standard` / `default` | Direct prompt, no special formatting |
| Structured CoT | `scot` / `structured-cot` | Step-by-step reasoning before code |
| Few-shot | `few-shot` / `fewshot` | Include solved examples in prompt |
| Code Gen Opt | `cgo` / `code-gen-opt` | Optimization-focused generation |
| Reflexion | `reflexion` / `reflect` | Generate → test → reflect → regenerate |

**N-samples:** `--n-samples N` generates N completions per problem, selects the best (maximizes pass@k). Default: 1.

#### 6.2.3 Finetune

```bash
# LoRA fine-tune with defaults (rank 16, lr 1e-4, 3 epochs)
apr-leaderboard finetune --model models/qwen-7b.apr --dataset data/code-instruct.jsonl

# Custom LoRA config
apr-leaderboard finetune --model models/qwen-7b.apr --dataset data/code-instruct.jsonl \
    --rank 32 --lr 0.001 --epochs 5
```

#### 6.2.4 Distill

```bash
# Progressive distillation (recommended for code models)
apr-leaderboard distill --teacher teacher-32b.apr --student student-7b.apr \
    --strategy progressive --temperature 3.0 --alpha 0.7 -o distilled-7b.apr

# Ensemble distillation from multiple teachers
apr-leaderboard distill --teacher ensemble.apr --student student-7b.apr \
    --strategy ensemble -o distilled-7b.apr
```

**Strategies:** `standard` (KL divergence), `progressive` (curriculum learning), `ensemble` (multi-teacher).

#### 6.2.5 Merge

```bash
# SLERP merge of two models
apr-leaderboard merge model-a.apr model-b.apr --strategy slerp -o merged.apr

# TIES merge of three models
apr-leaderboard merge a.apr b.apr c.apr --strategy ties -o merged.apr
```

**Strategies:** `slerp`, `ties` (TIES-Merging), `dare` (DARE-TIES), `linear` (linear average).

#### 6.2.6 Prune

```bash
# Wanda pruning with 20% sparsity (default)
apr-leaderboard prune --model tuned.apr --method wanda --target-ratio 0.2 -o pruned.apr

# SparseGPT with 30% sparsity
apr-leaderboard prune --model tuned.apr --method sparsegpt --target-ratio 0.3 -o pruned.apr
```

**Methods:** `wanda` (default), `magnitude`, `sparsegpt`. Target ratio: 0.0–1.0 (exclusive).

#### 6.2.7 Quantize

```bash
# INT4 quantization (default, best compression)
apr-leaderboard quantize --model pruned.apr --scheme int4 -o submit.apr

# Q6K quantization (better quality, larger size)
apr-leaderboard quantize --model pruned.apr --scheme q6k -o submit.apr
```

**Schemes:** `int4`, `int8`, `q4k`, `q5k`, `q6k`.

#### 6.2.8 Compare

```bash
# Check parity against HuggingFace reference implementation
apr-leaderboard compare --model models/qwen-7b.apr
```

#### 6.2.9 Submit

```bash
# Submit results to the Open LLM Leaderboard
apr-leaderboard submit --results results/humaneval_20260228.json \
    --model-id paiml/qwen-coder-7b-apr

# Submit to BigCodeBench leaderboard
apr-leaderboard submit --results results/bigcodebench_20260228.json \
    --model-id paiml/qwen-coder-7b-apr --leaderboard bigcode
```

#### 6.2.10 Pipeline (config-driven)

```bash
# Run entire pipeline from a TOML config file
apr-leaderboard pipeline --config configs/qwen-coder-7b.toml
```

Example pipeline config:

```toml
[model]
model_id = "Qwen/Qwen2.5-Coder-7B"
quantization = "fp16"

[eval]
benchmarks = ["humaneval", "mbpp", "bigcodebench"]
samples = 0  # full benchmark

[finetune]
enabled = true
dataset = "data/code-instruct.jsonl"
rank = 16
lr = 1e-4
epochs = 3

[submit]
model_id = "paiml/qwen-coder-7b-apr"
leaderboard = "open-llm-leaderboard"
```

### 6.3 CLI Surface Mapping

The full mapping between `apr-leaderboard` orchestration and `apr` ML operations:

```
apr-leaderboard pipeline --config pipeline.toml
    │
    ├── apr-leaderboard convert  ──►  apr import hf://... -o base.apr
    ├── apr-leaderboard distill  ──►  apr distill teacher.apr --student base.apr ...
    ├── apr-leaderboard finetune ──►  apr finetune base.apr --method qlora ...
    ├── apr-leaderboard merge    ──►  apr merge a.apr b.apr --strategy slerp ...
    ├── apr-leaderboard prune    ──►  apr prune model.apr --method wanda ...
    ├── apr-leaderboard quantize ──►  apr quantize model.apr --scheme int4 ...
    ├── apr-leaderboard eval     ──►  apr eval model.apr --benchmark humaneval ...
    └── apr-leaderboard submit   ──►  (HTTP POST to HF Hub API)
```

## 7. Technique Playbook

### 7.1 Knowledge Distillation

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

### 7.2 Model Merging

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

### 7.3 Pruning

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

### 7.4 Fine-tuning (LoRA)

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

### 7.5 Fine-tuning (QLoRA)

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

### 7.6 Quantization (Post-Training)

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

### 7.7 Hyperparameter Optimization (HPO)

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

## 8. Leaderboard-Winning Techniques

The techniques in §7 optimize the *model*. This section covers techniques that optimize *inference-time behavior* — how you extract the best score from a given model. These are the techniques that separate top-10 leaderboard entries from median ones.

### 8.1 Sampling Strategy Tuning

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

### 8.2 N-Sampling with Best-of-N Selection (pass@k Maximization)

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

### 8.3 Structured Prompting (System Prompt + Few-Shot + SCoT)

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

**Prompt strategies:**

| Strategy | Flag aliases | Description | Expected Impact |
|----------|-------------|-------------|-----------------|
| `standard` | `default` | Raw problem → code | Baseline |
| `scot` | `structured-cot` | Problem → structured reasoning → code | +5-14% pass@1 |
| `few-shot` | `fewshot` | N exemplars + problem → code | +3-8% pass@1 |
| `cgo` | `code-gen-opt` | Chain of Grounded Objectives — goal-oriented decomposition | +5-10% pass@1 |
| `reflexion` | `reflect` | Generate → test → reflect → regenerate (iterative self-correction) | +3-10% pass@1 |

**Implementation status:** `--prompt-strategy` flag is implemented in `apr-leaderboard eval` with all 5 strategies above. `--n-samples` enables best-of-N selection. The `--exemplars` flag (for few-shot) and `--system` flag (for custom system prompts) are available via the upstream `apr eval` command.

### 8.4 Speculative Decoding (Inference Speedup)

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

### 8.5 Preference Optimization (DPO/ORPO)

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

### 8.6 Continued Pretraining (Domain Adaptation)

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

### 8.7 Data Decontamination

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

### 8.8 Test-Time Compute Scaling

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

### 8.9 Technique Stacking: The Winning Formula

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

## 9. Composite Recipes

### 9.0 Step Zero: Establish Baseline (REQUIRED for all recipes)

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

### 9.1 Recipe A: "The Distilled Expert" (Maximum Quality)

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

### 9.2 Recipe B: "The Merge Alchemist" (Zero Training Compute)

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

### 9.3 Recipe C: "The Full Pipeline" (Kitchen Sink)

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

### 9.4 Recipe D: "Sovereign Binary" (The Differentiator)

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

## 10. Technique Interaction Matrix

Techniques are not independent. Order matters.

```
                      ┌──────────────────────────────────────────────┐
                      │          TECHNIQUE INTERACTION MATRIX        │
                      │                                              │
                      │  Column  │ distill  merge  prune  finetune  │
                      │  THEN    │                                   │
                      │  Row ↓   │                                   │
                      │──────────┼─────────────────────────────────  │
                      │ distill  │   —      ✗bad   ✓ok    ✗bad     │
                      │ merge    │  ✓ok      —     ✓ok    ✓✓best   │
                      │ prune    │  ✓ok     ✓ok     —     ✗bad     │
                      │ finetune │ ✓✓best  ✓ok    ✗bad    —        │
                      │ quantize │  ✓ok     ✓ok    ✓ok    ✓ok      │
                      └──────────────────────────────────────────────┘

  Legend: Read as "column THEN row" (column happens first)
    ✓✓best  = Optimal ordering
    ✓ok     = Works but not optimal
    ✗bad    = Harmful (degrades quality or wastes compute)

  Key asymmetries:
    distill→finetune = ✓✓best  (adapt distilled knowledge to task)
    finetune→distill = ✗bad    (distillation overwrites fine-tuned specialization)
    finetune→merge   = ✓✓best  (merge specialized variants)
    merge→finetune   = ✓ok     (works but loses merge diversity)
```

**Golden ordering:** distill → finetune → merge → prune → quantize

Rationale:
1. **Distill first** — Knowledge transfer works best on an unmodified student architecture
2. **Finetune second** — LoRA adapts the distilled weights to target benchmarks
3. **Merge third** — Combine fine-tuned variants while representations are still rich
4. **Prune fourth** — Remove redundancy AFTER merging (merged models have more redundancy)
5. **Quantize last** — Always final step; quantization is lossy and non-reversible

**Note on QLoRA as implicit QAT:** When the final deployment target is INT4, using QLoRA (§7.5) during the finetune step provides quantization-aware adaptation. The adapter trains against quantized base weights, making the final INT4 quantization less lossy than post-training quantization after full-precision LoRA.

**Anti-patterns:**
- Prune → Finetune: LoRA can't recover pruned knowledge effectively
- Finetune → Distill: Overwrites the fine-tuned specialization
- Quantize → anything: Quality loss compounds with every subsequent operation

## 11. Competitive Advantage: Why `apr` Wins

| Aspect | Python Ecosystem | `apr` CLI |
|--------|-----------------|-----------|
| Dependencies | transformers, torch, accelerate, bitsandbytes, peft, trl, vllm | Single binary |
| Setup time | 30-60 min (CUDA toolkit, conda, pip conflicts) | 0 min (`cargo install apr-cli`, trueno generates PTX natively) |
| Merge | 50-line Python script | `apr merge --strategy slerp` |
| Prune | 100+ lines, custom hooks | `apr prune --method wanda` |
| LoRA | peft + trl + custom training loop | `apr finetune --method lora` |
| Distill | Custom training loop, 200+ lines | `apr distill --strategy progressive` |
| Quantize | bitsandbytes or GPTQ, GPU required | `apr quantize --scheme int4` |
| Reproducibility | requirements.txt + CUDA version + random seeds | Deterministic Rust binary |
| Deployment | Docker + CUDA runtime + Python | `apr compile → single binary` |
| CI/CD | Complex, flaky GPU runners | `cargo test` on any machine |
| Auditability | Opaque Python state | `apr check` — 10-stage integrity pipeline |
| Correctness | pytest + hope | `pv proof-status` — Kani bounded model checking, 262 proof obligations |
| Quality gates | Ad-hoc linting | `pmat comply check --strict` — 30+ automated compliance checks |
| Contracts | None | `#[contract]` macro — compile-time binding to mathematical spec |
| Speculative decoding | vLLM config | `apr run --speculative` — native, no external runtime |
| N-sampling + rerank | Custom scripts | `apr eval --n-samples 50 --rerank` — single command |
| Preference optimization | trl + custom scripts | `apr align --method dpo/orpo` — integrated |

## 12. Data Strategy

The model is only as good as the fine-tuning data. Key datasets for code leaderboards:

| Dataset | Size | Purpose | Source | Format |
|---------|------|---------|--------|--------|
| Code Instruct (curated) | 50K | Instruction-following for code | Self-curated from OSS repos | JSONL (instruction, response) |
| Code Reasoning | 20K | Chain-of-thought for complex problems | Synthetic from teacher model | JSONL (problem, reasoning, code) |
| Code Tests | 10K | Test-driven examples (input→test→code) | HumanEval/MBPP-style | JSONL (prompt, tests, solution) |
| Multilingual Code | 30K | Python/Rust/TS/Go/Java coverage | MultiPL-E format | JSONL (language, prompt, solution) |
| Calibration | 128 | Wanda/SparseGPT calibration | Random code samples | JSONL (text) |

### 12.1 Decontamination Protocol

Training data MUST NOT overlap with evaluation benchmarks. This is critical for leaderboard integrity.

**n-gram decontamination:** Remove any training sample whose 10-gram overlap with any HumanEval/MBPP/BigCodeBench problem exceeds 50%. This is a hard gate — no exceptions.

```bash
# GATE: Decontamination check before training
apr validate --data training.jsonl --decontaminate \
    --reference humaneval.jsonl mbpp.jsonl bigcodebench.jsonl \
    --ngram 10 --threshold 0.50 --json > decontamination-report.json

# Verify <1% of training samples flagged
```

**Time-based decontamination for LiveCodeBench:** Any problem published within 90 days of training data generation is excluded. LiveCodeBench's rolling nature makes this mandatory.

### 12.2 Data Preparation Pipeline

```bash
# GATE: Validate teacher produces correct code BEFORE generating training data
apr eval teacher.apr --task classify --data humaneval.jsonl --json > teacher-baseline.json
# Verify teacher pass@1 meets minimum threshold (e.g., >60%) before proceeding

# Generate synthetic training data from validated teacher
apr chat teacher.apr --system "Generate code instruction pairs" \
    --batch instructions.txt --json > code-instruct-raw.jsonl

# Format validation
apr validate --data code-instruct-raw.jsonl --format jsonl

# Quality scoring (alimentar)
alimentar quality code-instruct-raw.jsonl --min-score 80 -o code-instruct-clean.jsonl

# Decontamination gate
apr validate --data code-instruct-clean.jsonl --decontaminate \
    --reference humaneval.jsonl mbpp.jsonl --ngram 10 --threshold 0.50
```

**Bootstrapping discipline:** Never generate training data from a teacher whose inference quality hasn't been verified. The pipeline is: import → eval teacher → generate data → validate data → decontaminate → train student.

## 13. Evaluation Protocol

Every recipe must be evaluated identically for fair comparison.

### 13.1 pass@k Computation

**Critical note on pass@k evaluation:** HumanEval and MBPP require *executing generated code* against test cases — not just token prediction. The pipeline is: (1) model generates k completions per problem, (2) completions are executed in a sandboxed environment, (3) pass@k is computed via the unbiased estimator.

The unbiased estimator for pass@k (Chen et al., 2021):

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where `n` = total completions generated, `c` = number that pass all tests, `k` = samples selected. This avoids biased estimation from sampling exactly k completions.

**apr-leaderboard eval flags that map to this:**

| Flag | Effect |
|---|---|
| `--samples N` | Number of benchmark problems to evaluate (0 = all) |
| `--n-samples N` | Completions per problem (for pass@k, best-of-N selection) |
| `--prompt-strategy S` | Prompt formatting (standard, scot, few-shot, cgo, reflexion) |

### 13.2 Code Execution Sandbox

aprender does not include a code execution sandbox. Generated completions must be evaluated externally via one of:

1. **EvalPlus harness** (recommended): Docker-based sandbox that runs Python completions against augmented test suites (80x more tests than vanilla HumanEval)
2. **Custom WASM sandbox**: CPython compiled to WASM for isolated execution (see Open Question §21.14)
3. **Direct Docker**: `docker run --network=none --memory=512m --timeout=10s python:3.11 -c "$CODE"`

### 13.3 Evaluation Steps

```bash
# Step 1: Perplexity baseline (pure inference, no code execution needed)
apr eval model.apr --dataset wikitext-2 --json > results/perplexity.json

# Step 2: Generate completions for code benchmarks
apr eval model.apr --task classify --data humaneval.jsonl --json > results/humaneval-completions.json
apr eval model.apr --task classify --data mbpp.jsonl --json > results/mbpp-completions.json

# Step 3: Execute completions in sandboxed environment (EXTERNAL)
# Using EvalPlus:
#   docker run -v ./results:/results evalplus/evalplus:latest \
#     --dataset humaneval --samples /results/humaneval-completions.json
# This produces pass@1, pass@10, pass@100 metrics.

# Step 4: Throughput benchmarking
apr bench model.apr --json > results/throughput.json

# Step 5: Cross-reference against HuggingFace
apr compare-hf model.apr --json > results/parity.json

# Step 6: Full QA gate before submission
apr qa model.apr --verbose
apr check model.apr
```

### 13.4 Evaluation Benchmarks (apr-leaderboard)

The `apr-leaderboard eval` command wraps the above steps for supported benchmarks:

```bash
# Run all HumanEval problems with 20 completions each, using structured CoT
apr-leaderboard eval --model models/qwen-7b.apr --benchmark humaneval \
    --samples 0 --n-samples 20 --prompt-strategy scot --output results/

# Quick sanity check: 10 problems, 1 completion each
apr-leaderboard eval --model models/qwen-7b.apr --benchmark humaneval --samples 10

# View results history
apr-leaderboard history --model qwen
```

## 14. Submission Flow

### 14.1 Leaderboard Targets

The `apr-leaderboard submit` command formats results for the target leaderboard's submission API:

| Leaderboard | Flag value | Submission method |
|---|---|---|
| Open LLM Leaderboard | `open-llm-leaderboard` (default) | HF Hub model upload → leaderboard evaluation queue |
| BigCodeBench | `bigcode` / `bigcodebench` | Direct result JSON submission |
| EvalPlus | `evalplus` | HF Hub model upload + EvalPlus-format results |

### 14.2 Submission Pipeline

```bash
# 1. Generate HuggingFace model card
apr eval final.apr --generate-card

# 2. Export to HuggingFace-compatible format
apr export final.apr --format safetensors -o submission/

# 3. Publish to HuggingFace Hub
apr publish submission/ --repo paiml/qwen-coder-7b-apr --private

# 4. Submit results via apr-leaderboard
apr-leaderboard submit --results results/humaneval_20260228.json \
    --model-id paiml/qwen-coder-7b-apr --leaderboard open-llm-leaderboard

# 5. Submit to leaderboard evaluation queue (via HF)
# The leaderboard pulls from your HF repo and runs its own evaluation
```

### 14.3 Model Card Template

The model card (`README.md` in the HF repo) MUST include:

- **Base model:** Qwen2.5-Coder-7B (with HF link)
- **Pipeline stages applied:** distill/finetune/merge/prune/quantize (which ones, in order)
- **Training data:** Summary with decontamination attestation
- **Evaluation results:** pass@1/pass@10 on HumanEval, MBPP, BigCodeBench
- **Infrastructure:** "Built with aprender (Rust, no Python dependencies)"
- **Quantization:** Scheme used, size reduction, quality impact
- **Reproducibility:** Link to pipeline config TOML

### 14.4 Pre-Submission Checklist

Before `apr-leaderboard submit`:

- [ ] `apr check model.apr` passes (format validation)
- [ ] `apr compare-hf model.apr` shows <5% parity gap
- [ ] `pmat comply check --strict` passes
- [ ] Decontamination report shows <1% n-gram overlap
- [ ] Model card generated and reviewed
- [ ] Results JSON includes all required benchmarks

## 15. Success Criteria

### 15.1 Primary Metrics

| Metric | Target | Stretch | Measurement | Notes |
|--------|--------|---------|-------------|-------|
| HumanEval pass@1 | ≥ apr baseline | ≥ HF reference | `apr-leaderboard eval --benchmark humaneval` | Relative to Step 0 baseline |
| MBPP pass@1 | ≥ apr baseline | ≥ HF reference | `apr-leaderboard eval --benchmark mbpp` | Relative to Step 0 baseline |
| BigCodeBench pass@1 | > 0 (eval works) | ≥ HF reference | `apr-leaderboard eval --benchmark bigcodebench` | Stretch: competitive |
| Inference parity | <5% gap vs HF | <2% gap vs HF | `apr compare-hf` | Perplexity gap on WikiText-2 |

### 15.2 Infrastructure Metrics

| Metric | Target | Stretch | Notes |
|--------|--------|---------|-------|
| Pipeline commands | ≤ 10 | ≤ 6 | Config-driven pipeline counts as 1 |
| Total binary size (compiled, 7B INT4) | < 5GB | < 4GB | 3.5GB weights + runtime |
| Wall-clock (import → submit) | < 24h (GPU) | < 8h (GPU) | CPU-only: much longer |
| Python dependencies | 0 | 0 | External sandbox for eval only |
| CUDA toolkit | Not required | Not required | trueno PTX generation handles GPU |
| GPU hardware | Recommended | Optional (≤7B) | Required for distill/finetune 32B teacher |

### 15.3 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test coverage | ≥ 95% | `cargo llvm-cov` |
| Clippy warnings | 0 | `cargo clippy -- -D warnings` |
| Source file size | < 500 lines each | `wc -l src/**/*.rs` |
| pmat comply | Pass | `pmat comply check --strict` |
| Contract binding coverage | ≥ 95% | `pv proof-status` |

### 15.4 Falsifiability

Every target above is falsifiable: it has a concrete measurement command, a numeric threshold, and a pass/fail outcome. If a metric cannot be measured, the spec has failed — not the implementation.

## 16. Provable Contracts (Design by Contract)

Every kernel in the pipeline MUST have a provable-contracts YAML contract binding it to its mathematical specification. This ensures the optimization techniques produce correct results, not just plausible ones.

### 16.1 Contract Coverage Requirements

The leaderboard pipeline touches these kernel equivalence classes from the provable-contracts registry:

| Kernel Class | Contracts Required | Pipeline Stage |
|---|---|---|
| **E (Qwen)** | RMSNorm, SwiGLU, GQA, RoPE | Inference (eval, distill, chat) |
| **Attention** | attention-kernel-v1, flash-attention-kernel-v1 | Inference, distillation |
| **Quantization** | quantization-ordering-v1, q4k-q6k-superblock-v1 | `apr quantize`, QLoRA base weights |
| **LoRA** | lora-algebra-v1 | `apr finetune --method lora/qlora` |
| **Softmax** | softmax-kernel-v1 | Attention, sampling |
| **Matmul** | matmul-kernel-v1 | All linear layers |
| **AdamW** | adamw-kernel-v1 | Training optimizer |

### 16.2 Contract Verification Gates

Each pipeline stage MUST pass its contract obligations before proceeding:

```bash
# Verify all kernel contracts are bound and implemented
pv proof-status ../provable-contracts/contracts/ \
    --binding ../provable-contracts/contracts/aprender/binding.yaml \
    --format json

# Verify Qwen2 architecture contracts specifically
pv audit ../provable-contracts/contracts/model/qwen35-shapes-v1.yaml \
    --binding ../provable-contracts/contracts/aprender/binding.yaml

# Run falsification tests for all pipeline-relevant kernels
cargo test --features kani -p aprender -- contract
```

### 16.3 Pipeline-Specific Proof Obligations

| Obligation | Property | Verification Level | Gate |
|---|---|---|---|
| PO-LB-001 | Distillation preserves architecture invariants | L2 (falsification) | Before `apr distill` |
| PO-LB-002 | Merge preserves tensor shape flow | L3 (proptest) | Before `apr merge` |
| PO-LB-003 | Prune maintains attention head structure | L2 (falsification) | Before `apr prune` |
| PO-LB-004 | Quantization ordering matches golden order §8 | L1 (type system) | Compile-time |
| PO-LB-005 | LoRA adapter rank ≤ hidden dim | L1 (Poka-Yoke) | Compile-time |
| PO-LB-006 | Q4K dequantize × quantize ≈ identity | L4 (Kani, bound=256) | CI |
| PO-LB-007 | Softmax normalization: sum(output) ≈ 1.0 | L4 (Kani, bound=16) | CI |
| PO-LB-008 | SLERP interpolation preserves weight norms | L3 (proptest) | Before `apr merge --strategy slerp` |

### 16.4 `#[contract]` Annotations

Every function in the apr-leaderboard pipeline that performs a mathematical operation MUST carry a `#[contract]` annotation linking it to its provable-contracts YAML:

```rust
use provable_contracts_macros::contract;

#[contract("quantization-ordering-v1", equation = "quantize_int4")]
pub fn quantize_model(model: &AprModel, scheme: QuantScheme) -> Result<AprModel> {
    // Implementation — contract macro enforces binding at compile time
}

#[contract("lora-algebra-v1", equation = "lora_forward")]
pub fn lora_forward(base: &Tensor, a: &Tensor, b: &Tensor, scale: f32) -> Tensor {
    // output = base @ x + scale * (B @ (A @ x))
}
```

If the binding is missing from `contracts/aprender/binding.yaml`, the build fails. Zero tolerance for unbound kernels.

## 17. Quality Gates (pmat comply)

Every pipeline step and every commit MUST pass the `pmat comply` quality gates. This is the enforcement mechanism for the claims in this spec.

### 17.1 Specification Compliance

This spec itself is validated by `pmat comply`:

```bash
# Score this specification (must achieve ≥95/100)
pmat spec score docs/specifications/leaderboard-spec.md --verbose

# Extract falsifiable claims and generate review checklist
pmat comply review docs/specifications/leaderboard-spec.md --format markdown

# Full compliance audit with signed evidence
pmat comply audit -o audit.json
```

### 17.2 Mandatory Pre-Commit Checks

```bash
# Full compliance check (blocks commit on failure)
pmat comply check --strict --format json

# Key checks enforced:
#   CB-200  TDG Grade Gate — no function below grade A
#   CB-303  Equation-Driven Development — contract bindings present
#   CB-125  Coverage quality — ≥95% with no exclusion gaming
#   CB-304  Dead code — 0% tolerance
#   CB-120  OIP Tarantula — no NaN, no unwrap in production paths
```

### 17.3 Pipeline Quality Gates

Each recipe step has a `pmat comply` gate:

| Pipeline Step | pmat Gate | Blocks On |
|---|---|---|
| Import | `apr check model.apr` + `pmat comply check` | Format validation failure, contract binding gaps |
| Distill | `pv proof-status` for attention/softmax contracts | Unverified kernel obligations |
| Finetune | `pmat comply check --strict` + coverage ≥95% | TDG regression, coverage drop |
| Merge | `pv audit` for merge strategy contracts | Unbound merge kernel |
| Prune | `apr eval` before/after + `pmat comply baseline` | Quality regression beyond threshold |
| Quantize | `pv proof-status` for Q4K/Q6K contracts | Kani proof failure |
| Eval | `pmat comply review` extracts claims → validates | Untested falsifiable claims |
| Submit | `pmat comply audit` signed evidence | Incomplete audit trail |

### 17.4 Cross-Crate Consistency

The sovereign stack (aprender, entrenar, trueno) MUST maintain cross-crate consistency:

```bash
# Detect API divergence and copy-paste duplication across stack
pmat comply cross-crate \
    --crates ../aprender ../entrenar ../trueno . \
    --similarity-threshold 0.80 \
    --strict

# Verify no contract drift between crates
pv diff ../provable-contracts/contracts/old/ ../provable-contracts/contracts/
```

### 17.5 Documentation Publishing

This specification is published as an [mdBook](https://rust-lang.github.io/mdBook/) via GitHub Actions. On every push to `main` that modifies `docs/` or `book.toml`, the workflow builds and deploys to GitHub Pages at:

> **https://paiml.github.io/apr-leaderboard/**

The mdBook source lives in `docs/src/` with chapters split from the canonical spec at `docs/specifications/leaderboard-spec.md`. The build output (`docs/book/`) is gitignored.

```bash
# Local preview
mdbook serve    # http://localhost:3000

# Build only
mdbook build    # outputs to docs/book/
```

## 18. Acceptance Criteria

Every criterion below is falsifiable. If any criterion cannot be demonstrated, this spec has failed.

- [ ] AC-001: `apr import hf://Qwen/Qwen2.5-Coder-7B` produces a valid `.apr` file that passes `apr check`
- [ ] AC-002: `apr eval` on imported model produces non-zero perplexity within 10% of HF reference
- [ ] AC-003: `apr distill` with progressive strategy produces a student model that outperforms the untrained student on perplexity
- [ ] AC-004: `apr finetune --method lora` completes training with decreasing loss curve
- [ ] AC-005: `apr finetune --method qlora` uses <50% VRAM compared to LoRA at equivalent rank
- [ ] AC-006: `apr merge --strategy slerp` preserves weight norms (L2 norm within 5% of inputs)
- [ ] AC-007: `apr merge --strategy ties` resolves sign conflicts (merged model has fewer conflicting task vectors than input sum)
- [ ] AC-008: `apr prune --method wanda` at conservative ratio degrades perplexity by <5%
- [ ] AC-009: `apr quantize --scheme int4` produces model <50% size of FP16 original
- [ ] AC-010: `apr compile` produces a standalone binary that runs inference without external dependencies
- [ ] AC-011: Full pipeline (Recipe C) completes end-to-end without manual intervention
- [ ] AC-012: `pv proof-status` shows ≥95% binding coverage for pipeline-relevant contracts
- [ ] AC-013: `pmat comply check --strict` passes with zero failures on the final submission
- [ ] AC-014: `apr compare-hf` shows <5% parity gap on perplexity for imported Qwen models
- [ ] AC-015: All falsification tests in provable-contracts pass for Kernel Class E (Qwen)
- [ ] AC-016: Training data has <1% n-gram overlap with HumanEval/MBPP test cases (`apr validate --decontaminate`)
- [ ] AC-017: `apr eval --n-samples 20` generates 20 distinct completions per problem (not duplicates)
- [ ] AC-018: Speculative decoding (`apr run --speculative`) achieves ≥1.5x throughput over standard decoding
- [ ] AC-019: `apr eval --prompt-strategy scot` produces structured reasoning before code output
- [ ] AC-020: `apr align --method dpo` reduces loss on preference pairs over 3 epochs
- [ ] AC-021: Qwen2.5-Coder-7B-Instruct imported via `apr import` achieves ≥85% HumanEval pass@1 (apr-native baseline ≥ HF reference - 5%)
- [ ] AC-022: Full pipeline on Qwen2.5-Coder-7B produces a model scoring ≥85% HumanEval, ≥82% HumanEval+, ≥80% MBPP
- [ ] AC-023: INT4 quantized model loses <2% pass@1 vs FP16 on HumanEval
- [ ] AC-024: Merged model (TIES of code-specialist + reasoning-specialist) scores ≥ best input specialist on at least one benchmark
- [ ] AC-025: `alimentar quality` scores all training data ≥80/100 before use in fine-tuning
- [ ] AC-026: `apr compile` of Qwen2.5-Coder-1.5B INT4 produces a binary <1GB that generates valid Python code
- [ ] AC-027: Every tooling gap in §5 has either a wire-in implementation or a documented external boundary

## 19. Implementation Status

Tracking table mapping spec sections to `apr-leaderboard` code implementation. Updated as code lands.

### 19.1 CLI Subcommands (§6.2)

| Subcommand | Source Module | Status | Tests | Notes |
|---|---|---|---|---|
| `convert` | `src/convert/mod.rs` | ✅ Scaffolded | 13 | HF download + APR v2 bundle |
| `eval` | `src/eval/mod.rs` | ✅ Scaffolded | 25 | pass@k metrics, prompt strategies, n-samples |
| `finetune` | `src/finetune/mod.rs` | ✅ Scaffolded | 10 | LoRA/QLoRA config + entrenar integration |
| `distill` | `src/optimize/mod.rs` | ✅ Scaffolded | 6 | Knowledge distillation (3 strategies) |
| `merge` | `src/optimize/mod.rs` | ✅ Scaffolded | 6 | Model merging (4 strategies) |
| `prune` | `src/optimize/mod.rs` | ✅ Scaffolded | 6 | Pruning (3 methods) |
| `quantize` | `src/optimize/mod.rs` | ✅ Scaffolded | 4 | Quantization (5 schemes) |
| `compare` | `src/optimize/mod.rs` | ✅ Scaffolded | 2 | HF parity check |
| `submit` | `src/submit/mod.rs` | ✅ Scaffolded | 12 | HF leaderboard submission |
| `benchmarks` | `src/harness/mod.rs` | ✅ Complete | 20+ | 10 benchmark definitions |
| `history` | `src/eval/mod.rs` | ✅ Complete | 3 | Result history viewer |
| `pipeline` | `src/pipeline/mod.rs` | ✅ Scaffolded | 10 | Config-driven TOML pipeline |

### 19.2 Prompt Strategies (§8.3)

| Strategy | Enum Variant | Aliases | Status |
|---|---|---|---|
| Standard | `PromptStrategy::Standard` | `default` | ✅ Implemented |
| Structured CoT | `PromptStrategy::SCoT` | `structured-cot` | ✅ Implemented |
| Few-shot | `PromptStrategy::FewShot` | `fewshot` | ✅ Implemented |
| Code Gen Opt | `PromptStrategy::Cgo` | `code-gen-opt` | ✅ Implemented |
| Reflexion | `PromptStrategy::Reflexion` | `reflect` | ✅ Implemented |

### 19.3 Optimization Operations (§7)

| Operation | Strategy/Method Enums | Validation | Status |
|---|---|---|---|
| Distill | `Standard`, `Progressive`, `Ensemble` | Empty path check | ✅ Scaffolded |
| Merge | `Slerp`, `Ties`, `Dare`, `LinearAvg` | Min 2 models, empty path check | ✅ Scaffolded |
| Prune | `Wanda`, `Magnitude`, `SparseGpt` | Ratio 0.0–1.0, empty path check | ✅ Scaffolded |
| Quantize | `Int4`, `Int8`, `Q4K`, `Q5K`, `Q6K` | Empty path check | ✅ Scaffolded |

### 19.4 Quality Metrics

| Metric | Current | Target | Gate |
|---|---|---|---|
| Test count | 131 | — | `cargo test` |
| Line coverage | 96.5% | ≥ 95% | `cargo llvm-cov` |
| Clippy warnings | 0 | 0 | `cargo clippy -- -D warnings` |
| Max file size | 491 lines | < 500 | `wc -l src/**/*.rs` |
| pmat pre-commit | ✅ Pass | ✅ Pass | git hook |

### 19.5 What "Scaffolded" Means

**Scaffolded** = CLI parsing, strategy/method enums, input validation, result serialization, and test coverage are all implemented. The actual ML operations (inference, training, merging) are delegated to upstream `apr` CLI calls which are currently printed but not executed. Wiring to real `apr` subprocess calls is tracked by PMAT-017.

## 20. Scientific Foundation (References)

[1] Sun et al., "A Simple and Effective Pruning Approach for Large Language Models" (Wanda), ICLR 2024.

[2] Yadav et al., "TIES-Merging: Resolving Interference When Merging Models", NeurIPS 2023.

[3] Yu et al., "Language Model is Sometimes a Knowledge Base" (DARE), arXiv:2311.03099, 2023.

[4] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models", NeurIPS 2023.

[5] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.

[6] Hinton et al., "Distilling the Knowledge in a Neural Network", arXiv:1503.02531, 2015.

[7] Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot", ICML 2023.

[8] Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ICLR 2023.

[9] Li et al., "Structured Chain-of-Thought Prompting for Code Generation", ACM TOSEM 2025.

[10] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023.

[11] Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model", EMNLP 2024.

[12] Rozière et al., "Code Llama: Open Foundation Models for Code", arXiv:2308.12950, 2023.

[13] Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023.

[14] Hui et al., "Qwen2.5-Coder Technical Report", arXiv:2409.12186, 2024.

[15] Jain et al., "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code", arXiv:2403.07974, 2024.

[16] Zhuo et al., "BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions", arXiv:2406.15877, 2024.

[17] NVIDIA, "OpenCodeReasoning: Advancing Data Distillation for Competitive Coding", arXiv:2504.01943, 2025.

[18] Goddard et al., "Arcee's MergeKit: A Toolkit for Merging Large Language Models", arXiv:2403.13257, 2024.

## 21. Open Questions

1. **Calibration data quality:** How much does Wanda calibration data selection affect code model pruning? Need ablation study.
2. **Merge tournament depth:** Is 2-round merging sufficient or do 3+ rounds compound gains?
3. **Distillation data volume:** What's the minimum code corpus size for progressive (curriculum) distillation to outperform standard KL?
4. **HPO budget:** Is 20-trial TPE scout sufficient to find good LoRA hyperparameters for code?
5. **Quantization floor:** At what pass@1 threshold does INT4 quantization degrade code generation quality measurably?
6. **Cross-architecture distillation:** Can we distill Qwen-32B into a different architecture (e.g., smaller custom model)?
7. **Inference parity gap:** What is the actual pass@1 gap between apr-native inference and PyTorch/HF for Qwen2.5-Coder models? This gates all absolute target setting.
8. **Code execution sandbox:** Should apr integrate a WASM-based sandbox for pass@k evaluation, or is external EvalPlus harness sufficient?
9. **CPU-only distillation feasibility:** Is progressive distillation from a 32B teacher on CPU practical within the 24h wall-clock budget, even with trueno SIMD? Likely needs GPU.
10. **Reasoning distillation transfer:** Does distilling from DeepSeek-R1 (or OCR-Nemotron) into Qwen2.5-Coder backbone require architecture adaptation, or does progressive distillation handle the mismatch?
11. **DPO data volume:** How many preference pairs are needed for measurable HumanEval+ improvement? Initial estimate: 5K-10K pairs.
12. **Merge across training regimes:** Can we TIES-merge a code-instruct model with a reasoning-distilled model effectively, given they were trained with different objectives?
13. **LiveCodeBench contamination window:** LiveCodeBench refreshes continuously. What's the minimum lag between problem publication and safe inclusion in training data?
14. **WASM sandbox for Python:** Is CPython-in-WASM viable for pass@k evaluation at scale (164-974 problems × N=50 completions × timeout per completion)?
