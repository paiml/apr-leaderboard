# What This Repo Does

## 1.1 Purpose

**apr-leaderboard** is a pipeline harness that proves the [sovereign AI stack](https://github.com/paiml) — aprender, entrenar, trueno — can compete on HuggingFace code generation leaderboards (HumanEval, MBPP, BigCodeBench) without Python, without the HuggingFace Transformers library, and without an external CUDA toolkit.

It is **not** a model training framework. It is **not** a general ML toolkit. It is a thin orchestration layer (~1,400 lines of Rust) that wires the sovereign stack's existing capabilities into a reproducible, config-driven leaderboard pipeline:

```
apr import → apr distill → apr finetune → apr merge → apr prune → apr quantize → apr eval → apr submit
```

Every command above is provided by **aprender** (`apr` CLI). This repo provides the pipeline config, benchmark metadata, result persistence, and the spec that defines the strategy.

## 1.2 What It Proves

This repo exists to answer one falsifiable question:

> **Can a single Rust binary (`apr`) match Python-ecosystem HumanEval/MBPP scores for Qwen2.5-Coder-7B, with zero Python dependencies?**

If the answer is yes, it proves:
1. **aprender** can import, infer, and evaluate HuggingFace models via the `.apr` format
2. **entrenar** can fine-tune those models with LoRA/QLoRA using its own autograd engine
3. **trueno** can run transformer attention at competitive throughput via SIMD/PTX
4. The full distill → finetune → merge → prune → quantize pipeline works end-to-end in pure Rust
5. **provable-contracts** kernel verification (Kani bounded model checking) doesn't prevent competitive performance — correctness and speed coexist

If the answer is no, it identifies exactly where the sovereign stack falls short (inference parity gap, training convergence, quantization quality loss) via `apr compare-hf`.

## 1.3 How It Relates to aprender

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

## 1.4 Current Implementation Status

**All 21 CLI subcommands are wired to real sovereign stack APIs.** Every operation produces valid APR v2 files that pass `check` validation end-to-end.

| Module | Status | What's wired |
|---|---|---|
| **convert/** | **Wired** | `aprender::format::v2::{AprV2Writer, AprV2Metadata}` + LZ4 + 4 quant formats |
| **eval/** | **Wired** | `entrenar::eval::pass_at_k` (Chen et al. estimator), 5 prompt strategies, N-sampling |
| **finetune/** | **Wired** | `entrenar::lora::{LoRALayer, QLoRALayer}`, `AdamW`, `WarmupCosineDecayLR`, adapter merge |
| **distill/** | **Wired** | `entrenar::distill::{DistillationLoss, ProgressiveDistiller, EnsembleDistiller}` |
| **merge/** | **Wired** | `entrenar::merge::{slerp_merge, ensemble_merge}` + APR v2 I/O via `apr_bridge` |
| **prune/** | **Wired** | `aprender::pruning::MagnitudeImportance` + `entrenar::prune::{PruningConfig, PruneFinetunePipeline}` |
| **quantize/** | **Wired** | `entrenar::quant::{Calibrator, quantize_tensor, quantization_mse}` |
| **align/** | **Wired** | `entrenar::train::{BCEWithLogitsLoss, CrossEntropyLoss, LossFn}` + DPO/ORPO preference loss |
| **validate/** | **Wired** | N-gram fingerprinting via `HashSet` + `harness::get_benchmark` integration |
| **inference/** | **Wired** | `entrenar::train::{CrossEntropyLoss, LossFn}` + token log-probs + temperature scaling |
| **compile/** | **Wired** | `aprender::format::v2::AprV2Reader` pre-compilation validation |
| **check/** | **Wired** | `aprender::format::v2::AprV2Reader` full validation |
| **acceptance/** | **Wired** | `provable_contracts::schema::{parse_contract, validate_contract}`, 27 ACs |
| **compare/** | **Wired** | `apr_bridge::load_apr_as_merge_model` + per-tensor weight statistics |
| **tune/** | **Wired** | `entrenar::train::CrossEntropyLoss` HPO trials + APR v2 I/O via `apr_bridge` |
| **submit/** | **Wired** | `aprender::format::v2::AprV2Reader` pre-submit validation + model card gen |
| **pipeline/** | **Wired** | 12-stage orchestration — all stages call wired backends |
| **export/** | **Wired** | `aprender::format::v2::AprV2Reader` tensor index export + metadata |
| **harness/** | Complete | All 10 benchmark definitions with metadata |

**Quality:** 368 tests, 0 clippy warnings, 96.1% line coverage, all files ≤500 lines, 13 source modules.

## 1.5 How People Use It

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
