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
