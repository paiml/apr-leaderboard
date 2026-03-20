# What This Repo Does

## 1.1 Purpose

**apr-leaderboard** is a pipeline harness that proves the [sovereign AI stack](https://github.com/paiml) — aprender, entrenar, trueno — can compete on HuggingFace code generation leaderboards (HumanEval, MBPP, BigCodeBench) without Python, without the HuggingFace Transformers library, and without any CUDA toolkit or GPU vendor lock-in.

It is **not** a model training framework. It is **not** a general ML toolkit. It is a thin orchestration layer — a Makefile, five shell scripts, YAML configs, a batuta playbook, and a forjar infrastructure manifest — that wires the sovereign stack's existing capabilities into a reproducible, config-driven leaderboard pipeline:

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
3. **trueno** can run transformer attention at competitive throughput via SIMD (CPU) and wgpu (any GPU)
4. The full distill → finetune → merge → prune → quantize pipeline works end-to-end in pure Rust — on any GPU vendor
5. **provable-contracts** kernel verification (Kani bounded model checking) doesn't prevent competitive performance — correctness and speed coexist

If the answer is no, it identifies exactly where the sovereign stack falls short (inference parity gap, training convergence, quantization quality loss) via `apr compare-hf`.

## 1.3 How It Relates to aprender

```
┌──────────────────────────────────────────────────────────┐
│                    apr-leaderboard                        │
│                                                          │
│  Makefile           YAML configs        Shell scripts    │
│  (dev convenience)  (models/recipes/   (10 scripts)     │
│                      eval/pipeline)                      │
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
│  │  │ LoRA    │  │  SIMD     │  │contracts│         │  │  │
│  │  │ QLoRA   │  │  AVX2/NEON│  │ Kani    │         │  │  │
│  │  │ AdamW   │  │  wgpu GPU │  │ L1-L4   │         │  │  │
│  │  │ autograd│  │  Q4K/Q6K  │  │ proofs  │         │  │  │
│  │  └─────────┘  └──────────┘  └─────────┘         │  │  │
│  └──────────────────────────────────────────────────┘  │  │
│                                                        │  │
└────────────────────────────────────────────────────────┘  │
                                                            │
   pmat comply ◄───── quality gate ─────────────────────────┘
```

**apr-leaderboard does NOT reimplement aprender.** It calls `apr` subcommands via Makefile targets and shell scripts. The relationship is:

| Layer | Repo | Responsibility |
|---|---|---|
| **Orchestration** | apr-leaderboard | Makefile targets, shell scripts, pipeline configs, benchmark metadata, result tracking, strategy spec |
| **ML Operations** | aprender (apr CLI) | Model import, inference, eval, distillation, merging, pruning, quantization |
| **Training** | entrenar | LoRA/QLoRA, autograd, optimizers, gradient checkpointing |
| **Compute** | trueno | SIMD tensor ops, wgpu GPU kernels, quantized matmul |
| **Correctness** | provable-contracts | Kernel contracts, Kani proofs, falsification tests |
| **Quality** | pmat comply | Compliance checks, spec scoring, cross-crate consistency |

## 1.4 Current Implementation Status

All orchestration is implemented via Makefile + shell scripts. Every `make` target calls real `apr` CLI subcommands.

| Component | Status | What It Does |
|---|---|---|
| **Makefile** | **Working** | Dev convenience: import, finetune, merge, prune, quantize, distill, compile, eval-*, export, publish, pipeline, verify, validate, dogfood, prove-wgpu |
| **scripts/eval-pass-at-k.sh** | **Working** | Downloads benchmark data, generates completions via `apr run`, executes in sandbox, computes pass@k |
| **scripts/pipeline.sh** | **Working** | Parses recipe YAML (bash-native, zero Python), runs stages sequentially, supports `--plan` dry-run and explicit `stages:` list |
| **scripts/submit.sh** | **Working** | Exports to SafeTensors, generates model card, publishes to HF Hub with dry-run confirmation |
| **scripts/import.sh** | **Working** | Wraps `apr import` with HF Hub reachability check and `apr check` validation |
| **scripts/prove-wgpu.sh** | **Working** | End-to-end wgpu training proof: import → QLoRA train → verify GPU backend |
| **configs/models/** | **Complete** | 6 YAML model configs (Qwen-7B, Qwen-32B, Qwen-1.5B, Qwen3-8B, DeepSeek-R1-7B, Phi-4) |
| **configs/recipes/** | **Complete** | 7 YAML recipe configs (quick-lora, merge-alchemist, full-pipeline, sovereign-binary, instruct-finetune, qwen3-qlora, wgpu-proof) |
| **configs/eval/** | **Complete** | Eval suite YAML with benchmark definitions, targets, and baselines |
| **configs/pipeline/** | **Complete** | Forjar infra manifest + batuta playbook DAG |
| **data_catalog.yaml** | **Complete** | Data governance: datasets, lineage, classification, lifecycle |
| **docs/** | **Complete** | Strategy spec (mdbook), 24 sections covering full pipeline |

**Quality:** All 19 YAML configs valid (`make validate`), 10 scripts, 19/19 `apr` subcommands verified. Real model import and inference tested with Qwen2.5-Coder-1.5B, 7B, 32B, and Qwen3-4B. Zero Python scripts. Zero TOML configs (migrated to YAML). Chen et al. unbiased pass@k estimator. 5 prompt strategies (standard, scot, few-shot, cgo, default). Best results: HumanEval **87.20%** (7B few-shot), MBPP **76.20%** (7B + test assertions).

**GPU sharing infrastructure:** 143 tests across 9 entrenar modules (VRAM guard, ledger, wait queue, profiler, MPS, cluster config, placement, coordinator, multi-adapter pipeline). See §22 for details.

## 1.5 How People Use It

**For leaderboard competitors:**

```bash
# 1. Verify the pipeline
make verify

# 2. Import a model from HuggingFace
make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

# 3. Evaluate on benchmarks
make eval-humaneval CHECKPOINT=checkpoints/qwen_qwen2.5-coder-7b-instruct.apr
make eval-all CHECKPOINT=checkpoints/qwen_qwen2.5-coder-7b-instruct.apr

# 4. Optimize (quantize, prune, merge, etc.)
make quantize CHECKPOINT=checkpoints/base.apr SCHEME=int4
make prune CHECKPOINT=checkpoints/base.apr PRUNE_METHOD=wanda SPARSITY=0.5

# 5. Run a full recipe pipeline
make pipeline RECIPE=recipe-a-quick-lora

# 6. Submit to HuggingFace Hub
make publish CHECKPOINT=checkpoints/model.apr HF_REPO=org/model-name
```

**For sovereign stack developers:**

This repo is an integration test for the sovereign stack. If `make pipeline` produces competitive scores, the stack works. If it doesn't, the per-step eval results pinpoint the weak component.

```bash
# Run baseline parity check
make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
apr run checkpoints/qwen_qwen2.5-coder-7b-instruct.apr \
    --prompt "def fibonacci(n):" --max-tokens 256 --no-gpu
apr eval checkpoints/qwen_qwen2.5-coder-7b-instruct.apr --dataset wikitext-2
apr bench checkpoints/qwen_qwen2.5-coder-7b-instruct.apr --json
```

**For researchers:**

The spec (this document) is the experimental protocol. The recipes in §9 are reproducible experiments. The acceptance criteria in §18 are the pass/fail conditions. Run them, report results, falsify or validate the thesis.
