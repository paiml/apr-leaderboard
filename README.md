<p align="center">
  <img src="docs/hero.svg" alt="apr-leaderboard pipeline" width="100%"/>
</p>

# apr-leaderboard

HuggingFace leaderboard pipeline for the sovereign Rust AI stack. Proves that a single `apr` binary — with zero Python, zero CUDA toolkit — can compete on code generation benchmarks (HumanEval, MBPP, BigCodeBench).

**[Read the full specification](https://paiml.github.io/apr-leaderboard/)**

## What This Proves

One falsifiable question:

> Can a single Rust binary (`apr`) match Python-ecosystem HumanEval/MBPP scores for Qwen2.5-Coder-7B, with zero Python dependencies?

If yes: [aprender](https://github.com/paiml/aprender), [entrenar](https://github.com/paiml/entrenar), and [trueno](https://github.com/paiml/trueno) work end-to-end as a sovereign AI stack.

If no: `apr compare-hf` pinpoints exactly where the stack falls short.

## Pipeline

```
apr import → apr distill → apr finetune → apr merge → apr prune → apr quantize → eval → apr publish
```

Every command is provided by the `apr` CLI (aprender). This repo provides the pipeline config, benchmark metadata, result persistence, and the strategy spec.

## Quick Start

```bash
# Verify apr CLI is available
make verify

# Import a model from HuggingFace
make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

# Evaluate on HumanEval
make eval-humaneval CHECKPOINT=checkpoints/qwen_qwen2.5-coder-7b-instruct.apr

# Run a full recipe pipeline
make pipeline RECIPE=recipe-a-quick-lora

# Dry-run a pipeline (validate config, show commands)
make pipeline-plan RECIPE=recipe-c-full-pipeline
```

## Sovereign Stack

| Crate | Role | Version |
|-------|------|---------|
| [aprender](https://crates.io/crates/aprender) | .apr format, inference, distillation, merging, pruning, quantization | 0.27 |
| [entrenar](https://crates.io/crates/entrenar) | LoRA/QLoRA training, autograd, AdamW, gradient checkpointing | 0.7 |
| [trueno](https://crates.io/crates/trueno) | SIMD tensor ops (AVX2/NEON), wgpu GPU, PTX generation | 0.16 |

## Benchmarks Supported

| Benchmark | Problems | Metric | Source |
|-----------|----------|--------|--------|
| HumanEval | 164 | pass@1 | OpenAI |
| HumanEval+ | 164 | pass@1 | EvalPlus |
| MBPP | 974 | pass@1 | Google Research |
| MBPP+ | 399 | pass@1 | EvalPlus |
| BigCodeBench | 1,140 | pass@1 | BigCode Project |
| LiveCodeBench | 500 | pass@1 | LiveCodeBench |
| MultiPL-E | 164 | pass@1 | 18 languages |
| DS-1000 | 1,000 | pass@1 | Data science |
| SWE-bench Lite | 300 | resolve_rate | GitHub issues |
| CRUXEval | 800 | pass@1 | I/O prediction |

## Project Structure

```
apr-leaderboard/
├── Makefile                    # All orchestration (top-level entry point)
├── scripts/
│   ├── import.sh               # HF model download + convert to .apr
│   ├── eval-pass-at-k.sh       # Generate completions → execute → score
│   ├── pipeline.sh             # Run a full recipe from TOML config
│   └── submit.sh               # Export + publish to HF Hub
├── configs/
│   ├── models/                 # Per-model configs (5 models)
│   │   ├── qwen-coder-7b.toml
│   │   ├── qwen-coder-1.5b.toml
│   │   ├── qwen-coder-32b.toml
│   │   ├── deepseek-r1-distill-7b.toml
│   │   └── phi-4.toml
│   └── recipes/                # Multi-stage pipeline recipes (4 recipes)
│       ├── recipe-a-quick-lora.toml
│       ├── recipe-b-merge-alchemist.toml
│       ├── recipe-c-full-pipeline.toml
│       └── recipe-d-sovereign-binary.toml
├── contracts/
│   └── pass-at-k.yaml          # Formal pass@k metric contract
├── checkpoints/                # .apr model files (gitignored)
├── results/                    # Evaluation result JSONs
├── data/                       # Training/calibration data (gitignored)
└── docs/                       # Specification (mdBook)
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make verify` | Check `apr` CLI available, all subcommands work |
| `make dogfood` | End-to-end smoke test |
| `make import MODEL=...` | Download HF model → .apr |
| `make eval-humaneval CHECKPOINT=...` | HumanEval pass@k evaluation |
| `make eval-mbpp CHECKPOINT=...` | MBPP pass@k evaluation |
| `make eval-all CHECKPOINT=...` | All benchmarks |
| `make finetune CHECKPOINT=...` | LoRA/QLoRA fine-tuning |
| `make merge MODELS="a.apr b.apr"` | Model merging (SLERP/TIES/DARE) |
| `make prune CHECKPOINT=...` | Pruning (Wanda/magnitude) |
| `make quantize CHECKPOINT=...` | Quantization (INT4/INT8/FP16) |
| `make distill TEACHER=... STUDENT=...` | Knowledge distillation |
| `make pipeline RECIPE=...` | Run a multi-stage recipe |
| `make pipeline-plan RECIPE=...` | Dry-run: show commands |
| `make publish CHECKPOINT=... HF_REPO=...` | Publish to HF Hub |

## Specification

The full specification is published as an [mdBook](https://paiml.github.io/apr-leaderboard/) via GitHub Actions. It covers:

- **§1** What this repo does and how it relates to aprender
- **§5** Technique playbook (distillation, merging, pruning, LoRA, quantization)
- **§6** Leaderboard-winning inference techniques (N-sampling, SCoT, speculative decoding, DPO)
- **§7** Composite recipes (4 end-to-end strategies)
- **§8** Technique interaction matrix and golden ordering
- **§14** Provable contracts (design by contract with Kani proofs)
- **§15** Quality gates (pmat comply enforcement)
- **§16** 20 falsifiable acceptance criteria

## License

MIT
