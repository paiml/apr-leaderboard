# CLAUDE.md

## Project Overview

APR Leaderboard is a thin orchestration layer over the `apr` CLI for building, evaluating, and submitting `.apr` models to HuggingFace leaderboards. Zero Rust — all computation is delegated to `apr` (aprender). Primary focus: Qwen-Coder models on coding benchmarks (HumanEval, MBPP, BigCodeBench, LiveCodeBench).

## Architecture

```
Makefile (orchestration)
├── scripts/import.sh         → apr import
├── scripts/eval-pass-at-k.sh → apr run + sandbox execution + scoring
├── scripts/pipeline.sh       → reads recipe TOML, runs stages in order
├── scripts/submit.sh         → apr export + apr publish
├── configs/models/            → per-model TOML configs
└── configs/recipes/           → multi-stage pipeline recipes
```

## Build Commands

```bash
make verify                                          # check apr CLI available
make dogfood                                         # end-to-end smoke test
make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct    # download + convert to .apr
make eval-humaneval CHECKPOINT=checkpoints/model.apr  # run HumanEval pass@k
make pipeline RECIPE=recipe-a-quick-lora              # run a full recipe pipeline
```

## Pipeline Flow

```
apr import → [apr distill] → [apr finetune] → [apr merge] → [apr prune] → [apr quantize] → eval → [apr publish]
```

Every command is provided by the `apr` CLI (aprender). This repo provides pipeline config, benchmark metadata, result persistence, and the strategy spec.

## Key Dependencies

- `apr` CLI v0.4.10+ (aprender) — all model operations
- `bash` — shell scripts
- `python3` — TOML parsing, benchmark data processing
- `curl` — HF Hub connectivity checks

## Quality Gates

```bash
make verify    # check apr CLI + all configs valid
make dogfood   # exercise all subcommands + validate configs
```
