# CLAUDE.md

## Project Overview

APR Leaderboard is a thin orchestration layer over the `apr` CLI for building, evaluating, and submitting `.apr` models to HuggingFace leaderboards. Zero Rust, zero Python — all computation is delegated to `apr` (aprender) and sovereign stack tooling. Primary focus: Qwen-Coder models on coding benchmarks (HumanEval, MBPP, BigCodeBench, LiveCodeBench).

## Architecture

```
Makefile (dev convenience)
├── scripts/import.sh         → apr import
├── scripts/eval-pass-at-k.sh → apr run + jq + awk scoring
├── scripts/pipeline.sh       → reads recipe YAML, runs stages in order
├── scripts/submit.sh         → apr export + apr publish
├── scripts/prove-wgpu.sh     → wgpu training proof
├── configs/models/            → per-model YAML configs
├── configs/recipes/           → multi-stage pipeline recipes (YAML)
├── configs/eval/              → benchmark evaluation suite (YAML)
├── configs/pipeline/          → forjar manifest + batuta playbook (YAML)
└── data_catalog.yaml          → data governance + lineage
```

## Build Commands

```bash
make verify                                          # check apr CLI available
make validate                                        # lint all configs (bashrs)
make dogfood                                         # end-to-end smoke test
make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct    # download + convert to .apr
make eval-humaneval CHECKPOINT=checkpoints/model.apr  # run HumanEval pass@k
make pipeline RECIPE=recipe-a-quick-lora              # run a full recipe pipeline
make prove-wgpu                                       # prove wgpu GPU training works
```

## Pipeline Flow

```
apr import → [apr distill] → [apr finetune] → [apr merge] → [apr prune] → [apr quantize] → eval → [apr publish]
```

Every command is provided by the `apr` CLI (aprender). This repo provides pipeline config, benchmark metadata, result persistence, and the strategy spec.

## Key Dependencies

- `apr` CLI v0.4.10+ (aprender) — all model operations
- `bashrs` — shell/config linting, Makefile validation
- `bash` — shell scripts (YAML parsing is bash-native, zero Python)
- `jq` — JSON processing in eval/submit scripts
- `curl` — HF Hub connectivity checks

## Constraints

- **Zero Python.** All config parsing, data processing, and orchestration uses sovereign stack tooling (`apr`, `bashrs`, `alimentar`) or bash builtins. No `python3` calls anywhere.
- **Zero CUDA.** GPU compute via wgpu (Vulkan/Metal/DX12). No CUDA toolkit, no nvcc, no vendor lock-in.
- **YAML-first.** All configs are YAML (albor pattern). Legacy TOML configs retained for backward compatibility only.

## Quality Gates

```bash
make verify    # check apr CLI + all subcommands
make validate  # bashrs lint all YAML configs + shell scripts + Makefile
make dogfood   # exercise all subcommands + validate configs (zero Python)
```
