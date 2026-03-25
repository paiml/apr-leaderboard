# CLAUDE.md

## Project Overview

APR Leaderboard is a thin orchestration layer over the `apr` CLI for building, evaluating, and submitting `.apr` models to HuggingFace leaderboards. Zero Rust, zero Python — all computation is delegated to `apr` (aprender) and sovereign stack tooling. Primary focus: Qwen-Coder models on coding benchmarks (HumanEval, MBPP, BigCodeBench, LiveCodeBench).

## Architecture

```
Makefile (45 targets)
├── scripts/import.sh           → apr import + validation
├── scripts/eval-pass-at-k.sh   → apr run + sandbox + Chen et al. pass@k
├── scripts/pipeline.sh         → reads recipe YAML, runs stages in order
├── scripts/submit.sh           → preflight checks + apr export + apr publish
├── scripts/prove-wgpu.sh       → wgpu training proof
├── scripts/download-benchmarks.sh → HumanEval/MBPP data
├── scripts/results-history.sh  → eval results viewer
├── scripts/eval-sweep.sh       → run eval across prompt strategies
├── scripts/compare-results.sh  → per-problem delta analysis
├── scripts/leaderboard-summary.sh → ranked markdown leaderboard
├── configs/models/              → per-model YAML configs
├── configs/recipes/             → multi-stage pipeline recipes (YAML)
├── configs/eval/                → benchmark evaluation suite (YAML)
├── configs/pipeline/            → forjar manifest + batuta playbook (YAML)
└── data_catalog.yaml            → data governance + lineage
```

## Build Commands

```bash
make verify                                                        # check apr CLI available
make validate                                                      # lint all configs (bashrs)
make dogfood                                                       # end-to-end smoke test
make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct                  # download + convert to .apr
make eval-humaneval CHECKPOINT=checkpoints/model.apr               # run HumanEval pass@k
make eval-humaneval CHECKPOINT=m.apr PROMPT_STRATEGY=scot           # structured CoT prompting
make pipeline RECIPE=recipe-a-quick-lora                           # run a full recipe pipeline
make results-history                                               # view eval results
make prove-wgpu                                                    # prove wgpu GPU training works
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

- **Zero Python in pipeline.** Config parsing, data processing, and orchestration use sovereign stack tooling (`apr`, `bashrs`, `alimentar`) or bash builtins. The only `python3` usage is sandbox execution in eval (external boundary per §5.3).
- **No GPU vendor lock-in.** GPU compute via wgpu (Vulkan/Metal/DX12) or optional CUDA backend. No vendor lock-in.
- **YAML-only.** All configs are YAML (albor pattern). Legacy TOML configs have been removed.

## Toyota Way: No Workarounds

**NEVER bypass a failing gate, check, or contract.** When something fails:
1. **Stop the line** — do not continue with a workaround
2. **Five-whys** — trace to root cause at the code/kernel level
3. **Fix the code** — not the gate, not with env vars, not with CPU fallback
4. **Verify** — the original gate/check must pass after the fix

**Forbidden patterns:**
- `SKIP_PARITY_GATE=1` — fix the GPU kernel instead
- `FP8_PREFILL=0 FP8_DECODE=0` — fix the FP8 detection code instead
- `CUDA_VISIBLE_DEVICES=""` — fix GPU, don't hide it
- Relaxing `AllImplemented` contract policy — implement the bindings
- "use CPU fallback" / "use worker mode" — fix the GPU/batch code

**Current blocker:** GPU inference on Blackwell sm_121 produces wrong tokens. F2 parity gate correctly blocks it. Root cause is in trueno-gpu CUDA kernels. Must be fixed there, not bypassed.

## Quality Gates

```bash
make verify    # check apr CLI + all subcommands
make validate  # bashrs lint all YAML configs + shell scripts + Makefile
make dogfood   # exercise all subcommands + validate configs (zero Python)
```

## Code Search

Use `pmat query` for semantic code search with quality annotations and `--faults`
for fault-pattern detection. NEVER use grep for code exploration — use `pmat query`
instead, which provides AST-aware results with TDG grades and defect annotations.

```bash
pmat query "pass@k computation" --faults   # find code with fault annotations
pmat query "pipeline stages"               # semantic search across codebase
```
