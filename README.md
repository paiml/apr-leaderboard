<p align="center">
  <img src="docs/hero.svg" alt="apr-leaderboard pipeline" width="100%"/>
</p>

# apr-leaderboard

[![CI](https://github.com/paiml/apr-leaderboard/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/apr-leaderboard/actions/workflows/ci.yml)

HuggingFace leaderboard pipeline for the sovereign Rust AI stack. Proves that a single `apr` binary — with zero Python and no GPU vendor lock-in — can compete on code generation benchmarks (HumanEval, MBPP, BigCodeBench). GPU compute via wgpu (Vulkan/Metal/DX12) or optional CUDA backend. Eval hardware: gx10 NVIDIA Blackwell GB10 (119 GB unified) + AMD Radeon Pro W5700X (wgpu/Vulkan).

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

## Installation

```bash
# Install the apr CLI (requires Rust toolchain)
cargo install apr-cli

# Clone this repo
git clone https://github.com/paiml/apr-leaderboard.git
cd apr-leaderboard

# Verify everything works
make verify
```

## Usage

```bash
# Import a model from HuggingFace
make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

# Evaluate on HumanEval
make eval-humaneval CHECKPOINT=checkpoints/qwen_qwen2.5-coder-7b-instruct.apr

# Batch inference (load model once — eliminates ~80s/problem JIT overhead)
apr run checkpoints/model.apr --batch-jsonl prompts.jsonl --max-tokens 512 --temperature 0.0

# N-sampling for unbiased pass@k (PMAT-003)
make eval-humaneval CHECKPOINT=checkpoints/model.apr NUM_SAMPLES=10 TEMPERATURE=0.8

# Sweep all eval results and compare
make eval-sweep
make compare-results BASE=results/humaneval_baseline.json NEW=results/humaneval_latest.json

# Run a full recipe pipeline
make pipeline RECIPE=recipe-a-quick-lora

# Dry-run a pipeline (validate config, show commands)
make pipeline-plan RECIPE=recipe-c-full-pipeline
```

## Sovereign Stack

| Crate | Role | Version |
|-------|------|---------|
| [aprender](https://crates.io/crates/aprender) | .apr format, inference, batch inference, distillation, merging, pruning, quantization | 0.4.11 |
| [entrenar](https://crates.io/crates/entrenar) | LoRA/QLoRA training, autograd, AdamW, gradient checkpointing | 0.7 |
| [trueno](https://crates.io/crates/trueno) | SIMD tensor ops (AVX2/NEON), wgpu GPU (Vulkan/Metal/DX12) | 0.16.3 |

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

## Current Results

HumanEval pass@1 (greedy decoding, temperature 0.0):

| Rank | Model | pass@1 | Passed | Backend | Notes |
|------|-------|--------|--------|---------|-------|
| 1 | Qwen2.5-Coder-32B-Instruct Q4K_M | **90.85%** | 149/164 | CPU (gx10) | Batch mode re-run |
| 2 | Qwen2.5-Coder-7B-Instruct Q4K (few-shot) | **87.20%** | 143/164 | CPU (gx10) | Few-shot prompting |
| 3 | Qwen2.5-Coder-7B-Instruct Q4K | **85.37%** | 140/164 | CPU/GPU (gx10) | GPU/CPU parity verified |
| 4 | Qwen2.5-Coder-7B-Instruct Q4K (SCoT) | **82.32%** | 135/164 | CPU (gx10) | Structured CoT |
| 5 | Qwen3-4B Q4K | **78.05%** | 128/164 | CPU (gx10) | Thinking model |
| 6 | Qwen2.5-Coder-1.5B Q4K | **59.15%** | 97/164 | CPU | Baseline |

MBPP pass@1 (greedy decoding, temperature 0.0):

| Rank | Model | pass@1 | Passed | Backend | Notes |
|------|-------|--------|--------|---------|-------|
| 1 | Qwen2.5-Coder-7B-Instruct Q4K | **76.20%** | 381/500 | CPU (gx10) | Standard + test assertions |

All results produced by `apr run` (zero Python inference). GPU via wgpu (Vulkan) on Blackwell GB10. Code execution sandbox uses python3.

## GPU Compute

`apr run --gpu` auto-dispatches to the best available backend:

| Backend | Compilation | Result on Blackwell sm_121 |
|---------|-------------|---------------------------|
| CUDA PTX (JIT) | Runtime JIT by NVIDIA driver | **FAIL** (cosine=-0.005, FP32 accumulation) |
| **wgpu (Vulkan)** | **Vulkan shader compiler** | **PASS** (cosine=0.999863) |
| PyTorch (cuBLAS) | Pre-compiled SASS | PASS (cosine=1.0, hardware verified) |
| CPU (SIMD) | Ahead-of-time | Always correct |

**Root cause (corrected 2026-03-27):** NOT a JIT bug. Our PTX loads correctly via Python ctypes (cosine=1.0). The cosine=-0.005 is FP32 non-associativity: parallel GPU accumulation order ≠ CPU sequential order, compounding through 280 operations. wgpu uses sequential accumulation matching CPU. See §25.

## Roadmap

**Completed:**
- 7B baseline (PMAT-006): 85.37% HumanEval pass@1 (≥85% gate met)
- Pipeline wiring (PMAT-017): all shell scripts call real `apr` CLI
- N-sampling (PMAT-003): `make eval-humaneval NUM_SAMPLES=10 TEMPERATURE=0.8`
- Preference pairs (PMAT-014): `make generate-preference-pairs`
- Synthetic training data (PMAT-004): `make generate-training-data`
- GPU fix (PMAT-037): wgpu Vulkan fallback, cosine=0.999863
- 22/22 contract falsification tests passing (6 contracts)
- Oracle analysis: 96.34% upper bound (158/164), only 6 unsolvable

**In progress:**
1. 32B→7B text-based distillation (PMAT-007): `make distill-generate` → `make distill-finetune`

**Pipeline ready (11 recipes A-K):**
2. BigCodeBench eval (`make eval-bigcodebench`) — first score
3. QLoRA fine-tune (Recipe I, PMAT-008): `make pipeline RECIPE=recipe-i-humaneval-qlora`
4. Specialist merge (Recipe J, PMAT-010): `make pipeline RECIPE=recipe-j-merge-specialists`
5. Final artifact (Recipe K, PMAT-011): `make pipeline RECIPE=recipe-k-final-artifact`
6. AC-022 success gate: `make validate-ac022` (≥85% HE, ≥80% MBPP)

**Blocked on upstream:**
7. DPO alignment (PMAT-001): `apr align --method dpo` needed in entrenar
8. HumanEval+ eval: EvalPlus harness integration

## Project Structure

```
apr-leaderboard/
├── Makefile                    # 54 orchestration targets
├── scripts/                    # 21 pipeline + 4 GPU canary scripts
│   ├── eval-pass-at-k.sh       # Generate → sandbox → Chen et al. pass@k (batch + N-sampling)
│   ├── eval-helpers.sh          # Extraction, scoring, batch generation helpers
│   ├── distill-generate.sh      # PMAT-007: 32B teacher → coding completions
│   ├── combine-training-data.sh # PMAT-008: merge + dedup + shuffle data sources
│   ├── validate-teacher.sh      # §12.2: verify teacher quality before distillation
│   ├── validate-ac022.sh        # AC-022: success gate (≥85% HE, ≥80% MBPP)
│   ├── failure-analysis.sh      # Always-fail / borderline / always-pass categorization
│   ├── oracle-analysis.sh       # Oracle upper bound across strategies
│   ├── pipeline.sh             # YAML-driven multi-stage pipeline
│   ├── submit.sh               # 6 preflight checks + export + HF Hub publish
│   └── ...                     # 10 more scripts (training data, proofs, benchmarks)
├── configs/
│   ├── models/                 # 7 per-model YAML configs
│   ├── recipes/                # 11 multi-stage pipeline recipes (A-K)
│   ├── eval/                   # Benchmark suite + prompt strategies
│   ├── distill/                # Text-based distillation config
│   └── pipeline/               # Forjar manifest + batuta playbook
├── data_catalog.yaml           # Data governance + lineage
├── contracts/                  # 6 provable contracts, 22/22 FTs
│   ├── pass-at-k.yaml          # Pass@k estimator (3 proofs, 5 tests)
│   ├── distillation.yaml       # Teacher quality + gain (3 proofs, 2 tests)
│   ├── decontamination.yaml    # N-gram overlap gate (3 proofs, 1 test)
│   ├── inference-throughput.yaml # Throughput + TTFT (2 proofs, 2 tests)
│   ├── lora-algebra.yaml       # LoRA correctness (3 proofs, pending)
│   ├── quantization.yaml       # Q4K dequant identity (3 proofs, pending)
│   └── CONTRACT_STATUS.md      # Audit trail
├── checkpoints/                # .apr model files (gitignored)
├── results/                    # Evaluation result JSONs
├── data/                       # Training/calibration data (gitignored)
└── docs/                       # Specification (mdBook)
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make verify` | Check `apr` CLI + all 19 subcommands |
| `make validate` | Lint all configs + scripts (bashrs) |
| `make dogfood` | End-to-end smoke test (zero Python) |
| `make import MODEL=...` | Download HF model → .apr |
| `make eval-humaneval CHECKPOINT=...` | HumanEval pass@k evaluation |
| `make eval-mbpp CHECKPOINT=...` | MBPP pass@k evaluation |
| `make eval-all CHECKPOINT=...` | All benchmarks |
| `make finetune CHECKPOINT=...` | LoRA/QLoRA fine-tuning |
| `make pipeline RECIPE=...` | Run a multi-stage recipe |
| `make pipeline-plan RECIPE=...` | Dry-run: show commands |
| `make publish CHECKPOINT=... HF_REPO=...` | Publish to HF Hub |
| `make prove-wgpu` | Dual GPU wgpu training proof |
| `make generate-training-data TEACHER=...` | PMAT-004: synthetic instruct pairs from teacher |
| `make generate-preference-pairs EVAL_WORK_DIR=...` | PMAT-014: DPO pairs from N-sampling |
| `make distill-generate` | PMAT-007: 32B teacher → coding completions |
| `make distill-finetune` | PMAT-007: QLoRA fine-tune 7B on teacher data |
| `make distill-eval` | PMAT-007: evaluate distilled model |
| `make check-contracts` | 22 falsification tests (pass@k, throughput, data, decon, eval, distill) |
| `make validate-ac022` | AC-022 success gate (≥85% HE, ≥80% MBPP) |
| `make failure-analysis` | Problem reliability analysis across all runs |
| `make validate-teacher TEACHER=...` | Verify teacher quality before distillation |
| `make prepare-calibration-data` | 128-sample Wanda/SparseGPT calibration set |
| `make eval-sweep` | Sweep all result JSONs, tabulate pass@k |
| `make leaderboard` | Generate ranked markdown leaderboard from results |

## Specification

The full specification is published as an [mdBook](https://paiml.github.io/apr-leaderboard/) via GitHub Actions. 25 sections covering:

- **S1-4** Architecture, thesis, target leaderboards, model selection
- **S5-6** Sovereign tooling map, CLI toolchain (19 subcommands, 47 targets)
- **S7-8** Technique playbook, leaderboard-winning techniques
- **S9-10** 8 composite recipes, technique interaction matrix + golden ordering
- **S12-14** Data strategy (incl. synthetic data §12.3), evaluation protocol (Chen et al. pass@k, N-sampling §13.5), submission flow
- **S16-18** Provable contracts (18 FTs + gpu-multi-backend-parity), quality gates, 29 acceptance criteria
- **S22-24** Dogfooding findings, training infrastructure, AC verification
- **S25** GPU Compute Architecture — hybrid wgpu/CUDA dispatch, parity gate, NVRTC strategy

## Contributing

1. Fork the repo and create a feature branch
2. Ensure `make verify && make validate && make dogfood` all pass
3. Keep `pmat check` clean (zero violations)
4. Submit a PR against `main`

All ML operations live in [aprender](https://github.com/paiml/aprender) — this repo is orchestration only.

## License

MIT
