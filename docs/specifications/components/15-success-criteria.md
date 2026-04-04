# Success Criteria

## 15.1 Primary Metrics

| Metric | Target | Stretch | Measurement | Notes |
|--------|--------|---------|-------------|-------|
| HumanEval pass@1 | ≥ apr baseline | ≥ HF reference | `make eval-humaneval` | Relative to Step 0 baseline |
| MBPP pass@1 | ≥ apr baseline | ≥ HF reference | `make eval-mbpp` | Relative to Step 0 baseline |
| BigCodeBench pass@1 | > 0 (eval works) | ≥ HF reference | `make eval-bigcodebench` | Stretch: competitive |
| Inference parity | <5% gap vs HF | <2% gap vs HF | `apr compare-hf` | Perplexity gap on WikiText-2 |

## 15.2 Infrastructure Metrics

| Metric | Target | Stretch | Notes |
|--------|--------|---------|-------|
| Makefile targets | 58 | — | Config-driven: `make pipeline RECIPE=...` wraps multi-stage pipeline. Includes `proof-status`, `status`, `check-contracts`. |
| Total binary size (compiled, 7B INT4) | < 5GB | < 4GB | 3.5GB weights + runtime |
| Wall-clock (import → submit) | < 24h (GPU) | < 8h (GPU) | CPU-only: much longer |
| Python dependencies | 0 | 0 | External sandbox for eval only |
| CUDA toolkit | Not required | Not required | wgpu handles GPU compute (any vendor) |
| GPU hardware | Recommended (any vendor) | Optional (≤7B) | Required for distill/finetune 32B teacher; NVIDIA, AMD, Intel, or Apple Silicon |

## 15.3 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test coverage | ≥ 95% | `cargo llvm-cov` (project source only — exclude path deps, see §19.7.1) |
| Clippy warnings | 0 | `cargo clippy -- -D warnings` |
| Source file size | < 500 lines each | `wc -l src/**/*.rs` |
| pmat comply | Pass | `pmat comply check --strict` |
| Contract binding coverage | ≥ 95% | `pv proof-status` |

## 15.4 Measured Baselines (apr-native)

Baselines measured via `apr run` + `scripts/eval-pass-at-k.sh` (greedy decoding, max_tokens=512):

| Model | Quant | HumanEval | MBPP | Backend | Notes |
|---|---|---|---|---|---|
| Qwen2.5-Coder-32B-Instruct | Q4K_M | **90.85%** (149/164) | — | CPU (gx10) | Batch mode re-run |
| Qwen2.5-Coder-7B-Instruct (few-shot) | Q4K | **87.20%** (143/164) | — | CPU (gx10) | Best 7B HumanEval strategy |
| Qwen2.5-Coder-7B-Instruct | Q4K | **85.37%** (140/164) | **76.20%** (381/500) | CPU/GPU (gx10) | GPU/CPU parity (HE) |
| Qwen2.5-Coder-7B-Instruct (SCoT) | Q4K | **82.32%** (135/164) | — | CPU (gx10) | Structured CoT |
| Qwen3-4B | Q4K | **78.05%** (128/164) | — | CPU (gx10) | Thinking model, 4096 tokens |
| Qwen2.5-Coder-1.5B | Q4K | 59.15% (97/164) | — | CPU | Baseline |

**HF parity (EvalPlus leaderboard reference):** HumanEval 7B gap = 0.60pp (87.20% few-shot vs 87.8%). MBPP 7B gap = 7.3pp (76.20% vs 83.5%). 32B HE gap = 1.65pp (90.85% vs 92.5%). Note: Qwen model card reports 88.4%/92.7% (different test harness).

**Oracle upper bounds:** HumanEval 96.34% (158/164, best-per-problem across all strategies). Only 6 problems never solved. See §24.19.

**Perplexity baseline:** 6.63 on WikiText-2 (1.5B Q4K, CPU). Cross-entropy: 1.89 nats.

**Contract gate:** `make check-contracts` — 67/68 passing. 1 failure: AC-022 MBPP gate (76.2% < 80%). See §17.6.

**Acceptance criteria:** 19/29 verified (66%). See §18. Critical path: PMAT-014 → PMAT-008 → PMAT-010 → PMAT-011 → AC-022.

## 15.5 Falsifiability

Every target above is falsifiable: it has a concrete measurement command, a numeric threshold, and a pass/fail outcome. If a metric cannot be measured, the spec has failed — not the implementation.
