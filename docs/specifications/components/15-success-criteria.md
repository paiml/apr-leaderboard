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
| Pipeline commands | ≤ 10 | ≤ 6 | Config-driven pipeline counts as 1 |
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

## 15.4 Falsifiability

Every target above is falsifiable: it has a concrete measurement command, a numeric threshold, and a pass/fail outcome. If a metric cannot be measured, the spec has failed — not the implementation.
