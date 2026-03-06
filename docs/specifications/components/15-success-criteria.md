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

## 15.4 Measured Baselines (apr-native)

Baselines measured via `apr run` + `scripts/eval-pass-at-k.sh` on CPU (no GPU):

| Model | Quant | HumanEval pass@1 | Throughput (tok/s) | TTFT (ms) | Notes |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | Q4K | 59.15% | ~2.5 | ~385 | GGUF import, greedy |
| Qwen2.5-Coder-7B-Instruct | Q4K | 68.9% | — | — | Pre-EOS fix, 128-token cap |
| Qwen2.5-Coder-1.5B-Instruct | Q4K | *eval in progress* | 2.5 | 385 | Chat mode, greedy |

**Key finding:** Instruct models via `--chat` append conversational trailing text after code. The `extract_python_code()` fix in `eval-pass-at-k.sh` (§22.15/§24.1) raised pass rate from 0% to ~70% on the 1.5B-Instruct model.

**Perplexity baseline:** 6.63 on WikiText-2 (1.5B Q4K, CPU). Cross-entropy: 1.89 nats.

## 15.5 Falsifiability

Every target above is falsifiable: it has a concrete measurement command, a numeric threshold, and a pass/fail outcome. If a metric cannot be measured, the spec has failed — not the implementation.
