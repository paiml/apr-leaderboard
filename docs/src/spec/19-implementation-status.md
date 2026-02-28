# Implementation Status

Tracking table mapping spec sections to `apr-leaderboard` code implementation. Updated as code lands.

## 19.1 CLI Subcommands (§6.2)

| Subcommand | Source Module | Status | Tests | Notes |
|---|---|---|---|---|
| `convert` | `src/convert/mod.rs` | ✅ Scaffolded | 13 | HF download + APR v2 bundle |
| `eval` | `src/eval/mod.rs` | ✅ Scaffolded | 25 | pass@k metrics, prompt strategies, n-samples |
| `finetune` | `src/finetune/mod.rs` | ✅ Scaffolded | 10 | LoRA/QLoRA config + entrenar integration |
| `distill` | `src/optimize/mod.rs` | ✅ Scaffolded | 6 | Knowledge distillation (3 strategies) |
| `merge` | `src/optimize/mod.rs` | ✅ Scaffolded | 6 | Model merging (4 strategies) |
| `prune` | `src/optimize/mod.rs` | ✅ Scaffolded | 6 | Pruning (3 methods) |
| `quantize` | `src/optimize/mod.rs` | ✅ Scaffolded | 4 | Quantization (5 schemes) |
| `compare` | `src/optimize/mod.rs` | ✅ Scaffolded | 2 | HF parity check |
| `submit` | `src/submit/mod.rs` | ✅ Scaffolded | 12 | HF leaderboard submission |
| `benchmarks` | `src/harness/mod.rs` | ✅ Complete | 20+ | 10 benchmark definitions |
| `history` | `src/eval/mod.rs` | ✅ Complete | 3 | Result history viewer |
| `pipeline` | `src/pipeline/mod.rs` | ✅ Scaffolded | 16 | Config-driven TOML pipeline (all 8 stages) |

## 19.2 Prompt Strategies (§8.3)

| Strategy | Enum Variant | Aliases | Status |
|---|---|---|---|
| Standard | `PromptStrategy::Standard` | `default` | ✅ Implemented |
| Structured CoT | `PromptStrategy::SCoT` | `structured-cot` | ✅ Implemented |
| Few-shot | `PromptStrategy::FewShot` | `fewshot` | ✅ Implemented |
| Code Gen Opt | `PromptStrategy::Cgo` | `code-gen-opt` | ✅ Implemented |
| Reflexion | `PromptStrategy::Reflexion` | `reflect` | ✅ Implemented |

## 19.3 Optimization Operations (§7)

| Operation | Strategy/Method Enums | Validation | Status |
|---|---|---|---|
| Distill | `Standard`, `Progressive`, `Ensemble` | Empty path check | ✅ Scaffolded |
| Merge | `Slerp`, `Ties`, `Dare`, `LinearAvg` | Min 2 models, empty path check | ✅ Scaffolded |
| Prune | `Wanda`, `Magnitude`, `SparseGpt` | Ratio 0.0–1.0, empty path check | ✅ Scaffolded |
| Quantize | `Int4`, `Int8`, `Q4K`, `Q5K`, `Q6K` | Empty path check | ✅ Scaffolded |

## 19.4 Quality Metrics

| Metric | Current | Target | Gate |
|---|---|---|---|
| Test count | 138 | — | `cargo test` |
| Line coverage | 96.5% | ≥ 95% | `cargo llvm-cov` |
| Clippy warnings | 0 | 0 | `cargo clippy -- -D warnings` |
| Max file size | 491 lines | < 500 | `wc -l src/**/*.rs` |
| pmat pre-commit | ✅ Pass | ✅ Pass | git hook |
| Pipeline configs | 4 | — | `configs/*.toml` |

## 19.5 What "Scaffolded" Means

**Scaffolded** = CLI parsing, strategy/method enums, input validation, result serialization, and test coverage are all implemented. The actual ML operations (inference, training, merging) are delegated to upstream `apr` CLI calls which are currently printed but not executed. Wiring to real `apr` subprocess calls is tracked by PMAT-017.
