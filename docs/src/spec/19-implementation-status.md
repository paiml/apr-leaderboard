# Implementation Status

Tracking table mapping spec sections to `apr-leaderboard` code implementation. Updated as code lands.

## 19.1 CLI Subcommands (§6.2)

| Subcommand | Source Module | Status | Tests | Notes |
|---|---|---|---|---|
| `convert` | `src/convert/mod.rs` | ✅ Scaffolded | 13 | HF download + APR v2 bundle |
| `eval` | `src/eval/mod.rs` | ✅ Scaffolded | 27 | pass@k, prompt strategies, n-samples, temperature, top-p, rerank |
| `finetune` | `src/finetune/mod.rs` | ✅ Scaffolded | 14 | LoRA/QLoRA/full, method selection, custom output |
| `distill` | `src/optimize/mod.rs` | ✅ Scaffolded | 7 | 3 strategies + epochs + data corpus |
| `merge` | `src/optimize/mod.rs` | ✅ Scaffolded | 9 | 4 strategies + weights + base-model + density + drop-rate |
| `prune` | `src/optimize/mod.rs` | ✅ Scaffolded | 7 | 3 methods + calibration dataset |
| `quantize` | `src/optimize/mod.rs` | ✅ Scaffolded | 5 | 5 schemes + calibration dataset |
| `compare` | `src/optimize/mod.rs` | ✅ Scaffolded | 2 | HF parity check |
| `submit` | `src/submit/mod.rs` | ✅ Scaffolded | 12 | HF leaderboard submission |
| `benchmarks` | `src/harness/mod.rs` | ✅ Complete | 20+ | 10 benchmark definitions |
| `history` | `src/eval/mod.rs` | ✅ Complete | 3 | Result history viewer |
| `pipeline` | `src/pipeline/mod.rs` | ✅ Scaffolded | 16 | Config-driven TOML pipeline (all 8 stages) |

## 19.1.1 CLI Flag Coverage Matrix

| Subcommand | Core Flags | Optional Flags | Status |
|---|---|---|---|
| `eval` | `--model`, `--benchmark`, `--samples`, `--output` | `--prompt-strategy`, `--n-samples`, `--temperature`, `--top-p`, `--rerank` | ✅ Complete |
| `finetune` | `--model`, `--dataset` | `--method`, `--rank`, `--lr`, `--epochs`, `-o` | ✅ Complete |
| `distill` | `--teacher`, `--student`, `-o` | `--strategy`, `--temperature`, `--alpha`, `--epochs`, `--data` | ✅ Complete |
| `merge` | `<models...>`, `-o` | `--strategy`, `--weights`, `--base-model`, `--density`, `--drop-rate` | ✅ Complete |
| `prune` | `--model`, `-o` | `--method`, `--target-ratio`, `--calibration` | ✅ Complete |
| `quantize` | `--model`, `-o` | `--scheme`, `--calibration` | ✅ Complete |
| `convert` | `--model-id` | `--output`, `--quantization` | ✅ Complete |
| `compare` | `--model` | — | ✅ Complete |
| `submit` | `--results`, `--model-id` | `--leaderboard` | ✅ Complete |
| `pipeline` | `--config` | — | ✅ Complete |

## 19.2 Prompt Strategies (§8.3)

| Strategy | Enum Variant | Aliases | Status |
|---|---|---|---|
| Standard | `PromptStrategy::Standard` | `default` | ✅ Implemented |
| Structured CoT | `PromptStrategy::SCoT` | `structured-cot` | ✅ Implemented |
| Few-shot | `PromptStrategy::FewShot` | `fewshot` | ✅ Implemented |
| Code Gen Opt | `PromptStrategy::Cgo` | `code-gen-opt` | ✅ Implemented |
| Reflexion | `PromptStrategy::Reflexion` | `reflect` | ✅ Implemented |

## 19.3 Optimization Operations (§7)

| Operation | Strategy/Method Enums | Extended Flags | Validation | Status |
|---|---|---|---|---|
| Distill | `Standard`, `Progressive`, `Ensemble` | `--epochs`, `--data` | Empty path check | ✅ Scaffolded |
| Merge | `Slerp`, `Ties`, `Dare`, `LinearAvg` | `--weights`, `--base-model`, `--density`, `--drop-rate` | Min 2 models, empty path check | ✅ Scaffolded |
| Prune | `Wanda`, `Magnitude`, `SparseGpt` | `--calibration` | Ratio 0.0–1.0, empty path check | ✅ Scaffolded |
| Quantize | `Int4`, `Int8`, `Q4K`, `Q5K`, `Q6K` | `--calibration` | Empty path check | ✅ Scaffolded |
| Finetune | `Lora`, `Qlora`, `Full` | `--method`, `-o` | Model file exists check | ✅ Scaffolded |

## 19.4 Quality Metrics

| Metric | Current | Target | Gate |
|---|---|---|---|
| Test count | 149 | — | `cargo test` |
| Line coverage | 96.5% | ≥ 95% | `cargo llvm-cov` |
| Clippy warnings | 0 | 0 | `cargo clippy -- -D warnings` |
| Max file size | 474 lines | < 500 | `wc -l src/**/*.rs` |
| pmat pre-commit | ✅ Pass | ✅ Pass | git hook |
| Pipeline configs | 4 | — | `configs/*.toml` |

## 19.5 What "Scaffolded" Means

**Scaffolded** = CLI parsing, strategy/method enums, input validation, result serialization, and test coverage are all implemented. The actual ML operations (inference, training, merging) are delegated to upstream `apr` CLI calls which are currently printed but not executed. Wiring to real `apr` subprocess calls is tracked by PMAT-017.
