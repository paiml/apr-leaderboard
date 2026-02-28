# Implementation Status

Tracking table mapping spec sections to `apr-leaderboard` code implementation. Updated as code lands.

## 19.1 CLI Subcommands (§6.2)

| Subcommand | Source Module | Status | Tests | Notes |
|---|---|---|---|---|
| `convert` | `src/convert/mod.rs` | ✅ Scaffolded | 15 | HF download + APR v2 bundle + Display roundtrip |
| `eval` | `src/eval/mod.rs` | ✅ Scaffolded | 31 | pass@k, prompt strategies, n-samples, temperature/top-p validation, rerank |
| `finetune` | `src/finetune/mod.rs` | ✅ Scaffolded | 16 | LoRA/QLoRA/full, method selection, Display roundtrip, custom output |
| `distill` | `src/optimize/mod.rs` | ✅ Scaffolded | 10 | 3 strategies + epochs + data + temperature/alpha validation |
| `merge` | `src/optimize/mod.rs` | ✅ Scaffolded | 18 | 4 strategies + weights sum-to-1 + base-model required for TIES/DARE + density/drop-rate range validation |
| `prune` | `src/optimize/mod.rs` | ✅ Scaffolded | 7 | 6 methods (wanda, magnitude, sparsegpt, structured, depth, width) + calibration |
| `quantize` | `src/optimize/mod.rs` | ✅ Scaffolded | 5 | 5 schemes + calibration dataset |
| `compare` | `src/optimize/mod.rs` | ✅ Scaffolded | 2 | HF parity check + --json flag |
| `submit` | `src/submit/mod.rs` | ✅ Scaffolded | 14 | HF leaderboard submission + Display roundtrip |
| `benchmarks` | `src/harness/mod.rs` | ✅ Complete | 20+ | 10 benchmark definitions |
| `history` | `src/eval/mod.rs` | ✅ Complete | 3 | Result history viewer |
| `pipeline` | `src/pipeline/mod.rs` | ✅ Scaffolded | 22 | Config-driven TOML pipeline (all 8 stages) + [eval] config + recipe B/D parsing |
| `align` | `src/align/mod.rs` | ✅ Scaffolded | 10 | DPO/ORPO preference optimization (§8.5) + beta validation |
| `validate` | `src/validate/mod.rs` | ✅ Scaffolded | 6 | Data decontamination checking (§8.7) + threshold validation |
| `tune` | `src/optimize/mod.rs` | ✅ Scaffolded | 6 | HPO: TPE/grid/random strategies (§7.7) + budget validation |
| `run` | `src/inference/mod.rs` | ✅ Scaffolded | 6 | Speculative decoding (§8.4) + draft model validation |
| `chat` | `src/inference/mod.rs` | ✅ Scaffolded | 5 | Batch generation (§8.6) + temperature validation |
| `check` | `src/compile/mod.rs` | ✅ Scaffolded | 4 | APR magic byte validation (§14.4) |
| `compile` | `src/compile/mod.rs` | ✅ Scaffolded | 4 | Binary compilation with --release --lto (§4.3.1, §9.4) |

## 19.1.1 CLI Flag Coverage Matrix

| Subcommand | Core Flags | Optional Flags | Status |
|---|---|---|---|
| `eval` | `--model`, `--benchmark`, `--samples`, `--output` | `--prompt-strategy`, `--n-samples`, `--temperature`, `--top-p`, `--rerank`, `--json`, `--exemplars`, `--system` | ✅ Complete |
| `finetune` | `--model`, `--dataset` | `--method`, `--rank`, `--lr`, `--epochs`, `-o` | ✅ Complete |
| `distill` | `--teacher`, `--student`, `-o` | `--strategy`, `--temperature`, `--alpha`, `--epochs`, `--data` | ✅ Complete |
| `merge` | `<models...>`, `-o` | `--strategy`, `--weights`, `--base-model`, `--density`, `--drop-rate` | ✅ Complete |
| `prune` | `--model`, `-o` | `--method`, `--target-ratio`, `--calibration`, `--analyze` | ✅ Complete |
| `quantize` | `--model`, `-o` | `--scheme`, `--calibration`, `--plan`, `--batch`, `--format` | ✅ Complete |
| `convert` | `--model-id` | `--output`, `--quantization` | ✅ Complete |
| `compare` | `--model` | `--json` | ✅ Complete |
| `submit` | `--results`, `--model-id` | `--leaderboard` | ✅ Complete |
| `pipeline` | `--config` | — | ✅ Complete |
| `align` | `--model`, `--data` | `--method`, `--beta`, `--epochs`, `--ref-model`, `-o` | ✅ Complete |
| `validate` | `--data`, `--benchmarks` | `--threshold`, `--decontaminate`, `-o` | ✅ Complete |
| `tune` | `--model`, `--data` | `--strategy`, `--budget`, `--max-epochs` | ✅ Complete |
| `run` | `--model`, `--prompt` | `--speculative`, `--speculation-k`, `--draft-model`, `--json` | ✅ Complete |
| `chat` | `--model` | `--batch`, `--prompt`, `--n-samples`, `--temperature`, `--system`, `--json` | ✅ Complete |
| `check` | `--model` | — | ✅ Complete |
| `compile` | `--model` | `--release`, `--lto`, `-o` | ✅ Complete |

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
| Distill | `Standard`, `Progressive`, `Ensemble` | `--epochs`, `--data` | temperature > 0, alpha 0.0–1.0, empty path check | ✅ Scaffolded |
| Merge | `Slerp`, `Ties`, `Dare`, `LinearAvg` (alias: `average`) | `--weights`, `--base-model`, `--density`, `--drop-rate` | Min 2 models, weights sum to 1.0, TIES/DARE require base-model, density/drop-rate 0.0–1.0 | ✅ Scaffolded |
| Prune | `Wanda`, `Magnitude`, `SparseGpt`, `Structured`, `Depth`, `Width` | `--calibration` | Ratio 0.0–1.0, empty path check | ✅ Scaffolded |
| Quantize | `Int4`, `Int8`, `Q4K`, `Q5K`, `Q6K` | `--calibration` | Empty path check | ✅ Scaffolded |
| Finetune | `Lora`, `Qlora`, `Full` | `--method`, `-o` | Model file exists check | ✅ Scaffolded |

## 19.4 Quality Metrics

| Metric | Current | Target | Gate |
|---|---|---|---|
| Test count | 244 | — | `cargo test` |
| CLI subcommands | 19 | — | All spec §6.2 subcommands implemented |
| Line coverage | 96.5% | ≥ 95% | `cargo llvm-cov` |
| Clippy warnings | 0 | 0 | `cargo clippy -- -D warnings` |
| Max file size | 494 lines | < 500 | `wc -l src/**/*.rs` |
| pmat pre-commit | ✅ Pass | ✅ Pass | git hook |
| Pipeline configs | 4 | — | `configs/*.toml` (recipes A–D) |
| Source modules | 11 | — | convert, eval, finetune, optimize, harness, pipeline, submit, align, validate, inference, compile |

## 19.5 What "Scaffolded" Means

**Scaffolded** = CLI parsing, strategy/method enums, input validation, result serialization, and test coverage are all implemented. The actual ML operations (inference, training, merging) are delegated to upstream `apr` CLI calls which are currently printed but not executed. Wiring to real `apr` subprocess calls is tracked by PMAT-017.
