# Implementation Status

Tracking table mapping spec sections to `apr-leaderboard` code implementation. Updated as code lands.

## 19.1 CLI Subcommands (§6.2)

| Subcommand | Source Module | Status | Tests | Notes |
|---|---|---|---|---|
| `convert` | `src/convert/mod.rs` | ✅ Scaffolded | 17 | HF download + APR v2 bundle + Display roundtrip |
| `eval` | `src/eval/mod.rs` | ✅ Scaffolded | 45 | pass@k estimator (§13.1), prompt strategies, n-samples, temperature/top-p validation, rerank, exemplars/system |
| `finetune` | `src/finetune/mod.rs` | ✅ Scaffolded | 15 | LoRA/QLoRA/full, method selection, Display roundtrip, custom output |
| `distill` | `src/optimize/mod.rs` | ✅ Scaffolded | 10 | 3 strategies + epochs + data + temperature/alpha validation + scaffold output |
| `merge` | `src/optimize/mod.rs` | ✅ Scaffolded | 18 | 4 strategies + weights sum-to-1 + base-model required for TIES/DARE + scaffold output |
| `prune` | `src/optimize/mod.rs` | ✅ Scaffolded | 7 | 6 methods (wanda, magnitude, sparsegpt, structured, depth, width) + scaffold output |
| `quantize` | `src/optimize/mod.rs` | ✅ Scaffolded | 5 | 5 schemes + calibration dataset + scaffold output |
| `compare` | `src/optimize/mod.rs` | ✅ Scaffolded | 2 | HF parity check + --json flag |
| `submit` | `src/submit/mod.rs` | ✅ Scaffolded | 32 | HF leaderboard submission + pre-submit validation (§14.4) + --generate-card (§14.3) + export metadata |
| `benchmarks` | `src/harness/mod.rs` | ✅ Complete | 21 | 10 benchmark definitions |
| `history` | `src/eval/mod.rs` | ✅ Complete | 3 | Result history viewer |
| `pipeline` | `src/pipeline/mod.rs` | ✅ Scaffolded | 49 | Config-driven TOML pipeline (12 stages) + [eval] config + recipe B/D + integration tests + ordering validation (§10) + config hash (§11) + --dry-run |
| `align` | `src/align/mod.rs` | ✅ Scaffolded | 12 | DPO/ORPO preference optimization (§8.5) + beta validation + output file creation |
| `validate` | `src/validate/mod.rs` | ✅ Scaffolded | 11 | Data decontamination checking (§8.7) + threshold validation + contamination report (§12.1) |
| `tune` | `src/optimize/mod.rs` | ✅ Scaffolded | 6 | HPO: TPE/grid/random strategies (§7.7) + budget validation |
| `run` | `src/inference/mod.rs` | ✅ Scaffolded | 9 | Speculative decoding (§8.4) + draft model validation + JSON output |
| `chat` | `src/inference/mod.rs` | ✅ Scaffolded | 6 | Batch generation (§8.6) + temperature validation + system prompt |
| `check` | `src/compile/mod.rs` | ✅ Scaffolded | 6 | APR magic byte validation (§14.4) + boundary tests |
| `compile` | `src/compile/mod.rs` | ✅ Scaffolded | 7 | Binary compilation with --release --lto --strip (§4.3.1, §9.4) |
| `export` | `src/submit/mod.rs` | ✅ Scaffolded | 5 | SafeTensors/GGUF metadata export (§14.2) + results bundling |
| `acceptance` | `src/acceptance/mod.rs` | ✅ Complete | 16 | 27 falsifiable ACs (§18) + category filter + scaffold verification |

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
| `submit` | `--results`, `--model-id` | `--leaderboard`, `--pre-submit-check`, `--generate-card` | ✅ Complete |
| `pipeline` | `--config` | `--dry-run` | ✅ Complete |
| `align` | `--model`, `--data` | `--method`, `--beta`, `--epochs`, `--ref-model`, `-o` | ✅ Complete |
| `validate` | `--data`, `--benchmarks` | `--threshold`, `--decontaminate`, `-o` | ✅ Complete |
| `tune` | `--model`, `--data` | `--strategy`, `--budget`, `--max-epochs` | ✅ Complete |
| `run` | `--model`, `--prompt` | `--speculative`, `--speculation-k`, `--draft-model`, `--json` | ✅ Complete |
| `chat` | `--model` | `--batch`, `--prompt`, `--n-samples`, `--temperature`, `--system`, `--json` | ✅ Complete |
| `check` | `--model` | — | ✅ Complete |
| `compile` | `--model` | `--release`, `--lto`, `--strip`, `--target`, `-o` | ✅ Complete |
| `export` | `--model` | `--format`, `-o`, `--results` | ✅ Complete |
| `acceptance` | — | `--category`, `--verify` | ✅ Complete |

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
| Test count | 343 | — | `cargo test` |
| CLI subcommands | 21 | — | All spec §6.2 subcommands + export + acceptance |
| Line coverage | 96.5% | ≥ 95% | `cargo llvm-cov` |
| Clippy warnings | 0 | 0 | `cargo clippy -- -D warnings` |
| Max file size | 498 lines | < 500 | `wc -l src/**/*.rs` |
| pmat pre-commit | ✅ Pass | ✅ Pass | git hook |
| Pipeline configs | 7 | — | `configs/*.toml` (recipes A–D + 3 model targets) |
| Pipeline stages | 12 | — | validate → convert → distill → finetune → align → merge → tune → prune → quantize → eval → compile → submit |
| Source modules | 12 | — | acceptance, align, compile, convert, eval, finetune, harness, inference, optimize, pipeline, submit, validate |
| Acceptance criteria | 27 | — | §18 falsifiable ACs (12 scaffolded, 11 pending, 4 external) |
| Pre-submit checks | 5 | — | APR format, results JSON, required benchmarks, model ID, model card |

## 19.5 Config Templates (§4)

| Config | Model | Tier | Strategy | Status |
|---|---|---|---|---|
| `qwen-coder-7b.toml` | Qwen2.5-Coder-7B | 1 | LoRA finetune → eval | ✅ Complete |
| `qwen-coder-32b.toml` | Qwen2.5-Coder-32B | 1 | Eval only (q8) | ✅ Complete |
| `qwen-coder-1.5b.toml` | Qwen2.5-Coder-1.5B | 3 | QLoRA → prune → INT4 → compile | ✅ Complete |
| `deepseek-r1-distill-7b.toml` | DeepSeek-R1-Distill-Qwen-7B | 2 | DPO align → prune → INT4 | ✅ Complete |
| `phi-4.toml` | Phi-4 | 2 | LoRA finetune → INT8 | ✅ Complete |
| `recipe-a-quick-lora.toml` | Qwen2.5-Coder-7B-Instruct | — | Quick LoRA (§9.1) | ✅ Complete |
| `recipe-b-merge-alchemist.toml` | Qwen2.5-Coder-7B-Instruct | — | Zero-training merge (§9.2) | ✅ Complete |
| `recipe-c-full-pipeline.toml` | Qwen2.5-Coder-7B | — | Full pipeline (§9.3) | ✅ Complete |
| `recipe-d-sovereign-binary.toml` | Qwen2.5-Coder-1.5B | — | Sovereign binary (§9.4) | ✅ Complete |

## 19.6 What "Scaffolded" Means

**Scaffolded** = CLI parsing, strategy/method enums, input validation, result serialization, and test coverage are all implemented. The actual ML operations (inference, training, merging) are delegated to upstream `apr` CLI calls which are currently printed but not executed. Wiring to real `apr` subprocess calls is tracked by PMAT-017.
