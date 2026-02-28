# CLAUDE.md

## Project Overview

APR Leaderboard is a pipeline for building, evaluating, and submitting `.apr` models to Hugging Face leaderboards. Primary focus: Qwen-Coder models on coding benchmarks (HumanEval, MBPP, BigCodeBench, LiveCodeBench).

## Build Commands

```bash
cargo build --release
cargo test --all-features
cargo run --release -- benchmarks        # list available benchmarks
cargo run --release -- pipeline --config configs/qwen-coder-7b.toml
```

## Architecture

```
CLI (main.rs)
├── convert/    HF Hub → .apr conversion (SafeTensors download + APR v2 bundle)
├── eval/       Benchmark evaluation harness (pass@k metrics)
├── harness/    Benchmark definitions (HumanEval, MBPP, BigCodeBench, etc.)
├── finetune/   LoRA/QLoRA fine-tuning via entrenar
└── submit/     HF leaderboard submission formatting
```

## Pipeline Flow

```
convert → [finetune] → eval → submit
   │          │          │        │
   ▼          ▼          ▼        ▼
 HF Hub    LoRA/QLoRA  pass@k   HF Hub
 → .apr    adapter     metrics  submission
```

## Key Dependencies

- `aprender 0.25`: .apr format, quantization
- `entrenar 0.5`: LoRA training, optimizers
- `trueno 0.14`: SIMD tensor backend
- `reqwest`: HF Hub API calls
- `tokio`: async runtime for downloads

## Quality Gates

```bash
make check   # fmt + lint + test
cargo clippy --all-targets -- -D warnings
```
