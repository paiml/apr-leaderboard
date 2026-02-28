<p align="center">
  <img src="docs/hero.svg" alt="apr-leaderboard pipeline" width="100%"/>
</p>

# apr-leaderboard

HuggingFace leaderboard pipeline for the sovereign Rust AI stack. Proves that a single `apr` binary — with zero Python, zero CUDA toolkit — can compete on code generation benchmarks (HumanEval, MBPP, BigCodeBench).

**[Read the full specification](https://paiml.github.io/apr-leaderboard/)**

## What This Proves

One falsifiable question:

> Can a single Rust binary (`apr`) match Python-ecosystem HumanEval/MBPP scores for Qwen2.5-Coder-7B, with zero Python dependencies?

If yes: [aprender](https://github.com/paiml/aprender), [entrenar](https://github.com/paiml/entrenar), and [trueno](https://github.com/paiml/trueno) work end-to-end as a sovereign AI stack.

If no: `apr compare-hf` pinpoints exactly where the stack falls short.

## Pipeline

```
apr import → apr distill → apr finetune → apr merge → apr prune → apr quantize → apr eval → apr submit
```

Every command is provided by the `apr` CLI (aprender). This repo provides the pipeline config, benchmark metadata, result persistence, and the strategy spec.

## Quick Start

```bash
# Build
cargo build --release

# List available benchmarks (10 code generation benchmarks)
cargo run --release -- benchmarks

# Run full pipeline from config
cargo run --release -- pipeline --config configs/qwen-coder-7b.toml

# View evaluation history
cargo run --release -- history
```

## Sovereign Stack

| Crate | Role | Version |
|-------|------|---------|
| [aprender](https://crates.io/crates/aprender) | .apr format, inference, distillation, merging, pruning, quantization | 0.27 |
| [entrenar](https://crates.io/crates/entrenar) | LoRA/QLoRA training, autograd, AdamW, gradient checkpointing | 0.7 |
| [trueno](https://crates.io/crates/trueno) | SIMD tensor ops (AVX2/NEON), wgpu GPU, PTX generation | 0.16 |
| [provable-contracts](https://crates.io/crates/provable-contracts) | Kernel correctness via Kani bounded model checking | 0.1 |

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

## Project Structure

```
apr-leaderboard/
├── src/
│   ├── main.rs          # CLI dispatcher + pipeline orchestrator
│   ├── convert/         # HF Hub → .apr conversion
│   ├── eval/            # Benchmark evaluation harness
│   ├── harness/         # 10 benchmark definitions
│   ├── finetune/        # LoRA training configuration
│   └── submit/          # HF leaderboard submission
├── configs/
│   ├── qwen-coder-7b.toml
│   └── qwen-coder-32b.toml
├── docs/
│   ├── specifications/
│   │   └── leaderboard-spec.md   # Full spec (published as mdBook)
│   └── hero.svg
└── Makefile
```

## Quality Gates

```bash
make check          # fmt + clippy + test
pmat comply check --strict   # 30+ compliance checks
pv proof-status     # Kani bounded model checking status
```

## Specification

The full specification is published as an [mdBook](https://paiml.github.io/apr-leaderboard/) via GitHub Actions. It covers:

- **§1** What this repo does and how it relates to aprender
- **§5** Technique playbook (distillation, merging, pruning, LoRA, quantization)
- **§6** Leaderboard-winning inference techniques (N-sampling, SCoT, speculative decoding, DPO)
- **§7** Composite recipes (4 end-to-end strategies)
- **§8** Technique interaction matrix and golden ordering
- **§14** Provable contracts (design by contract with Kani proofs)
- **§15** Quality gates (pmat comply enforcement)
- **§16** 20 falsifiable acceptance criteria

## License

MIT
