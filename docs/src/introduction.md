# APR Leaderboard Specification

**Status:** ACTIVE
**Version:** 2.2.0
**Date:** 2026-03-22
**Authors:** APR Team

## Quick Status

| Metric | Value |
|---|---|
| `apr` CLI subcommands verified | 19 |
| Makefile targets | 45 |
| Shell scripts | 10 |
| YAML configs | 19 (7 models + 8 recipes + 1 eval + 2 pipeline + 1 data) |
| Python scripts | 0 (zero-Python constraint) |
| TOML configs | 0 (YAML-only) |
| Provable contracts | 5 (pass-at-k, decontamination, throughput, lora-algebra, quantization) |
| GPU sharing tests | 143 (entrenar, 9 modules) |
| HumanEval pass@1 (best 7B) | **87.20%** (few-shot, 0.60pp from HF parity) |
| HumanEval pass@1 (best 32B) | **90.85%** (standard, CPU batch) |
| MBPP pass@1 (best 7B) | **76.20%** (standard + test assertions) |
| Perplexity (WikiText-2) | 6.63 (1.5B-Instruct Q4K) |
| ACs verified | 8 verified, 4 partial, 15 not tested, 2 blocked |
| Open issues | 6 (GH-8, GH-10, GH-11, GH-12, GH-13, GH-14) |

See [Implementation Status](./spec/19-implementation-status.md) for
detailed tracking.

> **Definitive spec:**
> [`docs/specifications/leaderboard-spec.md`](../specifications/leaderboard-spec.md)
> — single executive summary with
> [component files](../specifications/components/).
