# APR Leaderboard Specification

**Status:** ACTIVE
**Version:** 2.1.0
**Date:** 2026-03-06
**Authors:** APR Team

## Quick Status

| Metric | Value |
|---|---|
| `apr` CLI subcommands verified | 19 |
| Makefile targets | 41 |
| Shell scripts | 7 |
| YAML configs | 17 (6 models + 7 recipes + 1 eval + 2 pipeline + 1 data) |
| Python scripts | 0 (zero-Python constraint) |
| TOML configs | 0 (YAML-only) |
| Provable contracts | 4 (pass-at-k, decontamination, throughput, lora-algebra) |
| GPU sharing tests | 143 (entrenar, 9 modules) |
| HumanEval pass@1 (best) | 68.90% (7B Q4K, pre-EOS-fix) |
| HumanEval pass@1 (1.5B instruct) | ~70% (eval in progress) |
| Perplexity (WikiText-2) | 6.63 (1.5B-Instruct Q4K) |
| ACs verified | 7 verified, 4 partial, 16 not tested, 2 blocked |
| Open issues | 6 (GH-8, GH-10, GH-11, GH-12, GH-13, GH-14) |

See [Implementation Status](./spec/19-implementation-status.md) for
detailed tracking.

> **Definitive spec:**
> [`docs/specifications/leaderboard-spec.md`](../specifications/leaderboard-spec.md)
> — single executive summary with
> [component files](../specifications/components/).
