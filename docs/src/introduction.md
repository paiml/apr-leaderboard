# APR Leaderboard Specification

**Status:** ACTIVE
**Version:** 2.1.0
**Date:** 2026-03-05
**Authors:** APR Team

## Quick Status

| Metric | Value |
|---|---|
| `apr` CLI subcommands verified | 16 |
| Makefile targets | 33 |
| Shell scripts | 5 |
| YAML configs | 17 (6 models + 7 recipes + 1 eval + 2 pipeline + 1 data) |
| Python scripts | 0 (zero-Python constraint) |
| TOML configs | 0 (YAML-only) |
| Provable contracts | 1 (pass-at-k, 3 proof obligations) |
| GPU sharing tests | 143 (entrenar, 9 modules) |
| HumanEval pass@1 | 68.90% (7B Q4K, pre-EOS-fix) |

See [Implementation Status](./spec/19-implementation-status.md) for
detailed tracking.

> **Definitive spec:**
> [`docs/specifications/leaderboard-spec.md`](../specifications/leaderboard-spec.md)
> — single executive summary with
> [component files](../specifications/components/).
