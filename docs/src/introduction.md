# APR Leaderboard Specification

**Status:** ACTIVE
**Date:** 2026-03-04
**Authors:** APR Team
**Oracle:** `batuta oracle` — entrenar (90% confidence for LoRA), realizar (85% for serving), trueno/wgpu (80% for compute)

## Quick Status

| Metric | Value |
|---|---|
| CLI subcommands | 21 |
| Wired to real APIs | 21 (all subcommands) |
| Tests | 368 |
| Line coverage | 96.2% |
| Clippy warnings | 0 |
| Source modules | 13 |
| Pipeline configs | 13 (6 models + 7 recipes) |
| Provable contracts | 1 (pass-at-k, 3 proof obligations) |
| GPU sharing tests | 143 (entrenar, 9 modules) |
| HumanEval pass@1 | 68.90% (7B Q4K, pre-EOS-fix) |

See [Implementation Status](./spec/19-implementation-status.md) for detailed tracking.

> **Definitive spec:** [`docs/specifications/leaderboard-spec.md`](../specifications/leaderboard-spec.md) — single executive summary with [component files](../specifications/components/).
