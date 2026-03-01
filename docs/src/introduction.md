# APR Leaderboard Specification

**Status:** DRAFT
**Date:** 2026-03-01
**Authors:** APR Team
**Oracle:** `batuta oracle` — entrenar (90% confidence for LoRA), realizar (85% for serving), trueno (80% for compute)

## Quick Status

| Metric | Value |
|---|---|
| CLI subcommands | 21 |
| Wired to real APIs | 8 (eval, finetune, distill, merge, check, acceptance, apr_bridge) |
| Tests | 374 |
| Line coverage | 96.2% |
| Clippy warnings | 0 |
| Source modules | 13 |
| Pipeline configs | 9 (5 models + 4 recipes) |
| Provable contracts | 1 (pass-at-k, 3 proof obligations) |

See [Implementation Status](./spec/19-implementation-status.md) for detailed tracking.
