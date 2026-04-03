# PMAT Roadmap

Work item dependency graph and critical path to AC-022 (leaderboard submission gate).

## 27.1 Work Item Summary

| ID | Title | Status | Depends On | ACs |
|---|---|---|---|---|
| PMAT-006 | Baseline Evaluation Gate | **DONE** | — | AC-021 |
| PMAT-017 | Full Pipeline Orchestration | **DONE** | — | AC-011, AC-027 |
| PMAT-037 | GPU Training & Parity | **DONE** | — | AC-028, AC-029 |
| PMAT-007 | 32B→7B Text-Based Distillation | **DONE** (pipeline) | PMAT-006 | AC-003 |
| PMAT-014 | Preference Pair Generation | **IN PROGRESS** | PMAT-006 | AC-020 |
| PMAT-008 | DPO Alignment Pipeline | **READY** | PMAT-014 | AC-020, AC-022 |
| PMAT-010 | TIES Merge Specialists | **PENDING** | PMAT-007, PMAT-008 | AC-006, AC-007, AC-024 |
| PMAT-011 | Final Submission Artifact | **PENDING** | PMAT-010 | AC-008, AC-009, AC-022 |

## 27.2 Dependency DAG

```
PMAT-006 (DONE: 85.37% baseline)
├── PMAT-007 (DONE: adapter trained, merged, Q4K — awaiting eval)
│   └── PMAT-010 (PENDING: TIES merge)
│       └── PMAT-011 (PENDING: final artifact → AC-022)
├── PMAT-014 (IN PROGRESS: N-sampling preference pairs)
│   └── PMAT-008 (READY: DPO contract v2.0, pipeline defined)
│       └── PMAT-010 (PENDING: TIES merge)
└── PMAT-037 (DONE: wgpu training verified, 13 KAIZEN fixes)

PMAT-017 (DONE: 56 Makefile targets)
```

## 27.3 Critical Path

The shortest path to AC-022 (leaderboard submission):

```
PMAT-014 → PMAT-008 → PMAT-010 → PMAT-011 → AC-022
  (pairs)    (DPO)     (merge)    (quantize)   (gate)
```

**Parallel track:** PMAT-007 (distillation) feeds into PMAT-010 independently.

### Critical Path Estimates

| Step | Blocking On | Unblocks |
|---|---|---|
| PMAT-014: Generate N-sampling pairs | gx10 GPU (3h eval) | PMAT-008 |
| PMAT-008: DPO training on pairs | gx10 GPU (40 min) | PMAT-010 |
| PMAT-007: Distillation fine-tune | gx10 GPU (40 min) | PMAT-010 |
| PMAT-010: TIES merge two adapters | CPU (minutes) | PMAT-011 |
| PMAT-011: Prune → quantize → eval | gx10 GPU (3h eval) | AC-022 gate |

## 27.4 AC Coverage by PMAT

| AC | Required By | PMAT Item | Current Status |
|---|---|---|---|
| AC-002 | Perplexity baseline | PMAT-006 | **Verified** (6.63 PPL) |
| AC-003 | Distillation quality | PMAT-007 | **Verified** (99/99 completions) |
| AC-006 | Merge norm preservation | PMAT-010 | Contract written |
| AC-007 | TIES sign resolution | PMAT-010 | Not yet tested |
| AC-008 | Pruning quality | PMAT-011 | Not yet tested |
| AC-009 | Quantization size | PMAT-011 | FT-QUANT-001 PASS |
| AC-015 | All FTs pass | All | **54/55** (98.2%) |
| AC-020 | DPO alignment | PMAT-008 | **Verified** |
| AC-022 | Compound gate (HE+MBPP) | PMAT-011 | FAIL (MBPP 76.2%) |
| AC-024 | Merge > specialist | PMAT-010 | Not yet tested |

## 27.5 Contract Coverage

Each PMAT item has associated provable contracts:

| PMAT | Contracts | FTs | Makefile Tests | Status |
|---|---|---|---|---|
| PMAT-006 | pass-at-k, inference-throughput, perplexity-baseline | 8 | 7 | All passing |
| PMAT-017 | pipeline-validation | 3 | 3 | All passing |
| PMAT-037 | wgsl-gemm-tiled, nf4-dequantization, fused-cross-entropy, gpu-output-norm, wgsl-transpose, forward-pass-perf, qlora-training-loop | 29 | 0 (GPU) | pv L3 |
| PMAT-007 | distillation, lora-finetune-eval, tokenizer-preservation | 9 | 5 | Pipeline done, eval pending |
| PMAT-014 | preference-pairs | 3 | 0 (pending N-sampling) | Contract written |
| PMAT-008 | dpo-alignment v2.0, lora-finetune-eval | 8 | 0 (pending DPO) | Contract v2.0 with e2e pipeline |
| PMAT-010 | merge-weight-norm v2.0 | 6 | 0 (pending merge) | Contract v2.0 with AC-024 tests |
| PMAT-011 | leaderboard-gate, quantization, compile-binary | 9 | 4 (1 failing) | MBPP gate |

**Total: 22 contract YAMLs, 76 proof obligations, 76 falsification tests, 10 Kani harnesses. Makefile gate: 56/57 passing.**

## 27.6 Gap Analysis

### MBPP Gap (3.8pp to AC-022)

Current: 76.2% → Target: 80.0%

| Strategy | Expected Gain | Evidence |
|---|---|---|
| DPO on borderline problems | +2-4pp | HumanEval few-shot +1.83pp from standard |
| Teacher distillation (32B→7B) | +1-3pp | 32B is 90.85% vs 7B 85.37% on HumanEval |
| TIES merge (code + reasoning) | +1-2pp | Literature: TIES > single specialist |
| N-sampling with temperature | +0-1pp | pass@10 upper bound analysis |

**Conservative estimate:** DPO alone should close 2-3pp, combined with distillation gets to 80%+.

### Blocked Items

| Blocker | Affects | Resolution |
|---|---|---|
| naga SPIR-V bug | Cooperative matrix GEMM (perf) | Wait for naga fix or use tiled GEMM |
| ~~GH-14 tokenizer loss~~ | ~~AC-006, AC-008~~ | **FIXED: GH-580** — AprV2Writer preserves tokenizer |
| SafeTensors FP16 import | AC-014, AC-023 (parity) | Fix in realizar |

## 27.7 GH-580: Tokenizer Preservation Fix (2026-04-03)

**Root cause:** `run_merge()` used `AprWriter` (v1) which creates empty tokenizer. Base model is APR v2 with tokenizer in `AprV2Metadata.custom` HashMap.

**Fix:** Read base model with `AprV2Reader`, clone metadata (preserving tokenizer), use `AprV2Writer` for output. Also supports SafeTensors adapter input (wgpu training pipeline).

**Impact:** Unblocks PMAT-007 eval (distilled model can now run inference), PMAT-008 (DPO merge), PMAT-010 (TIES merge). All merge operations now preserve embedded tokenizer.

**Contract:** `tokenizer-preservation-v1.yaml` — 2 equations, 3 proof obligations, 3 falsification tests.

## 27.8 PMAT-007 Pipeline Artifacts (2026-04-03)

| Artifact | Size | Path (gx10) |
|---|---|---|
| Teacher completions | 240 KB | `data/distill/teacher-completions.jsonl` (99 prompts) |
| QLoRA adapter | 40 MB | `checkpoints/qwen2.5-coder-7b-distilled-qlora.apr` |
| Remapped adapter | 40 MB | `checkpoints/qwen2.5-coder-7b-distilled-qlora-remapped.safetensors` |
| Merged model (FP32) | 30 GB | `checkpoints/qwen2.5-coder-7b-distilled-merged.apr` |
| Quantized (Q4K) | 6.2 GB | `checkpoints/qwen2.5-coder-7b-distilled-q4k.apr` |
| Tokenizer | 7 MB | `checkpoints/qwen2.5-coder-7b-distilled-q4k.tokenizer.json` |

**Status:** Merge with GH-580 fix VERIFIED (tokenizer preserved, 10/10 `apr check`). Quantize path still loses tokenizer — needs GH-581 fix in `apr_convert()`. GGUF roundtrip workaround produces corrupted weights. Distilled model eval **BLOCKED** on quantize tokenizer fix.

**N-sampling launched (2026-04-03):** PMAT-014 HumanEval N-sampling (NUM_SAMPLES=10, T=0.8) running on gx10 with base 7B Q4K. 1640 prompts in CPU batch mode. ETA ~3h. Work dir preserved for preference pair extraction.
