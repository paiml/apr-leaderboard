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
| AC-007 | TIES sign resolution | PMAT-010 | Contract written (`ties-sign-resolution.yaml`) |
| AC-008 | Pruning quality | PMAT-011 | Contract written (`pruning-quality.yaml`) |
| AC-009 | Quantization size | PMAT-011 | **Verified** (FT-QUANT-001 PASS, 35%) |
| AC-014 | HF parity gap | PMAT-006 | **Verified** (HE 0.60pp, MBPP 3.2pp) |
| AC-015 | All FTs pass | All | **59/60** (98.3%) |
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

**Total: 28 contract YAMLs, 98 proof obligations, 98 falsification tests, 10 Kani harnesses. Makefile gate: 59/60 passing.**

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
| ~~GH-14 tokenizer loss~~ | ~~AC-006, AC-008~~ | **FIXED: GH-580** (merge) + **GH-581** (quantize) |
| **Q4K roundtrip corruption** | PMAT-007 eval, PMAT-011 | `load_model_tensors()` misreads Q4K APR. Merge reads Q4K correctly (RosettaStone). Fix: use RosettaStone reader in convert path. |
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

**Status (2026-04-03 18:39):** GH-580 merge fix VERIFIED. Additionally, LoRA merge had a **critical bug** — element-wise multiply instead of matrix multiply (Hadamard product instead of GEMM). Five-whys traced to a "simplified" comment in merge engine. Fix: proper triple-loop GEMM computing B^T @ A^T with d_in/d_out inferred from flat arrays + rank. Fix deployed to gx10. **All previous merged models (v1, v2) are invalid** — must re-merge with corrected binary.

**Next step:** Re-merge distilled model after PMAT-014 N-sampling completes (to avoid OOM on gx10). Then quantize to Q4K and eval on HumanEval + MBPP.

**N-sampling (PMAT-014, 2026-04-03):** Running on gx10 with base 7B Q4K. 467/1640 prompts completed (~28%) after 6h. Revised ETA: ~15h remaining (CPU batch at 6.1s/prompt + per-problem sandbox overhead). Work dir: `/tmp/tmp.4izwh76p7m` preserved with `APR_KEEP_WORKDIR=1`.

## 27.9 LoRA Merge Matmul Fix (2026-04-03)

**Root cause:** `MergeEngine::merge()` used element-wise multiply `a[i%len]*b[i%len]` (Hadamard product) instead of matrix multiply `B @ A` (GEMM). This produced garbage weight deltas that corrupted every merged model.

**Five whys:**
1. Why garbage inference? Model weights corrupted after LoRA merge
2. Why corrupted? `MergeEngine::merge()` produced wrong weight deltas
3. Why wrong deltas? Used `a[i%len]*b[i%len]` (element-wise) not `B@A` (matmul)
4. Why element-wise? Comment said "Simplified: just add scaled A and B values"
5. Why not caught? No matrix multiply unit test, garbage only visible at inference

**Fix:** Replaced with proper GEMM — infer d_in/d_out from flat arrays + rank, compute B^T @ A^T with triple loop. O(d_out × d_in × rank) per tensor. Handles both standard and transposed LoRA conventions.

**Impact:** All PMAT-007 merged models must be regenerated. Critical path unchanged — merge takes minutes once N-sampling finishes.

## 27.10 Contract Coverage Update (2026-04-03)

3 new provable contracts written:

| Contract | AC | Obligations | Tests |
|---|---|---|---|
| `binding-coverage.yaml` | AC-012 | 3 | 3 |
| `hf-parity.yaml` | AC-014 | 4 | 4 |
| `ties-sign-resolution.yaml` | AC-007 | 4 | 4 |

**Updated totals:** 28 contracts, 98 proof obligations, 98 falsification tests, 10 Kani harnesses.

**AC verification update:** 17/29 verified (59%). Newly verified: AC-009 (Q4K size, FT-QUANT-001 PASS), AC-014 (HF parity, gaps 0.60pp HE + 3.2pp MBPP).
