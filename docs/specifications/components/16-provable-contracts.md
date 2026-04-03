# Provable Contracts (Design by Contract)

Every kernel in the pipeline MUST have a provable-contracts YAML contract binding it to its mathematical specification. This ensures the optimization techniques produce correct results, not just plausible ones.

## 16.0 Implementation Status

The `provable-contracts` crate is wired into `apr-leaderboard` as a path dependency (`../provable-contracts/crates/provable-contracts`). Contract validation is integrated into the `acceptance --verify` command:

```bash
# Validate all contracts in contracts/ directory
apr-leaderboard acceptance --verify
# Output:
#   Acceptance Criteria Scaffold Verification:
#     Scaffolded: 12/27
#     Pending (needs real models): 11
#     External (needs tooling): 4
#
#   Contract validation:
#     contracts/pass-at-k.yaml — 1 equations, 3 obligations
```

**Wired APIs:**
- `provable_contracts::schema::parse_contract(path)` — Parse YAML contract files
- `provable_contracts::schema::validate_contract(&contract)` — Check equations, proof obligations, falsification tests
- `provable_contracts::error::Severity` — Filter validation violations by severity

**Current contracts (28 in `contracts/` directory, all parsed by `pv proof-status`):**

| Contract | Level | Obligs | Tests | Kani | Scope |
|---|---|---|---|---|---|
| `pass-at-k.yaml` | L2 | 3 | 3 | 0 | Eval estimator (Chen et al.) |
| `inference-throughput.yaml` | L2 | 2 | 2 | 0 | CPU/GPU throughput bounds |
| `decontamination.yaml` | L2 | 3 | 3 | 0 | N-gram overlap gate |
| `distillation.yaml` | L2 | 3 | 3 | 0 | Teacher→student quality (PMAT-007) |
| `lora-algebra.yaml` | L2 | 3 | 3 | 0 | LoRA rank/merge math |
| `quantization.yaml` | L2 | 3 | 3 | 0 | INT4/Q4K size + ordering |
| `dpo-alignment.yaml` v2.0 | L1 | 6 | 5 | 0 | DPO e2e pipeline + MBPP target (PMAT-008) |
| `qlora-training-loop.yaml` | L3 | 7 | 8 | 3 | Full training pipeline (§26) |
| `fused-cross-entropy.yaml` | L3 | 4 | 5 | 2 | Chunked CE loss |
| `nf4-dequantization.yaml` | L3 | 4 | 6 | 3 | NF4 codebook + roundtrip |
| `wgsl-gemm-tiled.yaml` | L3 | 4 | 5 | 2 | CUTLASS-derived WGSL GEMM |
| `wgsl-transpose.yaml` | L1 | 3 | 1 | 0 | GPU transpose shader |
| `gpu-output-norm.yaml` | L2 | 3 | 3 | 0 | GPU-resident RMSNorm |
| `forward-pass-perf.yaml` | L1 | 2 | 1 | 0 | Per-op layer timing |
| `lora-finetune-eval.yaml` | L2 | 3 | 3 | 0 | Train→merge→eval (PMAT-008) |
| `merge-weight-norm.yaml` v2.0 | L2 | 4 | 6 | 0 | SLERP/TIES norm + AC-024 (PMAT-010) |
| `leaderboard-gate.yaml` | L2 | 3 | 3 | 0 | AC-022 compound gate |
| `preference-pairs.yaml` | L1 | 4 | 3 | 0 | N-sampling→DPO pairs (PMAT-014) |
| `compile-binary.yaml` | L2 | 3 | 3 | 0 | apr compile (AC-010/026) |
| `pipeline-validation.yaml` | L2 | 3 | 3 | 0 | make verify/validate |
| `perplexity-baseline.yaml` | L2 | 3 | 3 | 0 | WikiText-2 PPL (AC-002) |
| `tokenizer-preservation.yaml` | L2 | 3 | 3 | 0 | GH-580/581 tokenizer in merge/quantize |
| `data-governance.yaml` | L2 | 3 | 3 | 0 | Data catalog + lineage |
| `quantization-quality.yaml` | L2 | 3 | 3 | 0 | INT4 pass@1 retention (AC-023) |
| `data-quality.yaml` | L2 | 4 | 4 | 0 | Training data quality (AC-025) |
| `pruning-quality.yaml` | L2 | 4 | 4 | 0 | Wanda pruning quality (AC-008) |
| `binding-coverage.yaml` | L2 | 3 | 3 | 0 | Contract binding coverage (AC-012) |
| `hf-parity.yaml` | L2 | 4 | 4 | 0 | HuggingFace parity gap (AC-014) |
| `ties-sign-resolution.yaml` | L2 | 4 | 4 | 0 | TIES sign conflict resolution (AC-007) |

**Totals:** 98 proof obligations, 98 falsification tests, 10 Kani harnesses. Levels: L1=4, L2=20, L3=4.

**Cross-project contracts (in `../provable-contracts/contracts/`):**

| Contract | Equations | Proof Obligations | Falsification Tests | Status |
|---|---|---|---|---|
| `gpu-multi-backend-parity-v1.yaml` | 4 (multi_backend_parity, backend_priority, bandwidth_bound, jit_correctness) | 6 (parity, no garbage, determinism, wgpu, nvrtc, bandwidth) | 7 (F-MBP-001..007) | Active |
| `gpu-context-health-v1.yaml` | 2 (fp8_guard, context_health) | 3 (FP8 disabled on Blackwell, no poison, Ada still enabled) | 3 (FT-GPU-CTX-001..003) | Verified |
| `ptx-target-parity-v1.yaml` | 3 (target_parity, no_hardcoded, jit_success) | 4 (target match, no emit_ptx, kernels with_target, JIT success) | 5 (FALSIFY-PTP-001..005) | **Violated on sm_121** |
| `gqa-kernel-v1.yaml` | 1 (GQA formula) | 8 (normalization, MHA equiv, convex bound, KV broadcast, SIMD, GPU, head mapping) | 9 (FALSIFY-GQ-001..009) | Active |

## 16.1 Contract Coverage Requirements

The leaderboard pipeline touches these kernel equivalence classes from the provable-contracts registry:

| Kernel Class | Contracts Required | Pipeline Stage |
|---|---|---|
| **E (Qwen)** | RMSNorm, SwiGLU, GQA, RoPE | Inference (eval, distill, chat) |
| **Attention** | attention-kernel-v1, flash-attention-kernel-v1 | Inference, distillation |
| **Quantization** | quantization-ordering-v1, q4k-q6k-superblock-v1 | `apr quantize`, QLoRA base weights |
| **LoRA** | lora-algebra-v1 | `apr finetune --method lora/qlora` |
| **Softmax** | softmax-kernel-v1 | Attention, sampling |
| **Matmul** | matmul-kernel-v1 | All linear layers |
| **AdamW** | adamw-kernel-v1 | Training optimizer |

## 16.2 Contract Verification Gates

Each pipeline stage MUST pass its contract obligations before proceeding:

```bash
# Verify all kernel contracts are bound and implemented
pv proof-status ../provable-contracts/contracts/ \
    --binding ../provable-contracts/contracts/aprender/binding.yaml \
    --format json

# Verify Qwen2 architecture contracts specifically
pv audit ../provable-contracts/contracts/model/qwen35-shapes-v1.yaml \
    --binding ../provable-contracts/contracts/aprender/binding.yaml

# Run falsification tests for all pipeline-relevant kernels
cargo test --features kani -p aprender -- contract
```

## 16.3 Pipeline-Specific Proof Obligations

| Obligation | Property | Verification Level | Gate |
|---|---|---|---|
| PO-LB-001 | Distillation preserves architecture invariants | L2 (falsification) | Before `apr distill` |
| PO-LB-002 | Merge preserves tensor shape flow | L3 (proptest) | Before `apr merge` |
| PO-LB-003 | Prune maintains attention head structure | L2 (falsification) | Before `apr prune` |
| PO-LB-004 | Quantization ordering matches golden order §8 | L1 (type system) | Compile-time |
| PO-LB-005 | LoRA adapter rank ≤ hidden dim | L1 (Poka-Yoke) | Compile-time |
| PO-LB-006 | Q4K dequantize × quantize ≈ identity (CPU + wgpu) | L4 (Kani, bound=256) | CI |
| PO-LB-007 | Softmax normalization: sum(output) ≈ 1.0 (CPU + wgpu) | L4 (Kani, bound=16) | CI |
| PO-LB-008 | SLERP interpolation preserves weight norms | L3 (proptest) | Before `apr merge --strategy slerp` |

## 16.4 `#[contract]` Annotations

Every function in the apr-leaderboard pipeline that performs a mathematical operation MUST carry a `#[contract]` annotation linking it to its provable-contracts YAML:

```rust
use provable_contracts_macros::contract;

#[contract("quantization-ordering-v1", equation = "quantize_int4")]
pub fn quantize_model(model: &AprModel, scheme: QuantScheme) -> Result<AprModel> {
    // Implementation — contract macro enforces binding at compile time
}

#[contract("lora-algebra-v1", equation = "lora_forward")]
pub fn lora_forward(base: &Tensor, a: &Tensor, b: &Tensor, scale: f32) -> Tensor {
    // output = base @ x + scale * (B @ (A @ x))
}
```

If the binding is missing from `contracts/aprender/binding.yaml`, the build fails. Zero tolerance for unbound kernels.

## 16.5 Falsification Test Results

Tests run via `make check-contracts` (**64 passed, 1 failed**, updated 2026-04-03):

| Category | Tests | Status | Details |
|---|---|---|---|
| pass@k | 5 | PASS | FT-001..005 (boundary, ratio, high-c) |
| throughput | 2 | PASS | 2.5 tok/s, 385ms TTFT |
| benchmark data | 3 | PASS | HumanEval 164, MBPP 974, BCB 1140 |
| decontamination | 1 | PASS | 0% HE/MBPP overlap |
| eval results | 3 | PASS | 90.85% best, 15 runs, latest >= 80% |
| distillation | 2 | PASS | 32B > 7B, 11 categories |
| MBPP eval | 1 | PASS | 76.2% >= 70% |
| AC-022 gate | 1 | **FAIL** | HE=90.85% MBPP=76.2% < 80% |
| quantization | 3 | PASS | Q4K 35% FP16, apr check, golden ordering |
| distillation data | 3 | PASS | 99 completions, valid JSONL, 99 prompts |
| oracle analysis | 2 | PASS | 96.34% upper bound, 6 never-solved |
| pipeline | 3 | PASS | 24 scripts, 22 configs, 57 targets |
| compile | 1 | PASS | apr compile available |
| data catalog | 2 | PASS | 9 contract bindings, 13 datasets |
| leaderboard coverage | 2 | PASS | 20 eval runs, 2 benchmarks |
| HF parity | 1 | PASS | 3.05pp gap (apr=90.85%, HF=87.8%) |
| contract coverage | 1 | PASS | 29 contract YAMLs >= 25 |
| structure | 29 | PASS | All 29 contract YAMLs valid |

**Makefile gate:** `make check-contracts` — **64 passed, 1 failed** (FT-GATE-001: MBPP 76.2% < 80%).

**pv proof-status:** 28/28 contracts parsed, 98 obligations, 98 tests, 10 Kani, 0/56 bindings.

See `contracts/CONTRACT_STATUS.md` for full audit trail.
