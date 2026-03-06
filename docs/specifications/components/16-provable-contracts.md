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

**Current contracts (in `contracts/` directory):**

| Contract | Equations | Proof Obligations | Falsification Tests |
|---|---|---|---|
| `pass-at-k.yaml` | 1 (`pass_at_k = 1 - C(n-c,k)/C(n,k)`) | 3 (bound, monotonicity, equivalence) | 3 (FT-001, FT-002, FT-003) |
| `decontamination.yaml` | 2 (ngram_overlap, contamination_rate) | 3 (bound, gate, monotonicity) | 3 (FT-DECON-001..003) |
| `inference-throughput.yaml` | 2 (tokens_per_second, time_to_first_token) | 2 (CPU tps >= 1.0, TTFT < 5s) | 2 (FT-TPUT-001..002) |
| `lora-algebra.yaml` | 3 (lora_forward, lora_merge, adapter_params) | 3 (rank bound, merge equivalence, param compression) | 3 (FT-LORA-001..003) |
| `quantization.yaml` | 3 (quantize_dequantize, size_reduction, ordering) | 3 (identity approx, size <50%, golden ordering) | 3 (FT-QUANT-001..003) |

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
