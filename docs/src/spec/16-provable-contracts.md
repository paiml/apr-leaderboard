#  Provable Contracts (Design by Contract)

Every kernel in the pipeline MUST have a provable-contracts YAML contract binding it to its mathematical specification. This ensures the optimization techniques produce correct results, not just plausible ones.

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
| PO-LB-006 | Q4K dequantize × quantize ≈ identity | L4 (Kani, bound=256) | CI |
| PO-LB-007 | Softmax normalization: sum(output) ≈ 1.0 | L4 (Kani, bound=16) | CI |
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
