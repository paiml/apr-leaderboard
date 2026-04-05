# 26. QLoRA Training Loop — Contracts and Implementation

## 26.6 Provable Contracts

### 26.6.1 Required Contracts (from `../provable-contracts`)

| Contract | File | Equations Used |
|----------|------|---------------|
| `lora-algebra-v1` | `lora-algebra-v1.yaml` | `lora_shape`, `task_vector` |
| `adamw-kernel-v1` | `adamw-kernel-v1.yaml` | `adam_moments`, `adam_variance`, `bias_correction`, `weight_update` |
| `loss-functions-v1` | `loss-functions-v1.yaml` | `nll` (causal LM loss = NLL on response tokens) |
| `classification-finetune-v1` | `classification-finetune-v1.yaml` | `softmax_sum`, `label_bounds` |
| `qlora-hyperparameters-v1` | `qlora-hyperparameters-v1.yaml` | `learning_rate_scaling`, `lora_alpha_ratio`, `warmup_fraction` |
| `batch-training-v1` | `batch-training-v1.yaml` | `gradient_accumulation`, `gradient_clipping`, `batch_loss` |
| `training-loop-v1` | `training-loop-v1.yaml` | `ema_loss`, `warmup_lr`, `val_split` |
| `lora-gradient-flow-v1` | `lora-gradient-flow-v1.yaml` | Autograd-aware transpose for LoRA gradient flow |

### 26.6.2 New Contracts

**Contract: `qlora-training-loop-v1`** (updated from v0)

```yaml
metadata:
  version: 2.0.0
  description: QLoRA training loop -- cuBLAS GEMM + frozen NF4 base + trainable BF16 LoRA
  depends_on:
    - lora-algebra-v1
    - adamw-kernel-v1
    - loss-functions-v1
    - wgsl-gemm-tiled-v1            # NEW (replaces cublas-gemm-wrapper-v1)
    - nf4-dequantization-v1         # NEW
    - fused-cross-entropy-v1        # NEW
equations:
  frozen_base:
    formula: dL/dW_base = 0 (no gradient flows to base weights)
    invariants:
      - Base weights unchanged after training step
      - Only LoRA A/B receive gradients
      - Autograd skips frozen subgraph (topological pruning)
  lora_forward_wgsl:
    formula: h = WGSL_GEMM(DequantF32(W_nf4), x) + WGSL_GEMM(WGSL_GEMM(x, A), B) * (a/r)
    invariants:
      - Output shape matches base layer output shape
      - LoRA contribution is zero when B is zero-initialized
      - cuBLAS result matches naive matmul within eps < 1e-5
  response_only_loss:
    formula: loss computed only on response tokens (positions P..T-1)
    invariants:
      - Prompt tokens do not contribute to loss
      - Loss is NLL (non-negative)
  loss_decreasing:
    formula: E[L(theta_{t+1})] < E[L(theta_t)] for sufficiently small lr
    invariants:
      - Training makes progress (loss decreasing in expectation)
  gradient_checkpoint:
    formula: backward(checkpoint_recompute(layer_i)) = backward(saved_activations(layer_i))
    invariants:
      - Recomputed activations match saved activations within eps < 1e-6
      - Only checkpoint boundary tensors persist across layers
  batch_training:
    formula: loss_batch = (1/B_s) * sum_{i=1}^{B_s} loss(sample_i)
    invariants:
      - Batch gradient = mean of per-sample gradients
      - No sample duplication or loss across micro-batches
```

**Contract: `wgsl-gemm-tiled-v1`** (NEW -- replaces cublas-gemm-wrapper-v1)

```yaml
metadata:
  version: 1.0.0
  description: >
    WGSL tiled GEMM for training -- CUTLASS-derived algorithm, zero unsafe.
    128x128 thread-block tiles, 8x8 thread micro-tiles, double-buffered shared memory.
    All via wgpu safe Rust API. No cuBLAS, no FFI.
  references:
    - "NVIDIA CUTLASS (MIT licensed) -- tiling algorithm reference"
    - "Burn/CubeCL -- proof that Vulkan GEMM can match 70-80% of cuBLAS"
  depends_on:
    - matmul-kernel-v1
equations:
  gemm_dimensions:
    formula: C[m,n] = alpha * op(A)[m,k] @ op(B)[k,n] + beta * C[m,n]
    invariants:
      - Output buffer has capacity >= m * n elements
      - Workgroup grid = ceil(m/128) * ceil(n/128)
      - Each thread computes 8x8 output elements
  tiled_naive_parity:
    formula: |WGSL_GEMM(A,B) - naive(A,B)| < eps for all elements
    invariants:
      - eps < 1e-4 for F32 (no precision loss from tiling)
      - No NaN or Inf in output when inputs are finite
  double_buffer_correctness:
    formula: smem[write_stage] and smem[read_stage] never alias during compute
    invariants:
      - workgroupBarrier() between write and read phases
      - write_stage ^= 1 toggles correctly
  zero_unsafe:
    formula: unsafe_block_count(wgsl_gemm_tiled) = 0
    invariants:
      - No extern "C" declarations
      - No raw pointer dereferencing
      - All GPU ops via wgpu safe API
falsification_tests:
  - id: FALSIFY-WGSL-GEMM-001
    rule: Dimension correctness
    prediction: WGSL tiled GEMM with m=128, n=3584, k=3584 produces [128,3584] output
    test: Compare output shape and values against CPU naive matmul
  - id: FALSIFY-WGSL-GEMM-002
    rule: Non-aligned dimensions
    prediction: m=97, n=3584, k=3584 produces correct output (non-power-of-2 M)
    test: WGSL result matches naive for odd M values (tile boundary handling)
  - id: FALSIFY-WGSL-GEMM-003
    rule: alpha/beta semantics
    prediction: alpha=2.0 doubles output; beta=1.0 adds to existing C
    test: Verify C_new = 2.0 * A @ B + 1.0 * C_old
  - id: FALSIFY-WGSL-GEMM-004
    rule: Tiled = untiled
    prediction: 128x128 tiled GEMM matches 16x16 naive GEMM within eps < 1e-6
    test: Same inputs, compare tiled vs naive WGSL shader outputs
kani_harnesses:
  - id: KANI-WGSL-GEMM-001
    property: Output buffer index m*N+n never exceeds m*n for all valid (m,n)
    bound: m,n in [1..256]
  - id: KANI-WGSL-GEMM-002
    property: Shared memory index never exceeds 2*TILE_M*TILE_K
    bound: tile_m,tile_k in [1..128]
```

**Contract: `nf4-dequantization-v1`** (NEW -- transpiled from bitsandbytes via decy)

```yaml
metadata:
  version: 1.0.0
  description: NF4 dequantization -- codebook LUT + blockwise scale (transpiled from bitsandbytes)
  references:
    - "Dettmers et al. 2023 QLoRA S3.1 NormalFloat4"
    - "bitsandbytes/csrc/kernels.cu:26-153 (source for decy transpilation)"
equations:
  nf4_codebook:
    formula: NF4_LUT[i] = InvNormCDF((i + 0.5) / 16) for i in [0..15], normalized to [-1, 1]
    invariants:
      - LUT has exactly 16 entries
      - LUT[0] = -1.0, LUT[7] = 0.0, LUT[15] = 1.0
      - LUT is monotonically increasing
  blockwise_dequant:
    formula: x_i = NF4_LUT[packed_byte >> 4] * absmax[i / blocksize] (high nibble)
    formula: x_{i+1} = NF4_LUT[packed_byte & 0x0F] * absmax[i / blocksize] (low nibble)
    invariants:
      - Output element count = 2 * input byte count
      - absmax index = floor(element_index / blocksize)
  quantize_roundtrip:
    formula: quantize(dequant(code)) = code for all 16 NF4 codes
    invariants:
      - Roundtrip preserves index (not value, since quantization is lossy)
      - dQuantizeNF4 binary search finds nearest codebook entry
falsification_tests:
  - id: FALSIFY-NF4-001
    rule: LUT ordering
    prediction: NF4_LUT is strictly monotonically increasing
    test: Assert LUT[i] < LUT[i+1] for all i in [0..14]
  - id: FALSIFY-NF4-002
    rule: Roundtrip fidelity
    prediction: dQuantizeNF4(dDequantizeNF4(code)) == code for all 16 codes
    test: Exhaustive test over all 16 values
  - id: FALSIFY-NF4-003
    rule: Blockwise scale
    prediction: max|dequant(quantize(x)) - x| < 2 * absmax / 16 (half-bin width)
    test: Property test with random vectors
  - id: FALSIFY-NF4-004
    rule: GPU/CPU parity
    prediction: |nf4_dequant_gpu(data) - nf4_dequant_cpu(data)| < 1e-6
    test: Compare PTX kernel output with CPU reference for 1M elements
kani_harnesses:
  - id: KANI-NF4-001
    property: dQuantizeNF4 returns value in [0..15]
    bound: exhaustive over 16 input codes
  - id: KANI-NF4-002
    property: Blockwise absmax index never exceeds absmax array bounds
    bound: n in [1..4096], blocksize in {32, 64, 128, 256}
```

**Contract: `fused-cross-entropy-v1`** (NEW)

```yaml
metadata:
  version: 1.0.0
  description: Fused cross-entropy loss -- chunked logsumexp, no full logit materialization
  depends_on:
    - cross-entropy-kernel-v1
    - loss-functions-v1
equations:
  chunked_logsumexp:
    formula: logsumexp(x) = logsumexp([logsumexp(chunk_1), ..., logsumexp(chunk_C)])
    invariants:
      - Algebraic decomposition is exact (not approximate)
      - Result matches unfused cross_entropy within eps < 1e-5
  fused_backward:
    formula: dCE/dx_i = softmax(x_i) - 1{i=label}
    invariants:
      - Gradient written in-place into logits buffer
      - No separate gradient tensor allocated
  memory_bound:
    formula: peak_memory = O(B_s * T) not O(B_s * T * V)
    invariants:
      - Only logsumexp scalars saved (not full softmax output)
      - For V=32000: saves ~500 MB per batch vs unfused
falsification_tests:
  - id: FALSIFY-FCE-001
    rule: Fused = unfused
    prediction: |fused_ce(logits, labels) - F.cross_entropy(logits, labels)| < 1e-5
    test: Compare for random logits with vocab_size in {1000, 32000, 128256}
  - id: FALSIFY-FCE-002
    rule: Backward parity
    prediction: fused backward gradient matches unfused backward within eps < 1e-4
    test: Compare gradients for random inputs
  - id: FALSIFY-FCE-003
    rule: Chunking correctness
    prediction: Single-chunk result = multi-chunk result (exact)
    test: Compare n_chunks=1 vs n_chunks=4 for vocab_size=65536
kani_harnesses:
  - id: KANI-FCE-001
    property: logsumexp decomposition is algebraically exact
    bound: chunks in [1..4], values in [-10.0..10.0]
```

### 26.6.3 Contract Annotations on Functions

```rust
#[provable_contracts_macros::contract("qlora-training-loop-v1", equation = "frozen_base")]
fn train_step(/* ... */) { /* ... */ }

#[provable_contracts_macros::contract("adamw-kernel-v1", equation = "weight_update")]
fn optimizer_step(/* ... */) { /* ... */ }

#[provable_contracts_macros::contract("loss-functions-v1", equation = "nll")]
fn compute_causal_lm_loss(/* ... */) { /* ... */ }

#[provable_contracts_macros::contract("lora-algebra-v1", equation = "lora_shape")]
fn create_lora_layer(/* ... */) { /* ... */ }
```

### 26.6.4 Falsification Tests

| ID | Rule | Prediction | Test |
|----|------|-----------|------|
| FT-001 | Frozen base | Base weights identical before/after `train_step` | Hash base weights, compare after N steps |
| FT-002 | LoRA zero init | First forward pass without training = base model output | Compare logits: model vs model+LoRA(B=0) |
| FT-003 | Response-only loss | Changing prompt tokens doesn't change loss gradient | Perturb prompt, verify same gradient on LoRA |
| FT-004 | Loss non-negative | NLL loss >= 0 for all inputs | proptest with random logits and labels |
| FT-005 | Loss decreasing | Loss at step N < loss at step 0 (averaged over 10 runs) | Train 100 steps, compare first vs last loss |
| FT-006 | AdamW decoupled | Weight decay applied to theta, not gradient | Compare with L2-regularized Adam |
| FT-007 | Shape preservation | LoRA output shape = base layer output shape | proptest with random dimensions |
| FT-008 | Gradient flow | dL/dA != 0 and dL/dB != 0 after first step (B no longer zero) | Check gradient norms after step 1 |
| FT-009 | WGSL tiled GEMM vs naive parity | Tiled GEMM matches naive matmul within eps < 1e-4 | Random F32 matrices, compare outputs |
| FT-010 | Gradient checkpoint correctness | Recomputed activations match saved within eps < 1e-6 | Compare with/without checkpointing |
| FT-011 | Fused CE = unfused CE | Fused cross-entropy matches standard within eps < 1e-5 | Random logits, multiple vocab sizes |
| FT-012 | Batch loss = mean per-sample | Batch loss equals average of individual sample losses | Compare batch vs sequential processing |
| FT-013 | NF4 roundtrip | dQuantizeNF4(dDequantizeNF4(i)) == i for all i in [0..15] | Exhaustive 16-value test |
| FT-014 | Decy transpilation parity | Rust NF4 dequant matches C reference within eps < 1e-7 | 1M random NF4-packed bytes, compare outputs |
| FT-015 | Zero unsafe | `grep -r "unsafe" trueno-gpu/src/` returns 0 matches | No unsafe blocks, no extern C, no raw pointers |
| FT-016 | CUDA FFI eliminated | `driver/sys/`, `driver/cublas*`, `ptx/` directories removed | No CUDA dependency in the crate |

## 26.7 Implementation Plan

For the full implementation plan (Phase 0 through Phase 4), including WGSL tiled GEMM, NF4 dequant, bridge pattern, model bridge, APR export, and merge support, see [QLoRA Blockers and Status (S26c)](26c-qlora-blockers-status.md).

## 26.8 Test Plan

| Test | Type | Validates |
|------|------|-----------|
| `test_train_step_decreases_loss` | Integration | Loss at step 10 < loss at step 0 |
| `test_base_weights_frozen` | Unit | Base model weights unchanged after training |
| `test_lora_zero_init` | Unit | B=0 init -> LoRA contribution = 0 |
| `test_response_only_loss` | Unit | Prompt tokens don't contribute to gradient |
| `test_adamw_decoupled` | Unit | AdamW != L2-regularized Adam |
| `test_export_reimport` | Integration | Export -> import -> same adapter weights |
| `test_merged_model_inference` | Integration | Merged model produces valid completions |
| `test_99_completions_training` | E2E | Train on teacher completions, verify loss decrease |
| `test_cublas_naive_parity` | Unit | cuBLAS GEMM matches naive matmul within eps < 1e-3 |
| `test_nf4_dequant_roundtrip` | Unit | dQuantizeNF4(dDequantizeNF4(i)) == i for all 16 codes |
| `test_nf4_decy_parity` | Unit | Rust transpiled NF4 matches C reference within eps < 1e-7 |
| `test_fused_ce_unfused_parity` | Unit | Fused cross-entropy = unfused within eps < 1e-5 |
| `test_gradient_checkpoint_parity` | Integration | With/without checkpointing produce same gradients |
| `test_batch_loss_mean` | Unit | Batch loss = mean of per-sample losses |
| `test_cublas_transpose_flags` | Unit | CUBLAS_OP_T matches explicit transpose + CUBLAS_OP_N |
| `test_batch4_throughput` | Perf | batch_size=4 achieves >= 4x throughput vs batch_size=1 |

## 26.9 Acceptance Criteria

- **AC-FT-001:** `apr finetune model.apr --method qlora --data train.jsonl` trains for N epochs with decreasing loss
- **AC-FT-002:** Training produces an APR file with trained LoRA weights (not random init)
- **AC-FT-003:** Merged model passes `apr check` and produces valid inference output
- **AC-FT-004:** All 16 falsification tests from S26.6.4 pass
- **AC-FT-005:** All 7 provable contracts annotated and verified (4 existing + 3 new)
- **AC-FT-006:** 7B QLoRA on 99 teacher completions completes in **< 30 minutes** on gx10
- **AC-FT-007:** Distilled 7B model achieves >= 85% pass@1 on HumanEval (no regression from baseline)
- **AC-FT-008:** Training throughput >= 50 tokens/sec on gx10 GB10
- **AC-FT-009:** All NF4 dequant functions transpiled via decy with **zero** `unsafe` blocks
- **AC-FT-010:** WGSL tiled GEMM passes all 4 FALSIFY-WGSL-GEMM tests + 2 Kani harnesses
- **AC-FT-011:** **Zero `unsafe` blocks** in trueno-gpu after CUDA FFI elimination (Step 0f)
- **AC-FT-012:** trueno-gpu has **zero `extern "C"` declarations** after Step 0f
- **AC-FT-013:** WgpuTrainingPipeline loss matches CUDA training loss within eps < 0.1 on 7B model (Step 0e)
- **AC-FT-014:** CUDA code deleted ONLY after AC-FT-013 passes (prove-then-delete)
- **AC-FT-015:** ALL 6 training operations on GPU via wgpu (forward, lm_head, loss, lm_head backward, layer backward, optimizer) -- no CPU fallback for any operation
- **AC-FT-016:** 6 new WGSL shaders (nf4_dequant, backward_rmsnorm, backward_swiglu, backward_attention, fused_cross_entropy, transpose) with falsification tests

## 26.10 References

- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" arXiv:2106.09685
- Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs" arXiv:2305.14314
- Loshchilov & Hutter (2017) "Decoupled Weight Decay Regularization" arXiv:1711.05101
- Eckart-Young-Mirsky theorem (1936) -- optimal low-rank approximation
- Unsloth (Han & Han, 2024) -- Triton kernel fusions for 2-5x QLoRA speedup (https://github.com/unslothai/unsloth)
- bitsandbytes (Dettmers, 2023) -- NF4 dequantization kernels (csrc/kernels.cu, transpiled via decy)
- Chen et al. (2016) "Training Deep Nets with Sublinear Memory Cost" arXiv:1604.06174 -- gradient checkpointing
- Vulkan VK_KHR_cooperative_matrix -- tensor core access from Vulkan (same hardware as CUDA wmma)
- Burn/CubeCL -- proof that Vulkan GEMM matches CUDA on same NVIDIA GPU
- decy (PAIML) -- C-to-Rust transpiler for bitsandbytes kernel transpilation
