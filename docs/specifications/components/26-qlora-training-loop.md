# 26. QLoRA Training Loop Specification

## 26.1 Problem Statement

`apr finetune --method qlora` creates LoRA adapter tensors with random initialization and writes an APR file, but does not execute training. The `execute_training()` function in `aprender/crates/apr-cli/src/commands/finetune.rs` is a stub — it calls `create_lora_tensors()` (Kaiming init) and exits.

**Root cause:** aprender has no training loop. The training loop exists in entrenar (`InstructPipeline::train_step`) but is not wired to the `apr finetune` CLI.

## 26.2 Existing Infrastructure Audit

### 26.2.1 What EXISTS (entrenar)

| Component | Location | Status |
|-----------|----------|--------|
| Autograd engine | `entrenar/src/autograd/` | Tape-based, backward ops for matmul, attention, activations, normalize |
| AdamW optimizer | `entrenar/src/optim/adamw.rs` | Full implementation with decoupled weight decay |
| LR schedulers | `entrenar/src/optim/scheduler/` | Cosine decay, linear warmup, step decay |
| Cross-entropy loss | `entrenar/src/finetune/classification.rs:577` | With autograd backward |
| Causal LM loss | `entrenar/src/finetune/instruct_pipeline.rs` | Response-only masking |
| LoRA layers | `entrenar/src/finetune/instruct_pipeline.rs` | `LoraLinear` with trainable A/B |
| Training loop | `entrenar/src/finetune/instruct_trainer.rs:156` | Epoch management, validation, checkpointing, early stopping |
| `train_step` | `entrenar/src/finetune/instruct_pipeline.rs:574` | Forward → loss → backward → optimizer, CPU + CUDA paths |
| Gradient clipping | `entrenar/src/finetune/instruct_pipeline.rs` | Max-norm clipping |
| CUDA training | `entrenar/src/autograd/cuda_training.rs` | NF4 QLoRA on GPU |
| Memory planner | `entrenar-lora/src/memory.rs` | VRAM estimation for QLoRA configs |
| Merge engine | `entrenar-lora/src/merge.rs` | Adapter merge into base model |

### 26.2.2 What EXISTS (aprender)

| Component | Location | Status |
|-----------|----------|--------|
| CLI `finetune` command | `apr-cli/src/commands/finetune.rs` | Parses args, plans config, creates adapter APR — **no training** |
| LoRA tensor creation | `apr-cli/src/commands/finetune.rs:create_lora_tensors` | Kaiming init A, zero B |
| APR writer | `aprender/src/serialization/apr.rs` | Writes .apr with metadata + tensors |
| Model loading | `realizar/src/gguf/` | `OwnedQuantizedModel` from .apr files |
| Autograd engine | `aprender/src/autograd/` | Tape-based reverse-mode AD (independent from entrenar) |
| Optimizers | `aprender/src/nn/optim/` | SGD, Adam, AdamW, RMSprop |
| Loss functions | `aprender/src/nn/loss.rs` | MSE, L1, SmoothL1, CrossEntropy |
| LoRA adapter | `aprender/src/transfer/lora.rs` | `LoRAAdapter` with `apply()` and `delta_weight()` |
| QLoRA example | `entrenar/examples/llama2/finetune_qlora.rs` | Complete QLoRA training example (~300 lines) |

### 26.2.3 What is MISSING

| Component | Gap | Required For |
|-----------|-----|-------------|
| Wiring `InstructPipeline` into `apr finetune` | `execute_training()` creates tensors but doesn't call entrenar | Training execution |
| APR model → entrenar model bridge | `OwnedQuantizedModel` → entrenar's model trait | Forward pass in training |
| Data loader for JSONL | Parse `{"instruction": ..., "response": ...}` → tokenized pairs | Training data |
| Checkpoint-to-APR export | Save trained LoRA weights back to .apr format | Output |
| Tokenizer integration | APR sibling tokenizer → entrenar tokenizer interface | Tokenization |

## 26.3 Architecture: Bridge Pattern

The fix is NOT reimplementing training in aprender. The fix is **bridging** aprender's model loading + CLI with entrenar's training loop.

```
apr finetune model.apr --method qlora --data train.jsonl --output distilled.apr
    │
    ├── 1. Load model: realizar::OwnedQuantizedModel::from_apr(path)
    ├── 2. Load tokenizer: sibling tokenizer.json
    ├── 3. Load data: parse JSONL → Vec<(instruction, response)>
    ├── 4. Create InstructPipeline with model + tokenizer + LoRA config
    ├── 5. Create InstructTrainer with pipeline + training config
    ├── 6. trainer.train() → epoch loop with loss/backward/optimizer
    ├── 7. Export trained LoRA weights → APR file
    └── 8. Optionally merge: base + adapter → merged APR
```

## 26.4 Mathematical Specification

### 26.4.1 QLoRA Forward Pass (Unsloth-informed, per Dettmers et al. 2023)

For each linear layer `W ∈ ℝ^{m×n}` in the transformer, with batch size `B_s`:

```
W_bf16 = DequantNF4→BF16(W_nf4)     # Custom kernel: NF4 LUT lookup × absmax (transpiled via decy)
h_base = cuBLAS_GEMM(x, W_bf16^T)   # cublasGemmEx: BF16 inputs, FP32 accumulation
h_lora = cuBLAS_GEMM(cuBLAS_GEMM(x, A), B) * (α/r)  # Two small GEMMs via cuBLAS
h = h_base + h_lora                  # Fused with cuBLAS addmm (alpha=s, beta=1)
```

Where:
- `A ∈ ℝ^{n×r}` — LoRA down-projection (Kaiming init), BF16
- `B ∈ ℝ^{r×m}` — LoRA up-projection (zero init), BF16
- `r` — LoRA rank (e.g., 32)
- `α` — LoRA alpha scaling (e.g., 64)
- `x ∈ ℝ^{B_s×n}` — batched input hidden states (batch_size × hidden_dim), BF16

**Critical architecture decision (from Unsloth analysis):** All GEMM operations use cuBLAS
(via `cublasGemmEx`), NOT custom PTX kernels. Custom kernels handle only the NF4→BF16
dequantization and element-wise fusions (RMSNorm, SwiGLU, RoPE, cross-entropy). This
matches the universal industry pattern: bitsandbytes, Unsloth, torchtune, and PEFT all
use cuBLAS for GEMM. Custom GEMM kernels are 10-50x slower than cuBLAS for training-sized
matrices.

**Transpilation via decy:** The NF4 dequantization kernels are transpiled from
bitsandbytes' `csrc/kernels.cu` (2400 LOC) using `../decy` (C-to-Rust transpiler).
Tier 1 functions (pure math: NF4 LUT, `dQuantizeNF4`, `dDequantizeNF4`) transpile
directly. Tier 3 functions (CUDA kernels) have their algorithms transpiled and
re-parallelized via trueno-gpu's PTX builder.

### 26.4.2 Causal Language Model Loss (Fused Cross-Entropy)

For a sequence batch `[t₁, t₂, ..., t_T]` with prompt length `P`:

```
# Fused: never materialize full [B_s × T, V] logit tensor
for chunk in chunks(hidden_states, CHUNK_SIZE=65536):
    logits_chunk = cuBLAS_GEMM(chunk, lm_head^T)    # [B_s, chunk, V]
    logsumexp_chunk = log(sum(exp(logits_chunk)))     # [B_s, chunk] scalar per token
    loss_chunk -= logits_chunk[labels] - logsumexp    # Accumulate NLL

loss = sum(loss_chunks) / R   # R = response tokens only
```

**Memory savings (from Unsloth):** Avoids materializing the full `[B_s × T, V]` logit
tensor (e.g., 4 × 2048 × 32000 × 2 = 500 MB). Instead, only `[B_s × T]` logsumexp
scalars are saved (~32 KB). Backward writes gradients in-place into the logits buffer.
For 256K-vocab models, this saves ~8 GB.

Where `R = T - P` is the number of response tokens.

### 26.4.3 Backward Pass (LoRA only, with gradient checkpointing)

Gradients flow only through LoRA A and B matrices. All backward GEMMs use cuBLAS:

```
# Re-dequantize base weight for backward (gradient checkpointing: not saved from forward)
W_bf16 = DequantNF4→BF16(W_nf4)

# Gradient w.r.t. input (for upstream layers)
∂L/∂x = cuBLAS_GEMM(∂L/∂h, W_bf16) + cuBLAS_GEMM(cuBLAS_GEMM(∂L/∂h, B^T), A^T) * (α/r)

# LoRA gradients (via cuBLAS addmm_ with fused scaling)
∂L/∂B = cuBLAS_GEMM((A^T @ x)^T, ∂L/∂h) * (α/r)   # addmm_(alpha=α/r, beta=0)
∂L/∂A = cuBLAS_GEMM(x^T, ∂L/∂h @ B^T) * (α/r)     # addmm_(alpha=α/r, beta=0)
```

Base weights `W_nf4` receive no gradient (frozen). The autograd engine skips the
entire frozen subgraph via topological pruning (per PyTorch autograd architecture).

**Gradient checkpointing:** Activations are NOT saved across layers. Each layer
boundary is a checkpoint; intermediate activations (RMSNorm output, attention scores,
FFN intermediates) are recomputed during the backward pass. This trades ~33% extra
compute for ~60% memory savings, enabling batch_size=4-8 instead of 1.

**In-place memory reuse (from Unsloth):** Input activation `X` is overwritten with
`∂L/∂X` when no longer needed. SwiGLU backward writes derivatives into input buffers.
Dequantized weights are immediately freed after each backward GEMM.

### 26.4.4 AdamW Update (per Loshchilov & Hutter 2017)

For each LoRA parameter `θ ∈ {A, B}`:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          # First moment
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          # Second moment
m̂_t = m_t / (1 - β₁ᵗ)                         # Bias-corrected first moment
v̂_t = v_t / (1 - β₂ᵗ)                         # Bias-corrected second moment
θ_t = θ_{t-1} - lr · (m̂_t / (√v̂_t + ε) + λ · θ_{t-1})  # Decoupled weight decay
```

Default hyperparameters: `β₁=0.9, β₂=0.999, ε=1e-8, λ=0.01`.

### 26.4.5 Learning Rate Schedule (Cosine with Warmup)

```
if step < warmup_steps:
    lr = lr_base * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + cos(π * progress))
```

## 26.5 Memory Model

For a model with `P` parameters, LoRA rank `r`, `L` adapted layers, batch size `B_s`:

```
Trainable params:    T = 2 · r · d · L · K    (A and B per layer per projection, K=7)
Base model:          P_bytes / 2               (NF4 = 0.5 bytes/param)
Dequant buffer:      max(m,n) × d × 2 bytes   (single BF16 weight, reused per layer)
LoRA adapters:       T × 2 bytes              (BF16)
Optimizer states:    T × 8 bytes              (m + v, both FP32)
Activations:         B_s × S × d × 2 bytes    (per checkpoint boundary, BF16)
Gradients:           T × 2 bytes              (BF16, FP32 accumulation in cuBLAS)
cuBLAS workspace:    ~256 MB                   (cuBLAS internal workspace)

Total ≈ P/2 + 12·T + B_s·S·d·2·√L + 256MB
```

Note: `√L` factor from gradient checkpointing (only checkpoint boundaries saved,
not all L layers).

For 7B Q4K, rank 32, 28 layers, batch_size=4:
- Base model: 3.75 GB (Q4K)
- Dequant buffer: 18944 × 3584 × 2 = 136 MB (reused, single largest weight matrix)
- LoRA: 2 × 32 × 3584 × 28 × 7 ≈ 45M params × 2 = 0.09 GB
- Optimizer: 45M × 8 = 0.36 GB
- Activations: 4 × 512 × 3584 × 2 × √28 ≈ 78 MB (with gradient checkpointing)
- cuBLAS workspace: 256 MB
- **Total: ~4.7 GB** (fits easily on gx10 119 GB, leaves room for batch_size=8)

**Comparison with v1 spec:** Previous spec had batch_size=1 with FP32 LoRA (5.5 GB).
New spec uses BF16 LoRA + gradient checkpointing + cuBLAS, achieving lower memory
at 4x batch size. The memory savings enable the throughput gains (cuBLAS GEMM
utilization scales with batch size).

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
  description: QLoRA training loop — cuBLAS GEMM + frozen NF4 base + trainable BF16 LoRA
  depends_on:
    - lora-algebra-v1
    - adamw-kernel-v1
    - loss-functions-v1
    - cublas-gemm-wrapper-v1        # NEW
    - nf4-dequantization-v1         # NEW
    - fused-cross-entropy-v1        # NEW
equations:
  frozen_base:
    formula: ∂L/∂W_base = 0 (no gradient flows to base weights)
    invariants:
      - Base weights unchanged after training step
      - Only LoRA A/B receive gradients
      - Autograd skips frozen subgraph (topological pruning)
  lora_forward_cublas:
    formula: h = cublasGemmEx(DequantBF16(W_nf4), x) + cublasGemmEx(cublasGemmEx(x, A), B) * (α/r)
    invariants:
      - Output shape matches base layer output shape
      - LoRA contribution is zero when B is zero-initialized
      - cuBLAS result matches naive matmul within ε < 1e-5
  response_only_loss:
    formula: loss computed only on response tokens (positions P..T-1)
    invariants:
      - Prompt tokens do not contribute to loss
      - Loss is NLL (non-negative)
  loss_decreasing:
    formula: E[L(θ_{t+1})] < E[L(θ_t)] for sufficiently small lr
    invariants:
      - Training makes progress (loss decreasing in expectation)
  gradient_checkpoint:
    formula: backward(checkpoint_recompute(layer_i)) = backward(saved_activations(layer_i))
    invariants:
      - Recomputed activations match saved activations within ε < 1e-6
      - Only checkpoint boundary tensors persist across layers
  batch_training:
    formula: loss_batch = (1/B_s) · Σ_{i=1}^{B_s} loss(sample_i)
    invariants:
      - Batch gradient = mean of per-sample gradients
      - No sample duplication or loss across micro-batches
```

**Contract: `cublas-gemm-wrapper-v1`** (NEW)

```yaml
metadata:
  version: 1.0.0
  description: cuBLAS GEMM wrapper for training — correct dimensions, no OOB
  depends_on:
    - matmul-kernel-v1
equations:
  gemm_dimensions:
    formula: C[m,n] = α · op(A)[m,k] @ op(B)[k,n] + β · C[m,n]
    invariants:
      - lda >= max(1, rows(A)) for column-major layout
      - ldb >= max(1, rows(B)) for column-major layout
      - ldc >= max(1, m)
      - Transpose flags match actual memory layout
  cublas_naive_parity:
    formula: |cuBLAS(A,B) - naive(A,B)| < ε for all elements
    invariants:
      - ε < 1e-3 for BF16 inputs with FP32 accumulation
      - No NaN or Inf in output when inputs are finite
  bf16_accumulation:
    formula: cuBLAS uses CUBLAS_COMPUTE_32F for BF16 inputs
    invariants:
      - Internal accumulation in FP32 (not BF16)
      - Output cast to BF16 after accumulation
falsification_tests:
  - id: FALSIFY-CUBLAS-001
    rule: Dimension correctness
    prediction: cublasGemmEx with m=128, n=3584, k=3584 produces [128,3584] output
    test: Compare output shape and values against naive matmul
  - id: FALSIFY-CUBLAS-002
    rule: Transpose correctness
    prediction: CUBLAS_OP_T produces same result as explicit transpose + CUBLAS_OP_N
    test: Compare both paths for random matrices
  - id: FALSIFY-CUBLAS-003
    rule: No OOB on non-aligned dimensions
    prediction: m=97, n=3584, k=3584 produces correct output (non-power-of-2 M)
    test: cuBLAS result matches naive for odd M values
  - id: FALSIFY-CUBLAS-004
    rule: alpha/beta semantics
    prediction: alpha=2.0 doubles output; beta=1.0 adds to existing C
    test: Verify C_new = 2.0 * A @ B + 1.0 * C_old
kani_harnesses:
  - id: KANI-CUBLAS-001
    property: Leading dimension >= max(1, row_count)
    bound: m,n,k in [1..64]
```

**Contract: `nf4-dequantization-v1`** (NEW — transpiled from bitsandbytes via decy)

```yaml
metadata:
  version: 1.0.0
  description: NF4 dequantization — codebook LUT + blockwise scale (transpiled from bitsandbytes)
  references:
    - "Dettmers et al. 2023 QLoRA §3.1 NormalFloat4"
    - "bitsandbytes/csrc/kernels.cu:26-153 (source for decy transpilation)"
equations:
  nf4_codebook:
    formula: NF4_LUT[i] = Φ⁻¹((i + 0.5) / 16) for i in [0..15], normalized to [-1, 1]
    invariants:
      - LUT has exactly 16 entries
      - LUT[0] = -1.0, LUT[7] = 0.0, LUT[15] = 1.0
      - LUT is monotonically increasing
  blockwise_dequant:
    formula: x_i = NF4_LUT[packed_byte >> 4] * absmax[i / blocksize] (high nibble)
    formula: x_{i+1} = NF4_LUT[packed_byte & 0x0F] * absmax[i / blocksize] (low nibble)
    invariants:
      - Output element count = 2 × input byte count
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
  description: Fused cross-entropy loss — chunked logsumexp, no full logit materialization
  depends_on:
    - cross-entropy-kernel-v1
    - loss-functions-v1
equations:
  chunked_logsumexp:
    formula: logsumexp(x) = logsumexp([logsumexp(chunk_1), ..., logsumexp(chunk_C)])
    invariants:
      - Algebraic decomposition is exact (not approximate)
      - Result matches unfused cross_entropy within ε < 1e-5
  fused_backward:
    formula: ∂CE/∂x_i = softmax(x_i) - 1{i=label}
    invariants:
      - Gradient written in-place into logits buffer
      - No separate gradient tensor allocated
  memory_bound:
    formula: peak_memory = O(B_s × T) not O(B_s × T × V)
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
    prediction: fused backward gradient matches unfused backward within ε < 1e-4
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
| FT-006 | AdamW decoupled | Weight decay applied to θ, not gradient | Compare with L2-regularized Adam |
| FT-007 | Shape preservation | LoRA output shape = base layer output shape | proptest with random dimensions |
| FT-008 | Gradient flow | ∂L/∂A ≠ 0 and ∂L/∂B ≠ 0 after first step (B no longer zero) | Check gradient norms after step 1 |
| FT-009 | cuBLAS vs naive parity | cuBLAS GEMM matches naive matmul within ε < 1e-3 | Random BF16 matrices, compare outputs |
| FT-010 | Gradient checkpoint correctness | Recomputed activations match saved within ε < 1e-6 | Compare with/without checkpointing |
| FT-011 | Fused CE = unfused CE | Fused cross-entropy matches standard within ε < 1e-5 | Random logits, multiple vocab sizes |
| FT-012 | Batch loss = mean per-sample | Batch loss equals average of individual sample losses | Compare batch vs sequential processing |
| FT-013 | NF4 roundtrip | dQuantizeNF4(dDequantizeNF4(i)) == i for all i in [0..15] | Exhaustive 16-value test |
| FT-014 | Decy transpilation parity | Rust NF4 dequant matches C reference within ε < 1e-7 | 1M random NF4-packed bytes, compare outputs |

## 26.7 Implementation Plan

### Phase 0: cuBLAS Training GEMM + NF4 Dequant (trueno-gpu + decy)

**Priority: HIGHEST — this is the 100-500x speedup.**

**Step 0a: Transpile bitsandbytes NF4 kernels via decy**

```bash
# Tier 1: Pure C math functions → Rust (direct transpilation)
decy transpile bitsandbytes/csrc/kernels.cu \
  --functions dDequantizeNF4,dQuantizeNF4,nf4_dequantization_lut \
  --output trueno-gpu/src/kernels/quantize/nf4_bnb.rs
```

Tier 1 functions (pure math, no CUDA deps):
- `nf4_dequantization_lut[16]` → `const NF4_LUT: [f32; 16]`
- `dDequantizeNF4(val)` → `fn dequantize_nf4(val: u8) -> f32`
- `dQuantizeNF4(x)` → `fn quantize_nf4(x: f32) -> u8`

Tier 3 algorithms (CUDA kernels → trueno PTX builder):
- `kDequantizeBlockwise` algorithm → PTX kernel via trueno's builder
- `kQuantizeBlockwise` algorithm → PTX kernel via trueno's builder

**Step 0b: cuBLAS GEMM wrapper in trueno-gpu (using existing FFI)**

trueno-gpu already has hand-written cuBLAS FFI bindings in `driver/cublaslt_sys.rs`
(same sovereign stack pattern as the CUDA driver API — `dlopen`, no external crates).
Add `cublas_gemm_bf16` using the existing cuBLASLt infrastructure:

```rust
/// cuBLAS GEMM for training: BF16 inputs, FP32 accumulation, BF16 output.
/// Transpiled calling convention from bitsandbytes/csrc/ops.cu:223-231.
#[provable_contracts_macros::contract("cublas-gemm-wrapper-v1", equation = "gemm_dimensions")]
pub fn cublas_gemm_bf16(
    handle: cublasHandle_t,
    m: u32, n: u32, k: u32,
    a: &GpuBuffer<bf16>,      // [m, k] or [k, m] depending on transpose
    b: &GpuBuffer<bf16>,      // [k, n] or [n, k] depending on transpose
    c: &mut GpuBuffer<bf16>,  // [m, n] output
    transpose_a: bool,
    transpose_b: bool,
    alpha: f32,
    beta: f32,
) -> Result<()> {
    // cublasGemmEx(handle, op_a, op_b, m, n, k,
    //   &alpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb,
    //   &beta, c, CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
}
```

**Step 0c: NF4 dequant → BF16 → cuBLAS GEMM pipeline**

```rust
/// Dequantize NF4 to BF16, then cuBLAS GEMM. Reuses global dequant buffer.
/// This is the Unsloth/bitsandbytes universal pattern.
#[provable_contracts_macros::contract("nf4-dequantization-v1", equation = "blockwise_dequant")]
pub fn nf4_gemm_cublas(
    nf4_weight: &NF4Weight,    // Packed NF4 + absmax
    input: &GpuBuffer<bf16>,   // [batch, hidden]
    output: &mut GpuBuffer<bf16>, // [batch, out_dim]
    dequant_buffer: &mut GpuBuffer<bf16>, // Reused across layers
) -> Result<()> {
    // 1. PTX kernel: dequant NF4 → BF16 into buffer (transpiled from bitsandbytes)
    // 2. cuBLAS GEMM: output = input @ dequant_buffer^T
}
```

**Step 0d: Batch collation**

Add batch_size parameter to training config. Collate multiple samples into
a single `[batch_size × seq_len, hidden_dim]` tensor. Pad shorter sequences,
mask padding in loss computation.

### Phase 1: Bridge `apr finetune` → entrenar (aprender change)

**File:** `aprender/crates/apr-cli/src/commands/finetune.rs`

Replace the stub `execute_training()` with:

```rust
fn execute_training(
    model_path: &Path,
    config: &OptimalConfig,
    data_path: &Path,
    output_path: &Path,
    epochs: u32,
    learning_rate: f64,
    json_output: bool,
) -> Result<()> {
    // 1. Load Q4K model via realizar
    let mapped = realizar::apr::MappedAprModel::from_path(model_path)?;
    let model = realizar::gguf::OwnedQuantizedModel::from_apr(&mapped)?;

    // 2. Load tokenizer (sibling .tokenizer.json)
    let tokenizer = load_sibling_tokenizer(model_path)?;

    // 3. Load JSONL training data
    let samples = load_instruct_jsonl(data_path)?;

    // 4. Create InstructPipeline (entrenar)
    let pipeline_config = InstructPipelineConfig {
        rank: config.rank,
        alpha: config.alpha,
        learning_rate: learning_rate as f32,
        max_seq_len: 512,
        gradient_clip_norm: Some(1.0),
        ..Default::default()
    };
    let pipeline = InstructPipeline::from_quantized_model(model, tokenizer, pipeline_config)?;

    // 5. Create InstructTrainer
    let train_config = InstructTrainingConfig {
        epochs: epochs as usize,
        val_split: 0.1,
        early_stopping_patience: 5,
        checkpoint_dir: output_path.parent().unwrap().join("checkpoints"),
        ..Default::default()
    };
    let mut trainer = InstructTrainer::new(pipeline, samples, train_config);

    // 6. Train
    let result = trainer.train();

    // 7. Export trained LoRA weights to APR
    export_lora_to_apr(trainer.pipeline(), output_path, model_path)?;

    // 8. Report
    report_training_result(&result, json_output);
    Ok(())
}
```

### Phase 2: Model Bridge (`InstructPipeline::from_quantized_model`)

**File:** `entrenar/src/finetune/instruct_pipeline.rs`

New constructor that accepts `OwnedQuantizedModel` instead of requiring SafeTensors:

```rust
/// Create InstructPipeline from a quantized APR/GGUF model.
/// Base weights stay in Q4K form (frozen). LoRA adapters are FP32 (trainable).
/// Forward: dequant(Q4K) @ x + (x @ A) @ B * (α/r)
#[provable_contracts_macros::contract("qlora-training-loop-v1", equation = "lora_forward")]
pub fn from_quantized_model(
    model: OwnedQuantizedModel,
    tokenizer: Tokenizer,
    config: InstructPipelineConfig,
) -> Result<Self> {
    // Wrap Q4K model in trait object that implements forward()
    // LoRA layers inject at q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    // Base weights frozen (no gradient). Only LoRA A/B are trainable.
    // ...
}
```

### Phase 3: APR Export

**File:** `aprender/crates/apr-cli/src/commands/finetune.rs`

```rust
/// Export trained LoRA A/B weights from pipeline to APR format.
#[provable_contracts_macros::contract("lora-algebra-v1", equation = "lora_shape")]
fn export_lora_to_apr(
    pipeline: &InstructPipeline,
    output_path: &Path,
    base_model_path: &Path,
) -> Result<()> {
    let mut writer = AprWriter::new();
    // Write metadata (base model, rank, alpha, training config)
    // Write LoRA A/B tensors (trained weights, not random init)
    // Copy tokenizer from base model
    // ...
}
```

### Phase 4: Merge Support

```bash
# Train adapter
apr finetune model.apr --method qlora --data train.jsonl --output adapter.apr

# Merge adapter into base
apr finetune model.apr --adapter adapter.apr --merge --output merged.apr

# Evaluate merged model
make eval-humaneval CHECKPOINT=checkpoints/merged.apr
```

## 26.8 Test Plan

| Test | Type | Validates |
|------|------|-----------|
| `test_train_step_decreases_loss` | Integration | Loss at step 10 < loss at step 0 |
| `test_base_weights_frozen` | Unit | Base model weights unchanged after training |
| `test_lora_zero_init` | Unit | B=0 init → LoRA contribution = 0 |
| `test_response_only_loss` | Unit | Prompt tokens don't contribute to gradient |
| `test_adamw_decoupled` | Unit | AdamW ≠ L2-regularized Adam |
| `test_export_reimport` | Integration | Export → import → same adapter weights |
| `test_merged_model_inference` | Integration | Merged model produces valid completions |
| `test_99_completions_training` | E2E | Train on teacher completions, verify loss decrease |
| `test_cublas_naive_parity` | Unit | cuBLAS GEMM matches naive matmul within ε < 1e-3 |
| `test_nf4_dequant_roundtrip` | Unit | dQuantizeNF4(dDequantizeNF4(i)) == i for all 16 codes |
| `test_nf4_decy_parity` | Unit | Rust transpiled NF4 matches C reference within ε < 1e-7 |
| `test_fused_ce_unfused_parity` | Unit | Fused cross-entropy = unfused within ε < 1e-5 |
| `test_gradient_checkpoint_parity` | Integration | With/without checkpointing produce same gradients |
| `test_batch_loss_mean` | Unit | Batch loss = mean of per-sample losses |
| `test_cublas_transpose_flags` | Unit | CUBLAS_OP_T matches explicit transpose + CUBLAS_OP_N |
| `test_batch4_throughput` | Perf | batch_size=4 achieves ≥ 4x throughput vs batch_size=1 |

## 26.9 Acceptance Criteria

- **AC-FT-001:** `apr finetune model.apr --method qlora --data train.jsonl` trains for N epochs with decreasing loss
- **AC-FT-002:** Training produces an APR file with trained LoRA weights (not random init)
- **AC-FT-003:** Merged model passes `apr check` and produces valid inference output
- **AC-FT-004:** All 14 falsification tests from §26.6.4 pass
- **AC-FT-005:** All 7 provable contracts annotated and verified (4 existing + 3 new)
- **AC-FT-006:** 7B QLoRA on 99 teacher completions completes in **< 10 minutes** on gx10 (cuBLAS + batch_size=4)
- **AC-FT-007:** Distilled 7B model achieves ≥ 85% pass@1 on HumanEval (no regression from baseline)
- **AC-FT-008:** Training throughput ≥ 100 tokens/sec on gx10 GB10 (vs ~2 tok/s current)
- **AC-FT-009:** All NF4 dequant functions transpiled via decy with < 5 `unsafe` blocks per 1000 LOC
- **AC-FT-010:** cuBLAS GEMM wrapper passes all 4 FALSIFY-CUBLAS tests + KANI-CUBLAS-001

## 26.10 References

- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" arXiv:2106.09685
- Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs" arXiv:2305.14314
- Loshchilov & Hutter (2017) "Decoupled Weight Decay Regularization" arXiv:1711.05101
- Eckart-Young-Mirsky theorem (1936) — optimal low-rank approximation
- Unsloth (Han & Han, 2024) — Triton kernel fusions for 2-5x QLoRA speedup (https://github.com/unslothai/unsloth)
- bitsandbytes (Dettmers, 2023) — NF4 dequantization kernels (csrc/kernels.cu, transpiled via decy)
- Chen et al. (2016) "Training Deep Nets with Sublinear Memory Cost" arXiv:1604.06174 — gradient checkpointing
- NVIDIA cuBLAS documentation — cublasGemmEx BF16 compute with FP32 accumulation
- decy (PAIML) — C-to-Rust transpiler for bitsandbytes kernel transpilation
