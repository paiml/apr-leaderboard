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
W_f32 = DequantNF4→F32(W_nf4)       # WGSL shader: NF4 LUT lookup × absmax (algorithm from decy)
h_base = WGSL_GEMM(x, W_f32^T)      # Tiled GEMM: CUTLASS-style 128×128, shared memory, safe Rust
h_lora = WGSL_GEMM(WGSL_GEMM(x, A), B) * (α/r)  # Two small GEMMs via same shader
h = h_base + h_lora                  # Fused add in epilogue (alpha=s, beta=1)
```

Where:
- `A ∈ ℝ^{n×r}` — LoRA down-projection (Kaiming init), BF16
- `B ∈ ℝ^{r×m}` — LoRA up-projection (zero init), BF16
- `r` — LoRA rank (e.g., 32)
- `α` — LoRA alpha scaling (e.g., 64)
- `x ∈ ℝ^{B_s×n}` — batched input hidden states (batch_size × hidden_dim), BF16

**Critical architecture decision (from Unsloth + CUTLASS analysis):** All GEMM operations
use a CUTLASS-style tiled GEMM implemented in WGSL compute shaders via wgpu (safe Rust
API). NO cuBLAS FFI, NO CUDA driver FFI, NO `unsafe` code. The tiling algorithm is
derived from NVIDIA's open-source CUTLASS library (MIT licensed) which achieves 90-95%
of cuBLAS throughput.

**Zero-unsafe mandate:** trueno-gpu currently has 68 `extern "C"` function pointers,
137 `unsafe` blocks, and 18 `unsafe impl` blocks — all for CUDA driver/cuBLAS/cuBLASLt
FFI. ALL of these are eliminated — not feature-gated, REMOVED. The replacement is wgpu
(safe Rust API for Vulkan/Metal/DX12 GPU compute). The PTX code generator (~5,500
lines), CUDA driver bindings, cuBLAS/cuBLASLt bindings — all deleted. All GPU compute
goes through WGSL compute shaders via wgpu.

**Single backend: wgpu only.** There is no CUDA feature flag, no dual-backend. wgpu
speaks Vulkan on NVIDIA GPUs, accessing the same hardware including tensor cores via
`VK_KHR_cooperative_matrix`. The CUDA driver API is just one FFI path to the GPU —
Vulkan is another, and wgpu makes it safe Rust. Burn/CubeCL proves Vulkan GEMM
matches CUDA performance on the same NVIDIA hardware. The 5-10% historical gap is
driver maturity, not hardware access — it closes as Vulkan extensions mature.

**CUTLASS algorithm in WGSL (not C++ transpilation):** CUTLASS is C++ templates — decy
handles C, not C++. Instead, we read the CUTLASS algorithm (MIT licensed, ~200 lines of
actual logic) and reimplement the tiling strategy in WGSL:
- Thread-block tile: 128×128×8 (output tile × K-step)
- Warp tile: 32×64 (per-warp output region)
- Thread micro-tile: 8×8 (per-thread output, outer-product accumulation)
- Double-buffered shared memory (load tile N+1 while computing tile N)
- Serpentine traversal for register reuse in inner loop
- Epilogue: transpose through shared memory for coalesced global stores
- Tensor cores via `VK_KHR_cooperative_matrix` when available (wgpu extension)

**NF4 transpilation via decy:** The NF4 dequantization kernels are transpiled from
bitsandbytes' `csrc/kernels.cu` (2400 LOC) using `../decy` (C-to-Rust transpiler).
Tier 1 functions (pure math: NF4 LUT, `dQuantizeNF4`, `dDequantizeNF4`) transpile
directly to safe Rust. Tier 3 functions (CUDA kernels) have their algorithms transpiled
and reimplemented as WGSL compute shaders for wgpu.

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

Gradients flow only through LoRA A and B matrices. All backward GEMMs use WGSL tiled GEMM:

```
# Re-dequantize base weight for backward (gradient checkpointing: not saved from forward)
W_f32 = DequantNF4→F32(W_nf4)     # WGSL dequant shader

# Gradient w.r.t. input (for upstream layers)
∂L/∂x = WGSL_GEMM(∂L/∂h, W_f32) + WGSL_GEMM(WGSL_GEMM(∂L/∂h, B^T), A^T) * (α/r)

# LoRA gradients (via WGSL GEMM with fused scaling in epilogue)
∂L/∂B = WGSL_GEMM((A^T @ x)^T, ∂L/∂h) * (α/r)   # epilogue alpha=α/r, beta=0
∂L/∂A = WGSL_GEMM(x^T, ∂L/∂h @ B^T) * (α/r)     # epilogue alpha=α/r, beta=0
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
    - wgsl-gemm-tiled-v1            # NEW (replaces cublas-gemm-wrapper-v1)
    - nf4-dequantization-v1         # NEW
    - fused-cross-entropy-v1        # NEW
equations:
  frozen_base:
    formula: ∂L/∂W_base = 0 (no gradient flows to base weights)
    invariants:
      - Base weights unchanged after training step
      - Only LoRA A/B receive gradients
      - Autograd skips frozen subgraph (topological pruning)
  lora_forward_wgsl:
    formula: h = WGSL_GEMM(DequantF32(W_nf4), x) + WGSL_GEMM(WGSL_GEMM(x, A), B) * (α/r)
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

**Contract: `wgsl-gemm-tiled-v1`** (NEW — replaces cublas-gemm-wrapper-v1)

```yaml
metadata:
  version: 1.0.0
  description: >
    WGSL tiled GEMM for training — CUTLASS-derived algorithm, zero unsafe.
    128×128 thread-block tiles, 8×8 thread micro-tiles, double-buffered shared memory.
    All via wgpu safe Rust API. No cuBLAS, no FFI.
  references:
    - "NVIDIA CUTLASS (MIT licensed) — tiling algorithm reference"
    - "Burn/CubeCL — proof that Vulkan GEMM can match 70-80% of cuBLAS"
  depends_on:
    - matmul-kernel-v1
equations:
  gemm_dimensions:
    formula: C[m,n] = α · op(A)[m,k] @ op(B)[k,n] + β · C[m,n]
    invariants:
      - Output buffer has capacity >= m × n elements
      - Workgroup grid = ceil(m/128) × ceil(n/128)
      - Each thread computes 8×8 output elements
  tiled_naive_parity:
    formula: |WGSL_GEMM(A,B) - naive(A,B)| < ε for all elements
    invariants:
      - ε < 1e-4 for F32 (no precision loss from tiling)
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
    prediction: 128×128 tiled GEMM matches 16×16 naive GEMM within ε < 1e-6
    test: Same inputs, compare tiled vs naive WGSL shader outputs
kani_harnesses:
  - id: KANI-WGSL-GEMM-001
    property: Output buffer index m*N+n never exceeds m*n for all valid (m,n)
    bound: m,n in [1..256]
  - id: KANI-WGSL-GEMM-002
    property: Shared memory index never exceeds 2*TILE_M*TILE_K
    bound: tile_m,tile_k in [1..128]
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
| FT-009 | WGSL tiled GEMM vs naive parity | Tiled GEMM matches naive matmul within ε < 1e-4 | Random F32 matrices, compare outputs |
| FT-010 | Gradient checkpoint correctness | Recomputed activations match saved within ε < 1e-6 | Compare with/without checkpointing |
| FT-011 | Fused CE = unfused CE | Fused cross-entropy matches standard within ε < 1e-5 | Random logits, multiple vocab sizes |
| FT-012 | Batch loss = mean per-sample | Batch loss equals average of individual sample losses | Compare batch vs sequential processing |
| FT-013 | NF4 roundtrip | dQuantizeNF4(dDequantizeNF4(i)) == i for all i in [0..15] | Exhaustive 16-value test |
| FT-014 | Decy transpilation parity | Rust NF4 dequant matches C reference within ε < 1e-7 | 1M random NF4-packed bytes, compare outputs |
| FT-015 | Zero unsafe | `grep -r "unsafe" trueno-gpu/src/` returns 0 matches | No unsafe blocks, no extern C, no raw pointers |
| FT-016 | CUDA FFI eliminated | `driver/sys/`, `driver/cublas*`, `ptx/` directories removed | No CUDA dependency in the crate |

## 26.7 Implementation Plan

### Phase 0: WGSL Tiled GEMM + NF4 Dequant + Eliminate Unsafe FFI (trueno-gpu + decy)

**Priority: HIGHEST — this is the 20-100x speedup + zero-unsafe compliance.**

**Step 0a: Transpile bitsandbytes NF4 math via decy**

```bash
# Tier 1: Pure C math functions → safe Rust (direct transpilation)
decy transpile bitsandbytes/csrc/kernels.cu \
  --functions dDequantizeNF4,dQuantizeNF4,nf4_dequantization_lut \
  --output trueno/src/quantize/nf4_bnb.rs
```

Tier 1 functions (pure math, zero unsafe):
- `nf4_dequantization_lut[16]` → `const NF4_LUT: [f32; 16]`
- `dDequantizeNF4(val)` → `fn dequantize_nf4(val: u8) -> f32`
- `dQuantizeNF4(x)` → `fn quantize_nf4(x: f32) -> u8`

Tier 3 algorithms (CUDA kernels → WGSL compute shaders for wgpu):
- `kDequantizeBlockwise` algorithm → WGSL compute shader
- `kQuantizeBlockwise` algorithm → WGSL compute shader

**Step 0b: CUTLASS-style tiled GEMM in WGSL (replaces cuBLAS entirely)**

Implement the CUTLASS tiling algorithm (MIT licensed, ~200 lines of logic) as a
WGSL compute shader, called via wgpu's safe Rust API. Zero `unsafe`, zero FFI.

```wgsl
// CUTLASS-derived tiled GEMM in WGSL
// Thread-block: 128×128 output tile, K-step: 8
// Each thread: 8×8 micro-tile (outer-product accumulation)
// Double-buffered workgroup shared memory
const TILE_M: u32 = 128u;
const TILE_N: u32 = 128u;
const TILE_K: u32 = 8u;
const THREAD_M: u32 = 8u;
const THREAD_N: u32 = 8u;

var<workgroup> smem_a: array<f32, 2 * 128 * 8>;  // double-buffered
var<workgroup> smem_b: array<f32, 2 * 8 * 128>;

@compute @workgroup_size(16, 16)  // 256 threads = 8 warps
fn tiled_gemm(...) {
    // 1. Each thread computes 8×8 output elements
    // 2. K-dimension loop with double-buffered shared memory tiles
    // 3. Inner loop: serpentine 8×8 outer product from shared memory
    // 4. Epilogue: coalesced store with alpha/beta scaling
}
```

```rust
/// WGSL tiled GEMM for training: F32, safe Rust via wgpu.
/// Algorithm from CUTLASS (MIT licensed). Zero unsafe.
#[provable_contracts_macros::contract("wgsl-gemm-tiled-v1", equation = "gemm_dimensions")]
pub fn wgsl_gemm_tiled(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: u32, n: u32, k: u32,
    a: &wgpu::Buffer,         // [m, k] F32
    b: &wgpu::Buffer,         // [k, n] F32
    c: &wgpu::Buffer,         // [m, n] output
    alpha: f32,
    beta: f32,
) -> Result<()> {
    // Pre-compiled pipeline (created once, reused per training step)
    // dispatch_workgroups(ceil(m/128), ceil(n/128), 1)
}
```

**Step 0c: NF4 dequant → F32 → WGSL GEMM pipeline**

```rust
/// Dequantize NF4 to F32, then tiled GEMM. All via wgpu, zero unsafe.
#[provable_contracts_macros::contract("nf4-dequantization-v1", equation = "blockwise_dequant")]
pub fn nf4_gemm_wgsl(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    nf4_weight: &wgpu::Buffer,    // Packed NF4 + absmax
    input: &wgpu::Buffer,         // [batch, hidden] F32
    output: &wgpu::Buffer,        // [batch, out_dim] F32
    dequant_buffer: &wgpu::Buffer, // Reused across layers
) -> Result<()> {
    // 1. WGSL shader: dequant NF4 → F32 (algorithm transpiled from bitsandbytes via decy)
    // 2. WGSL tiled GEMM: output = input @ dequant_buffer^T
}
```

**Step 0d: Delete ALL CUDA code from trueno-gpu**

Not feature-gated — REMOVED entirely. Single backend: wgpu.

| Delete | Files | Lines | Why |
|--------|-------|-------|-----|
| CUDA driver FFI | `driver/sys/mod.rs` | ~800 | wgpu replaces all CUDA driver calls |
| cuBLAS FFI | `driver/cublas_sys.rs` | ~200 | WGSL tiled GEMM replaces cuBLAS |
| cuBLASLt FFI | `driver/cublaslt_sys.rs` | ~300 | WGSL tiled GEMM replaces cuBLASLt |
| CUDA safe wrappers | `driver/context.rs`, `module.rs`, `stream.rs`, `graph.rs`, `cublas.rs`, `cublaslt.rs` | ~1500 | wgpu device/queue/buffer (safe) |
| CUDA memory | `driver/memory/buffer.rs`, `transfer.rs` | ~400 | wgpu::Buffer (safe) |
| PTX code generator | `ptx/` (entire directory) | ~5500 | WGSL shaders replace PTX |
| CUDA feature flags | `Cargo.toml`, `lib.rs` | ~50 | No `cuda` feature, no `cfg(feature = "cuda")` |
| **Total deleted** | **~23 files** | **~8750 lines** | **Zero unsafe, single backend** |

After deletion, trueno-gpu has:
- Zero `extern "C"` declarations
- Zero `unsafe` blocks
- Zero `unsafe impl` blocks
- One GPU backend: wgpu (safe Rust API → Vulkan/Metal/DX12)
- WGSL compute shaders for all GPU operations

**Step 0e: Batch collation**

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
- **AC-FT-004:** All 16 falsification tests from §26.6.4 pass
- **AC-FT-005:** All 7 provable contracts annotated and verified (4 existing + 3 new)
- **AC-FT-006:** 7B QLoRA on 99 teacher completions completes in **< 30 minutes** on gx10 (WGSL tiled GEMM + batch_size=4)
- **AC-FT-007:** Distilled 7B model achieves ≥ 85% pass@1 on HumanEval (no regression from baseline)
- **AC-FT-008:** Training throughput ≥ 50 tokens/sec on gx10 GB10 (vs ~2 tok/s current)
- **AC-FT-009:** All NF4 dequant functions transpiled via decy with **zero** `unsafe` blocks
- **AC-FT-010:** WGSL tiled GEMM passes all 4 FALSIFY-WGSL-GEMM tests + 2 Kani harnesses
- **AC-FT-011:** **Zero `unsafe` blocks** in trueno-gpu after CUDA FFI elimination
- **AC-FT-012:** trueno-gpu has **zero `extern "C"` declarations** (no FFI of any kind)

## 26.10 References

- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" arXiv:2106.09685
- Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs" arXiv:2305.14314
- Loshchilov & Hutter (2017) "Decoupled Weight Decay Regularization" arXiv:1711.05101
- Eckart-Young-Mirsky theorem (1936) — optimal low-rank approximation
- Unsloth (Han & Han, 2024) — Triton kernel fusions for 2-5x QLoRA speedup (https://github.com/unslothai/unsloth)
- bitsandbytes (Dettmers, 2023) — NF4 dequantization kernels (csrc/kernels.cu, transpiled via decy)
- Chen et al. (2016) "Training Deep Nets with Sublinear Memory Cost" arXiv:1604.06174 — gradient checkpointing
- Vulkan VK_KHR_cooperative_matrix — tensor core access from Vulkan (same hardware as CUDA wmma)
- Burn/CubeCL — proof that Vulkan GEMM matches CUDA on same NVIDIA GPU
- decy (PAIML) — C-to-Rust transpiler for bitsandbytes kernel transpilation
