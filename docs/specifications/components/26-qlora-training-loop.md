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
`VK_KHR_cooperative_matrix` (confirmed on gx10 GB10: revision 2, BF16+FP8 enabled).

**Falsified claims (corrected):** Vulkan GEMM does NOT match CUDA on discrete GPUs —
the gap is 20-50% on A100 due to architectural limits (no `cp.async` equivalent in
SPIR-V, smaller cooperative matrix sizes in KHR vs CUDA wmma, Vulkan vectorization
limited to line size 4 vs 8). However, on GB10 unified memory (our target hardware),
the gap effectively disappears because `cp.async` optimizes discrete GPU memory
transfers which are irrelevant on unified memory. llama.cpp benchmarks show Vulkan
matching or exceeding CUDA on GB10 for token generation.

**wgpu cooperative matrix status:** Shipped in wgpu v29.0.0 (2026-03-19), experimental.
Not yet standardized by W3C (gpuweb issue #4195 open). Load/store/multiply-add
implemented. CubeCL issue #1053: BF16+F32 mixed accumulation has known issues — use
F32 accumulation throughout until resolved.

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

**Step 0d: WgpuTrainingPipeline — complete replacement for CUDA training path**

NOT a hybrid/hack. A complete GPU training pipeline in wgpu that replaces the entire
`CudaTrainer` + `CudaBlock` + `CudaBlockScratch` + `GpuTraining` infrastructure.

The CUDA training path (`instruct_pipeline.rs:660-793`) does 6 operations ALL on GPU:
1. Forward: NF4 dequant → GEMM → RMSNorm → attention → SwiGLU × 28 layers
2. lm_head: GEMM (hidden → vocab logits)
3. Loss: fused causal cross-entropy (in-place gradient)
4. lm_head backward: GEMM (grad_logits → grad_hidden)
5. Backward: GEMM backward through 28 NF4 layers (LoRA gradients)
6. Optimizer: AdamW on LoRA weights

`WgpuTrainingPipeline` must do ALL 6 on wgpu. Architecture:

```
WgpuTrainingPipeline
├── WgslForwardPass (trueno)          — forward through 28 transformer layers
│   ├── WGSL NF4 dequant shader       — NF4 → F32 on GPU
│   ├── WGSL tiled GEMM shader        — CUTLASS-style 64×64
│   ├── WGSL RMSNorm shader           — already exists in wgsl_forward.rs
│   ├── WGSL SwiGLU shader            — already exists in wgsl_forward.rs
│   ├── WGSL RoPE shader              — already exists in wgsl_forward.rs
│   └── WGSL attention shader         — already exists in wgsl_forward.rs
├── WgslBackwardPass (NEW)            — backward through 28 layers
│   ├── Activation checkpointing      — save only layer boundaries
│   ├── WGSL backward GEMM            — same tiled GEMM with transposed args
│   ├── WGSL backward RMSNorm         — d/dx of x/rms(x)
│   ├── WGSL backward SwiGLU          — d/dx of SiLU(gate)×up
│   └── WGSL backward attention       — Q/K/V gradient through softmax
├── WgslCrossEntropy (NEW)            — fused loss + in-place gradient
│   ├── Chunked logsumexp             — never materialize full [T,V] softmax
│   └── In-place backward             — gradient overwrites logits buffer
├── WgpuTrainer (EXISTS)              — optimizer + gradient ops
│   ├── AdamW WGSL kernel             — decoupled weight decay
│   └── Gradient clipping WGSL        — scale by max_norm/grad_norm
└── WgpuBlockManager (NEW)            — GPU memory for 28 layers
    ├── NF4 weight buffers             — packed NF4 + absmax per layer
    ├── LoRA A/B buffers               — trainable F32 per layer
    ├── Activation checkpoint buffers  — reused across layers
    └── Dequant buffer                 — single reusable F32 buffer
```

**Implementation order (each builds on the previous):**

```
Step 0d.1: WgpuBlockManager — upload NF4 weights to wgpu::Buffer
Step 0d.2: WgslForwardPass training mode — save activations at layer boundaries
Step 0d.3: WgslBackwardPass — backward GEMM + RMSNorm + SwiGLU through 28 layers
Step 0d.4: WgslCrossEntropy — fused loss on GPU (chunked logsumexp)
Step 0d.5: Wire into InstructPipeline::wgpu_train_step (replaces cuda_train_step)
Step 0d.6: End-to-end test — 3-sample 7B training on gx10, compare loss with CUDA
```

**What already exists (proven):**
- WGSL tiled GEMM (forward + backward) — `ac65854f`, 375 GFLOPS on GB10
- WGSL RMSNorm, SwiGLU, RoPE, attention, residual — in `wgsl_forward.rs`
- NF4 dequant in safe Rust — `2d151d45`, 6/6 tests
- WgpuTrainer (AdamW + gradient clip) — `dae8a812`, 3/3 tests
- CUDA↔wgpu parity — 3/3 tests on gx10

**What needs building:**
- WgpuBlockManager — upload 28 layers of NF4 weights to wgpu buffers
- WgslForwardPass training mode — checkpoint activations
- WgslBackwardPass — backward through full transformer stack
- WgslCrossEntropy — fused chunked cross-entropy
- Pipeline integration — `InstructPipeline::wgpu_train_step`

**WGSL shaders needed (NEW):**
- `nf4_dequant.wgsl` — NF4 → F32 on GPU (algorithm from `nf4.rs`, already proven)
- `backward_rmsnorm.wgsl` — ∂L/∂x = (1/rms) × (γ × ∂L/∂y − x/rms² × mean(x·∂L/∂y·γ))
- `backward_swiglu.wgsl` — ∂L/∂gate = ∂L/∂h × up × σ(gate)×(1+gate×(1−σ(gate)))
- `backward_attention.wgsl` — ∂L/∂Q, ∂L/∂K, ∂L/∂V through scaled dot-product
- `fused_cross_entropy.wgsl` — chunked logsumexp + in-place gradient
- `transpose.wgsl` — GPU transpose for backward GEMM (avoids CPU roundtrip)

```
Prove-then-delete order:
1. ✅ Implement wgpu backward GEMM (tiled, same shader as forward) — dae8a812
2. ✅ Implement wgpu AdamW + gradient clipping (WGSL kernels) — dae8a812
3. Run 3-sample training via WgpuTrainer
4. Compare loss curve: wgpu vs CUDA (must match within ε < 0.1)
5. Run 100-sample training via wgpu (stability test)
6. ONLY THEN delete CUDA code from ALL repos
```

**DONE:** `WgpuTrainer` in `entrenar/src/autograd/wgpu_training.rs` provides:
- `matmul_forward()` — CUTLASS-style tiled GEMM via WGSL
- `matmul_backward()` — backward GEMM via transposed tiled GEMM
- `adamw_step()` — WGSL elementwise AdamW kernel
- `clip_gradients()` — WGSL gradient clipping
- 3/3 unit tests pass (forward parity, backward parity, AdamW direction)

**Step 0e: Parity gate — wgpu training matches CUDA training**

Before deleting ANY CUDA code, the following parity tests must pass:

| Test | Criterion | Status |
|------|-----------|--------|
| 3-sample loss match | `\|loss_wgpu - loss_cuda\| < 0.1` after 1 epoch | MUST PASS |
| Gradient norm match | `\|norm_wgpu - norm_cuda\| / norm_cuda < 0.05` | MUST PASS |
| 100-sample stability | No NaN/Inf over 1 epoch | MUST PASS |
| HumanEval inference parity | wgpu pass@1 = CUDA pass@1 (already proven: 84.15%) | **PASSED** |
| WgpuTrainer unit tests | Forward/backward/AdamW match CPU reference | **PASSED** (3/3) |
| **CUDA↔wgpu forward GEMM** | max error < 0.01 on gx10 GB10 | **PASSED** |
| **CUDA↔wgpu backward GEMM** | grad_a + grad_b max error < 0.01 | **PASSED** |
| **CUDA↔wgpu AdamW** | params max error < 1e-4 after 1 step | **PASSED** |

**Step 0f: Delete CUDA code from ALL affected repos (ONLY after 0e passes)**

Deletion spans 3 repos. All have wgpu replacements proven.

**trueno-gpu (primary — owns the CUDA FFI):**

| Delete | Files | Lines | Replacement |
|--------|-------|-------|-------------|
| CUDA driver FFI | `driver/sys/mod.rs` | ~800 | wgpu safe API |
| cuBLAS FFI | `driver/cublas_sys.rs` | ~200 | WGSL tiled GEMM |
| cuBLASLt FFI | `driver/cublaslt_sys.rs` | ~300 | WGSL tiled GEMM |
| CUDA safe wrappers | 6 files in `driver/` | ~1500 | wgpu wrappers |
| CUDA memory | `driver/memory/` | ~400 | wgpu::Buffer |
| PTX code generator | `ptx/` (entire directory) | ~5500 | WGSL shaders |
| CUDA feature flags | `Cargo.toml`, `lib.rs` | ~50 | Remove `cuda` feature |
| **Total** | **~23 files** | **~8750 lines** | |

**entrenar (training — depends on trueno-gpu CUDA):**

| Delete | Files | Lines | Replacement |
|--------|-------|-------|-------------|
| `CudaTrainer` | `autograd/cuda_training.rs` | ~350 | `WgpuTrainer` (already built) |
| CUDA backward ops | `autograd/cuda_backward/*.rs` | ~600 | `WgpuTrainer::matmul_backward()` |
| CUDA forward ops | `autograd/cuda_forward.rs` | ~200 | `WgpuTrainer::matmul_forward()` |
| CUDA optimizer | `autograd/cuda_optim.rs` | ~300 | `WgpuTrainer::adamw_step()` |
| `cuda` feature | `Cargo.toml` | ~10 | `gpu` feature (wgpu via trueno) |
| **Total** | **~8 files** | **~1460 lines** | |

**realizar (inference — depends on trueno-gpu CUDA):**

| Delete | Files | Lines | Replacement |
|--------|-------|-------|-------------|
| CUDA batch inference | `infer/batch_cuda.rs` | ~400 | `batch_wgpu.rs` (already default) |
| CUDA module loading | `infer/cuda_*.rs` | ~300 | wgpu forward pass |
| `cuda` feature | `Cargo.toml` | ~10 | `gpu` feature (wgpu via trueno) |
| **Total** | **~4 files** | **~710 lines** | |

**qwen-coder-deploy (config — no code changes):**

| Update | Files | Change |
|--------|-------|--------|
| forjar manifests | `forjar-gpu*.yaml` | `--features cuda` → `--features gpu` |
| Spec docs | `docs/specifications/*.yaml` | Reference wgpu not CUDA |

**apr-leaderboard (orchestration — no code changes):**

| Update | Files | Change |
|--------|-------|--------|
| `APR_NO_GPU` env var | scripts/*.sh | Still works (wgpu respects it) |
| MEMORY.md | memory/ | Update GPU status |

**Grand total across all repos: ~33 files, ~10,920 lines deleted.**

After deletion:
- Zero `extern "C"` declarations
- Zero `unsafe` blocks
- Zero `unsafe impl` blocks
- One GPU backend: wgpu (safe Rust API → Vulkan/Metal/DX12)
- WGSL compute shaders for all GPU operations

**Step 0g: Batch collation**

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
- **AC-FT-006:** 7B QLoRA on 99 teacher completions completes in **< 30 minutes** on gx10 (full WgpuTrainingPipeline, batch_size=4)
- **AC-FT-007:** Distilled 7B model achieves ≥ 85% pass@1 on HumanEval (no regression from baseline)
- **AC-FT-008:** Training throughput ≥ 50 tokens/sec on gx10 GB10 (benchmarked: 375 GFLOPS sustained for GEMM; blocked by 2 GB wgpu buffer limit on lm_head forcing CPU fallback — see §26.11)
- **AC-FT-009:** All NF4 dequant functions transpiled via decy with **zero** `unsafe` blocks
- **AC-FT-010:** WGSL tiled GEMM passes all 4 FALSIFY-WGSL-GEMM tests + 2 Kani harnesses
- **AC-FT-011:** **Zero `unsafe` blocks** in trueno-gpu after CUDA FFI elimination (Step 0f)
- **AC-FT-012:** trueno-gpu has **zero `extern "C"` declarations** after Step 0f
- **AC-FT-013:** WgpuTrainingPipeline loss matches CUDA training loss within ε < 0.1 on 7B model (Step 0e)
- **AC-FT-014:** CUDA code deleted ONLY after AC-FT-013 passes (prove-then-delete)
- **AC-FT-015:** ALL 6 training operations on GPU via wgpu (forward, lm_head, loss, lm_head backward, layer backward, optimizer) — no CPU fallback for any operation
- **AC-FT-016:** 6 new WGSL shaders (nf4_dequant, backward_rmsnorm, backward_swiglu, backward_attention, fused_cross_entropy, transpose) with falsification tests

## 26.11 Known Blockers and Status (2026-03-31)

### 26.11.1 wgpu 2 GB Buffer Binding Limit

**Status: BLOCKING full GPU training throughput**

wgpu's `max_storage_buffer_binding_size` is capped at `u32::MAX / 2 = 2,147,483,647`
bytes (2 GB - 1 byte) regardless of the Vulkan adapter's actual limit (4 GB on GB10).
The lm_head weight matrix for Qwen 7B is `152064 × 3584 × 4 = 2,179,989,504` bytes
(2.18 GB), exceeding this limit.

**Impact:** The entire forward pass falls back to CPU, making training ~20x slower
than it should be. The tiled GEMM (375 GFLOPS) is only used for sub-2GB matmuls.

**Fixes (in priority order):**
1. **Chunk lm_head matmul:** Split vocab into 2 halves (76032 × 3584 = 1.09 GB each).
   Compute two half-logit vectors, concatenate. Simple, no wgpu changes needed.
2. **Tie embeddings:** Many models (including Qwen) tie embed_tokens and lm_head.
   Use `embed_tokens^T` (3584 × 152064) for lm_head — same data, transposed access.
   Still > 2 GB but avoids a second copy.
3. **wgpu upstream:** Request `wgpu::Features::BUFFER_BINDING_SIZE` or equivalent.
   The WebGPU spec may eventually raise this limit.

**Recommended fix:** Option 1 (chunk). Estimated effort: ~50 lines in matmul.rs.
**Status: DONE** — chunked matmul committed (`6665d10e`).

### 26.11.3 Per-Call Buffer Creation in model.forward()

**Status: BLOCKING — primary performance bottleneck**

`InstructPipeline::wgpu_train_step` calls `model.forward()` which uses
`GpuDevice::matmul()` for each of 196 projections (7 per layer × 28 layers).
Each `matmul()` call creates shader, buffers, bind group, pipeline, dispatches,
reads back — ~8ms overhead per call = ~1.6 seconds overhead alone, plus the
actual compute.

`WgslForwardPass::forward_layer_training` already exists with:
- Persistent pre-uploaded weight buffers (no per-call alloc)
- Single command encoder for all 13 passes per layer
- Tiled GEMM (375 GFLOPS) for M>=4
- GPU-resident attention (no CPU readback)

**Fix:** Replace `model.forward(&full_ids)` with `WgslForwardPass::forward_layer_training`
loop. The weights are already uploaded to `WgslForwardPass` during model init.
This eliminates the per-call buffer overhead and enables the full 375 GFLOPS throughput.

**Estimated speedup:** ~100x (from ~1.6 hrs/sample to ~1 min/sample)

### 26.11.8 Final PROFILE Results (2026-03-31)

**315x speedup achieved. 5+ hours → 57 seconds. Loss correct.**

```
Pipeline ready in 20.4s (OwnedQuantizedModel, no Transformer)
Sample 1: loss=14.95  fwd=56ms  dl=10.3s  norm=4ms  gemm=83ms  ce=899ms  bwd=1.0s  total=12.3s
Sample 2: loss=14.71  fwd=49ms  dl=9.9s   norm=4ms  gemm=68ms  ce=836ms  bwd=1.0s  total=11.9s
Sample 3: loss=13.28  fwd=11ms  dl=2.9s   norm=0ms  gemm=7ms   ce=227ms  bwd=262ms total=3.4s
Training complete in 57.6s
```

**KAIZEN optimization chain (8 root causes found and fixed):**

| # | Root cause (five-whys) | Fix | Impact |
|---|---|---|---|
| 1 | CPU autograd replays entire forward | Saved activations, GPU-only backward | 5+ hrs → 7 min |
| 2 | Transformer::from_apr() 28GB CPU dequant | OwnedQuantizedModel → GPU direct | 20 min → 19s init |
| 3 | WgpuTrainer used 16×16 MATMUL_SHADER | Switch to 64×64 TILED_GEMM_SHADER | 20x GEMM |
| 4 | 1024 copy_buffer_to_buffer per step | WGSL scatter/gather shaders | 1 dispatch |
| 5 | Attention 3-pass QK^T recomputation | Store scores in shared memory | 7 min → 69s |
| 6 | Attention @workgroup_size(1) sequential | 128 threads parallel dot+V sum | 69s → 57s |
| 7 | 2GB wgpu buffer limit on lm_head | Pre-chunk at init, scatter on GPU | No crash |
| 8 | Per-step lm_head buffer allocation | Pre-upload at init, reuse | -2s/step |

**Remaining bottleneck:** GPU attention = 10.3s per step (28 layers × 370ms).
Sequential loop over seq positions per head. Next: Flash Attention tiling.

### 26.11.9 LoRA Weight Updates — Contract-First Design

**Status: NEXT — training computes loss but doesn't update LoRA weights.**

**Governing contracts:**
- `lora-algebra-v1 / lora_shape`: A[in, rank], B[rank, out]
- `wgpu-production-training-v1 / C-WGPU-LORA-BWD-001`:
  - `dL/dB = (α/r) * grad_output^T @ (saved_input @ A)` [rank, out]
  - `dL/dA = (α/r) * saved_input^T @ (grad_output @ B^T)` [in, rank]
- `adamw-kernel-v1 / weight_update`: decoupled weight decay
- `lora-gradient-flow-v1`: B_norm > 0 after step 1 (B starts at zero)

**Per layer, per projection (7 projections × 28 layers = 196 updates per step):**

```
For projection P with saved_input X[seq, in_dim] and grad_output G[seq, out_dim]:
  XA = X @ A                        [seq, rank]  — matmul_forward
  dB = (α/r) * XA^T @ G             [rank, out]  — matmul_backward
  dA = (α/r) * X^T @ (G @ B^T)      [in, rank]  — matmul_backward

  A = AdamW(A, dA, m_A, v_A, lr, step)
  B = AdamW(B, dB, m_B, v_B, lr, step)
```

**Falsification tests (from contracts):**
- FALSIFY-LORA-UPD-001: B_norm > 0 after step 1 (was zero-initialized)
- FALSIFY-LORA-UPD-002: dL/dA and dL/dB match CPU reference within ε < 1e-3
- FALSIFY-LORA-UPD-003: loss at step N < loss at step 0 (training makes progress)
- FALSIFY-LORA-UPD-004: base weights unchanged after step (frozen)

**Implementation (all via WgpuTrainer, zero unsafe):**
- LoRA A/B stored as wgpu::Buffer per projection per layer
- AdamW m/v states as wgpu::Buffer (6 buffers per projection × 7 × 28 = 1176 buffers)
- Gradient computation: 4 matmul_forward calls per projection per layer
- AdamW step: WgpuTrainer::adamw_step (existing WGSL kernel)

### 26.11.4 CPU Autograd Backward Negates GPU Forward (2026-03-31)

**Status: BLOCKING — current design flaw**

The current `wgpu_train_step` does:
1. GPU forward via `WgslForwardPass` — **fast** (~seconds)
2. GPU fused cross-entropy — **fast** (milliseconds)
3. GPU fused CE backward → grad_logits — **fast** (milliseconds)
4. **CPU `model.forward()` for autograd** — **SLOW** (~1.6 hrs/sample on 7B ARM)
5. CPU autograd backward — slow
6. CPU optimizer — fast

Step 4 negates the GPU forward speedup entirely. The CPU `model.forward()` exists
only to build the autograd graph needed for `backward()`. This is a design flaw:
the training loop should be **entirely GPU**, not GPU forward + CPU backward.

**Root cause:** entrenar's autograd is tape-based (CPU). To compute LoRA gradients,
it needs to replay the forward pass through autograd. The GPU forward
(`WgslForwardPass`) doesn't build an autograd graph — it's stateless.

**Fix: eliminate CPU autograd entirely.** Replace steps 4-6 with:

```
4. GPU lm_head backward: grad_hidden = grad_logits @ embed_weight   (WGSL tiled GEMM)
5. GPU backward through 28 layers via WgslBackwardPass              (already built)
   → LoRA gradients computed on GPU
6. GPU AdamW via WgpuTrainer::adamw_step                            (already built)
```

All components exist:
- `WgslBackwardPass` (`0e548e7e`) — backward through transformer layers
- `WgpuTrainer::matmul_backward` (`dae8a812`) — backward GEMM
- `WgpuTrainer::adamw_step` (`dae8a812`) — GPU optimizer
- `WgslCrossEntropy::backward` (`fed3f0ff`) — in-place grad_logits

**What's missing:** Wiring the backward output (grad_logits) through lm_head backward
→ WgslBackwardPass → WgpuTrainer::adamw_step in `wgpu_train_step`. The individual
pieces are tested (3/3 parity, 375 GFLOPS). Just need to compose them.

**Estimated effort:** Replace the CPU autograd block (steps 4-6) with ~50 lines of
GPU dispatch calls. No new shaders needed. No new components needed.

### 26.11.7 Model Loading Bottleneck: Transformer::from_apr() (2026-03-31)

**Status: BLOCKING — 20+ min model load on ARM before training starts**

`InstructPipeline::from_apr()` calls `Transformer::from_apr()` which dequantizes
ALL Q4K weights to F32 Tensors on CPU. For 7B: 28 layers × 9 weights × ~51-272 MB
each = ~28 GB of F32 data created on CPU. This takes ~20 min on ARM (gx10).

**Root cause:** The entrenar `Transformer` model requires F32 Tensors. The Q4K
quantized data from the `.apr` file must be fully dequantized before training starts.

**How realizar solves this:** `batch_wgpu.rs` loads the model via
`OwnedQuantizedModel` (keeps Q4K), then calls `dequant_model_weights()` which
dequantizes per-layer and uploads to GPU immediately. Total: ~2 min for streaming
dequant + upload vs ~20 min for full CPU dequant.

**Fix: bypass Transformer, use OwnedQuantizedModel + WgslForwardPass directly.**

```
Current (slow):
  .apr → Transformer::from_apr() [20 min CPU dequant] → F32 Tensors → wgpu upload

Fixed (fast):
  .apr → OwnedQuantizedModel [seconds, keeps Q4K] → dequant_model_weights()
       → WgslForwardPass.upload_weight() [streaming, ~2 min] → GPU training
```

This requires `InstructPipeline` to accept `OwnedQuantizedModel` instead of
building a `Transformer`. The `WgslForwardPass` handles the full forward pass
on GPU — no `Transformer` object needed for the forward computation.

**Implementation (no SATD / no TODO):**

```rust
// In aprender/crates/apr-cli/src/commands/finetune.rs:
// Complete path — no fallback, no TODO, no SATD.

// 1. Load Q4K model (seconds, keeps quantized)
let mapped = MappedAprModel::from_path(model_path)?;
let q_model = OwnedQuantizedModel::from_apr(&mapped)?;

// 2. Create WgslForwardPass + upload weights (streaming dequant, ~2 min)
let gpu = GpuDevice::new()?;
let mut fwd = WgslForwardPass::new(gpu.device, gpu.queue, ...);
let weights = dequant_model_weights(&q_model)?;
for (name, data, _, _) in weights {
    fwd.upload_weight(&name, &data);
}

// 3. Create InstructPipeline from WgslForwardPass (NEW constructor)
//    No Transformer::from_apr(). No CPU F32 tensors. No 20-min load.
let pipeline = InstructPipeline::from_wgsl_forward(fwd, tokenizer, instruct_config)?;

// 4. Train
let trainer = InstructTrainer::new(pipeline, samples, train_config);
trainer.train();
```

**New constructor in entrenar:**

```rust
/// §26.11.7: Create pipeline from pre-uploaded GPU weights.
/// No Transformer object. No CPU F32 tensors. All forward/backward on GPU.
///
/// Contract: qlora-training-loop-v1 / lora_forward_wgsl
#[provable_contracts_macros::contract(
    "qlora-training-loop-v1", equation = "lora_forward_wgsl"
)]
pub fn from_wgsl_forward(
    fwd: WgslForwardPass,
    tokenizer: Tokenizer,
    config: InstructConfig,
    model_config: TransformerConfig,
) -> Result<Self> {
    // WgslForwardPass IS the model. No Transformer needed.
    // LoRA adapters created and uploaded to GPU.
    // Optimizer states allocated on GPU.
    // Tokenizer for prompt/response encoding.
}
```

**Provable contract: `wgsl-training-pipeline-v1`**

```yaml
equations:
  fast_load:
    formula: "load_time(from_wgsl_forward) < load_time(from_apr) / 5"
    invariants:
      - "Q4K model stays quantized until GPU dequant"
      - "No F32 CPU tensor allocation for projection weights"
      - "Streaming dequant: one layer at a time, not all 28"
  no_transformer:
    formula: "from_wgsl_forward does not construct Transformer"
    invariants:
      - "No Transformer::from_apr() call"
      - "No Transformer::from_safetensors() call"
      - "Forward pass via WgslForwardPass only"
falsification_tests:
  - id: FALSIFY-WGSL-PIPE-001
    rule: Fast load
    prediction: "from_wgsl_forward loads 7B model in < 5 min on GB10"
    test: "Measure wall time, compare with from_apr (~20 min)"
  - id: FALSIFY-WGSL-PIPE-002
    rule: No SATD
    prediction: "grep -r 'TODO\|FIXME\|HACK\|workaround' in from_wgsl_forward = 0"
    test: "Static analysis"
```

### 26.11.5 GPU-Only Backward: Saved Activations Design (from research)

Based on PyTorch `derivatives.yaml`, Unsloth `fast_lora.py`, ggml backward graph,
QVAC-fabric-llm.cpp, and Korthikanti et al. (MLSys 2023 "Reducing Activation
Recomputation in Large Transformer Models", arxiv 2205.05198).

**Minimum saved activations per transformer layer for LoRA backward:**

| # | Tensor | Shape | Purpose |
|---|--------|-------|---------|
| 1 | `attn_norm_out` | [B, S, D] | Input to Q/K/V projections. For LoRA grad_A/grad_B. |
| 2 | `attn_output` | [B, S, D] | Input to O projection. For LoRA grad on o_proj. |
| 3 | `ffn_norm_out` | [B, S, D] | Input to gate/up. For LoRA grad on gate/up/down. |
| 4 | `silu_gate_output` | [B, S, D_ffn] | SiLU(gate)×up = input to down_proj. For LoRA grad. |
| 5 | `rstd_attn` | [B, S, 1] | RMSNorm reciprocal std. For RMSNorm backward. Tiny. |
| 6 | `rstd_ffn` | [B, S, 1] | FFN RMSNorm reciprocal std. Tiny. |
| 7 | `softmax_logsumexp` | [B, H, S] | Compact softmax stats for attention backward (FlashAttention-2 approach). Negligible memory. Required for correct Q/K/V LoRA gradients. |

**FALSIFIED (2026-03-31):** Original 6-tensor list was insufficient — missing
`softmax_logsumexp` required for correct attention backward. Without it, Q/K/V
LoRA gradients use a simplified approximation (grad_q ≈ grad_attn_out, grad_k =
grad_v = 0) which is WRONG. Added 7th tensor per FlashAttention-2 approach
(logsumexp is [B, H, S] = negligible memory).

**Memory: ~232 MB/layer in FP32 (for 7B, batch=1, seq=2048). 28 layers = ~6.5 GB.**
Fits easily in GB10's 119 GB unified memory.

**Key insight from research:** The frozen base weights do NOT need saving for backward
— they're read-only, already in memory. Dequantize NF4 on-the-fly during backward
(same as Unsloth). LoRA A/B are trainable parameters, always in memory.

**LoRA gradient formula (from Hu et al. 2021, verified in Unsloth):**
```
For h = W_base @ x + (x @ A) @ B * (α/r):
  grad_B = ((x @ A)^T @ grad_output) * (α/r)    [rank, out_dim]
  grad_A = (x^T @ (grad_output @ B^T)) * (α/r)  [in_dim, rank]
  grad_x = grad_output @ W_base^T + (grad_output @ B^T @ A^T) * (α/r)
```
Both LoRA gradients need only `x` (saved activation) and the LoRA weights (in memory).

**Backward pass order (mirrors forward in reverse):**
```
1. Fused CE backward → grad_logits (in-place, already done)
2. lm_head backward: grad_hidden = grad_logits @ embed_weight^T
3. For each layer L = 27..0:
   a. Residual backward: grad_output duplicated to BOTH FFN sublayer + identity path.
      After FFN backward, results SUMMED: grad_residual = grad_output + grad_ffn.
      (NOT split/divided — the same grad feeds both branches, results are added.)
   b. Down projection backward: grad_silu = grad @ W_down^T
   c. SwiGLU backward: grad_gate, grad_up from saved silu_gate_output
   d. Gate/Up backward: grad_ffn_norm = (grad_gate @ W_gate^T + grad_up @ W_up^T)
   e. FFN RMSNorm backward: using saved rstd_ffn
   f. Residual backward: grad duplicated to attention sublayer + identity path, results SUMMED.
   g. O projection backward: grad_attn = grad @ W_o^T
   h. Attention backward: recompute Q,K from saved attn_norm_out, use saved softmax_logsumexp
      for softmax Jacobian. grad_Q, grad_K, grad_V computed correctly (not approximated).
   i. Q/K/V backward: using saved attn_norm_out
   j. Attention RMSNorm backward: using saved rstd_attn
   k. Accumulate LoRA gradients for all 7 projections
4. GPU AdamW step on all LoRA A/B weights

### 26.11.6 Required Provable Contracts (from research)

**17+ existing backward contracts verified.** 3 new contracts needed:

| New Contract | Purpose | Falsification Test |
|---|---|---|
| `saved-activation-correctness-v1` | Cached activation == forward activation bit-identical | Corrupt one cached value, verify backward produces wrong gradient |
| `lora-backward-formula-v1` | grad_A, grad_B match Hu et al. closed-form vs CPU reference | Swap A/B in formula, verify test catches it |
| `residual-gradient-flow-v1` | dy/dx = I + d_sublayer/dx for residual connections | Remove residual identity path, verify gradient drops |

**Already well-covered (no new contract needed):**
- Backward GEMM transpose: `gemm-backward-tiled-v1` (10 falsification tests)
- Fused CE backward: `fused-cross-entropy-v1`, `inplace-cross-entropy-v1`
- SiLU/RMSNorm/RoPE backward: `wgpu-backward-training-v1` (6 GPU/CPU parity tests)
- AdamW: `adamw-kernel-v1` (11 falsification tests, 14 Kani harnesses)
- LoRA transpose chain: `lora-gradient-flow-v1` (3 tests passing)

### 26.11.2 End-to-End Training Verification

**Status: COMPLETED on gx10 (pre-chunking run: ~5.5 hrs, 8.77M GPU matmuls, no crash)**

The pre-chunking run completed successfully with CPU forward fallback:
- 8,770,000 GPU matmuls over ~5.5 hours — zero crashes, zero NaN
- Training loss output not captured (tail truncation), but process exited cleanly
- New run with chunked lm_head GPU matmul in progress

| Component | Path | Status |
|-----------|------|--------|
| Model load | CPU (Q4K dequant) | WORKING |
| Forward pass | CPU fallback (lm_head > 2GB) | WORKING (slow: ~1.6 hrs/sample) |
| wgpu matmuls | GPU (130K+ completed) | WORKING (no crash) |
| Fused cross-entropy | wgpu GPU | WORKING (FALSIFY-FCE-001 passed) |
| Backward pass | CPU autograd | WORKING |
| Optimizer | CPU AdamW | WORKING |
| Memory | 33 GB RSS (stable, no leak) | WORKING |

**Proven:**
- Pipeline wiring is correct (no crash, no NaN)
- wgpu GEMM is stable (130K+ matmuls)
- Fused CE matches naive (ε < 1e-4)
- CUDA↔wgpu parity (3/3 tests on gx10)
- End-to-end synthetic training (loss 0.14→0.13, 10 steps)
- 375 GFLOPS sustained on GB10 Vulkan

**Blocked by:** §26.11.1 (lm_head 2 GB limit). Once chunked, full GPU forward
will use tiled GEMM at 375 GFLOPS → estimated ~50 tok/s training throughput.

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
