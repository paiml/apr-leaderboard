# 26. QLoRA Training Loop Specification

## 26.1 Problem Statement

`apr finetune --method qlora` trains a LoRA adapter on GPU via `WgpuInstructPipeline` (wgpu 29, 592 GFLOPS tiled GEMM). Supports SFT (instruction/response JSONL) and DPO (preference pairs JSONL, auto-detected). 13 KAIZEN optimizations, 31 provable contracts, 8 Lean4 theorems.

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
| `train_step` | `entrenar/src/finetune/instruct_pipeline.rs:574` | Forward -> loss -> backward -> optimizer, CPU + CUDA paths |
| Gradient clipping | `entrenar/src/finetune/instruct_pipeline.rs` | Max-norm clipping |
| CUDA training | `entrenar/src/autograd/cuda_training.rs` | NF4 QLoRA on GPU |
| Memory planner | `entrenar-lora/src/memory.rs` | VRAM estimation for QLoRA configs |
| Merge engine | `entrenar-lora/src/merge.rs` | Adapter merge into base model |

### 26.2.2 What EXISTS (aprender)

| Component | Location | Status |
|-----------|----------|--------|
| CLI `finetune` command | `apr-cli/src/commands/finetune.rs` | Parses args, plans config, creates adapter APR -- **no training** |
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
| APR model -> entrenar model bridge | `OwnedQuantizedModel` -> entrenar's model trait | Forward pass in training |
| Data loader for JSONL | Parse `{"instruction": ..., "response": ...}` -> tokenized pairs | Training data |
| Checkpoint-to-APR export | Save trained LoRA weights back to .apr format | Output |
| Tokenizer integration | APR sibling tokenizer -> entrenar tokenizer interface | Tokenization |

## 26.3 Architecture: Bridge Pattern

The fix is NOT reimplementing training in aprender. The fix is **bridging** aprender's model loading + CLI with entrenar's training loop.

```
apr finetune model.apr --method qlora --data train.jsonl --output distilled.apr
    |
    +-- 1. Load model: realizar::OwnedQuantizedModel::from_apr(path)
    +-- 2. Load tokenizer: sibling tokenizer.json
    +-- 3. Load data: parse JSONL -> Vec<(instruction, response)>
    +-- 4. Create InstructPipeline with model + tokenizer + LoRA config
    +-- 5. Create InstructTrainer with pipeline + training config
    +-- 6. trainer.train() -> epoch loop with loss/backward/optimizer
    +-- 7. Export trained LoRA weights -> APR file
    +-- 8. Optionally merge: base + adapter -> merged APR
```

## 26.4 Mathematical Specification

### 26.4.1 QLoRA Forward Pass (Unsloth-informed, per Dettmers et al. 2023)

For each linear layer `W in R^{m x n}` in the transformer, with batch size `B_s`:

```
W_f32 = DequantNF4->F32(W_nf4)       # WGSL shader: NF4 LUT lookup x absmax (algorithm from decy)
h_base = WGSL_GEMM(x, W_f32^T)      # Tiled GEMM: CUTLASS-style 128x128, shared memory, safe Rust
h_lora = WGSL_GEMM(WGSL_GEMM(x, A), B) * (a/r)  # Two small GEMMs via same shader
h = h_base + h_lora                  # Fused add in epilogue (alpha=s, beta=1)
```

Where:
- `A in R^{n x r}` -- LoRA down-projection (Kaiming init), BF16
- `B in R^{r x m}` -- LoRA up-projection (zero init), BF16
- `r` -- LoRA rank (e.g., 32)
- `a` -- LoRA alpha scaling (e.g., 64)
- `x in R^{B_s x n}` -- batched input hidden states (batch_size x hidden_dim), BF16

**Critical architecture decision (from Unsloth + CUTLASS analysis):** All GEMM operations
use a CUTLASS-style tiled GEMM implemented in WGSL compute shaders via wgpu (safe Rust
API). NO cuBLAS FFI, NO CUDA driver FFI, NO `unsafe` code. The tiling algorithm is
derived from NVIDIA's open-source CUTLASS library (MIT licensed) which achieves 90-95%
of cuBLAS throughput.

**Zero-unsafe mandate:** trueno-gpu currently has 68 `extern "C"` function pointers,
137 `unsafe` blocks, and 18 `unsafe impl` blocks -- all for CUDA driver/cuBLAS/cuBLASLt
FFI. ALL of these are eliminated -- not feature-gated, REMOVED. The replacement is wgpu
(safe Rust API for Vulkan/Metal/DX12 GPU compute). The PTX code generator (~5,500
lines), CUDA driver bindings, cuBLAS/cuBLASLt bindings -- all deleted. All GPU compute
goes through WGSL compute shaders via wgpu.

**Single backend: wgpu only.** There is no CUDA feature flag, no dual-backend. wgpu
speaks Vulkan on NVIDIA GPUs, accessing the same hardware including tensor cores via
`VK_KHR_cooperative_matrix` (confirmed on gx10 GB10: revision 2, BF16+FP8 enabled).

**Falsified claims (corrected):** Vulkan GEMM does NOT match CUDA on discrete GPUs --
the gap is 20-50% on A100 due to architectural limits (no `cp.async` equivalent in
SPIR-V, smaller cooperative matrix sizes in KHR vs CUDA wmma, Vulkan vectorization
limited to line size 4 vs 8). However, on GB10 unified memory (our target hardware),
the gap effectively disappears because `cp.async` optimizes discrete GPU memory
transfers which are irrelevant on unified memory. llama.cpp benchmarks show Vulkan
matching or exceeding CUDA on GB10 for token generation.

**wgpu cooperative matrix status:** Upgraded to wgpu 29.0 (2026-04-02). Feature confirmed
on gx10 GB10: `EXPERIMENTAL_COOPERATIVE_MATRIX = true`, 6 configurations available.
Best config: **M=16, K=16, N=16, F16 input, F32 accumulation** (config 3).
No F32xF32 -- requires F32->F16 conversion for inputs, F32 accumulation for precision.
Contract: cooperative-matrix-gemm-v1.

**CUTLASS algorithm in WGSL (not C++ transpilation):** CUTLASS is C++ templates -- decy
handles C, not C++. Instead, we read the CUTLASS algorithm (MIT licensed, ~200 lines of
actual logic) and reimplement the tiling strategy in WGSL:
- Thread-block tile: 128x128x8 (output tile x K-step)
- Warp tile: 32x64 (per-warp output region)
- Thread micro-tile: 8x8 (per-thread output, outer-product accumulation)
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

For a sequence batch `[t_1, t_2, ..., t_T]` with prompt length `P`:

```
# Fused: never materialize full [B_s x T, V] logit tensor
for chunk in chunks(hidden_states, CHUNK_SIZE=65536):
    logits_chunk = cuBLAS_GEMM(chunk, lm_head^T)    # [B_s, chunk, V]
    logsumexp_chunk = log(sum(exp(logits_chunk)))     # [B_s, chunk] scalar per token
    loss_chunk -= logits_chunk[labels] - logsumexp    # Accumulate NLL

loss = sum(loss_chunks) / R   # R = response tokens only
```

**Memory savings (from Unsloth):** Avoids materializing the full `[B_s x T, V]` logit
tensor (e.g., 4 x 2048 x 32000 x 2 = 500 MB). Instead, only `[B_s x T]` logsumexp
scalars are saved (~32 KB). Backward writes gradients in-place into the logits buffer.
For 256K-vocab models, this saves ~8 GB.

Where `R = T - P` is the number of response tokens.

### 26.4.3 Backward Pass (LoRA only, with gradient checkpointing)

Gradients flow only through LoRA A and B matrices. All backward GEMMs use WGSL tiled GEMM:

```
# Re-dequantize base weight for backward (gradient checkpointing: not saved from forward)
W_f32 = DequantNF4->F32(W_nf4)     # WGSL dequant shader

# Gradient w.r.t. input (for upstream layers)
dL/dx = WGSL_GEMM(dL/dh, W_f32) + WGSL_GEMM(WGSL_GEMM(dL/dh, B^T), A^T) * (a/r)

# LoRA gradients (via WGSL GEMM with fused scaling in epilogue)
dL/dB = WGSL_GEMM((A^T @ x)^T, dL/dh) * (a/r)   # epilogue alpha=a/r, beta=0
dL/dA = WGSL_GEMM(x^T, dL/dh @ B^T) * (a/r)     # epilogue alpha=a/r, beta=0
```

Base weights `W_nf4` receive no gradient (frozen). The autograd engine skips the
entire frozen subgraph via topological pruning (per PyTorch autograd architecture).

**Gradient checkpointing:** Activations are NOT saved across layers. Each layer
boundary is a checkpoint; intermediate activations (RMSNorm output, attention scores,
FFN intermediates) are recomputed during the backward pass. This trades ~33% extra
compute for ~60% memory savings, enabling batch_size=4-8 instead of 1.

**In-place memory reuse (from Unsloth):** Input activation `X` is overwritten with
`dL/dX` when no longer needed. SwiGLU backward writes derivatives into input buffers.
Dequantized weights are immediately freed after each backward GEMM.

### 26.4.4 AdamW Update (per Loshchilov & Hutter 2017)

For each LoRA parameter `theta in {A, B}`:

```
m_t = b1 * m_{t-1} + (1 - b1) * g_t          # First moment
v_t = b2 * v_{t-1} + (1 - b2) * g_t^2        # Second moment
m_hat_t = m_t / (1 - b1^t)                    # Bias-corrected first moment
v_hat_t = v_t / (1 - b2^t)                    # Bias-corrected second moment
theta_t = theta_{t-1} - lr * (m_hat_t / (sqrt(v_hat_t) + eps) + lambda * theta_{t-1})
```

Default hyperparameters: `b1=0.9, b2=0.999, eps=1e-8, lambda=0.01`.

### 26.4.5 Learning Rate Schedule (Cosine with Warmup)

```
if step < warmup_steps:
    lr = lr_base * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + cos(pi * progress))
```

## 26.5 Memory Model

For a model with `P` parameters, LoRA rank `r`, `L` adapted layers, batch size `B_s`:

```
Trainable params:    T = 2 * r * d * L * K    (A and B per layer per projection, K=7)
Base model:          P_bytes / 2               (NF4 = 0.5 bytes/param)
Dequant buffer:      max(m,n) * d * 2 bytes   (single BF16 weight, reused per layer)
LoRA adapters:       T * 2 bytes              (BF16)
Optimizer states:    T * 8 bytes              (m + v, both FP32)
Activations:         B_s * S * d * 2 bytes    (per checkpoint boundary, BF16)
Gradients:           T * 2 bytes              (BF16, FP32 accumulation in cuBLAS)
cuBLAS workspace:    ~256 MB                   (cuBLAS internal workspace)

Total ~ P/2 + 12*T + B_s*S*d*2*sqrt(L) + 256MB
```

Note: `sqrt(L)` factor from gradient checkpointing (only checkpoint boundaries saved,
not all L layers).

For 7B Q4K, rank 32, 28 layers, batch_size=4:
- Base model: 3.75 GB (Q4K)
- Dequant buffer: 18944 x 3584 x 2 = 136 MB (reused, single largest weight matrix)
- LoRA: 2 x 32 x 3584 x 28 x 7 ~ 45M params x 2 = 0.09 GB
- Optimizer: 45M x 8 = 0.36 GB
- Activations: 4 x 512 x 3584 x 2 x sqrt(28) ~ 78 MB (with gradient checkpointing)
- cuBLAS workspace: 256 MB
- **Total: ~4.7 GB** (fits easily on gx10 119 GB, leaves room for batch_size=8)

**Comparison with v1 spec:** Previous spec had batch_size=1 with FP32 LoRA (5.5 GB).
New spec uses BF16 LoRA + gradient checkpointing + cuBLAS, achieving lower memory
at 4x batch size. The memory savings enable the throughput gains (cuBLAS GEMM
utilization scales with batch size).

## 26.6 Provable Contracts

For the full set of provable contracts (qlora-training-loop-v1, wgsl-gemm-tiled-v1, nf4-dequantization-v1, fused-cross-entropy-v1), falsification tests, and Kani harnesses, see [QLoRA Contracts (S26b)](26b-qlora-contracts-implementation.md).
