# 26. QLoRA Training Loop — Implementation and Status

## 26.7 Implementation Plan

### Phase 0: WGSL Tiled GEMM + NF4 Dequant + Eliminate Unsafe FFI (trueno-gpu + decy)

**Priority: HIGHEST -- this is the 20-100x speedup + zero-unsafe compliance.**

**Step 0a: Transpile bitsandbytes NF4 math via decy**

```bash
# Tier 1: Pure C math functions -> safe Rust (direct transpilation)
decy transpile bitsandbytes/csrc/kernels.cu \
  --functions dDequantizeNF4,dQuantizeNF4,nf4_dequantization_lut \
  --output trueno/src/quantize/nf4_bnb.rs
```

Tier 1 functions (pure math, zero unsafe):
- `nf4_dequantization_lut[16]` -> `const NF4_LUT: [f32; 16]`
- `dDequantizeNF4(val)` -> `fn dequantize_nf4(val: u8) -> f32`
- `dQuantizeNF4(x)` -> `fn quantize_nf4(x: f32) -> u8`

Tier 3 algorithms (CUDA kernels -> WGSL compute shaders for wgpu):
- `kDequantizeBlockwise` algorithm -> WGSL compute shader
- `kQuantizeBlockwise` algorithm -> WGSL compute shader

**Step 0b: CUTLASS-style tiled GEMM in WGSL (replaces cuBLAS entirely)**

Implement the CUTLASS tiling algorithm (MIT licensed, ~200 lines of logic) as a
WGSL compute shader, called via wgpu's safe Rust API. Zero `unsafe`, zero FFI.

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

**Step 0c: NF4 dequant -> F32 -> WGSL GEMM pipeline**

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
    // 1. WGSL shader: dequant NF4 -> F32 (algorithm transpiled from bitsandbytes via decy)
    // 2. WGSL tiled GEMM: output = input @ dequant_buffer^T
}
```

**Step 0d: WgpuTrainingPipeline -- complete replacement for CUDA training path**

NOT a hybrid/hack. A complete GPU training pipeline in wgpu that replaces the entire
`CudaTrainer` + `CudaBlock` + `CudaBlockScratch` + `GpuTraining` infrastructure.

The CUDA training path (`instruct_pipeline.rs:660-793`) does 6 operations ALL on GPU:
1. Forward: NF4 dequant -> GEMM -> RMSNorm -> attention -> SwiGLU x 28 layers
2. lm_head: GEMM (hidden -> vocab logits)
3. Loss: fused causal cross-entropy (in-place gradient)
4. lm_head backward: GEMM (grad_logits -> grad_hidden)
5. Backward: GEMM backward through 28 NF4 layers (LoRA gradients)
6. Optimizer: AdamW on LoRA weights

**What already exists (proven):**
- WGSL tiled GEMM (forward + backward) -- `ac65854f`, 375 GFLOPS on GB10
- WGSL RMSNorm, SwiGLU, RoPE, attention, residual -- in `wgsl_forward.rs`
- NF4 dequant in safe Rust -- `2d151d45`, 6/6 tests
- WgpuTrainer (AdamW + gradient clip) -- `dae8a812`, 3/3 tests
- CUDA<->wgpu parity -- 3/3 tests on gx10

**What needs building:**
- WgpuBlockManager -- upload 28 layers of NF4 weights to wgpu buffers
- WgslForwardPass training mode -- checkpoint activations
- WgslBackwardPass -- backward through full transformer stack
- WgslCrossEntropy -- fused chunked cross-entropy
- Pipeline integration -- `InstructPipeline::wgpu_train_step`

**Step 0e: Parity gate -- wgpu training matches CUDA training**

Before deleting ANY CUDA code, the following parity tests must pass:

| Test | Criterion | Status |
|------|-----------|--------|
| 3-sample loss match | \|loss_wgpu - loss_cuda\| < 0.1 after 1 epoch | MUST PASS |
| Gradient norm match | \|norm_wgpu - norm_cuda\| / norm_cuda < 0.05 | MUST PASS |
| 100-sample stability | No NaN/Inf over 1 epoch | MUST PASS |
| HumanEval inference parity | wgpu pass@1 = CUDA pass@1 (already proven: 84.15%) | **PASSED** |
| WgpuTrainer unit tests | Forward/backward/AdamW match CPU reference | **PASSED** (3/3) |
| **CUDA<->wgpu forward GEMM** | max error < 0.01 on gx10 GB10 | **PASSED** |
| **CUDA<->wgpu backward GEMM** | grad_a + grad_b max error < 0.01 | **PASSED** |
| **CUDA<->wgpu AdamW** | params max error < 1e-4 after 1 step | **PASSED** |

**Step 0f: Delete CUDA code from ALL affected repos (ONLY after 0e passes)**

Deletion spans 3 repos. All have wgpu replacements proven.

**Grand total across all repos: ~33 files, ~10,920 lines deleted.**

After deletion: Zero `extern "C"` declarations, zero `unsafe` blocks, zero `unsafe impl` blocks. One GPU backend: wgpu (safe Rust API -> Vulkan/Metal/DX12). WGSL compute shaders for all GPU operations.

### Phase 1: Bridge `apr finetune` -> entrenar (aprender change)

**File:** `aprender/crates/apr-cli/src/commands/finetune.rs`

Replace the stub `execute_training()` with bridge to entrenar's InstructPipeline.

### Phase 2: Model Bridge (`InstructPipeline::from_quantized_model`)

**File:** `entrenar/src/finetune/instruct_pipeline.rs`

New constructor that accepts `OwnedQuantizedModel` instead of requiring SafeTensors.

### Phase 3: APR Export

Export trained LoRA A/B weights from pipeline to APR format.

### Phase 4: Merge Support

```bash
# Train adapter
apr finetune model.apr --method qlora --data train.jsonl --output adapter.apr

# Merge adapter into base
apr finetune model.apr --adapter adapter.apr --merge --output merged.apr

# Evaluate merged model
make eval-humaneval CHECKPOINT=checkpoints/merged.apr
```

## 26.11 Known Blockers and Status (2026-03-31)

### 26.11.1 wgpu 2 GB Buffer Binding Limit

**Status: RESOLVED -- lm_head pre-chunked at init, GPU scatter/gather shaders.**

wgpu's `max_storage_buffer_binding_size` capped at 2 GB. lm_head for Qwen 7B = 2.18 GB.
Fix: pre-chunk into <2 GB pieces at pipeline init. GPU scatter/gather shaders
assemble/extract per-chunk results without CPU roundtrip.

### 26.11.3 Per-Call Buffer Creation in model.forward()

**Status: RESOLVED -- WgpuInstructPipeline uses WgslForwardPass with persistent
weight buffers, single command encoder per layer, tiled GEMM (375 GFLOPS).**

### 26.11.8 Final PROFILE Results (2026-03-31)

**315x speedup achieved. 5+ hours -> 57 seconds. Loss correct.**

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
| 1 | CPU autograd replays entire forward | Saved activations, GPU-only backward | 5+ hrs -> 7 min |
| 2 | Transformer::from_apr() 28GB CPU dequant | OwnedQuantizedModel -> GPU direct | 20 min -> 19s init |
| 3 | WgpuTrainer used 16x16 MATMUL_SHADER | Switch to 64x64 TILED_GEMM_SHADER | 20x GEMM |
| 4 | 1024 copy_buffer_to_buffer per step | WGSL scatter/gather shaders | 1 dispatch |
| 5 | Attention 3-pass QK^T recomputation | Store scores in shared memory | 7 min -> 69s |
| 6 | Attention @workgroup_size(1) sequential | 128 threads parallel dot+V sum | 69s -> 57s |
| 7 | 2GB wgpu buffer limit on lm_head | Pre-chunk at init, scatter on GPU | No crash |
| 8 | Per-step lm_head buffer allocation | Pre-upload at init, reuse | -2s/step |

### 26.11.9 LoRA Weight Updates -- Contract-First Design

**Status: IMPLEMENTED -- GPU transpose + matmul_forward path (2026-04-01). Adapter export in PEFT format.**

**Governing contracts:**
- `lora-algebra-v1 / lora_shape`: A[in, rank], B[rank, out]
- `wgpu-production-training-v1 / C-WGPU-LORA-BWD-001`
- `adamw-kernel-v1 / weight_update`: decoupled weight decay
- `lora-gradient-flow-v1`: B_norm > 0 after step 1 (B starts at zero)

**Falsification tests (from contracts):**
- FALSIFY-LORA-UPD-001: B_norm > 0 after step 1 (was zero-initialized)
- FALSIFY-LORA-UPD-002: dL/dA and dL/dB match CPU reference within eps < 1e-3
- FALSIFY-LORA-UPD-003: loss at step N < loss at step 0 (training makes progress)
- FALSIFY-LORA-UPD-004: base weights unchanged after step (frozen)
- FALSIFY-LORA-GRAD-001: dB non-zero when XA and G are non-zero (NEW, passes)

### 26.11.10 KAIZEN Optimization Chain (2026-04-01)

**13 root causes fixed. Fully GPU-resident pipeline -- zero CPU downloads during training.**

| # | Root Cause | Fix | Speedup |
|---|-----------|-----|---------|
| 1 | 16x16 GEMM shader (MATMUL) | Switch to 64x64 tiled GEMM (CUTLASS) | 1200x |
| 2 | 1024 copy_buffer_to_buffer/step | WGSL scatter/gather shaders | ~10x |
| 3 | Attention @workgroup_size(1) | 128-thread parallel dot + softmax | ~100x |
| 4 | 20 min Transformer::from_apr() | OwnedQuantizedModel direct upload | 60x |
| 5 | Per-step lm_head download (189s) | Pre-chunk at init, GPU scatter | ~100x |
| 6 | LoRA after attention consumed Q/K/V | Inline LoRA addmm before attention | correctness |
| 7 | RMSNorm dispatch(1,1,1) | Multi-row via workgroup_id.y | correctness |
| 8 | WgpuTrainer::new() creates 2nd device | from_device() shares device | correctness |
| 9 | CPU RMSNorm roundtrip (44s download) | GPU RMSNorm, hidden stays on GPU | 626x on norm |
| 10 | LoRA addmm shader 0.11 GFLOPS | Two tiled GEMM dispatches + residual add | 151x |
| 11 | **CE forward blocks 10.7s on GPU sync** | **forward_async() + deferred read_loss()** | **async** |
| 12 | **lm_head backward CPU download (11.6s)** | **GPU-resident accumulate via residual add** | **174x** |
| 13 | **LoRA backward CPU transpose (16.5s)** | **WGSL GPU transpose shader** | **12.9x** |

**Current performance (gx10 GB10, 7B Q4K, seq_len<=512, 2026-04-02):**
- Pipeline init: 20s (model load + dequant + upload)
- JIT warmup: first step ~1.4s (shader compilation), first B!=0 step ~13s
- Steady state: 300-800ms/step (short sequences); 11.9s/step average (mixed lengths)
- All operations async: ce=0, lm_bwd=65ms. ONE sync point: `read_loss()` at step end.
- **50 samples x 3 epochs: 29.7 min (11.9s/step avg)**

**Training results (50 samples, 3 epochs, 2026-04-02):**
- Loss: 17.17 -> 16.31 -> **16.09** (decreasing across all epochs)
- B_norm: 0.000 -> 0.071 -> 0.268 -> 0.549 (growing correctly)
- FALSIFY-LORA-UPD-001: **PASSED** (B_norm > 0 after step 1)
- FALSIFY-LORA-UPD-003: **PASSED** (loss epoch 3 < epoch 1)
- Adapter export: 392 tensors (617 MB safetensors), merge into .apr verified
- End-to-end inference on merged model verified (CUDA, generates tokens)

### 26.11.7 Model Loading Bottleneck: Transformer::from_apr() (2026-03-31)

**Status: RESOLVED -- WgpuInstructPipeline bypasses Transformer entirely (20s init).**

### 26.11.5 GPU-Only Backward: Saved Activations Design (from research)

**Minimum saved activations per transformer layer for LoRA backward:**

| # | Tensor | Shape | Purpose |
|---|--------|-------|---------|
| 1 | `attn_norm_out` | [B, S, D] | Input to Q/K/V projections. For LoRA grad_A/grad_B. |
| 2 | `attn_output` | [B, S, D] | Input to O projection. For LoRA grad on o_proj. |
| 3 | `ffn_norm_out` | [B, S, D] | Input to gate/up. For LoRA grad on gate/up/down. |
| 4 | `silu_gate_output` | [B, S, D_ffn] | SiLU(gate) x up = input to down_proj. For LoRA grad. |
| 5 | `rstd_attn` | [B, S, 1] | RMSNorm reciprocal std. For RMSNorm backward. Tiny. |
| 6 | `rstd_ffn` | [B, S, 1] | FFN RMSNorm reciprocal std. Tiny. |
| 7 | `softmax_logsumexp` | [B, H, S] | Compact softmax stats for attention backward. |

**Memory: ~232 MB/layer in FP32 (for 7B, batch=1, seq=2048). 28 layers = ~6.5 GB.**

### 26.11.6 Required Provable Contracts (from research)

**17+ existing backward contracts verified.** 3 new contracts needed:

| New Contract | Purpose | Falsification Test |
|---|---|---|
| `saved-activation-correctness-v1` | Cached activation == forward activation bit-identical | Corrupt one cached value, verify backward produces wrong gradient |
| `lora-backward-formula-v1` | grad_A, grad_B match Hu et al. closed-form vs CPU reference | Swap A/B in formula, verify test catches it |
| `residual-gradient-flow-v1` | dy/dx = I + d_sublayer/dx for residual connections | Remove residual identity path, verify gradient drops |

### 26.11.2 End-to-End Training Verification

**Status: COMPLETED on gx10 (pre-chunking run: ~5.5 hrs, 8.77M GPU matmuls, no crash)**

The pre-chunking run completed successfully with CPU forward fallback:
- 8,770,000 GPU matmuls over ~5.5 hours -- zero crashes, zero NaN
- New run with chunked lm_head GPU matmul in progress

**The pipeline is GPU-bound.** The 28-layer forward compute (238.7 GFLOP/layer)
dominates. wgpu upgraded to 29.0 (2026-04-02) -- tiled GEMM improved from
375->592 GFLOPS (+58%) from the wgpu upgrade alone. Cooperative matrix WGSL shader
compiles but naga 29 SPIR-V backend crashes (known bug). Deferred until naga fix.
