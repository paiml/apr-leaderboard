# Training Infrastructure

Training bricks, QLoRA readiness, GPU sharing, and wgpu proof findings. Split from [Dogfooding Findings](22a-dogfooding-baselines.md) for file size compliance.

## 23.1 Training & Serving Bricks (QLoRA Foundation)

Added 7 new `ComputeBrick` types to realizar and wired them into `apr bench --brick`. These provide measurable performance contracts for the QLoRA training loop (Recipe F) and serving path.

### 23.1.1 Training Bricks

All training bricks read **real model architecture** from `.apr` metadata. Tested on `qwen2.5-coder-7b-instruct-q4k.apr` (Qwen2 architecture):

| Brick | CLI Name | Dimensions from Model | Result | Type |
|-------|----------|----------------------|--------|------|
| LoRA forward | `apr bench <model> --brick lora_forward` | d_in=3584, d_out=3584, rank=16 | **54us** | Real matmul |
| Optimizer step | `apr bench <model> --brick optimizer` | 6,422,528 LoRA params (28 layers x rank-16 x Q,V) | 50us | Analytical |
| Loss compute | `apr bench <model> --brick loss` | vocab=152,064, seq=128 | 20us | Analytical |
| Training step | `apr bench <model> --brick train_step` | hidden=3584, 28 layers, rank=16 | 5,000us | Analytical |

Key findings:
- `lora_forward` runs an actual two-stage matmul using model-accurate dimensions. The 54us CPU result for a 3584-dim rank-16 projection is consistent with expected FLOP count (~230K FLOPs).
- LoRA parameter count formula: `num_layers x 2 x rank x hidden_dim x 2` = 28 x 2 x 16 x 3584 x 2 = 6,422,528 trainable parameters (Q and V projections).
- All bricks correctly parse APR v2 metadata JSON to extract `hidden_dim`, `num_layers`, `vocab_size`, and `architecture` fields.

### 23.1.2 Serving Bricks

Serving bricks load the **real 7.5 GiB model** and run actual autoregressive generation:

| Brick | CLI Name | Config | Result | Notes |
|-------|----------|--------|--------|-------|
| TTFT | `apr bench <model> --brick ttft` | 7 prompt tokens -> 1 output token | **761ms** | CPU 7B, CV=1.6% |
| Throughput | `apr bench <model> --brick throughput` | 7 prompt -> 32 output tokens | **~8 tok/s** | CV=1.7% |
| Batch | `apr bench <model> --brick batch` | 4 x 16 tokens sequential | **~6 tok/s** | CV=3.1% |

Key findings:
- Serving bricks are statistically stable (CV < 5% on all measurements, 5 iterations with 3 warmup).
- 8 tok/s CPU decode for 7B Q4K is consistent with full-model benchmark results.
- TTFT of 761ms on CPU includes full prefill + first decode step. GPU TTFT via wgpu should be ~10-50ms.
- Budget targets (500us TTFT, 50 tok/s decode) are GPU-oriented. CPU results serve as baseline.

### 23.1.3 QLoRA Readiness Checklist

| Prerequisite | Status | Evidence |
|---|---|---|
| Qwen3-8B imported (FP16) | Done | `checkpoints/qwen_qwen3-8b.apr` (16 GB) |
| Instruction corpus prepared | Done | `data/instruct-corpus.jsonl` (15,494 pairs) |
| Training loop validated | Done | S22.7: tiny model, loss decreasing over 2 epochs |
| BPE tokenizer fast enough | Done | S22.12: 70us/encode (2x faster than before, 1.49x faster than HF) |
| Tokenizer loading fast enough | Done | S22.12.1: 142ms load (1.43x faster than HF) |
| Training bricks benchmarked | Done | S23.1.1: real dimensions, parameter counts validated |
| Serving bricks benchmarked | Done | S23.1.2: real inference, stable measurements |
| EOS termination working | Done | S22.10: GH-373 fixed, stop tokens resolve correctly |
| Token generation uncapped | Done | S22.9: GH-372 fixed, max_tokens passes through |
| Recipe YAML configured | Done | `configs/recipes/recipe-f-qwen3-qlora.yaml` |
| QLoRA in InstructPipeline | Done | S23.1.4: NF4 quantization wired via wgpu |
| .apr weight loading | Done | `from_apr()` loading implemented |
| GPU inference (wgpu) | Done | wgpu backend -- any GPU vendor (Vulkan/Metal/DX12) |

### 23.1.4 QLoRA Instruct (Resolved)

**Problem:** `apr finetune --task instruct --method qlora --quantize-nf4` did not work. The `--task instruct` dispatch exited before the qlora method handling.

**Root cause:** `InstructPipeline` (entrenar) only supported full-precision LoRA. QLoRA (NF4 base weights + FP16 adapters) existed in `ClassifyPipeline` but was not plumbed into instruction fine-tuning.

**Status (2026-03-02): RESOLVED** -- All changes implemented and verified.

Commits:
- `entrenar@9e4d442`: QLoRA NF4 instruct fine-tuning with wgpu acceleration
- `aprender@ea586a31`: Wire QLoRA params through run_instruct()

Verification results (1.5B Q4K, 50 samples, max_seq_len=128, RTX 4090 via wgpu/Vulkan):
- 2 epochs completed in 137.6s (40 train, 10 val)
- Train loss: 15.06, Val loss: 53.99
- Checkpoints saved: `best/`, `epoch-0/`, `epoch-1/` (8.4 MB each, SafeTensors)

Verification results (7B Q4K, 40 samples, max_seq_len=128, RTX 4090 via wgpu/Vulkan):
- 1 epoch completed in 272.5s
- Train loss: 15.12, Val loss: 33.12

## 23.2 GPU-SHARE Multi-Adapter Training (Phase 2)

**Problem:** Training N LoRA adapters on the same base model required N separate processes, each loading the full 7B model to GPU (~7.3 GB each). 3 adapters = 21.9 GB VRAM.

**Solution:** MultiAdapterPipeline trains N independent LoRA adapter sets on a single frozen NF4 base model. Base model loaded once to GPU; each adapter maintains independent LoRA A/B matrices, optimizer state, and training data.

**VRAM savings:** 3 adapters on 7B: MPS = 21.9 GB vs multi-adapter = 7.36 GB (**3x savings**).

Implementation (2026-03-04):
- entrenar PR #208: `MultiAdapterPipeline` with RoundRobin/Synchronized/PriorityValLoss scheduling
- entrenar PR #209: Per-adapter checkpointing (metadata.json + model.safetensors per adapter slot)
- aprender PR #399: `--adapters DATA:CHECKPOINT` CLI flag with multi-adapter dispatch

```bash
apr finetune model.apr --task instruct --method qlora --quantize-nf4 \
  --adapters data/corpus-a.jsonl:checkpoints/adapter-a \
  --adapters data/corpus-b.jsonl:checkpoints/adapter-b \
  --rank 16 --epochs 3
```

**Spec status:** Complete. 143 GPU tests pass. Zero SATD across all 3 phases:
- Phase 1: VRAM guard, ledger, wait queue, profiler, MPS
- Phase 2: Multi-adapter pipeline, scheduling, adapters-config TOML
- Phase 3: Cluster config, placement, coordinator, SSH transport

## 23.3 Dual wgpu Training Proof (Recipe G)

**Goal:** Prove that the entire training pipeline runs on dual wgpu GPUs (Vulkan) without any CUDA toolkit dependency.

**Hardware:** 2x AMD Radeon Pro W5700X (Navi10), 16 GB VRAM each, Vulkan 1.3.255, RADV Mesa driver.

```
GPU0: /dev/dri/renderD128 -- AMD Radeon Pro W5700X (RADV NAVI10)
GPU1: /dev/dri/renderD129 -- AMD Radeon Pro W5700X (RADV NAVI10)
```

**Recipe:** `configs/recipes/recipe-g-wgpu-proof.yaml`

**What it proves:**
1. `apr import` produces a checkpoint that works with wgpu inference
2. `apr run --gpu` uses wgpu/Vulkan backend on **both** GPUs (not CUDA)
3. `apr finetune --method qlora` trains on GPU via wgpu with decreasing loss
4. Inference verified independently on GPU0 and GPU1 via `DRI_PRIME`
5. Post-training model produces valid code output
6. No CUDA toolkit is installed or referenced at any point

**Dual GPU strategy:**
- **GPU0 (renderD128):** Training workloads (`apr finetune`, `apr distill`)
- **GPU1 (renderD129):** Concurrent evaluation (`apr eval`, `apr run` for benchmarks)
- `DRI_PRIME=0` / `DRI_PRIME=1` selects GPU for each process

**How to run:** `make prove-wgpu`

**Success criteria:**
- [ ] Vulkan enumerates 2 discrete GPUs (verified: `vulkaninfo --summary`)
- [ ] Training completes with exit code 0 on GPU0
- [ ] Inference works on GPU0 AND GPU1 independently
- [ ] Loss values present in output and decreasing
- [ ] GPU backend indicators in verbose output (Vulkan/RADV/Navi)
- [ ] No `nvcc`, `libcudart`, or CUDA toolkit referenced in process
- [ ] `apr run --gpu` produces valid Python code post-training

**Verification:** `make prove-wgpu` runs all checks. See `scripts/prove-wgpu.sh`.

**Status:** READY to run. Dual GPU hardware confirmed.
