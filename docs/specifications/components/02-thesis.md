# Thesis

## 2.1 The Claim

> Can a single Rust binary (`apr`) match Python-ecosystem HumanEval/MBPP scores
> for Qwen2.5-Coder-7B, with zero Python dependencies?

This is the one falsifiable question that drives the entire project. If the answer
is yes, the sovereign Rust AI stack works end-to-end. If no, `apr compare-hf`
pinpoints exactly where it falls short.

## 2.2 The Problem with the Status Quo

The Python ML ecosystem requires:
- **200+ transitive dependencies** (transformers, torch, accelerate, bitsandbytes, peft, trl, vllm)
- **Vendor-locked CUDA toolchains** (nvcc, libcudart, cuDNN — NVIDIA only)
- **Multi-GB Docker images** (pytorch/pytorch: ~6 GB; vllm: ~15 GB)
- **30-60 minute setup** (CUDA toolkit install, conda env, pip conflicts)

These are not engineering choices — they are historical accidents. Nothing about
LoRA fine-tuning, weight merging, or INT4 quantization requires Python or CUDA.

## 2.3 The Constraint

Every optimization step must be expressible as an `apr` subcommand:

```
apr import → apr distill → apr finetune → apr merge → apr prune → apr quantize → apr eval → apr publish
```

**Hard rules:**
- No Python. No notebooks. No HuggingFace Transformers library.
- No CUDA toolkit. No nvcc. No libcudart.
- GPU compute via wgpu only (Vulkan/Metal/DX12) — any vendor.
- Pure sovereign stack: aprender, entrenar, trueno.

## 2.4 Compute Reality

| Resource | Value |
|----------|-------|
| GPUs | 2x AMD Radeon Pro W5700X (Navi10) |
| VRAM | 16 GB per GPU, 32 GB total |
| GPU backend | wgpu / Vulkan 1.3.255 (RADV) |
| Render devices | `/dev/dri/renderD128`, `/dev/dri/renderD129` |
| CPU | 16 cores, 64 GB RAM |
| GPU selection | `DRI_PRIME=0` (train), `DRI_PRIME=1` (eval) |

No NVIDIA hardware. No CUDA toolkit installed. No vendor lock-in.

Training and inference run on any GPU: NVIDIA, AMD, Intel Arc, or Apple Silicon
(Metal). The wgpu abstraction makes this portable by default.

## 2.5 Inference Without GPU

Inference-only techniques (merging, quantization) and small-model inference
(≤7B quantized) run on CPU via trueno SIMD (AVX2/NEON). GPU is recommended
for training-phase techniques (distillation, fine-tuning) but not required
for evaluation.

## 2.6 Falsification Criteria

The thesis is falsified if any of these hold after applying the full pipeline:
1. HumanEval pass@1 < 80% for Qwen2.5-Coder-7B (below "Strong" tier)
2. Inference parity gap > 5% vs HuggingFace reference implementation
3. Any pipeline stage requires Python or CUDA toolkit to complete
4. wgpu training fails to produce decreasing loss on Qwen2.5-Coder-1.5B

See §15 for complete success criteria and §18 for acceptance criteria.
