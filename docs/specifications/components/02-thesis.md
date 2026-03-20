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
- No GPU vendor lock-in. Primary backend: wgpu (Vulkan/Metal/DX12). Optional: CUDA for hardware that lacks wgpu support (e.g., Blackwell sm_121).
- Pure sovereign stack: aprender, entrenar, trueno.

## 2.4 Compute Reality

| Resource | Dev Workstation | gx10 (Eval Server) |
|----------|----------------|-------------------|
| GPUs | 2x AMD Radeon Pro W5700X (Navi10) | NVIDIA Blackwell GB10 (sm_121) |
| VRAM/Memory | 16 GB per GPU, 32 GB total | 119 GB unified |
| GPU backend | wgpu / Vulkan 1.3.255 (RADV) | CUDA 13.0 |
| CPU | 16 cores, 64 GB RAM | aarch64, 10 cores |
| Best HumanEval | — | **87.20%** (7B few-shot) |

No GPU vendor lock-in. wgpu is the primary backend (any vendor); CUDA is optional for hardware where wgpu support lags. CPU/GPU parity verified: 7B produces identical 85.37% on both backends.

## 2.5 Inference Without GPU

Inference-only techniques (merging, quantization) and small-model inference
(≤7B quantized) run on CPU via trueno SIMD (AVX2/NEON). GPU is recommended
for training-phase techniques (distillation, fine-tuning) but not required
for evaluation.

## 2.6 Falsification Criteria

The thesis is falsified if any of these hold after applying the full pipeline:
1. HumanEval pass@1 < 80% for Qwen2.5-Coder-7B (below "Strong" tier) — **NOT FALSIFIED: 87.20%** ✅
2. Inference parity gap > 5% vs HuggingFace reference implementation — **NOT FALSIFIED: 0.60pp gap** ✅
3. Any pipeline stage requires Python to complete — **NOT FALSIFIED: zero Python** ✅
4. wgpu training fails to produce decreasing loss on Qwen2.5-Coder-1.5B — **NOT FALSIFIED: loss decreases** ✅

See §15 for complete success criteria and §18 for acceptance criteria.
