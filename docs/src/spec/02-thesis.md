# Thesis

The Python ML ecosystem requires 200+ dependencies, vendor-locked CUDA toolchains, and multi-GB Docker images to compete on HuggingFace leaderboards. We will demonstrate that a single-binary Rust pipeline — using only `apr` CLI commands — can match or exceed these results with zero Python, zero CUDA toolkit, zero vendor lock-in, and 10x smaller deployment artifacts.

**Constraint:** Every optimization step must be expressible as an `apr` subcommand. No Python. No notebooks. No HuggingFace Transformers library. Pure sovereign stack.

**Compute reality:** GPU hardware is recommended for training-phase techniques (distillation, fine-tuning). Inference-only techniques (merging, quantization) and small-model inference (≤7B quantized) run on CPU via trueno SIMD (AVX2/NEON). GPU compute uses **wgpu** — the cross-platform GPU abstraction over Vulkan, Metal, and DX12. No `nvcc`, no `libcudart`, no CUDA toolkit, no vendor lock-in. Training and inference run on any GPU: NVIDIA, AMD, Intel Arc, or Apple Silicon (Metal).
