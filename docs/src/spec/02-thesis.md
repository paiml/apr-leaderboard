# Thesis

The Python ML ecosystem requires 200+ dependencies, GPU-locked CUDA toolchains, and multi-GB Docker images to compete on HuggingFace leaderboards. We will demonstrate that a single-binary Rust pipeline — using only `apr` CLI commands — can match or exceed these results with zero Python, zero external CUDA toolkit (trueno generates PTX natively), and 10x smaller deployment artifacts.

**Constraint:** Every optimization step must be expressible as an `apr` subcommand. No Python. No notebooks. No HuggingFace Transformers library. Pure sovereign stack.

**Compute reality:** GPU hardware is recommended for training-phase techniques (distillation, fine-tuning). Inference-only techniques (merging, quantization) and small-model inference (≤7B quantized) run on CPU via trueno SIMD (AVX2/NEON). The "zero CUDA" claim means no `nvcc`, no `libcudart`, no CUDA toolkit install — trueno's PTX backend generates GPU kernels in pure Rust.
