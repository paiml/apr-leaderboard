# Open Questions

Questions marked ✅ have been partially or fully answered by dogfooding.

1. **Calibration data quality:** How much does Wanda calibration data selection affect code model pruning? Need ablation study.
2. **Merge tournament depth:** Is 2-round merging sufficient or do 3+ rounds compound gains?
3. **Distillation data volume:** What's the minimum code corpus size for progressive (curriculum) distillation to outperform standard KL?
4. ✅ **HPO budget:** Is 20-trial TPE scout sufficient to find good LoRA hyperparameters for code? *Partial answer: Even 4 trials identify the correct LR regime (5e-5 beats 1e-3). The search space for LR is coarser than expected — budget 10-20 is likely sufficient for LR+rank. Interaction effects (LR × rank × epochs) may need more.*
5. **Quantization floor:** At what pass@1 threshold does INT4 quantization degrade code generation quality measurably? *Note: INT4 MSE on small tensors (256-dim) is 0.000033; production tensors (4096+) will differ.*
6. **Cross-architecture distillation:** Can we distill Qwen-32B into a different architecture (e.g., smaller custom model)?
7. ✅ **Inference parity gap:** What is the actual pass@1 gap between apr-native inference and PyTorch/HF for Qwen2.5-Coder models? *Partial answer: 7B Q4K achieves 68.90% pass@1 on HumanEval (pre-EOS-fix, 128-token cap). After GH-372 (128-token cap removal) and GH-373 (EOS termination fix), expected to improve. HF reference is 88.4%.*
8. ✅ **Code execution sandbox:** Should apr integrate a WASM-based sandbox for pass@k evaluation, or is external EvalPlus harness sufficient? *Answered: External sandbox implemented in eval script (python3 with 10s timeout or Docker with network=none + 512MB memory limit). WASM sandbox remains a stretch goal (§5.3 Option B). The external approach works for all three benchmarks.*
9. **CPU-only distillation feasibility:** Is progressive distillation from a 32B teacher on CPU practical within the 24h wall-clock budget, even with trueno SIMD? Likely needs GPU.
10. **Reasoning distillation transfer:** Does distilling from DeepSeek-R1 (or OCR-Nemotron) into Qwen2.5-Coder backbone require architecture adaptation, or does progressive distillation handle the mismatch?
11. **DPO data volume:** How many preference pairs are needed for measurable HumanEval+ improvement? Initial estimate: 5K-10K pairs. *Note: untrained DPO loss = 0.70 ≈ -ln(0.5), confirming the loss function works. The question is now purely about data volume.*
12. **Merge across training regimes:** Can we TIES-merge a code-instruct model with a reasoning-distilled model effectively, given they were trained with different objectives?
13. **LiveCodeBench contamination window:** LiveCodeBench refreshes continuously. What's the minimum lag between problem publication and safe inclusion in training data?
14. **WASM sandbox for Python:** Is CPython-in-WASM viable for pass@k evaluation at scale (164-974 problems × N=50 completions × timeout per completion)?

## New Questions from Dogfooding

15. ✅ **GGUF vs SafeTensors import path:** SafeTensors imports produce F16/BF16 tensors that realizar cannot run inference on (fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K). *Answered: Use GGUF import path (pre-quantized Q4_K_M). This is the only working path for end-to-end inference today.*
16. ✅ **GPU inference readiness:** Historical CUDA path replaced with wgpu. GPU compute now uses Vulkan/Metal/DX12 via wgpu for vendor-agnostic operation. *GPU QLoRA training works correctly (wgpu NF4 blocks, LoRA backward GEMM verified). `--no-gpu` flag available for CPU-only fallback.*
17. **`apr serve` for .apr files:** `apr serve` loads .apr models but HTTP server doesn't bind. Is this a missing feature or a configuration issue? Does it only work with raw GGUF?
18. **Import prerequisites:** `apr import` requires config.json and tokenizer.json in the HF cache. Should the import command auto-download these, or is manual download expected for non-standard model formats?
19. **Pruning precision at scale:** Wanda achieves 19.9% at 20% target on 256 params. Does floor rounding error vanish at 7B+ parameter counts, or do per-layer targets need adjustment?
20. **Tensor naming conventions:** The pipeline is sensitive to tensor name consistency across stages. What naming convention should import/distill/merge/finetune standardize on? *Partial answer: aprender 0.27.2 checkpoint spec (v1.4.0) uses HF convention (`model.layers.N.*`) for base model tensors.*

## Answered by GPU-SHARE Implementation (2026-03-04)

21. ✅ **Multi-GPU sharing:** Can multiple QLoRA jobs share a single GPU safely? *Answered: Yes, via GPU-SHARE multi-adapter pipeline. Single process loads base model once, trains N LoRA adapters concurrently. 3x VRAM savings for 3 adapters. 143 tests. VRAM ledger (flock + JSON) prevents OOM. MPS available as `--experimental-mps` opt-in but not recommended (fault propagation risk).*
22. ✅ **Heterogeneous cluster training:** Can we train across 4090 + Jetson + CPU-only nodes? *Answered: Yes, via GPU-SHARE Phase 3. YAML cluster config, VRAM-aware job placement (scoring: free_vram/budget x flops x 1/load), SSH transport (BatchMode, ConnectTimeout), checkpoint coordination with leaderboard ranking. CPU-only nodes limited to small models (≤350M).*
23. ✅ **GPU backward pass correctness (GH-378):** Are `gemm_backward_a` dimensions correct in LoRA backward? *Answered: Four calls had k/n swapped, causing 256x buffer overflow. Fixed: `(s,qd,r)→(s,r,qd)`, `(s,r,h)→(s,h,r)`, etc. 7B QLoRA training now completes without GPU errors. Compute now via wgpu.*
24. ✅ **Model perplexity sanity:** Does a Q4K GGUF-imported model produce non-degenerate perplexity? *Answered: Qwen2.5-Coder-1.5B-Instruct Q4K achieves perplexity 6.63 on WikiText-2 (cross-entropy 1.89). Non-zero, plausible for a code-tuned model on general text.*
25. **QA format parity (GH-13):** `apr qa` doesn't recognize .apr-wrapped GGUF for cross-format parity testing. Should `apr qa` introspect `original_format` metadata?
26. **CPU throughput floor:** 2.5 tok/s on CPU for 1.5B Q4K — is this acceptable for batch eval, or should eval always target GPU? Current HumanEval eval takes ~2.7h on CPU (164 problems x ~60s each).
