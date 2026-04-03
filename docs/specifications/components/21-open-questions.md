# Open Questions

Questions marked ✅ have been partially or fully answered by dogfooding.

1. **Calibration data quality:** How much does Wanda calibration data selection affect code model pruning? Need ablation study.
2. **Merge tournament depth:** Is 2-round merging sufficient or do 3+ rounds compound gains?
3. **Distillation data volume:** What's the minimum code corpus size for progressive (curriculum) distillation to outperform standard KL?
4. ✅ **HPO budget:** Is 20-trial TPE scout sufficient to find good LoRA hyperparameters for code? *Partial answer: Even 4 trials identify the correct LR regime (5e-5 beats 1e-3). The search space for LR is coarser than expected — budget 10-20 is likely sufficient for LR+rank. Interaction effects (LR × rank × epochs) may need more.*
5. **Quantization floor:** At what pass@1 threshold does INT4 quantization degrade code generation quality measurably? *Note: INT4 MSE on small tensors (256-dim) is 0.000033; production tensors (4096+) will differ.*
6. **Cross-architecture distillation:** Can we distill Qwen-32B into a different architecture (e.g., smaller custom model)?
7. ✅ **Inference parity gap:** What is the actual pass@1 gap between apr-native inference and PyTorch/HF for Qwen2.5-Coder models? *Answered: 7B Q4K achieves **87.20%** (few-shot). HF reference 87.8%, gap = **0.60pp**. 32B Q4K_M achieves **90.85%** vs HF 92.5%, gap = **1.65pp**. Gap attributable to Q4K quantization loss + greedy-only decoding. GPU/CPU parity confirmed.*
8. ✅ **Code execution sandbox:** Should apr integrate a WASM-based sandbox for pass@k evaluation, or is external EvalPlus harness sufficient? *Answered: External sandbox implemented in eval script (python3 with 10s timeout or Docker with network=none + 512MB memory limit). WASM sandbox remains a stretch goal (§5.3 Option B). The external approach works for all three benchmarks.*
9. **CPU-only distillation feasibility:** Is progressive distillation from a 32B teacher on CPU practical within the 24h wall-clock budget, even with trueno SIMD? Likely needs GPU.
10. **Reasoning distillation transfer:** Does distilling from DeepSeek-R1 (or OCR-Nemotron) into Qwen2.5-Coder backbone require architecture adaptation, or does progressive distillation handle the mismatch?
11. **DPO data volume:** How many preference pairs are needed for measurable HumanEval+ improvement? Initial estimate: 5K-10K pairs. *Note: untrained DPO loss = 0.70 ≈ -ln(0.5), confirming the loss function works. The question is now purely about data volume.*
12. **Merge across training regimes:** Can we TIES-merge a code-instruct model with a reasoning-distilled model effectively, given they were trained with different objectives?
13. **LiveCodeBench contamination window:** LiveCodeBench refreshes continuously. What's the minimum lag between problem publication and safe inclusion in training data?
14. **WASM sandbox for Python:** Is CPython-in-WASM viable for pass@k evaluation at scale (164-974 problems × N=50 completions × timeout per completion)?

## New Questions from Dogfooding

15. ✅ **GGUF vs SafeTensors import path:** SafeTensors imports produce F16/BF16 tensors that realizar cannot run inference on (fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K). *Answered: Use GGUF import path (pre-quantized Q4_K_M). This is the only working path for end-to-end inference today.*
16. ✅ **GPU inference readiness:** *Answered (2026-03-27): FIXED via wgpu.* `apr run --gpu` auto-dispatches CUDA → wgpu → CPU. wgpu cosine=0.999863 on Blackwell sm_121. Root cause: FP32 non-associativity in parallel accumulation (NOT a JIT bug — falsified). PyTorch canary proves hardware correct. wgpu uses Vulkan compute shaders with sequential accumulation matching CPU. See §25.
17. **`apr serve` for .apr files:** `apr serve` loads .apr models but HTTP server doesn't bind. Is this a missing feature or a configuration issue? Does it only work with raw GGUF?
18. **Import prerequisites:** `apr import` requires config.json and tokenizer.json in the HF cache. Should the import command auto-download these, or is manual download expected for non-standard model formats?
19. **Pruning precision at scale:** Wanda achieves 19.9% at 20% target on 256 params. Does floor rounding error vanish at 7B+ parameter counts, or do per-layer targets need adjustment?
20. ✅ **Tensor naming conventions:** *Answered (2026-04-03): CONFIRMED as a real issue.* wgpu training saves adapters as `layer.N.proj.lora_a` while GGUF base uses `model.layers.N.self_attn.proj.weight`. Merge matched 0/339 layers until tensors were remapped. Fix: `scripts/remap-adapter-tensors.py` normalizes names. Upstream fix needed in `entrenar::merge` for automatic remapping. See §24.21.

## Answered by GPU-SHARE Implementation (2026-03-04)

21. ✅ **Multi-GPU sharing:** Can multiple QLoRA jobs share a single GPU safely? *Answered: Yes, via GPU-SHARE multi-adapter pipeline. Single process loads base model once, trains N LoRA adapters concurrently. 3x VRAM savings for 3 adapters. 143 tests. VRAM ledger (flock + JSON) prevents OOM. MPS available as `--experimental-mps` opt-in but not recommended (fault propagation risk).*
22. ✅ **Heterogeneous cluster training:** Can we train across 4090 + Jetson + CPU-only nodes? *Answered: Yes, via GPU-SHARE Phase 3. YAML cluster config, VRAM-aware job placement (scoring: free_vram/budget x flops x 1/load), SSH transport (BatchMode, ConnectTimeout), checkpoint coordination with leaderboard ranking. CPU-only nodes limited to small models (≤350M).*
23. ✅ **GPU backward pass correctness (GH-378):** Are `gemm_backward_a` dimensions correct in LoRA backward? *Answered: Four calls had k/n swapped, causing 256x buffer overflow. Fixed: `(s,qd,r)→(s,r,qd)`, `(s,r,h)→(s,h,r)`, etc. 7B QLoRA training now completes without GPU errors. Compute now via wgpu.*
24. ✅ **Model perplexity sanity:** Does a Q4K GGUF-imported model produce non-degenerate perplexity? *Answered: Qwen2.5-Coder-1.5B-Instruct Q4K achieves perplexity 6.63 on WikiText-2 (cross-entropy 1.89). Non-zero, plausible for a code-tuned model on general text.*
25. **QA format parity (GH-13):** `apr qa` doesn't recognize .apr-wrapped GGUF for cross-format parity testing. Should `apr qa` introspect `original_format` metadata?
26. ✅ **CPU throughput floor:** 2.5 tok/s on CPU for 1.5B Q4K — is this acceptable for batch eval, or should eval always target GPU? *Answered: CPU eval works. 7B batch mode: model loads once (5.2s), inference ~45-60s/prompt on gx10 aarch64 (competing with concurrent eval). HumanEval 7B batch: ~3h CPU. MBPP 7B batch (500 problems): ~8h CPU. GPU required for production eval at scale. Batch mode eliminates ~80s/problem JIT overhead on GPU.*
27. ✅ **SCoT on small models:** Does structured chain-of-thought prompting improve code quality on ≤7B models? *Answered: No. SCoT **hurts** 7B: 82.32% vs 85.37% standard (-3.05pp). On 1.5B, reasoning consumes all tokens. Few-shot is the best ≤7B strategy: 87.20% (+1.83pp). SCoT may help ≥32B where reasoning is more concise.*
28. ✅ **HF parity via compare-hf:** `apr compare-hf` returns 0 comparisons on GGUF Q4K imports (dtype mismatch with HF FP16). *Answered: Expected behavior — Q4K uses different dtypes than HF FP16/BF16. Parity verified via benchmark scores: 7B HumanEval 87.20% vs 87.8% HF (0.60pp gap), MBPP 76.20% vs 83.5% HF (7.3pp gap).*

## New Questions from Distillation Pipeline (2026-03-28)

29. **Text-based distillation effectiveness on Q4K:** Does the 32B teacher (90.85%) generate sufficiently diverse completions at temperature=0.8 to improve the 7B student beyond its 85.37% baseline? The 99 targeted prompts cover 11 categories derived from HumanEval failure analysis. Falsifiable: if HumanEval stays below 86% after QLoRA training, text-based distillation is insufficient.
30. ✅ **Combined data optimality:** *Answered (2026-04-03): 15K combined training is impractical (~153h ETA).* Targeted 99 teacher completions alone take 66.5 min. The 15K combined corpus would require batching or multi-epoch scheduling. Recommendation: train on 99 targeted samples first (PMAT-007), then optionally fine-tune further on a small instruct subset (1K-2K samples).
31. **QLoRA rank selection for distillation:** Recipe I uses rank 32 (same as Recipe E). Should distillation QLoRA use higher rank (64+) to capture more of the teacher's reasoning patterns, or does the Q4K quantization bottleneck make higher rank wasteful?
