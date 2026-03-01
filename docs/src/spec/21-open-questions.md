# Open Questions

Questions marked ✅ have been partially or fully answered by dogfooding.

1. **Calibration data quality:** How much does Wanda calibration data selection affect code model pruning? Need ablation study.
2. **Merge tournament depth:** Is 2-round merging sufficient or do 3+ rounds compound gains?
3. **Distillation data volume:** What's the minimum code corpus size for progressive (curriculum) distillation to outperform standard KL?
4. ✅ **HPO budget:** Is 20-trial TPE scout sufficient to find good LoRA hyperparameters for code? *Partial answer: Even 4 trials identify the correct LR regime (5e-5 beats 1e-3). The search space for LR is coarser than expected — budget 10-20 is likely sufficient for LR+rank. Interaction effects (LR × rank × epochs) may need more.*
5. **Quantization floor:** At what pass@1 threshold does INT4 quantization degrade code generation quality measurably? *Note: INT4 MSE on small tensors (256-dim) is 0.000033; production tensors (4096+) will differ.*
6. **Cross-architecture distillation:** Can we distill Qwen-32B into a different architecture (e.g., smaller custom model)?
7. **Inference parity gap:** What is the actual pass@1 gap between apr-native inference and PyTorch/HF for Qwen2.5-Coder models? This gates all absolute target setting.
8. **Code execution sandbox:** Should apr integrate a WASM-based sandbox for pass@k evaluation, or is external EvalPlus harness sufficient?
9. **CPU-only distillation feasibility:** Is progressive distillation from a 32B teacher on CPU practical within the 24h wall-clock budget, even with trueno SIMD? Likely needs GPU.
10. **Reasoning distillation transfer:** Does distilling from DeepSeek-R1 (or OCR-Nemotron) into Qwen2.5-Coder backbone require architecture adaptation, or does progressive distillation handle the mismatch?
11. **DPO data volume:** How many preference pairs are needed for measurable HumanEval+ improvement? Initial estimate: 5K-10K pairs. *Note: untrained DPO loss = 0.70 ≈ -ln(0.5), confirming the loss function works. The question is now purely about data volume.*
12. **Merge across training regimes:** Can we TIES-merge a code-instruct model with a reasoning-distilled model effectively, given they were trained with different objectives?
13. **LiveCodeBench contamination window:** LiveCodeBench refreshes continuously. What's the minimum lag between problem publication and safe inclusion in training data?
14. **WASM sandbox for Python:** Is CPython-in-WASM viable for pass@k evaluation at scale (164-974 problems × N=50 completions × timeout per completion)?

## New Questions from Dogfooding

15. ✅ **Coverage measurement methodology:** `cargo llvm-cov` includes path dependencies, inflating missed lines. Must filter to project source. *Answered: filter to `apr-leaderboard/src/` for accurate project coverage (96.1% vs 71.7% total).*
16. ✅ **Tensor naming conventions:** The pipeline is sensitive to tensor name consistency across stages. What naming convention should convert/distill/merge/finetune standardize on? *Answered by aprender 0.27.2 checkpoint spec (v1.4.0): use HF convention (`model.layers.N.*`) for base model tensors, `lora.*` for adapters, `__training__.*` prefix for optimizer state. The checkpoint taxonomy (`.apr` / `.adapter.apr` / `.ckpt.apr`) with namespace filtering via `AprReader::open_filtered()` resolves the naming ambiguity.*
17. **ndarray vs Tensor type unification:** `entrenar::distill` uses `ndarray::Array2<f32>` while `entrenar::train` uses `entrenar::Tensor`. Can these be unified, or does the bridge reshape remain necessary?
18. **Pruning precision at scale:** Wanda achieves 19.9% at 20% target on 256 params. Does floor rounding error vanish at 7B+ parameter counts, or do per-layer targets need adjustment?
