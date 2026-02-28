#  Open Questions

1. **Calibration data quality:** How much does Wanda calibration data selection affect code model pruning? Need ablation study.
2. **Merge tournament depth:** Is 2-round merging sufficient or do 3+ rounds compound gains?
3. **Distillation data volume:** What's the minimum code corpus size for progressive (curriculum) distillation to outperform standard KL?
4. **HPO budget:** Is 20-trial TPE scout sufficient to find good LoRA hyperparameters for code?
5. **Quantization floor:** At what pass@1 threshold does INT4 quantization degrade code generation quality measurably?
6. **Cross-architecture distillation:** Can we distill Qwen-32B into a different architecture (e.g., smaller custom model)?
7. **Inference parity gap:** What is the actual pass@1 gap between apr-native inference and PyTorch/HF for Qwen2.5-Coder models? This gates all absolute target setting.
8. **Code execution sandbox:** Should apr integrate a WASM-based sandbox for pass@k evaluation, or is external EvalPlus harness sufficient?
9. **CPU-only distillation feasibility:** Is progressive distillation from a 32B teacher on CPU practical within the 24h wall-clock budget, even with trueno SIMD? Likely needs GPU.
10. **Reasoning distillation transfer:** Does distilling from DeepSeek-R1 (or OCR-Nemotron) into Qwen2.5-Coder backbone require architecture adaptation, or does progressive distillation handle the mismatch?
11. **DPO data volume:** How many preference pairs are needed for measurable HumanEval+ improvement? Initial estimate: 5K-10K pairs.
12. **Merge across training regimes:** Can we TIES-merge a code-instruct model with a reasoning-distilled model effectively, given they were trained with different objectives?
13. **LiveCodeBench contamination window:** LiveCodeBench refreshes continuously. What's the minimum lag between problem publication and safe inclusion in training data?
14. **WASM sandbox for Python:** Is CPython-in-WASM viable for pass@k evaluation at scale (164-974 problems × N=50 completions × timeout per completion)?
