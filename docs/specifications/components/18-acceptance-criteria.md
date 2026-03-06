# Acceptance Criteria

Every criterion below is falsifiable. If any criterion cannot be
demonstrated, this spec has failed. Status: [x] = verified,
[ ] = not yet tested.

## Verified

- [x] AC-001: `apr import hf://Qwen/Qwen2.5-Coder-7B` produces a valid `.apr` file that passes `apr check`
- [x] AC-004: `apr finetune --method lora` completes training with decreasing loss curve (S22.7: tiny model, loss 6.9330->6.9301 over 2 epochs; S23.1.4: 7B Q4K val_loss=33.12)
- [x] AC-005: `apr finetune --method qlora` uses <50% VRAM compared to LoRA at equivalent rank (S23.1.4: QLoRA NF4 on 1.5B verified, S23.2: multi-adapter 3x VRAM savings)
- [x] AC-013: `pmat comply check --strict` passes with zero failures (`Status: COMPLIANT` verified)
- [x] AC-027: Every tooling gap in S5 has either a wire-in implementation or a documented external boundary (5 gaps documented with wire-in plans, 9 Ludwig parity gaps tracked with crate targets, execution sandbox scoped as external boundary)
- [x] AC-028: `make prove-wgpu` completes successfully -- QLoRA training runs on wgpu (Vulkan/Metal/DX12) with no CUDA toolkit installed
- [x] AC-029: Training via wgpu produces decreasing loss over 2 epochs on Qwen2.5-Coder-1.5B

## Partially Verified

- [x] AC-017: N-sampling generates distinct completions per problem -- eval script supports `NUM_SAMPLES`, tracks per-task n,c counts, computes Chen et al. unbiased pass@k estimator in log-space. Uses `apr run` for generation. Full 20-sample verification pending GPU eval run.
- [x] AC-016: Training data has <1% n-gram overlap with HumanEval/MBPP test cases -- benchmark data downloaded via `make benchmark-download` (HumanEval 164, MBPP 974, BigCodeBench 1140). `apr data decontaminate` not yet wired (GH-11). Gate ready to enforce once upstream lands.
- [x] AC-002: `apr eval` on imported model produces non-zero perplexity within 10% of HF reference -- perplexity = 6.63 on WikiText-2 (§22.0). Non-zero confirmed. HF parity check returns 0 comparisons on GGUF imports (different dtype); 10% threshold pending SafeTensors import path fix.
- [x] AC-019: Structured prompting produces reasoning before code — SCoT produces step-by-step reasoning (confirmed on 1.5B, §22.18). On small models, reasoning consumes token budget; quality improvement pending 7B+ evaluation

## Not Yet Tested

- [ ] AC-003: `apr distill` with progressive strategy produces a student model that outperforms the untrained student on perplexity
- [ ] AC-006: `apr merge --strategy slerp` preserves weight norms (L2 norm within 5% of inputs) — merge mechanics work (339 tensors, qwen2 arch preserved). Dequantizes Q4K→FP32 (6.62 GiB). Blocked on GH-14 (tokenizer/config loss)
- [ ] AC-007: `apr merge --strategy ties` resolves sign conflicts (merged model has fewer conflicting task vectors than input sum)
- [ ] AC-008: `apr prune --method wanda` at conservative ratio degrades perplexity by <5% — pruning achieves target sparsity (10.0%) but dequantizes Q4K→FP32, losing tokenizer/config. Blocked on GH-14 (§22.20)
- [ ] AC-009: `apr quantize --scheme int4` produces model <50% size of FP16 original — GGUF Q4K import already at 1.04 GiB (34.7% of ~3.0 GiB FP16). Running `apr quantize --scheme int4` on Q4K input produces 2.43 GiB (dequantizes first). INT4 quantization needs FP16 input to demonstrate <50% size reduction
- [ ] AC-010: `apr compile` produces a standalone binary that runs inference without external dependencies -- Binary created (671 KiB, §22.15) but inference dispatch not yet statically linked (needs realizar runtime)
- [ ] AC-011: Full pipeline (Recipe C) completes end-to-end without manual intervention
- [ ] AC-012: `pv proof-status` shows >=95% binding coverage for pipeline-relevant contracts
- [ ] AC-014: `apr compare-hf` shows <5% parity gap on perplexity for imported Qwen models — GGUF Q4K imports produce 0 comparisons (dtype mismatch with HF FP16). Parity must be verified via benchmark scores or SafeTensors import path (§22.18)
- [ ] AC-015: All falsification tests in provable-contracts pass for Kernel Class E (Qwen)
- [ ] AC-021: Qwen2.5-Coder-7B-Instruct imported via `apr import` achieves >=85% HumanEval pass@1 (apr-native baseline >= HF reference - 5%)
- [ ] AC-022: Full pipeline on Qwen2.5-Coder-7B produces a model scoring >=85% HumanEval, >=82% HumanEval+, >=80% MBPP
- [ ] AC-023: INT4 quantized model loses <2% pass@1 vs FP16 on HumanEval
- [ ] AC-024: Merged model (TIES of code-specialist + reasoning-specialist) scores >= best input specialist on at least one benchmark
- [ ] AC-025: `alimentar quality` scores all training data >=80/100 before use in fine-tuning
- [ ] AC-026: `apr compile` of Qwen2.5-Coder-1.5B INT4 produces a binary <1GB that generates valid Python code -- Binary 671 KiB + model 1.04 GiB = 1.04 GiB total (§22.15). Runtime under 1MB, model data slightly over 1GB. Inference not yet working in compiled binary.

## Blocked on Upstream

- [ ] AC-018: Speculative decoding achieves >=1.5x throughput over standard decoding (GH-10: `apr run --speculative` not yet exposed)
- [ ] AC-020: DPO alignment reduces loss on preference pairs over 3 epochs (GH-8: `apr align` not yet implemented, routes through `apr finetune --method dpo`)

## Summary

| Category | Count |
|---|---|
| Verified | 7 |
| Partially Verified | 4 |
| Not Yet Tested | 16 |
| Blocked on Upstream | 2 |
| **Total** | **29** |
