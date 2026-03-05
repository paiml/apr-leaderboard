# Acceptance Criteria

Every criterion below is falsifiable. If any criterion cannot be
demonstrated, this spec has failed. Status: ✅ = verified,
⬚ = not yet tested, 🔧 = blocked on upstream.

## Verified

- [x] AC-001: `apr import hf://Qwen/Qwen2.5-Coder-7B` produces a valid `.apr` file that passes `apr check`
- [x] AC-028: `make prove-wgpu` completes successfully — QLoRA training runs on wgpu (Vulkan/Metal/DX12) with no CUDA toolkit installed
- [x] AC-029: Training via wgpu produces decreasing loss over 2 epochs on Qwen2.5-Coder-1.5B

## Not Yet Tested

- [ ] AC-002: `apr eval` on imported model produces non-zero perplexity within 10% of HF reference
- [ ] AC-003: `apr distill` with progressive strategy produces a student model that outperforms the untrained student on perplexity
- [ ] AC-004: `apr finetune --method lora` completes training with decreasing loss curve
- [ ] AC-005: `apr finetune --method qlora` uses <50% VRAM compared to LoRA at equivalent rank
- [ ] AC-006: `apr merge --strategy slerp` preserves weight norms (L2 norm within 5% of inputs)
- [ ] AC-007: `apr merge --strategy ties` resolves sign conflicts (merged model has fewer conflicting task vectors than input sum)
- [ ] AC-008: `apr prune --method wanda` at conservative ratio degrades perplexity by <5%
- [ ] AC-009: `apr quantize --scheme int4` produces model <50% size of FP16 original
- [ ] AC-010: `apr compile` produces a standalone binary that runs inference without external dependencies
- [ ] AC-011: Full pipeline (Recipe C) completes end-to-end without manual intervention
- [ ] AC-012: `pv proof-status` shows ≥95% binding coverage for pipeline-relevant contracts
- [ ] AC-013: `pmat comply check --strict` passes with zero failures on the final submission
- [ ] AC-014: `apr compare-hf` shows <5% parity gap on perplexity for imported Qwen models
- [ ] AC-015: All falsification tests in provable-contracts pass for Kernel Class E (Qwen)
- [ ] AC-021: Qwen2.5-Coder-7B-Instruct imported via `apr import` achieves ≥85% HumanEval pass@1 (apr-native baseline ≥ HF reference - 5%)
- [ ] AC-022: Full pipeline on Qwen2.5-Coder-7B produces a model scoring ≥85% HumanEval, ≥82% HumanEval+, ≥80% MBPP
- [ ] AC-023: INT4 quantized model loses <2% pass@1 vs FP16 on HumanEval
- [ ] AC-024: Merged model (TIES of code-specialist + reasoning-specialist) scores ≥ best input specialist on at least one benchmark
- [ ] AC-025: `alimentar quality` scores all training data ≥80/100 before use in fine-tuning
- [ ] AC-026: `apr compile` of Qwen2.5-Coder-1.5B INT4 produces a binary <1GB that generates valid Python code
- [ ] AC-027: Every tooling gap in §5 has either a wire-in implementation or a documented external boundary

## Blocked on Upstream

- [ ] AC-016: Training data has <1% n-gram overlap with HumanEval/MBPP test cases (GH-9: `apr validate --decontaminate` not yet implemented)
- [ ] AC-017: N-sampling generates 20 distinct completions per problem — sampling works via eval script, but `apr eval --n-samples` flag not yet implemented
- [ ] AC-018: Speculative decoding achieves ≥1.5x throughput over standard decoding (GH-10: `apr run --speculative` not yet exposed)
- [ ] AC-019: Structured prompting produces reasoning before code (PMAT-005: `--prompt-strategy` not yet implemented)
- [ ] AC-020: DPO alignment reduces loss on preference pairs over 3 epochs (GH-8: `apr align` not yet implemented, routes through `apr finetune --method dpo`)
