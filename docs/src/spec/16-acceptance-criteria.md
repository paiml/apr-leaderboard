# Acceptance Criteria

Every criterion below is falsifiable. If any criterion cannot be demonstrated, this spec has failed.

- [ ] AC-001: `apr import hf://Qwen/Qwen2.5-Coder-7B` produces a valid `.apr` file that passes `apr check`
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
- [ ] AC-016: Training data has <1% n-gram overlap with HumanEval/MBPP test cases (`apr validate --decontaminate`)
- [ ] AC-017: `apr eval --n-samples 20` generates 20 distinct completions per problem (not duplicates)
- [ ] AC-018: Speculative decoding (`apr run --speculative`) achieves ≥1.5x throughput over standard decoding
- [ ] AC-019: `apr eval --prompt-strategy scot` produces structured reasoning before code output
- [ ] AC-020: `apr align --method dpo` reduces loss on preference pairs over 3 epochs
