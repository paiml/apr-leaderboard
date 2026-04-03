# Contract Status

Audit trail for all provable contracts. Run `make check-contracts` to verify.

## Contract Summary

| Contract | Equations | Proofs | FTs | Makefile Tests | Status |
|----------|-----------|--------|-----|----------------|--------|
| pass-at-k.yaml | 1 | 3 | 3 | FT-001..005 (5) | Active |
| inference-throughput.yaml | 2 | 2 | 2 | FT-TPUT-001..002 (2) | Active |
| decontamination.yaml | 2 | 3 | 3 | FT-DECON-001 (1) | Active |
| lora-algebra.yaml | 3 | 3 | 3 | — | Pending |
| quantization.yaml | 3 | 3 | 3 | — | Pending |
| distillation.yaml | 2 | 3 | 3 | FT-DIST-001..002 (2) | Active |
| dpo-alignment.yaml | 2 | 3 | 2 | — | Active (impl) |
| forward-pass-perf.yaml | 2 | 2 | 1 | — | Active |
| fused-cross-entropy.yaml | 5 | 4 | 5 | — | Active |
| gpu-output-norm.yaml | 1 | 3 | 3 | — | Active |
| lora-finetune-eval.yaml | 2 | 3 | 3 | — | Active |
| nf4-dequantization.yaml | 4 | 4 | 6 | — | Active |
| qlora-training-loop.yaml | — | — | — | — | Active |
| wgsl-gemm-tiled.yaml | 4 | 4 | 5 | — | Active |
| wgsl-transpose.yaml | 2 | 3 | 1 | — | Active |
| merge-weight-norm.yaml | 3 | 4 | 4 | — | Active |
| leaderboard-gate.yaml | 3 | 3 | 3 | FT-GATE-001 (1) | Active |
| preference-pairs.yaml | 3 | 4 | 3 | — | Active |
| compile-binary.yaml | 3 | 3 | 3 | FT-COMPILE-001 (1) | Active |
| pipeline-validation.yaml | 3 | 3 | 3 | FT-PIPE-001..003 (3) | Active |
| perplexity-baseline.yaml | 2 | 3 | 3 | — | Active |

**Active:** Contract YAML valid with all required sections (metadata, equations, proof_obligations, falsification_tests).
**Pending:** Contract YAML exists but falsification tests require upstream `apr` features (LoRA merge, quantization round-trip).

## Makefile Test Breakdown

| Test ID | Contract | Rule | Status |
|---------|----------|------|--------|
| FT-001 | pass-at-k | Zero correct = 0 pass rate | PASS |
| FT-002 | pass-at-k | All correct = 1 pass rate | PASS |
| FT-003 | pass-at-k | pass@1 = c/n | PASS |
| FT-004 | pass-at-k | pass@10 boundary (n=10, c=5) | PASS |
| FT-005 | pass-at-k | pass@10 high-c (n=20, c=10) | PASS |
| FT-TPUT-001 | throughput | tok/s >= 1.0 | PASS |
| FT-TPUT-002 | throughput | TTFT < 500ms | PASS |
| FT-DATA-HE | data | HumanEval >= 100 problems | PASS |
| FT-DATA-MBPP | data | MBPP >= 100 problems | PASS |
| FT-DATA-BCB | data | BigCodeBench >= 100 problems | PASS |
| FT-DECON-001 | decontamination | No HE/MBPP prompt overlap | PASS |
| FT-EVAL-001 | eval | Best pass@1 >= 85% | PASS |
| FT-EVAL-002 | eval | >= 3 HumanEval runs | PASS |
| FT-EVAL-003 | eval | Latest run >= 80% | PASS |
| FT-DIST-001 | distillation | Teacher > student | PASS |
| FT-DIST-002 | distillation | >= 10 prompt categories | PASS |
| FT-MBPP-001 | MBPP eval | Best MBPP pass@1 >= 70% | PASS (76.2%) |
| FT-GATE-001 | leaderboard-gate | AC-022: HE>=85% AND MBPP>=80% | FAIL (HE=90.85%, MBPP=76.2%) |
| FT-QUANT-001 | quantization | Q4K < 50% of FP16 | PASS (35.0%) |
| FT-QUANT-002 | quantization | apr check passes on Q4K | PASS |
| FT-QUANT-003 | quantization | Golden ordering enforced | PASS |
| FT-DISTDATA-001 | distillation | >= 50 teacher completions | PASS (99) |
| FT-DISTDATA-002 | distillation | Valid JSONL format | PASS (99/99) |
| FT-DISTDATA-003 | distillation | >= 50 distill prompts | PASS (99) |
| FT-COMPILE-001 | compile-binary | apr compile available | PASS |
| FT-PIPE-001 | pipeline-validation | >= 15 scripts | PASS (22) |
| FT-PIPE-002 | pipeline-validation | >= 15 configs | PASS (22) |
| FT-PIPE-003 | pipeline-validation | >= 40 Make targets | PASS (56) |
| FT-ORACLE-001 | oracle | Oracle pass@1 >= 90% | PASS (96.34%) |
| FT-ORACLE-002 | oracle | <= 10 never-solved problems | PASS (6) |
| FT-CATALOG-001 | data catalog | >= 5 contract bindings | PASS (9) |
| FT-CATALOG-002 | data catalog | >= 8 datasets documented | PASS (13) |
| FT-LB-001 | leaderboard | >= 10 eval runs total | PASS (20) |
| FT-LB-002 | leaderboard | >= 2 benchmarks with results | PASS (2) |
| Structure | all | Valid YAML with required sections | PASS (×21) |

**Total: 54 passed, 1 failed** (updated 2026-04-03)

**pv proof-status:** 21/21 contracts parsed, 70 obligations, 70 tests, 10 Kani, 0/56 bindings.

**Note:** FT-GATE-001 is informational — correctly identifies MBPP 3.8pp gap from 80% threshold. Closing strategy: DPO training (PMAT-008) + text-based distillation (PMAT-007).

## Cross-Project Contracts

| Contract | Location | Scope |
|----------|----------|-------|
| gpu-multi-backend-parity-v1.yaml | ../provable-contracts/ | wgpu/CUDA/CPU parity (§25) |
| gpu-context-health-v1.yaml | ../provable-contracts/ | FP8 architecture guard (GH-542) |
| ptx-target-parity-v1.yaml | ../provable-contracts/ | PTX .target directive |
| gqa-kernel-v1.yaml | ../provable-contracts/ | GQA attention correctness |
