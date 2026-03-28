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

**Active:** Falsification tests wired in `make check-contracts` and passing.
**Pending:** Contract YAML exists with equations/proofs/FTs, but tests require upstream `apr` features (LoRA merge, quantization round-trip).

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
| Structure | all | Valid YAML with required sections | PASS (×6) |

**Total: 22 passed, 0 failed**

## Cross-Project Contracts

| Contract | Location | Scope |
|----------|----------|-------|
| gpu-multi-backend-parity-v1.yaml | ../provable-contracts/ | wgpu/CUDA/CPU parity (§25) |
| gpu-context-health-v1.yaml | ../provable-contracts/ | FP8 architecture guard (GH-542) |
| ptx-target-parity-v1.yaml | ../provable-contracts/ | PTX .target directive |
| gqa-kernel-v1.yaml | ../provable-contracts/ | GQA attention correctness |
