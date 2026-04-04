# Contract Binding Registry

Maps proof obligations in contract YAMLs to their implementing functions in the
aprender/entrenar/trueno codebase. Required for AC-012 (≥95% binding coverage).

## Binding Status

| Contract | Obligations | Bound | Coverage | Notes |
|----------|------------|-------|----------|-------|
| pass-at-k.yaml | 3 | 3 | 100% | `scripts/eval-pass-at-k.sh` (awk pass@k estimator) |
| inference-throughput.yaml | 2 | 2 | 100% | `realizar::inference` (tok/s, TTFT) |
| decontamination.yaml | 3 | 3 | 100% | `alimentar::quality::decontaminate` |
| distillation.yaml | 3 | 2 | 67% | `entrenar::distill` (teacher>student, progressive) |
| lora-algebra.yaml | 3 | 3 | 100% | `aprender::transfer::lora` (LoRA forward, merge, rank) |
| quantization.yaml | 3 | 3 | 100% | `aprender::format::converter` (Q4K, ordering, size) |
| dpo-alignment.yaml | 6 | 4 | 67% | `entrenar::dpo_step` (loss, gradient). Pipeline FTs pending. |
| qlora-training-loop.yaml | 7 | 7 | 100% | `entrenar::qlora` + `trueno::wgpu` (NF4, AdamW, CE, GEMM) |
| fused-cross-entropy.yaml | 4 | 4 | 100% | `trueno::loss::cross_entropy_chunked` |
| nf4-dequantization.yaml | 4 | 4 | 100% | `trueno::quantize::nf4_dequant` |
| wgsl-gemm-tiled.yaml | 4 | 4 | 100% | `trueno::wgpu::gemm_tiled` (CUTLASS-derived) |
| wgsl-transpose.yaml | 3 | 3 | 100% | `trueno::wgpu::transpose` |
| gpu-output-norm.yaml | 3 | 3 | 100% | `trueno::wgpu::rmsnorm` |
| forward-pass-perf.yaml | 2 | 2 | 100% | `realizar::forward_pass` (per-op timing) |
| lora-finetune-eval.yaml | 3 | 3 | 100% | `entrenar::finetune` → `realizar::eval` |
| merge-weight-norm.yaml | 4 | 3 | 75% | `aprender::format::converter::merge` (SLERP, norm). TIES sign election pending. |
| leaderboard-gate.yaml | 3 | 3 | 100% | `scripts/eval-pass-at-k.sh` (HE+MBPP compound) |
| preference-pairs.yaml | 4 | 3 | 75% | `scripts/generate-preference-pairs.sh`. N-sampling integration pending. |
| compile-binary.yaml | 3 | 2 | 67% | `aprender::compile` (binary, size). Inference dispatch pending. |
| pipeline-validation.yaml | 3 | 3 | 100% | `Makefile` + `scripts/*.sh` (count gates) |
| perplexity-baseline.yaml | 3 | 3 | 100% | `realizar::eval::perplexity` |
| tokenizer-preservation.yaml | 3 | 3 | 100% | `aprender::format::apr_v2` (AprV2Reader/Writer) |
| data-governance.yaml | 3 | 3 | 100% | `data_catalog.yaml` (lineage, classification) |
| quantization-quality.yaml | 3 | 3 | 100% | `aprender::format::converter` (retention, size, check) |
| data-quality.yaml | 4 | 3 | 75% | `alimentar::quality`. Score metric pending. |
| pruning-quality.yaml | 4 | 2 | 50% | `aprender::prune::wanda`. Eval pipeline pending. |
| binding-coverage.yaml | 3 | 1 | 33% | Meta: `pv proof-status`. Full binding check pending. |
| hf-parity.yaml | 4 | 3 | 75% | `scripts/eval-pass-at-k.sh` + `apr compare-hf`. Tokenizer equiv pending. |
| ties-sign-resolution.yaml | 4 | 1 | 25% | `aprender::format::converter::merge::ties`. Awaiting PMAT-010. |

## Summary

| Metric | Count |
|--------|-------|
| Total obligations | 98 |
| Bound | 80 |
| Coverage | **81.6%** |
| Target (AC-012) | ≥95% |
| Gap | 18 obligations need binding |

## Unbound Obligations

Priority list for reaching 95% (need 13 more bindings):

1. **ties-sign-resolution** (3 unbound): Implement TIES merge, add sign election tests
2. **pruning-quality** (2 unbound): Wire `apr prune` eval pipeline
3. **binding-coverage** (2 unbound): Meta — needs pv proof-status integration
4. **dpo-alignment** (2 unbound): Wire DPO pipeline (PMAT-008)
5. **hf-parity** (1 unbound): Tokenizer equivalence test
6. **data-quality** (1 unbound): alimentar quality score metric
7. **merge-weight-norm** (1 unbound): TIES sign election audit
8. **compile-binary** (1 unbound): Inference dispatch in compiled binary
9. **preference-pairs** (1 unbound): N-sampling batch integration
10. **distillation** (1 unbound): Progressive distillation temperature annealing

## How to Add Bindings

Each contract YAML can include a `bindings` section mapping obligations to code:

```yaml
bindings:
  - obligation: "PO-001"
    crate: "aprender"
    module: "format::converter::merge"
    function: "merge_tensors"
    annotation: "#[contract(merge-weight-norm, PO-001)]"
```

Run `pv proof-status --check-bindings` to validate all bindings resolve to existing functions.
