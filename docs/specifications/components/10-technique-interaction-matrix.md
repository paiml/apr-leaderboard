# Technique Interaction Matrix

Techniques are not independent. Order matters.

```
                      ┌──────────────────────────────────────────────┐
                      │          TECHNIQUE INTERACTION MATRIX        │
                      │                                              │
                      │  Column  │ distill  merge  prune  finetune  │
                      │  THEN    │                                   │
                      │  Row ↓   │                                   │
                      │──────────┼─────────────────────────────────  │
                      │ distill  │   —      ✗bad   ✓ok    ✗bad     │
                      │ merge    │  ✓ok      —     ✓ok    ✓✓best   │
                      │ prune    │  ✓ok     ✓ok     —     ✗bad     │
                      │ finetune │ ✓✓best  ✓ok    ✗bad    —        │
                      │ quantize │  ✓ok     ✓ok    ✓ok    ✓ok      │
                      └──────────────────────────────────────────────┘

  Legend: Read as "column THEN row" (column happens first)
    ✓✓best  = Optimal ordering
    ✓ok     = Works but not optimal
    ✗bad    = Harmful (degrades quality or wastes compute)

  Key asymmetries:
    distill→finetune = ✓✓best  (adapt distilled knowledge to task)
    finetune→distill = ✗bad    (distillation overwrites fine-tuned specialization)
    finetune→merge   = ✓✓best  (merge specialized variants)
    merge→finetune   = ✓ok     (works but loses merge diversity)
```

**Golden ordering:** distill → finetune → merge → prune → quantize

Rationale:
1. **Distill first** — Knowledge transfer works best on an unmodified student architecture
2. **Finetune second** — LoRA adapts the distilled weights to target benchmarks
3. **Merge third** — Combine fine-tuned variants while representations are still rich
4. **Prune fourth** — Remove redundancy AFTER merging (merged models have more redundancy)
5. **Quantize last** — Always final step; quantization is lossy and non-reversible

**Note on QLoRA as implicit QAT:** When the final deployment target is INT4,
using QLoRA (§7.5) during the finetune step provides quantization-aware
adaptation. The adapter trains against quantized base weights, making the
final INT4 quantization less lossy than post-training quantization after
full-precision LoRA.

**Anti-patterns:**
- Prune → Finetune: LoRA can't recover pruned knowledge effectively
- Finetune → Distill: Overwrites the fine-tuned specialization
- Quantize → anything: Quality loss compounds with every subsequent operation

**Prompt strategy (§7.6) is orthogonal** — it applies at eval time after all model modifications. No interaction with the training pipeline. Dogfooding shows prompt strategy yields +1.83pp (HumanEval) and +25.4pp (MBPP) at zero compute cost. Always optimize prompts before starting the training pipeline.
