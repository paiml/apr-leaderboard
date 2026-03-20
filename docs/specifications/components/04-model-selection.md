# Model Selection & Improvement Strategy

## 4.1 WHAT Models We Will Improve

We select models based on three criteria: (1) competitive baseline scores, (2) permissive licensing (Apache-2.0 or MIT), (3) architecture support in aprender.

**Primary targets (Tier 1 — submit to leaderboards):**

| Model | Size | Why This Model | Baseline HE | Target HE | Strategy |
|-------|------|----------------|-------------|-----------|----------|
| Qwen2.5-Coder-7B-Instruct | 7B | Best 7B code model. Apache-2.0. Beats CodeLlama-70B. | 87.8% | **90%+** | Distill + LoRA + DPO |
| Qwen2.5-Coder-32B-Instruct | 32B | Best open code model overall. Matches GPT-4o. | 92.7% | **94%+** | DPO + merge + speculative |
| Qwen2.5-Coder-7B (base) | 7B | Distillation target. Prove 32B→7B transfer works. | ~65% | **85%+** | Full pipeline (Recipe C) |

**Secondary targets (Tier 2 — prove stack generality):**

| Model | Size | Why This Model | Strategy |
|-------|------|----------------|----------|
| OCR-Nemotron-7B | 7B | Best 7B for LiveCodeBench (51.3%). Reasoning distilled. | Import + eval parity check |
| Phi-4 | 14B | Strong at 14B. Different architecture than Qwen. | Import + merge with Qwen variants |
| DeepSeek-R1-Distill-Qwen-7B | 7B | Reasoning-enhanced Qwen. Merge candidate. | Merge with Qwen2.5-Coder-7B |

**Stretch target (Tier 3 — marketing win):**

| Model | Size | Why This Model | Strategy |
|-------|------|----------------|----------|
| Qwen2.5-Coder-1.5B | 1.5B | Smallest competitive code model. `apr compile` → single binary demo. | LoRA + quantize + compile |

## 4.2 WHY We Will Improve Them

**The falsifiable claim:** A single Rust binary can produce models that score in the "Strong" tier or above on every target benchmark.

Five specific improvement hypotheses, each falsifiable:

**H1: Reasoning distillation closes the LiveCodeBench gap.**
- Qwen2.5-Coder-7B scores 18.2% on LiveCodeBench. OCR-Nemotron-7B (reasoning-distilled) scores 51.3%. Distilling from a reasoning teacher should lift LiveCodeBench by 2-3x without hurting HumanEval.
- *Falsified if:* LiveCodeBench stays below 30% after distillation.

**H2: DPO with execution feedback pushes HumanEval+ past 87%.**
- Current Qwen2.5-Coder-7B scores 84.1% on HumanEval+. The 84→87% gap is alignment, not capability. DPO using (correct_code, incorrect_code) pairs from execution feedback should close it.
- *Falsified if:* HumanEval+ stays below 86% after DPO.

**H3: Merge specialists beat any single model.**
- Merging a code-instruct specialist with a code-reasoning specialist (via TIES on the same Qwen2.5 backbone) should exceed either specialist alone.
- *Falsified if:* Merged model scores below the best input specialist on all benchmarks.

**H4: Quantization to INT4 loses <2% pass@1.**
- Conservative quantization (INT4 with calibration) should preserve almost all accuracy for code generation.
- *Falsified if:* INT4 model drops more than 2% pass@1 vs FP16 on HumanEval.

**H5: The full pipeline (distill→finetune→merge→prune→quantize) compounds gains.**
- Each technique contributes independently. Stacked in the golden ordering (§10), they should compound.
- *Falsified if:* Full pipeline scores lower than the best single-technique result.

## 4.3 HOW We Will Improve Each Model

### 4.3.1 Qwen2.5-Coder-7B: "The Complete Proof" (Primary Target)

This is the model that proves the thesis. Every technique applied, every claim validated.

```
Phase 1: Baseline
  apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct → baseline.apr
  apr eval baseline.apr → establish apr-native HumanEval/MBPP scores
  apr compare-hf baseline.apr → measure parity gap

Phase 2: Reasoning Distillation (H1)
  apr import hf://Qwen/Qwen2.5-Coder-32B-Instruct → teacher.apr
  apr distill teacher.apr --student base.apr --strategy progressive
  → Expected: +5-13% on HumanEval, +15-30% on LiveCodeBench

Phase 3: LoRA Fine-tuning on Curated Code Data
  apr finetune distilled.apr --method qlora --rank 32 --data code-instruct.jsonl
  → Expected: +3-5% from domain-specific tuning

Phase 4: DPO Alignment (H2)
  apr align distilled-tuned.apr --method dpo --data preference-pairs.jsonl
  → Expected: +2-4% on HumanEval+ from execution-feedback alignment

Phase 5: Merge with Reasoning Variant (H3)
  apr merge code-specialist.apr reasoning-specialist.apr --strategy ties
  → Expected: best-of-both-worlds across benchmarks

Phase 6: Prune + Quantize (H4)
  apr prune merged.apr --method wanda --target-ratio 0.2
  apr quantize pruned.apr --scheme int4
  → Expected: <2% pass@1 loss, 4x smaller, 2x faster inference

Phase 7: Compile & Ship
  apr compile final.apr -o qwen-coder-7b --release --lto
  → Standalone binary, zero runtime deps
```

**Success gate:** Final model achieves ≥85% HumanEval, ≥82% HumanEval+, ≥80% MBPP, all via `apr` commands only.

**Current status (2026-03-20):** Phase 1 complete.
- HumanEval: 7B **87.20%** (few-shot, 0.60pp from HF parity), 32B **89.63%** (GPU)
- MBPP: 7B **76.20%** (7.3pp from HF parity, fixed by adding test assertions to prompt)
- Success gate: HumanEval ≥85% ✅, MBPP ≥80% — **3.8pp short**, needs 32B eval or prompt tuning
- Next: 32B MBPP eval (GPU), 32B few-shot HumanEval, per-problem failure analysis

### 4.3.2 Qwen2.5-Coder-32B: "The Crown" (Maximum Score)

The 32B model is already at 92.7% HumanEval. The goal is to push past the ceiling using techniques that benefit from the model's existing strength.

```
Phase 1: Baseline + parity verification
Phase 2: DPO with execution feedback (primary lever)
Phase 3: Merge with reasoning variant (R1-Distill-Qwen-32B)
Phase 4: Speculative decoding for faster eval iteration
Phase 5: N-sampling (N=50) + reranking for maximum pass@1
```

**Success gate:** ≥94% HumanEval, ≥88% HumanEval+, ≥45% BigCodeBench.

### 4.3.3 Qwen2.5-Coder-1.5B: "The Sovereign Binary" (Marketing Win)

```
Phase 1: Import + baseline
Phase 2: LoRA fine-tune on curated instruction data
Phase 3: INT4 quantize
Phase 4: apr compile → single static binary (~800MB)
Phase 5: Ship as downloadable executable
```

**Success gate:** ≥60% HumanEval in a standalone binary with zero dependencies. The demo: `./qwen-coder "def fibonacci(n):"` just works.

## 4.4 What Happens When Improvement Fails

Each hypothesis above has a falsification criterion. When falsified:

1. **Diagnose with five-whys:** `apr diagnose model.apr --method five-whys` identifies root cause (inference bug? data quality? technique misconfigured?)
2. **Compare against HF reference:** `apr compare-hf model.apr` — if parity gap is >5%, fix inference first, don't optimize on a broken baseline
3. **Ablation:** Remove the last technique applied and re-evaluate. If removal improves score, the technique was destructive in this combination.
4. **Escalate to next tier:** If a technique fundamentally doesn't work at world-class level, the tooling must improve (see §5 Sovereign Tooling Map)
