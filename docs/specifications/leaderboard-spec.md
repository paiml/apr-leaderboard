---
title: "APR Leaderboard Specification"
version: "2.2.0"
status: "Active"
created: "2026-02-28"
updated: "2026-03-20"
---

# APR Leaderboard Specification

**THE** definitive spec for the apr-leaderboard project. All details live in
[components/](components/) — this document is the executive summary and TOC.

---

## Table of Contents

| # | Component | File | Summary |
|---|-----------|------|---------|
| 1 | [What This Repo Does](#1-what-this-repo-does) | [01](components/01-what-this-repo-does.md) | Purpose, architecture, implementation status |
| 2 | [Thesis](#2-thesis) | [02](components/02-thesis.md) | Falsifiable claim and compute reality |
| 3 | [Target Leaderboards](#3-target-leaderboards) | [03](components/03-target-leaderboards.md) | Benchmarks, competitive thresholds |
| 4 | [Model Selection](#4-model-selection) | [04](components/04-model-selection.md) | Which models, why, improvement strategies |
| 5 | [Sovereign Tooling Map](#5-sovereign-tooling-map) | [05](components/05-sovereign-tooling-map.md) | Stack coverage, gaps, parity checks |
| 6 | [CLI Toolchain](#6-cli-toolchain) | [06](components/06-cli-toolchain.md) | `apr` subcommands, Makefile targets, scripts |
| 7 | [Technique Playbook](#7-technique-playbook) | [07](components/07-technique-playbook.md) | LoRA, QLoRA, distillation, pruning, quantization |
| 8 | [Leaderboard Techniques](#8-leaderboard-winning-techniques) | [08](components/08-leaderboard-winning-techniques.md) | Speculative decoding, FIM, prompt strategies |
| 9 | [Composite Recipes](#9-composite-recipes) | [09](components/09-composite-recipes.md) | Recipes A-H pipeline definitions |
| 10 | [Technique Interaction Matrix](#10-technique-interaction-matrix) | [10](components/10-technique-interaction-matrix.md) | Compatibility matrix for stacked techniques |
| 11 | [Competitive Advantage](#11-competitive-advantage) | [11](components/11-competitive-advantage.md) | Why sovereign stack wins |
| 12 | [Data Strategy](#12-data-strategy) | [12](components/12-data-strategy.md) | Training corpora, data lineage |
| 13 | [Evaluation Protocol](#13-evaluation-protocol) | [13](components/13-evaluation-protocol.md) | Benchmark execution, scoring |
| 14 | [Submission Flow](#14-submission-flow) | [14](components/14-submission-flow.md) | Export, model card, HF Hub publish |
| 15 | [Success Criteria](#15-success-criteria) | [15](components/15-success-criteria.md) | Pass/fail conditions |
| 16 | [Provable Contracts](#16-provable-contracts) | [16](components/16-provable-contracts.md) | Kani bounded model checking |
| 17 | [Quality Gates](#17-quality-gates) | [17](components/17-quality-gates.md) | CI/CD gates, regression checks |
| 18 | [Acceptance Criteria](#18-acceptance-criteria) | [18](components/18-acceptance-criteria.md) | AC-001 through AC-029 |
| 19 | [Implementation Status](#19-implementation-status) | [19](components/19-implementation-status.md) | Tracking table for all targets |
| 20 | [Scientific Foundation](#20-scientific-foundation) | [20](components/20-scientific-foundation.md) | Papers, references |
| 21 | [Open Questions](#21-open-questions) | [21](components/21-open-questions.md) | Unresolved decisions |
| 22 | [Dogfooding Findings](#22-dogfooding-findings) | [22](components/22-dogfooding-findings.md) | Real results, baseline scores, upstream fixes |
| 23 | [Training Infrastructure](#23-training-infrastructure) | [23](components/23-training-infrastructure.md) | Training bricks, QLoRA, GPU sharing, wgpu proof |
| 24 | [AC Verification](#24-ac-verification) | [24](components/24-ac-verification.md) | Detailed acceptance criteria verification findings |

---

## 1. What This Repo Does

**apr-leaderboard** is a pipeline harness that proves the sovereign Rust AI stack
can compete on HuggingFace code generation leaderboards without Python, without
CUDA, and without GPU vendor lock-in.

```
apr import -> apr distill -> apr finetune -> apr merge -> apr prune -> apr quantize -> eval -> apr publish
```

Every command is provided by `apr` CLI (aprender). This repo provides:
- YAML configs (models, recipes, eval suites, pipeline DAG)
- Shell scripts (pipeline execution, evaluation, submission)
- Strategy spec (this document + components)

**Architecture:**

```
Makefile (dev convenience)
+-- scripts/pipeline.sh       -> reads recipe YAML, runs stages
+-- scripts/eval-pass-at-k.sh -> apr run + sandbox + Chen et al. pass@k
+-- scripts/submit.sh         -> preflight checks + apr export + apr publish
+-- scripts/prove-wgpu.sh     -> dual GPU wgpu training proof
+-- scripts/results-history.sh -> eval results viewer
+-- configs/models/            -> 7 YAML model configs
+-- configs/recipes/           -> 8 YAML recipe configs
+-- configs/eval/              -> benchmark suite definitions
+-- configs/pipeline/          -> forjar manifest + batuta playbook
+-- data_catalog.yaml          -> data governance + lineage
```

**Constraints:**
- Zero Python (bash-native YAML parsing, `awk`/`jq` for math/JSON)
- Zero CUDA (wgpu only — Vulkan/Metal/DX12)
- YAML-only (albor pattern, legacy TOML removed)

-> [Full details](components/01-what-this-repo-does.md)

---

## 2. Thesis

> Can a single Rust binary (`apr`) match Python-ecosystem HumanEval/MBPP scores
> for Qwen2.5-Coder-7B, with zero Python dependencies?

**Compute reality:** Primary eval on gx10 (NVIDIA Blackwell GB10, 119 GB unified, CUDA sm_121). Also verified on AMD Radeon Pro W5700X (Navi10) via wgpu/Vulkan. CPU/GPU parity confirmed on HumanEval (85.37% on both backends).

-> [Full details](components/02-thesis.md)

---

## 3. Target Leaderboards

| Leaderboard | Primary Benchmark | Target Score |
|---|---|---|
| BigCode (EvalPlus) | HumanEval / MBPP | pass@1 >= 80% (7B) |
| BigCodeBench | BigCodeBench | pass@1 >= 40% |
| LiveCodeBench | Competitive programming | Contamination-free eval |
| Open LLM Leaderboard v2 | Arc/HellaSwag/MMLU | Architecture diversity |

-> [Full details](components/03-target-leaderboards.md)

---

## 4. Model Selection

| Tier | Model | Params | Strategy | Goal |
|---|---|---|---|---|
| 1 (Primary) | Qwen2.5-Coder-7B | 7B | LoRA + distill + merge + prune + quantize | Complete proof |
| 1 (Max Score) | Qwen2.5-Coder-32B | 32B | Eval only (Q8) | Crown score |
| 2 (Sovereign) | Qwen2.5-Coder-1.5B | 1.5B | QLoRA + prune + INT4 + compile | <1GB binary |
| 2 (Reasoning) | DeepSeek-R1-Distill-7B | 7B | DPO + prune + INT4 | LiveCodeBench |
| 2 (Diversity) | Phi-4 | 14B | LoRA + INT8 | Architecture diversity |
| 3 (QLoRA) | Qwen3-8B | 8B | QLoRA + NF4 | VRAM-efficient proof |

-> [Full details](components/04-model-selection.md)

---

## 5. Sovereign Tooling Map

| Capability | Tool | Status |
|---|---|---|
| Model import/export | aprender (`apr`) | Working |
| Training (LoRA/QLoRA) | entrenar | Working |
| GPU compute | trueno (wgpu) | Working |
| SIMD tensor ops | trueno (AVX2/NEON) | Working |
| Kernel verification | provable-contracts (Kani) | Working |
| Config validation | bashrs | Working |
| Data loading | alimentar | Working |
| Pipeline orchestration | batuta | Working |
| Inference | realizar | Working |

**Hard rule:** No CUDA toolkit. No Python. GPU compute is wgpu only.

-> [Full details](components/05-sovereign-tooling-map.md)

---

## 6. CLI Toolchain

19 `apr` subcommands verified. 10 shell scripts. 45 Makefile targets.
Zero Python scripts — all data prep uses `apr` CLI.

```bash
make verify                              # smoke-test all subcommands
make validate                            # bashrs lint all configs
make pipeline RECIPE=recipe-a-quick-lora # run recipe pipeline
make prove-wgpu                          # dual GPU wgpu proof
```

-> [Full details](components/06-cli-toolchain.md)

---

## 7. Technique Playbook

| Technique | Impact | VRAM | Time |
|---|---|---|---|
| LoRA (rank 16-64) | +5-15 HumanEval | 8-16 GB | 2-6 hrs |
| QLoRA (NF4 + FP16 adapters) | +5-12 HumanEval | 4-8 GB | 3-8 hrs |
| Knowledge distillation | +3-8 HumanEval | 16-24 GB | 4-12 hrs |
| SLERP/TIES merge | +2-5 HumanEval | 16 GB | Minutes |
| Wanda pruning (50%) | -1-3 HumanEval | 8 GB | Minutes |
| Q4K quantization | -1-2 HumanEval | 4 GB | Minutes |

-> [Full details](components/07-technique-playbook.md)

---

## 8. Leaderboard-Winning Techniques

Speculative decoding, fill-in-middle (FIM), prompt engineering,
n-sampling with reranking, code-specific tokenization.

-> [Full details](components/08-leaderboard-winning-techniques.md)

---

## 9. Composite Recipes

| Recipe | Pipeline | Target |
|---|---|---|
| A: Quick LoRA | import -> LoRA -> eval | Fast iteration |
| B: Merge Alchemist | import -> TIES merge -> prune -> quantize -> eval | Zero training |
| C: Full Pipeline | distill -> LoRA -> merge -> prune -> quantize -> eval | Maximum score |
| D: Sovereign Binary | import -> QLoRA -> prune -> INT4 -> compile | <1GB binary |
| E: Instruct LoRA | import -> prep-data -> instruct LoRA -> eval | BigCodeBench |
| F: Qwen3 QLoRA | import -> QLoRA (NF4) -> eval | VRAM-efficient |
| G: wgpu Proof | import -> QLoRA (wgpu) -> eval -> verify | GPU proof |
| H: Reasoning Distill | 32B teacher -> progressive distill -> 7B student -> eval | Knowledge transfer |

-> [Full details](components/09-composite-recipes.md)

---

## 10. Technique Interaction Matrix

Compatibility rules for stacking techniques. Key constraint: quantize
AFTER merge (never before). Prune AFTER finetune. Distill BEFORE finetune.

-> [Full details](components/10-technique-interaction-matrix.md)

---

## 11. Competitive Advantage

Zero setup time (no CUDA toolkit, no conda). Single binary. Any GPU vendor.
Sovereign — no Python runtime, no cloud API dependency.

-> [Full details](components/11-competitive-advantage.md)

---

## 12. Data Strategy

15.5K instruction/response pairs from 4 ground truth corpora (depyler, hf-gtc,
jax-gtc, vllm-gtc). Data lineage tracked in `data_catalog.yaml`.

-> [Full details](components/12-data-strategy.md)

---

## 13. Evaluation Protocol

```
download benchmark -> apr run --json -> score pass/fail -> compute pass@k -> write JSON
```

Benchmarks: HumanEval (164), MBPP (500), BigCodeBench (1140).
Suite config: `configs/eval/coding-benchmarks.yaml`.

-> [Full details](components/13-evaluation-protocol.md)

---

## 14. Submission Flow

`apr export` -> SafeTensors + model card -> dry-run confirmation -> `apr publish` to HF Hub.

-> [Full details](components/14-submission-flow.md)

---

## 15. Success Criteria

| Metric | Target | Gate | Current |
|---|---|---|---|
| HumanEval pass@1 (7B) | >= 80% | Hard | **87.20%** ✅ |
| MBPP pass@1 (7B) | >= 70% | Hard | **76.20%** ✅ |
| BigCodeBench pass@1 | >= 35% | Soft | — |
| Pipeline stages | All 12 working | Hard | ✅ |
| Zero Python deps | 0 | Hard | ✅ |
| wgpu GPU compute | Dual GPU proven | Hard | ✅ |

-> [Full details](components/15-success-criteria.md)

---

## 16. Provable Contracts

Kani bounded model checking. 8 pipeline-specific proof obligations (PO-LB-001
through PO-LB-008). `#[contract]` annotations bind every kernel to its YAML spec.
5 contract files with 14 proof obligations + 14 falsification tests deployed.

-> [Full details](components/16-provable-contracts.md)

---

## 17. Quality Gates

```bash
make verify    # apr CLI + subcommands
make validate  # bashrs lint (YAML + shell + Makefile)
make dogfood   # end-to-end smoke test (zero Python)
```

-> [Full details](components/17-quality-gates.md)

---

## 18. Acceptance Criteria

29 falsifiable acceptance criteria (AC-001 through AC-029). Every criterion has
a concrete command and a pass/fail threshold. 7 verified, 4 partially verified,
2 blocked on upstream, 16 not yet tested. Key milestones:
- AC-001: `apr import` produces valid `.apr` file ✅
- AC-004: LoRA training with decreasing loss ✅
- AC-005: QLoRA <50% VRAM vs LoRA ✅
- AC-011: Full pipeline (Recipe C) completes end-to-end
- AC-021: Qwen-7B imported achieves >= 85% HumanEval pass@1
- AC-028: `make prove-wgpu` passes — QLoRA training on wgpu ✅
- AC-029: wgpu training produces decreasing loss over 2 epochs ✅

-> [Full details](components/18-acceptance-criteria.md)

---

## 19. Implementation Status

**All orchestration implemented.** 42 Makefile targets, 7 shell scripts,
17 YAML configs, 19/19 `apr` subcommands verified. Zero Python scripts.
Zero TOML configs (migrated to YAML-only).

| Component | Count | Status |
|---|---|---|
| YAML model configs | 6 | Complete |
| YAML recipe configs | 7 | Complete |
| YAML eval suite | 1 | Complete |
| YAML pipeline configs | 2 | Complete |
| Data catalog | 1 | Complete |
| Shell scripts | 7 | Complete |
| Makefile targets | 42 | Complete |
| Provable contracts | 5 | Complete |

-> [Full details](components/19-implementation-status.md)

---

## 20. Scientific Foundation

19 references grouped by pipeline stage: training (LoRA, QLoRA, DPO, ORPO),
compression (Wanda, SparseGPT, TIES, DARE), inference (speculative decoding,
self-consistency), benchmarks (HumanEval, LiveCodeBench, BigCodeBench).

-> [Full details](components/20-scientific-foundation.md)

---

## 21. Open Questions

28 questions tracked, 13 answered by dogfooding (marked ✅). Key open:
- Calibration data quality for Wanda pruning (ablation needed)
- DPO data volume for measurable HumanEval+ improvement
- CPU-only distillation feasibility within 24h wall-clock

-> [Full details](components/21-open-questions.md)

---

## 22. Dogfooding Findings

**Baseline results:**

| Model | pass@1 | Backend |
|---|---|---|
| Qwen2.5-Coder-1.5B Q4K | 59.15% | CPU |
| Qwen2.5-Coder-7B Q4K | 68.90% | CPU |

**Hardware:** 2x AMD Radeon Pro W5700X (Navi10), 16 GB each, Vulkan 1.3.255.
**wgpu dual GPU proof:** Ready to run. Both GPUs enumerated and tested.

-> [Full details](components/22-dogfooding-findings.md)

---

## 23. Training Infrastructure

GPU sharing (VRAM guard, ledger, wait queue, MPS), multi-adapter pipeline,
cluster config with SSH transport, checkpoint coordination. 143 tests across
all GPU modules. QLoRA NF4 verified on wgpu.

-> [Full details](components/23-training-infrastructure.md)

---

## 24. AC Verification

Detailed acceptance criteria verification findings split from §22:
compile binary (AC-026), throughput benchmarks, SCoT prompting (AC-019),
HF parity (AC-014), MBPP function names, Wanda pruning (AC-008),
pass@k falsification tests (3/3 pass), golden ordering enforcement.

-> [Full details](components/24-ac-verification.md)

---

## Infrastructure Summary

```
Dependencies: apr (v0.4.10+) + bashrs + bash + jq + curl
GPU:          2x AMD Radeon Pro W5700X (wgpu/Vulkan)
Config:       YAML-first (albor pattern)
Python:       Zero
CUDA:         Zero
```

## File Map

```
docs/specifications/
+-- leaderboard-spec.md           <- THIS FILE (executive summary + TOC)
+-- components/
    +-- 01-what-this-repo-does.md  <- Architecture, status
    +-- 02-thesis.md               <- Falsifiable claim
    +-- 03-target-leaderboards.md  <- Benchmarks, thresholds
    +-- 04-model-selection.md      <- Model tiers, strategies
    +-- 05-sovereign-tooling-map.md <- Stack coverage, gaps
    +-- 06-cli-toolchain.md        <- apr subcommands, scripts
    +-- 07-technique-playbook.md   <- LoRA, QLoRA, distill, prune
    +-- 08-leaderboard-winning-techniques.md
    +-- 09-composite-recipes.md    <- Recipes A-G
    +-- 10-technique-interaction-matrix.md
    +-- 11-competitive-advantage.md
    +-- 12-data-strategy.md        <- Training data, lineage
    +-- 13-evaluation-protocol.md  <- Benchmark execution
    +-- 14-submission-flow.md      <- HF Hub publish
    +-- 15-success-criteria.md     <- Pass/fail gates
    +-- 16-provable-contracts.md   <- Kani proofs
    +-- 17-quality-gates.md        <- CI/CD gates
    +-- 18-acceptance-criteria.md  <- AC-001 to AC-029
    +-- 19-implementation-status.md <- Tracking table
    +-- 20-scientific-foundation.md <- Papers
    +-- 21-open-questions.md       <- Unresolved
    +-- 22-dogfooding-findings.md  <- Real results, GPU proofs
    +-- 23-training-infrastructure.md <- GPU sharing, QLoRA, wgpu
    +-- 24-ac-verification.md     <- AC verification findings
```
