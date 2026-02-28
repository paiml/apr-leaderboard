#  Target Leaderboards & Competitive Thresholds

| Leaderboard | Primary Metric | Benchmarks | Why |
|-------------|---------------|------------|-----|
| EvalPlus | pass@1 | HumanEval+, MBPP+ | Rigorous test suites (80x/35x more tests than originals) expose real quality — the gold standard |
| BigCodeBench | pass@1 | 1,140 practical tasks | Tests library usage, I/O, and dependencies — not yet saturated (GPT-4o scores ~61%) |
| LiveCodeBench | pass@1 | 1,055 fresh competitive problems | Continuously refreshed from LeetCode/CodeForces — contamination-resistant |
| BigCode Models | pass@1 | HumanEval, MBPP, MultiPL-E | Code generation visibility — our primary use case |

## 3.1 Competitive Score Thresholds (2025-2026)

HumanEval is approaching saturation (SOTA 92.7%). BigCodeBench and LiveCodeBench differentiate more meaningfully.

| Benchmark | Not Competitive | Entry | Strong | SOTA (Open) |
|-----------|-----------------|-------|--------|-------------|
| HumanEval (pass@1) | <60% | 60-75% | 75-85% | **85-93%** |
| HumanEval+ (pass@1) | <70% | 70-80% | 80-85% | **85-89%** |
| MBPP (pass@1) | <70% | 70-80% | 80-85% | **85-91%** |
| BigCodeBench-Full (pass@1) | <30% | 30-40% | 40-50% | **50%+** |
| LiveCodeBench (pass@1) | <20% | 20-40% | 40-60% | **60%+** |

## 3.2 The Landscape: Who Holds the Crown

**32B class — current SOTA:**

| Model | HumanEval | HE+ | MBPP | LiveCode | License |
|-------|-----------|------|------|----------|---------|
| Qwen2.5-Coder-32B-Instruct | **92.7%** | **87.2%** | **90.2%** | 31.4% | Apache-2.0 |
| OCR-Nemotron-32B | — | — | — | **61.8%** | Apache-2.0 |
| R1-Distill-Qwen-32B | — | — | — | 58.1% | MIT |
| DeepSeek-Coder-V2 (236B MoE) | 85.4% | 82.3% | — | — | Restricted |
| Codestral 25.01 (22B) | 86.6% | — | 91.2% | — | Restricted |

**7B class — current SOTA:**

| Model | HumanEval | HE+ | MBPP | LiveCode | License |
|-------|-----------|------|------|----------|---------|
| Qwen2.5-Coder-7B-Instruct | **88.4%** | **84.1%** | **83.5%** | 18.2% | Apache-2.0 |
| OCR-Nemotron-7B | — | — | — | **51.3%** | Apache-2.0 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 81.1% | — | — | — | Restricted |
| Phi-4 (14B) | 82.6% | — | — | — | MIT |

**Critical gap:** Qwen2.5-Coder dominates standard benchmarks (HumanEval, MBPP) but falls behind on LiveCodeBench. The gap is reasoning: OCR-Nemotron-32B (distilled from DeepSeek-R1) nearly doubles Qwen's LiveCodeBench score. This is the improvement vector.
