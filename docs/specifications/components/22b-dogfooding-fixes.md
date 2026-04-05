# Dogfooding Findings (Fixes and Integration)

## 22.10 EOS Termination (GH-373)

**Problem:** After removing the 128-token cap, models generated all max_tokens of garbage after producing valid output. The APR CPU generation loop never terminated early on EOS.

**Root cause:** The APR transformer loader hardcoded `eos_token_id: None`. The EOS check `validated.config.eos_token_id == Some(next_token)` never matched.

**Fix:** Added `resolve_apr_stop_tokens()` in realizar which merges EOS from three sources:
1. Model config (`eos_token_id` from metadata)
2. Caller-provided stop tokens (`InferenceConfig.stop_tokens`)
3. Sibling tokenizer.json (ChatML markers: `<|im_end|>` = 151645, `<|endoftext|>` = 151643)

Commit: realizar `e9ac04d`. Verified: Qwen2.5-Coder-7B now correctly resolves `Stop tokens: [151643, 151645]` and terminates at EOS.

## 22.11 Upstream Issues Identified

| Issue | Component | Severity | Status |
|---|---|---|---|
| F16/BF16 passthrough ignores --quantize | aprender | High | **Fixed** (GH-205) |
| Flat Q4K quantization wrong block alignment | aprender | High | **Fixed** (GH-370) |
| No generative finetune path | entrenar/aprender | High | **Fixed** (GH-371) |
| Hardcoded .min(128) token cap | realizar | High | **Fixed** (GH-372) |
| APR EOS termination broken | realizar | Critical | **Fixed** (GH-373) |
| GPU backend migration | realizar | Medium | Migrated from CUDA to wgpu |
| `apr serve` doesn't bind HTTP for .apr | aprender | Medium | Use `apr run --batch-jsonl` for batch inference |
| O(n^2) BPE merge bottleneck | aprender | High | **Fixed** (GH-378) |
| InstructPipeline lacks QLoRA/NF4 | entrenar | High | **Fixed** — wgpu NF4 support |
| InstructPipeline can't load .apr weights | entrenar/aprender | High | **Fixed** — `from_apr()` loading |
| Chat mode trailing text breaks eval | eval script | High | **Fixed** — `extract_python_code()` strips non-Python |
| Prune/merge lose tokenizer and config on GGUF models | aprender | High | **Open** (GH-14) |
| `apr compare-hf` returns 0 comparisons on Q4K vs FP16 | aprender | Medium | Expected — dtype mismatch |
| `apr qa` format parity on .apr-wrapped GGUF | aprender | Medium | **Open** (GH-13) |
| 32B batch GPU crash — FP8 poisons CUDA context on sm_121 | realizar | Critical | **Fixed** (GH-542) — `cc >= 89 && cc < 100` auto-disables FP8 on Blackwell |
| Blackwell GPU garbage (misdiagnosed) | eval test | Low | **Closed** (GH-550) — bare prompt without chat template hit max_tokens, not GPU numerics. GPU inference correct (90.85% HE verified). |
| Stale apr binary blocks --batch-jsonl | gx10 ops | High | **Fixed** — removed .local/bin/apr |

## 22.12 BPE Tokenizer Performance (GH-378)

**Problem:** O(n^2) BPE merge bottleneck. **Fix:** Priority-queue + doubly-linked symbol list. O(n + m log m).

| Metric | Before | After | HF v0.22 |
|--------|--------|-------|----------|
| Encode latency | 145 us | **70 us** (2.06x faster) | 104 us |
| Load latency | 272ms | **142ms** (1.43x faster than HF) | 204ms |
| Allocations | ~825K | ~225K | — |

## 22.13 Training Infrastructure

Training bricks, QLoRA readiness, GPU sharing (multi-adapter), and dual wgpu training proof are documented in [Training Infrastructure (S23)](23-training-infrastructure.md).

## 22.14 QA Gate Results

`apr qa checkpoints/qwen2.5-coder-1.5b-instruct-q4k.apr --verbose` results:

| Check | Status | Details |
|-------|--------|---------|
| Capability Match | PASS | Non-GGUF format check N/A |
| Tensor Contract | PASS | 339 tensors passed PMAT-235 gates |
| Metadata Plausibility | PASS | arch=qwen2, rope_theta=1M, max_pos=32768 |
| Golden Output | PASS | 2 golden test cases passed |
| Throughput | PASS | 2.0 tok/s >= 1 tok/s threshold |
| Perf Regression | PASS | Baseline established |
| Format Parity | FAIL | Expects GGUF format for cross-format parity |
| GPU Speedup | SKIP | CUDA not available |
| Ollama Parity | SKIP | Non-GGUF format |
| PTX Parity | SKIP | Non-GGUF format |
| GPU State Isolation | SKIP | CUDA not available |
| Classifier Head | SKIP | Not requested |

6 PASS, 1 FAIL, 5 SKIP. The Format Parity failure is because .apr wraps GGUF internally but `apr qa` doesn't recognize it as GGUF for the cross-format test. All functional checks pass.

## 22.15 Instruct Model Conversational Trailing Text

**Problem:** Instruct models (Qwen2.5-Coder-1.5B-Instruct via `--chat`) generate correct Python code but append conversational text like `Human\nCan you explain...` or `**Explanation**:`. This causes Python syntax errors in the test harness, producing 0% pass rate despite correct code generation.

**Root cause:** The `--chat` flag causes `apr run` to use chat template formatting. The model completes the instruction correctly, then continues generating in chat turn format. EOS termination (GH-373) helps but doesn't always prevent this.

**Fix:** Added `extract_python_code()` to the eval script that stops at non-Python markers (`Human`, `Assistant`, `**`, `###`, `---`). Applied after markdown fence stripping, before test assembly.

**Impact:** Without fix: 0% pass rate. With fix: expected to match or exceed the 1.5B base model's 59.15%.

## 22.16 MBPP Function Name Fix Impact

**Before fix:** MBPP pass rate 5% (1/20). Model generated correct code but used wrong function names (e.g., `solve()` instead of `min_cost()`), causing all `assert min_cost(...)` tests to fail with `NameError`.

**After fix (function name only):** MBPP pass rate 50.80% (254/500). 10x improvement from extracting the expected function name from `test_list[0]` and including it in the prompt.

**After fix (function name + test assertions):** MBPP pass rate **76.20%** (381/500). Additional +25.4pp from including `test_list` assertions as examples in the prompt, giving the model exact I/O format.

**Five Whys:**
1. Why 5% pass rate? → Tests fail with `NameError`
2. Why NameError? → Model uses wrong function name
3. Why wrong name? → Prompt doesn't specify the expected name
4. Why no name in prompt? → `build_instruction()` didn't parse MBPP `test_list`
5. Why not? → MBPP format was only partially understood (S24.5)

## 22.17 Qwen3 Thinking Model Evaluation (GH-479)

**Model:** Qwen3-4B Q4K (imported from GGUF, 2.5 GB)

### 22.17.1 Thinking Mode Behavior

Qwen3 models use a "thinking" mode where the model generates reasoning tokens before producing code:
```
[151667]   <- <think> token
...reasoning text (1000-6000 tokens)...
[151668]   <- </think> token
...actual code answer...
```

**Critical finding: Thinking is mandatory for code quality.**

| Mode | pass@1 | Notes |
|------|--------|-------|
| With thinking (4096 tokens) | **78.05%** | 128/164 passed (full run), 4 timeouts |
| Without thinking (`/no_think`) | **5%** | 8/164 passed — model produces garbage |
| Without thinking (disabled in prompt) | **5%** | `/no_think` not respected by Q4K model |

The 17x accuracy difference proves that Qwen3-4B relies entirely on chain-of-thought reasoning for code generation. Without thinking, the model is essentially non-functional.

### 22.17.2 Thinking Overflow Problem

At 4096 max_tokens, ~9% of problems overflow (model spends all tokens reasoning without reaching `[151668]`). These produce no code and are scored as failures.

Pathological example: HumanEval/1 (parentheses grouping) — model spiraled for 4096+ tokens analyzing the string character by character, never producing code.

### 22.17.3 Eval Script Adaptations

Three additions to `eval-pass-at-k.sh`:
1. **`strip_thinking_tokens()`** — extracts code after `[151668]`, falls back to parsing ` ```python ` blocks from reasoning
2. **Effective max_tokens override** — auto-increases to 4096 for Qwen3 models
3. **Scaled timeout** — `max_tokens/2 + 60` seconds (~35 min for 4096 tokens at ~3 tok/s CPU)

### 22.17.4 Parallel Evaluation Architecture

Rewrote eval script from sequential to parallel (Phase 1-4 architecture):
1. **Prepare** — split benchmark into per-problem JSON files
2. **Generate** — N parallel workers claim problems via flock queue
3. **Test** — sequential sandbox execution
4. **Score** — Chen et al. pass@k

Worker count limited by model memory: each `apr run` instance loads ~20 GB for Qwen3-4B. 2 workers safe on 119 GB system; 4 workers caused OOM risk (109/119 GB used).

### 22.17.5 GH-479 Fix: `head_dim` vs `hidden_dim / num_heads`

Qwen3 uses `head_dim=128` with `hidden_dim=2560` and `num_heads=32`, making `hidden_dim/num_heads=80 != head_dim`. 25+ instances of `hidden_dim / num_heads` across 18 files in realizar were replaced with `config.head_dim()` accessor methods. All 15,064 realizar tests pass. Fix committed as realizar `016bcb9` + `0284c3e`.

### 22.17.6 Performance Characteristics

| Metric | Value |
|--------|-------|
| CPU inference (gx10 aarch64) | ~3-4 tok/s |
| GPU inference (local CUDA) | ~1.6 tok/s (slower than CPU) |
| Model load time | ~25s per invocation |
| Avg thinking tokens | ~2000-4000 per problem |
| Avg code tokens | ~100-300 per problem |
| Memory per instance | ~20 GB (Q4K + KV cache) |

### 22.17.7 Key Insights

1. **Thinking models need different eval infrastructure** — timeout, token budget, and post-processing all require thinking-aware logic
2. **Model size != capability with thinking** — 4B thinking model achieves 78.05% pass@1, below 7B non-thinking (85.37%) but strong for its size
3. **Q4K quantization doesn't break thinking** — the model still produces structured `[151667]...[151668]` reasoning despite 4-bit quantization
4. **Token efficiency is terrible** — 80-95% of generated tokens are thinking (discarded). A 4096-token generation yields ~200 tokens of actual code
5. **CPU > GPU for this model** — GPU inference 2.5x slower than CPU, likely due to Q4K kernel overhead or PCIe transfer costs

## 22.18 AC Verification Results

Detailed AC verification findings (compile, throughput, SCoT, HF parity, pruning, MBPP function names, submit fix) have been moved to [AC Verification (S24)](24-ac-verification.md) for file size compliance.

## 22.19 Batch Inference Mode (GH-batch)

**Problem:** Each `apr run` invocation on gx10 (Blackwell sm_121) incurs ~80s of CUDA JIT compilation overhead. For 164 HumanEval problems, this means ~3.6 hours of JIT alone, dominating eval wall-clock time.

**Solution:** `apr run --batch-jsonl` loads the model and CUDA kernels once, then processes all prompts sequentially. Implemented in realizar (`batch.rs`) and wired through aprender CLI.

Detailed batch inference architecture, testing results, performance projections, and eval script integration are documented in [AC Verification (S24)](24-ac-verification.md) S24.12-S24.13.

## 22.20 Lessons Learned (2026-04-03)

Key insights from 6 weeks of end-to-end dogfooding:

1. **GGUF Q4K is the working import path.** SafeTensors FP16/BF16 models cannot run inference in realizar (fused matmul requires Q4K/Q6K/Q8K types). GGUF pre-quantized imports produce runnable models with embedded tokenizers. This is not a bug — it's a deliberate architecture choice for inference efficiency.

2. **Oracle analysis reveals the ceiling.** Best-per-problem across all strategies and runs: 96.34% (158/164). Only 6 problems are never solved by any strategy. The gap between best single-run (90.85% 32B) and oracle (96.34%) is 5.49pp — strategy routing or ensemble decoding could close 3-4pp of this.

3. **Few-shot beats reasoning prompts for small models.** For 7B: few-shot (+1.83pp) > standard > CGO (-1.83pp) > SCoT (-3.05pp). Structured reasoning overhead costs more than it gains at 7B scale. This reverses at 32B where reasoning helps.

4. **Batch mode is essential for evaluation.** Per-invocation overhead (model load + CUDA JIT) dominates. Batch mode eliminates ~80s overhead per invocation. Without it, 164 HumanEval problems x 80s = 3.6 hours of pure overhead.

5. **wgpu training works but needs the right data size.** 99 samples x 3 epochs = 39 min on gx10. 15K samples x 3 epochs = 150+ hours — impractical for single-session training. Targeted small datasets from failure analysis are the right approach.

6. **Provable contracts catch real bugs.** FT-GATE-001 (AC-022 MBPP gate) correctly identified the 3.8pp gap before any manual analysis. The contract-first approach surfaces issues automatically through falsification tests.
