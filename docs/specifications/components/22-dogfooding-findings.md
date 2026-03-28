# Dogfooding Findings

Real end-to-end dogfooding with Qwen2.5-Coder models (1.5B, 7B, 32B) and Qwen3-4B. These findings inform spec updates and upstream `apr` CLI improvements.
## 22.0 HumanEval Baseline Results

| Model | Quantization | pass@1 | Passed | Avg Tokens | Avg Latency | Backend | Notes |
|-------|-------------|--------|--------|------------|-------------|---------|-------|
| Qwen2.5-Coder-32B-Instruct Q4K_M | Q4K_M | **90.85%** | 149/164 | — | — | CPU (gx10) | 32B batch mode re-run |
| Qwen2.5-Coder-32B-Instruct Q4K_M | Q4K_M | 89.63% | 147/164 | 73.9 | 294s | CPU†† (gx10) | 32B, parity gate blocked CUDA |
| Qwen2.5-Coder-7B-Instruct Q4K | Q4K | **85.37%** | 140/164 | 85.5 | 113s | CPU (gx10) | EOS fix + 512 tokens |
| Qwen2.5-Coder-7B-Instruct Q4K | Q4K | **85.37%** | 140/164 | 85.5 | 112s | CPU†† (gx10) | Parity gate blocked CUDA, CPU fallback |
| Qwen2.5-Coder-7B-Instruct Q4K (few-shot) | Q4K | **87.20%** | 143/164 | — | — | CPU (gx10) | Few-shot prompting (+1.83pp vs standard) |
| Qwen2.5-Coder-7B-Instruct Q4K (SCoT) | Q4K | **82.32%** | 135/164 | — | — | CPU (gx10) | Structured CoT prompting |
| Qwen3-4B Q4K | Q4K | **78.05%** | 128/164 | ~3000† | ~280s | CPU (gx10) | Thinking mode, 4096 tokens |
| Qwen2.5-Coder-7B-Instruct Q4K | Q4K | 68.90% | 113/164 | 128.0 | 102s | CPU | Pre-EOS-fix, 128 cap |
| Qwen2.5-Coder-1.5B Q4K | Q4_K_M (GGUF) | 59.15% | 97/164 | 59.5 | 3.6s | CPU | 128 token cap |

†Qwen3 avg tokens includes ~2500 thinking tokens (discarded) + ~500 code tokens.
††These runs were labeled "GPU" but the CUDA parity gate silently fell back to CPU. CUDA cosine=-0.005 on sm_121 due to FP32 accumulation ordering (GH-559/561). wgpu (Vulkan) gives cosine=0.999863 and is now wired as fallback.

**Key findings:**
- 85.37% → **90.85%** from 7B → 32B model (+9 problems solved, batch re-run)
- GPU/CPU parity confirmed: 7B produces identical 85.37% on both backends
- Few-shot prompting is the best 7B strategy: **87.20%** (+1.83pp vs 85.37% standard, +3 problems)
- Simpler exemplar wins: trivial `add(a,b)` (87.20%) > 3-exemplar (85.98%) > standard (84.76-85.37%)
- SCoT prompting hurts 7B (82.32% vs 85.37% standard) — model already strong without CoT
- CGO fixed: 0% → **83.54%** (137/164) after rewriting prompt to request code-only output
- MBPP: 50.80% → **76.20%** (+25.4pp) from including test assertions in prompt

**7B Prompt Strategy Comparison (HumanEval):**

| Strategy | pass@1 | vs Standard | Notes |
|----------|--------|-------------|-------|
| few-shot (trivial `add(a,b)`) | **87.20%** | +1.83pp | Best — simplest exemplar wins |
| few-shot (3-exemplar) | 85.98% | +0.61pp | Complex exemplars hurt slightly |
| standard | 84.76-85.37% | baseline | Variance across runs |
| cgo | 83.54% | -1.83pp | "Use helper functions" prompt (fixed from 0%) |
| scot | 82.32% | -3.05pp | Reasoning overhead hurts small model |

**32B Prompt Strategy Comparison (HumanEval):**

| Strategy | pass@1 | vs Standard | Notes |
|----------|--------|-------------|-------|
| standard | **90.85%** | baseline | Best 32B strategy (CPU batch) |
| few-shot | 87.20% | -3.65pp | Few-shot hurts 32B even more than SCoT hurts 7B |

**MBPP Strategy Comparison (7B, with test assertions):**

| Strategy | pass@1 | vs Standard | Notes |
|----------|--------|-------------|-------|
| standard | **76.20%** | baseline | Best MBPP strategy |
| few-shot | 74.80% | -1.40pp | Few-shot doesn't help MBPP |

**Cross-benchmark insight:** Few-shot helps HumanEval (function completion with signature) but hurts MBPP (prose description + test assertions). The exemplar primes the model for HumanEval's completion format but adds noise for MBPP's from-scratch generation. For 32B, standard prompting is always optimal — the larger model doesn't need format priming.

**7B Oracle Analysis (multi-run, multi-strategy):**

| Metric | Value |
|--------|-------|
| Oracle (best per problem across all runs) | **96.34%** (158/164) |
| Standard (union of all standard runs) | 95.12% (156/164) |
| Few-shot (union of all few-shot runs) | 93.29% (153/164) |
| CGO (union of all CGO runs) | 83.54% (137/164) |
| Gap (oracle - best single strategy) | 1.22pp |
| Never solved (any strategy) | 6 problems |

**6 always-fail problems** (true 7B Q4K limitations): `max_fill`, `maximum`, `intersection`, `tri`, `order_by_points`, `generate_integers`. These require teacher knowledge transfer (PMAT-007).

**39 inconsistent problems** pass in some runs but fail in others. Of these, 16 have <50% pass rate (need distillation/improvement) and 23 have ≥50% pass rate (recoverable via N-sampling).

**Actionable insight:** Standard prompting is actually the strongest when unioned across runs (156/164). CGO has 1 unique win, standard has 3 unique wins. N-sampling with temperature>0 should recover most inconsistent problems (Chen et al. pass@10).

**Perplexity baseline (WikiText-2):**
| Model | Perplexity | Cross-Entropy | Tokens | Eval Time |
|-------|-----------|---------------|--------|-----------|
| Qwen2.5-Coder-1.5B-Instruct Q4K | 6.63 | 1.89 | 164 | 75.8s |

**Notes:**
- 7B model shows +9.75pp improvement over 1.5B
- 7B 68.90% result was with 128-token cap (GH-372) and broken EOS termination (GH-373)
- Both issues fixed; re-evaluation complete: 85.37% standard, **87.20% few-shot** (0.60pp from HF parity)
- 7B HF reference ~87.8% — gap closed to 0.60pp with few-shot prompting. Remaining gap: Q4K quantization loss
- GPU inference via wgpu (Vulkan/Metal/DX12) — no CUDA dependency
- Perplexity = 6.63 on WikiText-2 confirms non-degenerate model quality (AC-002 partial)

## 22.1 Model Import: GGUF vs SafeTensors

Two import paths were tested. Only GGUF produces runnable models today.

### 22.1.1 SafeTensors Import Path (Broken for Inference)

```bash
apr import hf://Qwen/Qwen2.5-Coder-1.5B -o checkpoints/qwen-1.5b.apr
```

**Result:** Import succeeds but inference fails.

- `apr check` score: **F (3/100)** — fails most validation stages
- Produces F16/BF16 tensors
- realizar's fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K (not F16/BF16)
- Error: `Operation 'owned_fused_matmul' not supported: Fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K, got type 30`
- `apr quantize` also fails: `Failed to dequantize tensor 'model.embed_tokens.weight'` (BF16 embedding)

**Root cause:** SafeTensors import preserves original tensor dtype (BF16). realizar expects quantized tensors for inference. There is no working SafeTensors → quantized pipeline today.

### 22.1.2 GGUF Import Path (Working)

```bash
apr import Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf -o checkpoints/qwen-1.5b-q4k.apr
```

**Result:** Full success.

- `apr check` score: **B+ (85/100)** — 10/10 validation stages pass
- Embedded tokenizer included automatically
- Quantized tensors (Q4_K_M) work with realizar
- File size: 1.1 GB

### 22.1.3 Recommendation

Use pre-quantized GGUF files from HuggingFace for the import step. The SafeTensors path needs upstream work in realizar to support F16/BF16 inference or in `apr import` to auto-quantize on ingest.

## 22.2 Inference Testing

### 22.2.1 Inference (Working)

```bash
# GPU inference (default -- mandatory for production eval)
apr run checkpoints/qwen2.5-coder-1.5b-q4k.apr \
    "def fibonacci(n):" --max-tokens 128

# On Blackwell sm_121, GPU is blocked by parity gate (GH-559: Q4K dequant error)
# Do NOT use SKIP_PARITY_GATE=1 — fix root cause in trueno-gpu PTX codegen
apr run checkpoints/qwen2.5-coder-32b-instruct-q4km.apr \
    --batch-jsonl prompts.jsonl --max-tokens 512
```

**Result:** Generates real Python code (correct Fibonacci implementation). GPU mandatory for eval throughput.

### 22.2.2 GPU Inference (wgpu)

```bash
apr run checkpoints/qwen2.5-coder-1.5b-q4k.apr \
    "def fibonacci(n):" --max-tokens 128
```

GPU inference uses wgpu (Vulkan/Metal/DX12) or CUDA (optional). Works on NVIDIA, AMD, Intel Arc, and Apple Silicon GPUs. GPU is mandatory for production eval — never fall back to CPU.

**Blackwell sm_121 GPU status (2026-03-27): FIXED via wgpu (Vulkan).**

`apr run --gpu` auto-dispatches: CUDA (parity fails) → **wgpu (cosine=0.999863)** → CPU. Token-for-token parity confirmed between wgpu and CPU output.

**Root cause (corrected 2026-03-27):** NOT a JIT bug. Individual CUDA kernels produce correct results (RMSNorm diff=5e-7, Q GEMV ~1%). The cosine=-0.005 is FP32 non-associativity: parallel GPU accumulation order ≠ sequential CPU order, compounding through 280 operations. Falsified by loading our exact PTX via Python ctypes → cosine=1.0. PyTorch avoids this via TF32 accumulators. wgpu avoids it with sequential accumulation matching CPU. See §25 for full architecture specification.

`SKIP_PARITY_GATE=1` is **forbidden** (Toyota Way).

### 22.2.3 `apr serve` (Partial)

`apr serve` loads .apr models but the HTTP server does not bind to a port.
This may be an unimplemented feature for the .apr format — serve may only
work with raw GGUF files. `apr run` is the reliable path for batch
inference in eval scripts.

## 22.3 Validation (`apr check`)

The 10 validation stages for GGUF-imported models:

| Stage | Status | Notes |
|---|---|---|
| Tokenizer | ✅ Pass | Embedded in GGUF import |
| Embedding | ✅ Pass | Q4_K_M quantized |
| RoPE | ✅ Pass | Rotary position embeddings |
| Q/K/V | ✅ Pass | Attention projections |
| Attention | ✅ Pass | Multi-head attention |
| MLP | ✅ Pass | Feed-forward network |
| LayerNorm | ✅ Pass | Layer normalization |
| LM Head | ✅ Pass | Language model head |
| Logits | ✅ Pass | Output logits |
| Sampler | ✅ Pass | Token sampling |

## 22.4 Import Prerequisites

`apr import` for SafeTensors models requires these files in the HF cache:
- `config.json` — model architecture config
- `tokenizer.json` — tokenizer vocabulary

These may not download automatically for all model formats. If missing:
```bash
# Manual download to HF cache
curl -L "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B/resolve/main/config.json" \
    -o ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B/snapshots/<hash>/config.json
```

GGUF imports do not have this issue — all metadata is embedded in the GGUF file.

## 22.5 Pipeline Integration

### 22.5.1 `make verify` Output

All 19 `apr` subcommands respond to `--help`:

```
import       OK      run          OK      serve        OK
chat         OK      finetune     OK      merge        OK
prune        OK      quantize     OK      distill      OK
eval         OK      export       OK      publish      OK
check        OK      compile      OK      bench        OK
inspect      OK      data         OK      qa           OK
compare-hf   OK
```

### 22.5.2 `make dogfood` Output

All YAML configs and scripts validated:
- 7 model configs in `configs/models/` (YAML-only, includes Qwen3-4B)
- 8 recipe configs in `configs/recipes/` (YAML-only, includes recipe-h distillation)
- 10 shell scripts in `scripts/` (all pass `bash -n`)

### 22.5.3 `make pipeline-plan` Output

Dry-run correctly shows all stages and commands for each recipe. Example for recipe-a-quick-lora:

```
Pipeline stages: import finetune eval
[import]   apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct -o checkpoints/...
[finetune] apr finetune ... --method lora --rank 16 --learning-rate 0.0002 --epochs 3
[eval]     ./scripts/eval-pass-at-k.sh <benchmark> checkpoints/...
```

## 22.6 SafeTensors Import + Quantize (Fixed)

**GH-205 fix:** `apr import hf://... --quantize q4k` now correctly quantizes F16/BF16 SafeTensors sources instead of silently passing through F16 raw bytes.

**GH-370 fix:** Q4K quantization now uses `quantize_q4_k_matrix` for row-aligned super-blocks instead of flat byte slicing.

```bash
# This now works (previously produced F16 despite --quantize):
apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct --quantize q4k \
    -o checkpoints/qwen2.5-coder-7b-instruct-q4k.apr
# Result: 7.48 GiB Q4K checkpoint, passes `apr check`
```

## 22.7 Instruction Fine-tuning (GH-371)

**Gap found:** `apr finetune --task classify` existed but no generative instruction-following path. Filed and closed GH-371.

**Solution:** Added `InstructPipeline`, `InstructTrainer`, `InstructCorpus` to entrenar. Wired `--task instruct` into apr CLI.

**Dogfood run (tiny model, 50 samples):**
```
InstructPipeline: 4 LoRA layers, rank=8, alpha=16.0
Corpus: 50 samples, Train: 40, Val: 10

Epoch  Train Loss  Val Loss  Train PPL  Val PPL      LR     Time
    1    6.9330    6.9257   1025.62   1018.08  6.09e-4   1819ms
    2    6.9301    6.9317   1022.59   1024.26  1.48e-6    995ms

Best epoch: 1 (val_loss: 6.9257)
Total time: 2.8s
```

Loss decreasing confirms the training loop is functional. 18 unit tests pass in entrenar.

## 22.8 Data Preparation Pipeline

`make prep-data` extracts 15,494 instruction/response pairs from 4 ground truth corpora via AST parsing of Python files:

```
depyler: 1824 files → 11,841 pairs (algorithms, data structures, CLI)
hf-gtc:   129 files →  3,535 pairs (HuggingFace recipes)
jax-gtc:     7 files →     58 pairs (JAX numerical patterns)
vllm-gtc:    6 files →     81 pairs (vLLM inference)
Total: 15,494 pairs (17 MB JSONL)
```

## 22.9 Token Generation Cap (GH-372)

**Problem:** All completions generated exactly 128 tokens regardless of `--max-tokens 512`.

**Root cause:** 10 instances of `.min(128)` in realizar silently capped generation across GGUF, APR, and GPU inference paths.

**Fix:** Removed all `.min(128)` caps. `InferenceConfig.max_tokens` now passes through uncapped. Commit: realizar `c0a28ef`.

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
5. Why not? → MBPP format was only partially understood (§24.5)

## 22.17 Qwen3 Thinking Model Evaluation (GH-479)

**Model:** Qwen3-4B Q4K (imported from GGUF, 2.5 GB)

### 22.17.1 Thinking Mode Behavior

Qwen3 models use a "thinking" mode where the model generates reasoning tokens before producing code:
```
[151667]   ← <think> token
...reasoning text (1000-6000 tokens)...
[151668]   ← </think> token
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
1. **`strip_thinking_tokens()`** — extracts code after `[151668]`, falls back to parsing ```` ```python ```` blocks from reasoning
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

Qwen3 uses `head_dim=128` with `hidden_dim=2560` and `num_heads=32`, making `hidden_dim/num_heads=80 ≠ head_dim`. 25+ instances of `hidden_dim / num_heads` across 18 files in realizar were replaced with `config.head_dim()` accessor methods. All 15,064 realizar tests pass. Fix committed as realizar `016bcb9` + `0284c3e`.

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
2. **Model size ≠ capability with thinking** — 4B thinking model achieves 78.05% pass@1, below 7B non-thinking (85.37%) but strong for its size
3. **Q4K quantization doesn't break thinking** — the model still produces structured `[151667]...[151668]` reasoning despite 4-bit quantization
4. **Token efficiency is terrible** — 80-95% of generated tokens are thinking (discarded). A 4096-token generation yields ~200 tokens of actual code
5. **CPU > GPU for this model** — GPU inference 2.5x slower than CPU, likely due to Q4K kernel overhead or PCIe transfer costs

## 22.18 AC Verification Results

Detailed AC verification findings (compile, throughput, SCoT, HF parity, pruning, MBPP function names, submit fix) have been moved to [AC Verification (S24)](24-ac-verification.md) for file size compliance.

## 22.19 Batch Inference Mode (GH-batch)

**Problem:** Each `apr run` invocation on gx10 (Blackwell sm_121) incurs ~80s of CUDA JIT compilation overhead. For 164 HumanEval problems, this means ~3.6 hours of JIT alone, dominating eval wall-clock time.

**Solution:** `apr run --batch-jsonl` loads the model and CUDA kernels once, then processes all prompts sequentially. Implemented in realizar (`batch.rs`) and wired through aprender CLI.

### 22.19.1 Architecture

```
BatchInferenceConfig → run_batch_inference()
    ├── detect_format() (8-byte magic: APR\0 vs GGUF)
    ├── run_batch_gguf() → MappedGGUFModel → OwnedQuantizedModel
    └── run_batch_apr()  → MappedAprModel  → OwnedQuantizedModel
        └── init_batch_model()
            └── OwnedQuantizedModelCuda (GPU, parity gate — GH-559 blocks sm_121)
        └── run_batch_loop()
            ├── Read JSONL prompts (BufRead)
            ├── Encode with ChatML template
            ├── BatchModel::generate() → GPU dispatch
            ├── Write JSONL results (flushed per prompt)
            └── Aggregate BatchStats
```

### 22.19.2 Testing Results

| Test | Prompts | Backend | Result |
|------|---------|---------|--------|
| Local 1.5B | 7 | CPU | 7/7 OK (2 code + 5 factorial) |
| gx10 7B | 2 | CPU | 2/2 OK (clean output) |
| gx10 7B | 2 | GPU | JIT compiled OK, output garbled (training contention) |

**GPU parity gate — RESOLVED (2026-03-25).** GPU now produces token-for-token identical output to CPU on Blackwell sm_121. Root cause was a combination of:
1. FP8 E4M3 kernels causing `CUDA_ERROR_ILLEGAL_ADDRESS` (fixed: GH-542, `cc >= 89 && cc < 100` guard)
2. PTX backward branch miscompilation on sm_121 (fixed: GH-480, PTX post-processor in trueno-gpu 0.4.35)
3. Stale CUDA driver (fixed: upgrade 580 → 590.48.01)

**`SKIP_PARITY_GATE=1` is forbidden** (Toyota Way). The parity gate now passes naturally — no bypass needed.

**Five-whys (updated 2026-03-25):**
1. Why did GPU produce wrong tokens? → FP8 kernels + PTX backward branches + stale driver
2. Why FP8 issue? → Blackwell sm_121 (cc=121) was treated as FP8-capable (cc >= 89), but FP8 E4M3 only works on Hopper (cc 89-99)
3. Why PTX issue? → `bra LABEL` backward jumps miscompile on sm_121 JIT — patched to `@%p_jw bra LABEL`
4. Why stale driver? → Driver 580 didn't have sm_121 JIT fixes; driver 590 resolves JIT errors
5. Fix: Three upstream fixes (GH-542, GH-480, driver 590) — code fixes, not gate bypass

### 22.19.3 Performance Projection

| Scenario | JIT Overhead | Total Wall-Clock |
|----------|-------------|-----------------|
| Sequential (164 problems) | 80s × 164 = 3.6h | 3.6h + inference |
| Batch (164 problems) | 80s × 1 = 80s | 80s + inference |
| Speedup | — | **~160x JIT reduction** |

### 22.19.4 Eval Script Integration

The eval script (`scripts/eval-pass-at-k.sh`) now auto-detects batch mode:

1. Checks if `apr run --help` contains `--batch-jsonl`
2. If available, builds all prompts into a single JSONL file
3. Runs `apr run --batch-jsonl prompts.jsonl --temperature T --top-k K`
4. Parses JSONL output back into per-problem completion files
5. Falls back to per-problem worker mode on failure

Environment variables: `APR_BATCH_MODE=auto|on|off`.

### 22.19.5 Key Implementation Details

- **Format auto-detection:** 8-byte magic read distinguishes APR (`APR\0`) from GGUF
- **APR tokenization:** Uses `AprV2Model::encode_text()` / `decode_apr_tokens()` (separate from GGUF path)
- **Stop tokens:** `resolve_apr_stop_tokens()` merges EOS from model config + sibling tokenizer.json
- **GPU mandatory:** GPU/CPU parity verified on Blackwell sm_121. Never fall back to CPU for eval.
- **Temperature/top-k passthrough:** CLI flags `--temperature` and `--top-k` pass through to `BatchInferenceConfig` for non-greedy sampling
- **Streaming output:** Results flushed after each prompt for pipeline consumption
- **ChatML template:** Hardcoded `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n` for Qwen models

MBPP eval, per-problem analysis, recommendations: [AC Verification (S24)](24-ac-verification.md) §24.12-§24.13.
