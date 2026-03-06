# Acceptance Criteria Verification

Detailed verification findings for individual acceptance criteria. Split from [Dogfooding Findings (S22)](22-dogfooding-findings.md) for file size compliance.

## 24.1 Compile to Binary (AC-026)

`apr compile` creates a standalone launcher binary:

```bash
apr compile checkpoints/qwen2.5-coder-1.5b-instruct-q4k.apr \
    --release --strip -o checkpoints/qwen-1.5b-binary
```

| Component | Size |
|-----------|------|
| Binary (runtime) | 671 KiB |
| Model (embedded ref) | 1.04 GiB |
| **Total** | **~1.04 GiB** |

The binary shows model info and accepts `--prompt` but reports "Full inference dispatch requires the aprender runtime." The compile command creates a launcher that packages the model reference, but full inference requires realizar crates to be statically linked. AC-026 target was <1GB — the runtime binary itself (671 KiB) is well under, but with model data it's 1.04 GiB. This is a GGUF Q4K model; INT4 quantization might bring it under 1GB.

**LTO note:** `--lto` flag conflicts with `embed-bitcode=no` in the generated Cargo project. Use `--release --strip` without `--lto`.

## 24.2 Throughput Benchmarks

`apr bench` results on CPU (no GPU):

| Model | Backend | Tok/s | TTFT | Median Latency | Iterations |
|-------|---------|-------|------|----------------|------------|
| Qwen2.5-Coder-1.5B-Instruct Q4K | CPU | 2.5 | 385ms | 12,982ms | 5 |

TTFT = time to first token. CPU throughput is expected to be low — wgpu GPU inference would significantly improve these numbers.

## 24.3 Structured Prompting (AC-019)

Tested `standard` vs `scot` (structured chain-of-thought) prompt strategies on HumanEval problem 0 (`has_close_elements`):

| Strategy | Output | Code Correct | Notes |
|---|---|---|---|
| `standard` | Direct code (O(n²) brute force) + trailing text | Yes | `extract_python_code()` strips trailing text |
| `scot` | Step-by-step reasoning (sort + adjacent) | No code produced | Reasoning consumed all 512 tokens |

**Finding:** SCoT produces reasoning before code as expected, and the reasoning is correct (identified O(n log n) optimization via sorting). However, on 1.5B models with 512-token budgets, reasoning text consumes too many tokens — the model doesn't reach code generation.

**Recommendation:** For SCoT to work on small models, either:
1. Increase `MAX_TOKENS` to 1024+ (doubles eval time per problem)
2. Use SCoT only on 7B+ models where reasoning is more concise
3. Post-process to extract code from mixed reasoning+code output

AC-019 status: Structured prompting **does** produce reasoning before code. Quality improvement pending larger model evaluation.

## 24.4 HF Parity Check (AC-014)

`apr compare-hf` on GGUF-imported model vs HF reference:

```bash
apr compare-hf --hf "Qwen/Qwen2.5-Coder-1.5B-Instruct" --json \
    checkpoints/qwen2.5-coder-1.5b-instruct-q4k.apr
```

**Result:** 0 tensor comparisons performed. The GGUF Q4K model uses Q4K/Q6K dtypes while HF reference uses FP16/BF16 — no tensors have matching dtypes to compare element-wise.

**AC-014 status:** Cannot verify <5% parity gap via `compare-hf` on GGUF imports. Parity must be verified indirectly via benchmark scores or perplexity comparison.

## 24.5 MBPP Function Name Extraction

**Problem:** MBPP eval showed 5% pass rate (1/20) despite the model generating correct code.

**Five Whys:**
1. Why 5% pass rate? Tests fail with `NameError: name 'min_cost' is not defined`
2. Why NameError? Model defines `solve()` but test asserts `min_cost(...)`
3. Why wrong function name? Prompt didn't specify the expected function name
4. Why no name in prompt? `build_instruction()` didn't extract names from MBPP test_list
5. Why not? MBPP format was only partially understood

**Fix:** Extract function name from first test assertion via `grep -oP '(?<=assert )\w+'` and include it in the prompt: "Write a Python function called \`min_cost\` to solve this task."

## 24.6 Wanda Pruning on GGUF Models (AC-008)

`apr prune --method wanda --target-ratio 0.1` on Qwen2.5-Coder-1.5B-Instruct Q4K:

| Metric | Value |
|--------|-------|
| Input size | 1.04 GiB (Q4K) |
| Output size | 6.62 GiB (FP32, dequantized) |
| Sparsity | 10.0% (matches target) |

**Key finding:** Wanda pruning dequantizes Q4K → FP32, inflating output 6.4x. Pruned model loses embedded tokenizer and config. Needs prune → re-quantize → re-package pipeline (GH-14).

## 24.7 Submit Script Preflight Fix

**Problem:** `scripts/submit.sh` pmat check always failed even when COMPLIANT.

**Root cause:** pmat returns exit code 2 for COMPLIANT-with-advisories. Script treated any non-zero as failure.

**Fix:** Accept both exit 0 (clean) and exit 2 (advisories-only) as PASS.

## 24.8 Pipeline Verification (2026-03-05)

`make verify`: 19/19 subcommands OK, 17 YAML configs, 7 scripts. Eval script handles HumanEval (function completion), MBPP (assert-based test_list), and BigCodeBench (instruct mode) with benchmark-specific test assembly. Chen et al. unbiased pass@k estimator with per-task sample tracking. `make validate`: all configs pass `bashrs` lint.
