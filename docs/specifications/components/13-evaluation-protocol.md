# Evaluation Protocol

Every recipe must be evaluated identically for fair comparison.

## 13.1 pass@k Computation

**Critical note on pass@k evaluation:** HumanEval and MBPP require *executing generated code* against test cases — not just token prediction. The pipeline is: (1) model generates k completions per problem, (2) completions are executed in a sandboxed environment, (3) pass@k is computed via the unbiased estimator.

The unbiased estimator for pass@k (Chen et al., 2021):

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where `n` = total completions generated, `c` = number that pass all tests, `k` = samples selected. This avoids biased estimation from sampling exactly k completions.

**Implementation:** `scripts/eval-pass-at-k.sh` implements the Chen et al. estimator in bash/awk (log-space computation). The upstream `entrenar::eval::pass_at_k(n, c, k)` provides a Rust implementation validated by a provable-contracts YAML (`contracts/pass-at-k.yaml`) with 3 proof obligations (bound [0,1], monotonicity, pass@1 equivalence) and 3 falsification tests.

**Eval parameters:**

| Flag | Effect |
|---|---|
| `--samples N` | Number of benchmark problems to evaluate (0 = all) |
| `--n-samples N` | Completions per problem (for pass@k, best-of-N selection) |
| `--prompt-strategy S` | Prompt formatting (standard, scot, few-shot, cgo) |

## 13.2 Code Execution Sandbox

aprender does not include a code execution sandbox. Generated completions must be evaluated externally via one of:

1. **EvalPlus harness** (recommended): Docker-based sandbox that runs Python completions against augmented test suites (80x more tests than vanilla HumanEval)
2. **Custom WASM sandbox**: CPython compiled to WASM for isolated execution (see Open Question §21.14)
3. **Direct Docker**: `docker run --network=none --memory=512m --timeout=10s python:3.11 -c "$CODE"`

## 13.3 Evaluation Steps

```bash
# Step 1: Perplexity baseline (pure inference, no code execution needed)
make eval-perplexity CHECKPOINT=checkpoints/model.apr

# Step 2: Code benchmark evaluation (generate + execute + score)
# Each problem: apr run → strip markdown fences → python3/Docker sandbox → pass@k
make eval-humaneval CHECKPOINT=checkpoints/model.apr
make eval-mbpp CHECKPOINT=checkpoints/model.apr
make eval-bigcodebench CHECKPOINT=checkpoints/model.apr

# Step 3: Throughput benchmarking
apr bench checkpoints/model.apr --json > results/throughput.json

# Step 4: Cross-reference against HuggingFace
apr compare-hf checkpoints/model.apr --json > results/parity.json

# Step 5: Full QA gate before submission
apr qa checkpoints/model.apr --verbose
apr check checkpoints/model.apr
```

**Sandbox boundary (§5.3):** Code execution uses python3 (preferred) or Docker (`--network=none --memory=512m`) as an external dependency. This is the only non-sovereign step in the pipeline.

## 13.4 Evaluation via Makefile Targets

The eval pipeline is driven by `scripts/eval-pass-at-k.sh` via Makefile targets:

```bash
# Run all HumanEval problems with 1 completion each (default)
make eval-humaneval CHECKPOINT=checkpoints/qwen-7b.apr

# 20 completions per problem with structured CoT prompting
make eval-humaneval CHECKPOINT=checkpoints/qwen-7b.apr \
    NUM_SAMPLES=20 PROMPT_STRATEGY=scot MAX_TOKENS=1024

# Full benchmark suite (HumanEval + MBPP + BigCodeBench)
make eval-all CHECKPOINT=checkpoints/qwen-7b.apr

# View results history
make results-history
```

The eval script handles: (1) benchmark download, (2) completion generation via `apr run --batch-jsonl` (batch mode, auto-detected) or `apr run --json --chat` (worker mode), (3) markdown fence stripping + trailing text extraction, (4) python3/Docker sandbox execution with timeout, (5) Chen et al. unbiased pass@k computation, (6) JSON result output.

## 13.5 N-Sampling for pass@k (PMAT-003)

When `NUM_SAMPLES > 1`, the eval pipeline generates N completions per problem using temperature sampling:

```bash
# Generate 10 samples per HumanEval problem with temperature=0.8
make eval-humaneval CHECKPOINT=checkpoints/qwen-7b.apr \
    NUM_SAMPLES=10 TEMPERATURE=0.8
```

**Implementation details:**
- Batch mode duplicates each prompt N times (task_id format: `{idx}_s{sample}`)
- Temperature > 0 automatically enables top-k=40 sampling (greedy for T=0)
- Each sample is tested independently in the sandbox
- Results: `task_id  N  num_passed` (TSV) → Chen et al. estimator
- `pass@1` with N>1 gives the unbiased estimate: `E[1 - (n-c)/n]`
- `pass@10` requires `N >= 10` and gives: `E[1 - C(n-c,10)/C(n,10)]`

**Recommended configurations:**

| Configuration | N | Temperature | Top-k | Use Case |
|---|---|---|---|---|
| Greedy (default) | 1 | 0.0 | 1 | Deterministic baseline |
| pass@1 (unbiased) | 10 | 0.8 | 40 | Publication-grade pass@1 |
| pass@10 | 100 | 0.8 | 40 | Pass@10 for leaderboard |

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `APR_BATCH_MODE` | `auto` | Batch mode: `auto` (detect), `on` (force), `off` (disable) |

## 13.5 Instruct Model Post-Processing

Instruct models (via `--chat`) often append conversational text after generating correct Python code — e.g., "Human\n...", "**Explanation**:...", or markdown headers. This trailing text causes Python syntax errors in the sandbox.

The eval script applies two post-processing steps to all completions:

1. **`strip_markdown_fences()`** — Removes ` ```python ` / ` ``` ` wrappers
2. **`extract_python_code()`** — Stops at lines that are clearly not Python: `Human`, `Assistant`, `User`, `**...`, `###`, `---`

This is critical for instruct model evaluation. Without it, valid completions fail due to trailing conversational text (observed: 0% → ~70% pass rate on Qwen2.5-Coder-1.5B-Instruct).

## 13.6 Batch Inference Mode

For large eval suites (164 HumanEval + 974 MBPP problems), per-invocation model loading dominates wall-clock time. On gx10 (Blackwell sm_121), each `apr run` invocation incurs ~80s of CUDA JIT compilation overhead.

**Batch mode** (`--batch-jsonl`) loads the model and compiles CUDA kernels once, then processes all prompts sequentially:

```bash
# Prepare JSONL input (one prompt per line)
jq -c '{prompt: .prompt, task_id: .task_id, max_tokens: 512}' problems/*.json > batch.jsonl

# Run batch inference (model loads once, ~80s JIT amortized across all prompts)
apr run checkpoints/model.apr --batch-jsonl batch.jsonl --max-tokens 512 --verbose

# Output: JSONL with per-prompt results (text, tokens_generated, tok_per_sec, inference_ms, used_gpu)
```

**Performance impact:**

| Mode | Model Load | Per-Problem Overhead | 164 Problems (HumanEval) |
|------|-----------|---------------------|--------------------------|
| Sequential `apr run` | ~80s × 164 | ~80s JIT + inference | ~3.6 hours JIT alone |
| Batch `--batch-jsonl` | ~80s × 1 | inference only | ~80s JIT + inference time |

Auto-detects APR vs GGUF format. GPU is mandatory for eval. On Blackwell sm_121, GPU is blocked by parity gate (GH-559). Never bypass the gate — fix the root cause. Results stream as JSONL (one line per prompt, flushed after each).

## 13.7 MBPP Function Name Extraction

MBPP test assertions reference specific function names (e.g., `assert min_cost(...) == 42`). If the model generates a function with a different name, all tests fail even if the logic is correct.

The eval script extracts the expected function name from the first test assertion:

```bash
func_name="$(jq -r '.test_list[0]' <<< "$problem_json" | grep -oP '(?<=assert )\w+')"
```

This is included in the prompt: *"Write a Python function called \`min_cost\` to solve this task."*

Additionally, test assertions from `test_list` are appended to the prompt as examples, giving the model the exact function signature, argument types, and expected output format.

**Impact:** Without function name extraction or test assertions, MBPP pass rate was 5%. With function name only: 50.80%. With function name + test assertions: **76.20%** (381/500). The 25.4pp improvement from test assertions confirms that MBPP requires explicit I/O examples for strong performance.
