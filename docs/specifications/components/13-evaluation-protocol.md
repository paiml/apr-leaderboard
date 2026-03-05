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
| `--prompt-strategy S` | Prompt formatting (standard, scot, few-shot, cgo, reflexion) |

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

The eval script handles: (1) benchmark download, (2) completion generation via `apr run --json --chat`, (3) markdown fence stripping, (4) python3/Docker sandbox execution with timeout, (5) Chen et al. unbiased pass@k computation, (6) JSON result output.
