# Evaluation Protocol

Every recipe must be evaluated identically for fair comparison.

## 13.1 pass@k Computation

**Critical note on pass@k evaluation:** HumanEval and MBPP require *executing generated code* against test cases — not just token prediction. The pipeline is: (1) model generates k completions per problem, (2) completions are executed in a sandboxed environment, (3) pass@k is computed via the unbiased estimator.

The unbiased estimator for pass@k (Chen et al., 2021):

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where `n` = total completions generated, `c` = number that pass all tests, `k` = samples selected. This avoids biased estimation from sampling exactly k completions.

**apr-leaderboard eval flags that map to this:**

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
apr eval model.apr --dataset wikitext-2 --json > results/perplexity.json

# Step 2: Generate completions for code benchmarks
apr eval model.apr --task classify --data humaneval.jsonl --json > results/humaneval-completions.json
apr eval model.apr --task classify --data mbpp.jsonl --json > results/mbpp-completions.json

# Step 3: Execute completions in sandboxed environment (EXTERNAL)
# Using EvalPlus:
#   docker run -v ./results:/results evalplus/evalplus:latest \
#     --dataset humaneval --samples /results/humaneval-completions.json
# This produces pass@1, pass@10, pass@100 metrics.

# Step 4: Throughput benchmarking
apr bench model.apr --json > results/throughput.json

# Step 5: Cross-reference against HuggingFace
apr compare-hf model.apr --json > results/parity.json

# Step 6: Full QA gate before submission
apr qa model.apr --verbose
apr check model.apr
```

## 13.4 Evaluation Benchmarks (apr-leaderboard)

The `apr-leaderboard eval` command wraps the above steps for supported benchmarks:

```bash
# Run all HumanEval problems with 20 completions each, using structured CoT
apr-leaderboard eval --model models/qwen-7b.apr --benchmark humaneval \
    --samples 0 --n-samples 20 --prompt-strategy scot --output results/

# Quick sanity check: 10 problems, 1 completion each
apr-leaderboard eval --model models/qwen-7b.apr --benchmark humaneval --samples 10

# View results history
apr-leaderboard history --model qwen
```
