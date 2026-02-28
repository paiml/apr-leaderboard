# Evaluation Protocol

Every recipe must be evaluated identically for fair comparison.

**Critical note on pass@k evaluation:** HumanEval and MBPP require *executing generated code* against test cases — not just token prediction. The pipeline is: (1) model generates k completions per problem, (2) completions are executed in a sandboxed environment, (3) pass@k is computed via the unbiased estimator. aprender does not include a code execution sandbox; generated completions must be evaluated externally (e.g., via containerized Python execution using the EvalPlus harness).

```bash
# Step 1: Perplexity baseline (pure inference, no code execution needed)
apr eval model.apr --dataset wikitext-2 --json > results/perplexity.json

# Step 2: Generate completions for code benchmarks
apr eval model.apr --task classify --data humaneval.jsonl --json > results/humaneval-completions.json
apr eval model.apr --task classify --data mbpp.jsonl --json > results/mbpp-completions.json

# Step 3: Execute completions in sandboxed environment (EXTERNAL)
# This step requires a code execution harness — not provided by apr.
# Use EvalPlus or a custom Docker-based sandbox to run generated code
# against test cases and compute pass@k.

# Step 4: Throughput benchmarking
apr bench model.apr --json > results/throughput.json

# Step 5: Cross-reference against HuggingFace
apr compare-hf model.apr --json > results/parity.json

# Step 6: Full QA gate before submission
apr qa model.apr --verbose
apr check model.apr
```
