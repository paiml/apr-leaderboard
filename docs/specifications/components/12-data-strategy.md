# Data Strategy

The model is only as good as the fine-tuning data. Our primary data comes from four ground truth corpora in the paiml ecosystem.

## 12.0 Ground Truth Corpora (Tier 1)

Extracted via `make prep-data` → `apr data prep` (GH-7). These are high-quality,
hand-crafted Python implementations with full type annotations, docstrings,
and test coverage.

| Corpus | Raw Pairs | Description | Source Repo |
|--------|-----------|-------------|-------------|
| depyler | ~11,841 | Algorithms, data structures, CLI patterns, TDD examples | `~/src/depyler/` |
| hf-gtc | ~3,535 | HuggingFace production recipes (training, inference, RAG) | `~/src/hf-ground-truth-corpus/` |
| jax-gtc | ~58 | JAX numerical computing (autodiff, transforms, training) | `~/src/jax-ground-truth-corpus/` |
| vllm-gtc | ~81 | vLLM inference optimization (KV cache, sampling, serving) | `~/src/vllm-ground-truth-corpus/` |
| **Total** | **~15,494** | | |

**Extraction method:** AST parsing extracts function/class definitions with docstrings. Instruction = signature + docstring reformulated as natural language. Response = full source code. Filtered by response length (3–200 lines).

## 12.0.1 Supplemental Datasets (Tier 2)

| Dataset | Size | Purpose | Source | Format |
|---------|------|---------|--------|--------|
| Code Reasoning | 20K | Chain-of-thought for complex problems | Synthetic from teacher model | JSONL (problem, reasoning, code) |
| Code Tests | 10K | Test-driven examples (input→test→code) | HumanEval/MBPP-style | JSONL (prompt, tests, solution) |
| Multilingual Code | 30K | Python/Rust/TS/Go/Java coverage | MultiPL-E format | JSONL (language, prompt, solution) |
| Calibration | 128 | Wanda/SparseGPT calibration | Random code samples | JSONL (text) |

## 12.1 Decontamination Protocol

Training data MUST NOT overlap with evaluation benchmarks. This is critical for leaderboard integrity.

**n-gram decontamination:** Remove any training sample whose 10-gram overlap with any HumanEval/MBPP/BigCodeBench problem exceeds 50%. This is a hard gate — no exceptions.

```bash
# GATE: Decontamination check before training (GH-9: not yet implemented)
apr validate --data training.jsonl --decontaminate \
    --reference humaneval.jsonl mbpp.jsonl bigcodebench.jsonl \
    --ngram 10 --threshold 0.50 --json > decontamination-report.json

# Verify <1% of training samples flagged
```

**Time-based decontamination for LiveCodeBench:** Any problem published within 90 days of training data generation is excluded. LiveCodeBench's rolling nature makes this mandatory.

## 12.2 Data Preparation Pipeline

```bash
# GATE: Validate teacher produces correct code BEFORE generating training data
apr eval teacher.apr --task classify --data humaneval.jsonl --json > teacher-baseline.json
# Verify teacher pass@1 meets minimum threshold (e.g., >60%) before proceeding

# Generate synthetic training data from validated teacher
apr chat teacher.apr --system "Generate code instruction pairs" \
    --batch instructions.txt --json > code-instruct-raw.jsonl

# Format validation
apr validate --data code-instruct-raw.jsonl --format jsonl

# Quality scoring (alimentar)
alimentar quality code-instruct-raw.jsonl --min-score 80 -o code-instruct-clean.jsonl

# Decontamination gate
apr validate --data code-instruct-clean.jsonl --decontaminate \
    --reference humaneval.jsonl mbpp.jsonl --ngram 10 --threshold 0.50
```

**Bootstrapping discipline:** Never generate training data from a teacher whose inference quality hasn't been verified. The pipeline is: import → eval teacher → generate data → validate data → decontaminate → train student.
