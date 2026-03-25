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
# GATE: Decontamination check before training
apr data decontaminate training.jsonl \
    --reference humaneval.jsonl mbpp.jsonl bigcodebench.jsonl \
    --ngram 10 --threshold 0.50 --json

# Or via Makefile:
make decontaminate DATA=data/instruct-corpus.jsonl
```

**Implementation:** `alimentar::quality::decontaminate` (alimentar#30)
wired into `apr data decontaminate` (aprender#415). Enforces AC-016
gate: fails if contamination rate >= 1%.

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
apr data decontaminate code-instruct-clean.jsonl \
    --reference humaneval.jsonl mbpp.jsonl --ngram 10 --threshold 0.50
```

**Bootstrapping discipline:** Never generate training data from a teacher whose inference quality hasn't been verified. The pipeline is: import → eval teacher → generate data → validate data → decontaminate → train student.

## 12.3 Preference Pair Generation (PMAT-014)

DPO alignment requires preference pairs: (prompt, chosen, rejected) triples where "chosen" is a correct completion and "rejected" is an incorrect one. We generate these from N-sampling eval results.

```bash
# Step 1: Run N-sampling eval (generates N completions per problem)
make eval-humaneval CHECKPOINT=checkpoints/model.apr NUM_SAMPLES=10 TEMPERATURE=0.8

# Step 2: Generate preference pairs from eval results
make generate-preference-pairs EVAL_WORK_DIR=/tmp/eval-work-dir
# Output: data/preference-pairs.jsonl

# Step 3: Use for DPO training
apr finetune checkpoint.apr --method dpo --data data/preference-pairs.jsonl
```

**Pair generation strategy:** For each problem with at least 1 passing and 1 failing sample, create all (passing, failing) pairs. A problem with 3 passing and 7 failing samples produces 21 preference pairs. This maximizes training signal from each eval run.

**Expected yield from 164 HumanEval problems at 85% pass@1 (N=10, T=0.8):**
- ~140 problems with at least 1 pass → usable for pairs
- ~120 problems with mixed pass/fail → source of pairs
- ~500-1000 preference pairs per eval run

**Implementation:** `scripts/generate-preference-pairs.sh` reads the eval work directory, re-tests each sample to classify pass/fail, and outputs JSONL.
