# Data Strategy

The model is only as good as the fine-tuning data. Key datasets for code leaderboards:

| Dataset | Size | Purpose | Source | Format |
|---------|------|---------|--------|--------|
| Code Instruct (curated) | 50K | Instruction-following for code | Self-curated from OSS repos | JSONL (instruction, response) |
| Code Reasoning | 20K | Chain-of-thought for complex problems | Synthetic from teacher model | JSONL (problem, reasoning, code) |
| Code Tests | 10K | Test-driven examples (input→test→code) | HumanEval/MBPP-style | JSONL (prompt, tests, solution) |
| Multilingual Code | 30K | Python/Rust/TS/Go/Java coverage | MultiPL-E format | JSONL (language, prompt, solution) |
| Calibration | 128 | Wanda/SparseGPT calibration | Random code samples | JSONL (text) |

## 12.1 Decontamination Protocol

Training data MUST NOT overlap with evaluation benchmarks. This is critical for leaderboard integrity.

**n-gram decontamination:** Remove any training sample whose 10-gram overlap with any HumanEval/MBPP/BigCodeBench problem exceeds 50%. This is a hard gate — no exceptions.

```bash
# GATE: Decontamination check before training
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
