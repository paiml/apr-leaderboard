#  Data Strategy

The model is only as good as the fine-tuning data. Key datasets for code leaderboards:

| Dataset | Size | Purpose | Source |
|---------|------|---------|--------|
| Code Instruct (curated) | 50K | Instruction-following for code | Self-curated from OSS repos |
| Code Reasoning | 20K | Chain-of-thought for complex problems | Synthetic from teacher model |
| Code Tests | 10K | Test-driven examples (input→test→code) | HumanEval/MBPP-style |
| Multilingual Code | 30K | Python/Rust/TS/Go/Java coverage | MultiPL-E format |
| Calibration | 128 | Wanda/SparseGPT calibration | Random code samples |

**Data preparation via apr CLI:**

```bash
# GATE: Validate teacher produces correct code BEFORE generating training data
apr eval teacher.apr --task classify --data humaneval.jsonl --json > teacher-baseline.json
# Verify teacher pass@1 meets minimum threshold (e.g., >60%) before proceeding

# Generate synthetic training data from validated teacher
apr chat teacher.apr --system "Generate code instruction pairs" \
    --batch instructions.txt --json > code-instruct-raw.jsonl

# Format validation
apr validate --data code-instruct-raw.jsonl --format jsonl

# GATE: Decontamination — remove any samples overlapping with eval benchmarks
# (HumanEval, MBPP problem descriptions must not appear in training data)
```

**Bootstrapping discipline:** Never generate training data from a teacher whose inference quality hasn't been verified. The pipeline is: import → eval teacher → generate data → validate data → train student.
