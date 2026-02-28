#  The `apr` CLI Toolchain

Every technique maps to a single shell command. This is the differentiator — our competitors use 500-line Python scripts; we use one-liners.

## 4.1 Import (HF → APR)

```bash
# Import from HuggingFace Hub — auto-detects architecture
apr import hf://Qwen/Qwen2.5-Coder-7B -o qwen-7b.apr --arch qwen2

# Import with quantization on ingest
apr import hf://Qwen/Qwen2.5-Coder-32B -o qwen-32b-q8.apr --quantize int8

# Import GGUF with provenance enforcement
apr import qwen-7b.gguf -o qwen-7b.apr --enforce-provenance
```

## 4.2 Evaluate (Baseline)

```bash
# Perplexity baseline
apr eval qwen-7b.apr --dataset wikitext-2 --threshold 20.0

# Classification eval with custom data
apr eval qwen-7b.apr --task classify --data humaneval.jsonl --json
```

## 4.3 Full Optimization Pipeline (preview)

```bash
# The complete leaderboard recipe in 6 commands (follows golden ordering §10):
apr import hf://Qwen/Qwen2.5-Coder-7B -o base.apr
apr distill teacher.apr --student base.apr --strategy progressive --temperature 3.0 -o distilled.apr
apr finetune distilled.apr --method qlora --rank 32 --data code-instruct.jsonl -o tuned.apr
apr merge tuned.apr variant-b.apr --strategy slerp -o merged.apr
apr prune merged.apr --method wanda --target-ratio 0.2 --calibration calib.jsonl -o pruned.apr
apr quantize pruned.apr --scheme int4 -o submit.apr
```
