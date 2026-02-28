# CLI Toolchain

Two CLIs work together: **`apr`** (upstream aprender — ML operations) and **`apr-leaderboard`** (this repo — orchestration). Every technique maps to a single shell command. Our competitors use 500-line Python scripts; we use one-liners.

## 6.1 The `apr` CLI (aprender)

The upstream `apr` binary provides all ML operations. `apr-leaderboard` calls these under the hood.

### 6.1.1 Import (HF → APR)

```bash
# Import from HuggingFace Hub — auto-detects architecture
apr import hf://Qwen/Qwen2.5-Coder-7B -o qwen-7b.apr --arch qwen2

# Import with quantization on ingest
apr import hf://Qwen/Qwen2.5-Coder-32B -o qwen-32b-q8.apr --quantize int8

# Import GGUF with provenance enforcement
apr import qwen-7b.gguf -o qwen-7b.apr --enforce-provenance
```

### 6.1.2 Evaluate (Baseline)

```bash
# Perplexity baseline
apr eval qwen-7b.apr --dataset wikitext-2 --threshold 20.0

# Classification eval with custom data
apr eval qwen-7b.apr --task classify --data humaneval.jsonl --json
```

### 6.1.3 Full Optimization Pipeline (preview)

```bash
# The complete leaderboard recipe in 6 commands (follows golden ordering §10):
apr import hf://Qwen/Qwen2.5-Coder-7B -o base.apr
apr distill teacher.apr --student base.apr --strategy progressive --temperature 3.0 -o distilled.apr
apr finetune distilled.apr --method qlora --rank 32 --data code-instruct.jsonl -o tuned.apr
apr merge tuned.apr variant-b.apr --strategy slerp -o merged.apr
apr prune merged.apr --method wanda --target-ratio 0.2 --calibration calib.jsonl -o pruned.apr
apr quantize pruned.apr --scheme int4 -o submit.apr
```

## 6.2 The `apr-leaderboard` CLI (this repo)

The orchestration layer that drives the pipeline. Each subcommand maps to one or more upstream `apr` operations.

| Subcommand | Maps to | Description |
|---|---|---|
| `convert` | `apr import` | Download HF model → `.apr` format |
| `eval` | `apr eval` | Run benchmark suite with pass@k metrics |
| `finetune` | `apr finetune` (entrenar) | LoRA/QLoRA fine-tuning |
| `distill` | `apr distill` | Knowledge distillation (teacher → student) |
| `merge` | `apr merge` | Model merging (SLERP, TIES, DARE, linear) |
| `prune` | `apr prune` | Structured/unstructured pruning |
| `quantize` | `apr quantize` | Post-training quantization |
| `compare` | `apr compare-hf` | Parity check against HF reference |
| `submit` | — | Format + push results to HF leaderboard |
| `benchmarks` | — | List available benchmark suites |
| `history` | — | Show past evaluation results |
| `pipeline` | all of the above | Config-driven end-to-end pipeline |

### 6.2.1 Convert

```bash
# Convert a HuggingFace model to .apr format
apr-leaderboard convert --model-id Qwen/Qwen2.5-Coder-7B

# With custom output and quantization
apr-leaderboard convert --model-id Qwen/Qwen2.5-Coder-7B --output models/ --quantization int8
```

### 6.2.2 Eval

```bash
# Run HumanEval with defaults (standard prompt, 1 sample)
apr-leaderboard eval --model models/qwen-7b.apr --benchmark humaneval

# Full benchmark with structured CoT and best-of-20 selection
apr-leaderboard eval --model models/qwen-7b.apr --benchmark humaneval \
    --samples 0 --prompt-strategy scot --n-samples 20

# Subset evaluation on BigCodeBench
apr-leaderboard eval --model models/qwen-7b.apr --benchmark bigcodebench \
    --samples 100 --output results/
```

**Prompt strategies** (§8.3):

| Strategy | Flag value | Description |
|---|---|---|
| Standard | `standard` / `default` | Direct prompt, no special formatting |
| Structured CoT | `scot` / `structured-cot` | Step-by-step reasoning before code |
| Few-shot | `few-shot` / `fewshot` | Include solved examples in prompt |
| Code Gen Opt | `cgo` / `code-gen-opt` | Optimization-focused generation |
| Reflexion | `reflexion` / `reflect` | Generate → test → reflect → regenerate |

**N-samples:** `--n-samples N` generates N completions per problem, selects the best (maximizes pass@k). Default: 1.

### 6.2.3 Finetune

```bash
# LoRA fine-tune with defaults (rank 16, lr 1e-4, 3 epochs)
apr-leaderboard finetune --model models/qwen-7b.apr --dataset data/code-instruct.jsonl

# Custom LoRA config
apr-leaderboard finetune --model models/qwen-7b.apr --dataset data/code-instruct.jsonl \
    --rank 32 --lr 0.001 --epochs 5
```

### 6.2.4 Distill

```bash
# Progressive distillation (recommended for code models)
apr-leaderboard distill --teacher teacher-32b.apr --student student-7b.apr \
    --strategy progressive --temperature 3.0 --alpha 0.7 -o distilled-7b.apr

# Ensemble distillation from multiple teachers
apr-leaderboard distill --teacher ensemble.apr --student student-7b.apr \
    --strategy ensemble -o distilled-7b.apr
```

**Strategies:** `standard` (KL divergence), `progressive` (curriculum learning), `ensemble` (multi-teacher).

### 6.2.5 Merge

```bash
# SLERP merge of two models
apr-leaderboard merge model-a.apr model-b.apr --strategy slerp -o merged.apr

# TIES merge of three models
apr-leaderboard merge a.apr b.apr c.apr --strategy ties -o merged.apr
```

**Strategies:** `slerp`, `ties` (TIES-Merging), `dare` (DARE-TIES), `linear` (linear average).

### 6.2.6 Prune

```bash
# Wanda pruning with 20% sparsity (default)
apr-leaderboard prune --model tuned.apr --method wanda --target-ratio 0.2 -o pruned.apr

# SparseGPT with 30% sparsity
apr-leaderboard prune --model tuned.apr --method sparsegpt --target-ratio 0.3 -o pruned.apr
```

**Methods:** `wanda` (default), `magnitude`, `sparsegpt`. Target ratio: 0.0–1.0 (exclusive).

### 6.2.7 Quantize

```bash
# INT4 quantization (default, best compression)
apr-leaderboard quantize --model pruned.apr --scheme int4 -o submit.apr

# Q6K quantization (better quality, larger size)
apr-leaderboard quantize --model pruned.apr --scheme q6k -o submit.apr
```

**Schemes:** `int4`, `int8`, `q4k`, `q5k`, `q6k`.

### 6.2.8 Compare

```bash
# Check parity against HuggingFace reference implementation
apr-leaderboard compare --model models/qwen-7b.apr
```

### 6.2.9 Submit

```bash
# Submit results to the Open LLM Leaderboard
apr-leaderboard submit --results results/humaneval_20260228.json \
    --model-id paiml/qwen-coder-7b-apr

# Submit to BigCodeBench leaderboard
apr-leaderboard submit --results results/bigcodebench_20260228.json \
    --model-id paiml/qwen-coder-7b-apr --leaderboard bigcode
```

### 6.2.10 Pipeline (config-driven)

```bash
# Run entire pipeline from a TOML config file
apr-leaderboard pipeline --config configs/qwen-coder-7b.toml
```

Example pipeline config:

```toml
[model]
model_id = "Qwen/Qwen2.5-Coder-7B"
quantization = "fp16"

[eval]
benchmarks = ["humaneval", "mbpp", "bigcodebench"]
samples = 0  # full benchmark

[finetune]
enabled = true
dataset = "data/code-instruct.jsonl"
rank = 16
lr = 1e-4
epochs = 3

[submit]
model_id = "paiml/qwen-coder-7b-apr"
leaderboard = "open-llm-leaderboard"
```

## 6.3 CLI Surface Mapping

The full mapping between `apr-leaderboard` orchestration and `apr` ML operations:

```
apr-leaderboard pipeline --config pipeline.toml
    │
    ├── apr-leaderboard convert  ──►  apr import hf://... -o base.apr
    ├── apr-leaderboard distill  ──►  apr distill teacher.apr --student base.apr ...
    ├── apr-leaderboard finetune ──►  apr finetune base.apr --method qlora ...
    ├── apr-leaderboard merge    ──►  apr merge a.apr b.apr --strategy slerp ...
    ├── apr-leaderboard prune    ──►  apr prune model.apr --method wanda ...
    ├── apr-leaderboard quantize ──►  apr quantize model.apr --scheme int4 ...
    ├── apr-leaderboard eval     ──►  apr eval model.apr --benchmark humaneval ...
    └── apr-leaderboard submit   ──►  (HTTP POST to HF Hub API)
```
