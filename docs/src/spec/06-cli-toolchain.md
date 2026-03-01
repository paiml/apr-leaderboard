# CLI Toolchain

Two CLIs work together: **`apr`** (upstream aprender — ML operations) and **`apr-leaderboard`** (this repo — orchestration). Every technique maps to a single shell command. Our competitors use 500-line Python scripts; we use one-liners.

## 6.1 The `apr` CLI (aprender)

The upstream `apr` binary provides all ML operations. `apr-leaderboard` calls these under the hood.

## 6.1.1 Import (HF → APR)

```bash
# Import from HuggingFace Hub — auto-detects architecture
apr import hf://Qwen/Qwen2.5-Coder-7B -o qwen-7b.apr --arch qwen2

# Import with quantization on ingest
apr import hf://Qwen/Qwen2.5-Coder-32B -o qwen-32b-q8.apr --quantize int8

# Import GGUF with provenance enforcement
apr import qwen-7b.gguf -o qwen-7b.apr --enforce-provenance
```

## 6.1.2 Evaluate (Baseline)

```bash
# Perplexity baseline
apr eval qwen-7b.apr --dataset wikitext-2 --threshold 20.0

# Classification eval with custom data
apr eval qwen-7b.apr --task classify --data humaneval.jsonl --json
```

## 6.1.3 Full Optimization Pipeline (preview)

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

| Subcommand | Maps to | Description | Status |
|---|---|---|---|
| `convert` | `apr import` | Download HF model → `.apr` format | Scaffolded |
| `eval` | `apr eval` | Run benchmark suite with pass@k metrics | **Wired** (`entrenar::eval::pass_at_k`) |
| `finetune` | `apr finetune` (entrenar) | LoRA/QLoRA fine-tuning | Scaffolded |
| `distill` | `apr distill` | Knowledge distillation (teacher → student) | **Wired** (`entrenar::distill`) |
| `merge` | `apr merge` | Model merging (SLERP, TIES, DARE, linear) | **Wired** (`entrenar::merge` + `apr_bridge`) |
| `prune` | `apr prune` | Structured/unstructured pruning | Scaffolded |
| `quantize` | `apr quantize` | Post-training quantization | Scaffolded |
| `compare` | `apr compare-hf` | Parity check against HF reference | Scaffolded |
| `submit` | — | Format + push results to HF leaderboard | Scaffolded |
| `benchmarks` | — | List available benchmark suites | Complete |
| `history` | — | Show past evaluation results | Complete |
| `pipeline` | all of the above | Config-driven end-to-end pipeline (12 stages) | Scaffolded |
| `align` | `apr align` (entrenar) | DPO/ORPO preference optimization | Scaffolded |
| `validate` | `alimentar` | Data decontamination checking | Scaffolded |
| `tune` | `apr tune` (entrenar) | HPO: TPE/grid/random strategies | Scaffolded |
| `run` | `apr run` (realizar) | Inference with speculative decoding | Scaffolded |
| `chat` | `apr chat` (realizar) | Batch generation / chat completions | Scaffolded |
| `check` | `apr check` | Validate APR format and integrity | **Wired** (`aprender::format::v2::AprV2Reader`) |
| `compile` | `apr compile` | Compile model to standalone binary | Scaffolded |
| `export` | — | SafeTensors/GGUF metadata export | Scaffolded |
| `acceptance` | — | List/verify 27 acceptance criteria (§18) | **Wired** (`provable_contracts`) |

## 6.2.1 Convert

```bash
# Convert a HuggingFace model to .apr format
apr-leaderboard convert --model-id Qwen/Qwen2.5-Coder-7B

# With custom output and quantization
apr-leaderboard convert --model-id Qwen/Qwen2.5-Coder-7B --output models/ --quantization int8
```

## 6.2.2 Eval

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

## 6.2.3 Finetune

```bash
# LoRA fine-tune with defaults (method=lora, rank 16, lr 1e-4, 3 epochs)
apr-leaderboard finetune --model models/qwen-7b.apr --dataset data/code-instruct.jsonl

# QLoRA with custom config and explicit output
apr-leaderboard finetune --model models/qwen-7b.apr --dataset data/code-instruct.jsonl \
    --method qlora --rank 32 --lr 0.001 --epochs 5 -o tuned.apr

# Full fine-tune (all parameters)
apr-leaderboard finetune --model models/qwen-7b.apr --dataset data/code-instruct.jsonl \
    --method full --lr 1e-5 --epochs 2
```

**Methods:** `lora` (default), `qlora` (quantized LoRA), `full` (all parameters).

| Flag | Default | Description |
|---|---|---|
| `--method` | `lora` | Fine-tuning method |
| `--rank` | `16` | LoRA rank |
| `--lr` | `1e-4` | Learning rate |
| `--epochs` | `3` | Number of epochs |
| `-o` | `<input>_finetuned.apr` | Output model path |

## 6.2.4 Distill

```bash
# Progressive distillation (recommended for code models)
apr-leaderboard distill --teacher teacher-32b.apr --student student-7b.apr \
    --strategy progressive --temperature 3.0 --alpha 0.7 -o distilled-7b.apr

# Distillation with training data and custom epochs
apr-leaderboard distill --teacher teacher-32b.apr --student student-7b.apr \
    --strategy progressive --epochs 10 --data code-corpus.jsonl -o distilled-7b.apr
```

**Strategies:** `standard` (KL divergence), `progressive` (curriculum learning), `ensemble` (multi-teacher).

| Flag | Default | Description |
|---|---|---|
| `--strategy` | `progressive` | Distillation strategy |
| `--temperature` | `3.0` | Softmax temperature |
| `--alpha` | `0.7` | Mixing coefficient (0=student, 1=teacher) |
| `--epochs` | `5` | Number of distillation epochs |
| `--data` | — | Training data corpus |

## 6.2.5 Merge

```bash
# SLERP merge of two models with weights
apr-leaderboard merge model-a.apr model-b.apr --strategy slerp \
    --weights 0.7,0.3 -o merged.apr

# TIES merge with base model and density
apr-leaderboard merge a.apr b.apr --strategy ties \
    --base-model base.apr --density 0.2 -o merged.apr

# DARE merge with drop rate
apr-leaderboard merge a.apr b.apr --strategy dare \
    --base-model base.apr --drop-rate 0.3 -o merged.apr
```

**Strategies:** `slerp`, `ties` (TIES-Merging), `dare` (DARE-TIES), `linear` (linear average).

| Flag | Default | Description |
|---|---|---|
| `--strategy` | `slerp` | Merge strategy |
| `--weights` | — | Comma-separated merge weights (e.g., `0.7,0.3`) |
| `--base-model` | — | Base model for task-vector strategies (ties, dare) |
| `--density` | — | Sparse mask density for TIES (0.0–1.0) |
| `--drop-rate` | — | Drop rate for DARE (0.0–1.0) |

## 6.2.6 Prune

```bash
# Wanda pruning with 20% sparsity (default)
apr-leaderboard prune --model tuned.apr --method wanda --target-ratio 0.2 -o pruned.apr

# SparseGPT with calibration data
apr-leaderboard prune --model tuned.apr --method sparsegpt --target-ratio 0.3 \
    --calibration calib-code.jsonl -o pruned.apr
```

**Methods:** `wanda` (default), `magnitude`, `sparsegpt`. Target ratio: 0.0–1.0 (exclusive).

| Flag | Default | Description |
|---|---|---|
| `--method` | `wanda` | Pruning method |
| `--target-ratio` | `0.2` | Target sparsity ratio |
| `--calibration` | — | Calibration dataset for Wanda/SparseGPT |

## 6.2.7 Quantize

```bash
# INT4 quantization (default, best compression)
apr-leaderboard quantize --model pruned.apr --scheme int4 -o submit.apr

# Q6K quantization with calibration data
apr-leaderboard quantize --model pruned.apr --scheme q6k \
    --calibration calib-code.jsonl -o submit.apr
```

**Schemes:** `int4`, `int8`, `q4k`, `q5k`, `q6k`.

| Flag | Default | Description |
|---|---|---|
| `--scheme` | `int4` | Quantization scheme |
| `--calibration` | — | Calibration dataset for quality-aware quantization |

## 6.2.8 Compare

```bash
# Check parity against HuggingFace reference implementation
apr-leaderboard compare --model models/qwen-7b.apr
```

## 6.2.9 Submit

```bash
# Submit results to the Open LLM Leaderboard
apr-leaderboard submit --results results/humaneval_20260228.json \
    --model-id paiml/qwen-coder-7b-apr

# Submit to BigCodeBench leaderboard
apr-leaderboard submit --results results/bigcodebench_20260228.json \
    --model-id paiml/qwen-coder-7b-apr --leaderboard bigcode
```

## 6.2.10 Pipeline (config-driven)

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
    ├── apr-leaderboard validate ──►  alimentar decontamination check
    ├── apr-leaderboard convert  ──►  apr import hf://... -o base.apr
    ├── apr-leaderboard distill  ──►  entrenar::distill (DistillationLoss, ProgressiveDistiller)
    ├── apr-leaderboard finetune ──►  apr finetune base.apr --method qlora ...
    ├── apr-leaderboard align    ──►  DPO/ORPO preference optimization
    ├── apr-leaderboard merge    ──►  entrenar::merge (slerp_merge, ensemble_merge) + apr_bridge
    ├── apr-leaderboard tune     ──►  HPO (TPE/grid/random)
    ├── apr-leaderboard prune    ──►  apr prune model.apr --method wanda ...
    ├── apr-leaderboard quantize ──►  apr quantize model.apr --scheme int4 ...
    ├── apr-leaderboard eval     ──►  entrenar::eval::pass_at_k + prompt strategies
    ├── apr-leaderboard compile  ──►  apr compile --release --lto --strip
    └── apr-leaderboard submit   ──►  (HTTP POST to HF Hub API)
```
