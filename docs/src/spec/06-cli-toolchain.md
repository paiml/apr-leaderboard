# CLI Toolchain

Two layers work together: **`apr`** (upstream aprender — ML operations) and **`make`** (this repo — orchestration via Makefile + shell scripts). Every technique maps to a single shell command. Our competitors use 500-line Python scripts; we use one-liners.

## 6.1 The `apr` CLI (aprender)

The upstream `apr` binary provides all ML operations. The Makefile and shell scripts call these under the hood.

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

## 6.1.3 Instruction Fine-tuning (GH-371)

```bash
# Instruction fine-tuning with LoRA on Q/V projections
apr finetune model.apr --task instruct --data instruct.jsonl --epochs 3 --rank 16

# QLoRA on consumer GPU (NF4 base + FP16 adapters, ~4.5 GB VRAM)
apr finetune model.apr --task instruct --method qlora --quantize-nf4 \
    --data instruct.jsonl --rank 16 --vram 8 --max-seq-len 512

# Multi-adapter concurrent training (GPU-SHARE)
apr finetune model.apr --task instruct --method qlora --quantize-nf4 \
    --adapters-config adapters.toml

# With experimental MPS (multi-process GPU sharing)
apr finetune model.apr --task instruct --experimental-mps --gpu-share 50

# Plan-only mode (shows config without training)
apr finetune --task instruct --model-size 7B --plan
```

**Corpus format (JSONL):**
```json
{"instruction": "Write a function that...", "response": "def foo():\n    ..."}
{"instruction": "...", "response": "...", "system": "You are...", "metadata": {"source": "depyler"}}
```

**Adapters config format (TOML):**
```toml
[[adapter]]
data = "data/corpus-a.jsonl"
checkpoint = "checkpoints/adapter-a"
label = "code-review"
rank = 16
learning_rate = 0.0002
```

**Contracts:**
- F-INST-001: Non-empty instruction and response
- F-INST-002: Cross-entropy loss computed only on response tokens
- F-INST-003: Perplexity reported per epoch
- F-INST-004: Qwen chat template (`<|im_start|>` / `<|im_end|>`)
- GPU-SHARE-002: VRAM reservation via ledger before allocation

## 6.1.4 Full Optimization Pipeline (preview)

```bash
# The complete leaderboard recipe in 6 commands (follows golden ordering §10):
apr import hf://Qwen/Qwen2.5-Coder-7B -o base.apr
apr distill teacher.apr --student base.apr --strategy progressive --temperature 3.0 -o distilled.apr
apr finetune distilled.apr --method qlora --rank 32 --data code-instruct.jsonl -o tuned.apr
apr merge tuned.apr variant-b.apr --strategy slerp -o merged.apr
apr prune merged.apr --method wanda --target-ratio 0.2 --calibration calib.jsonl -o pruned.apr
apr quantize pruned.apr --scheme int4 -o submit.apr
```

## 6.2 The `make` Orchestration Layer (this repo)

The orchestration layer that drives the pipeline. Each Makefile target maps to one or more `apr` CLI subcommands or shell scripts.

| Make Target | Calls | Description |
|---|---|---|
| `make import` | `apr import` | Download HF model → `.apr` format |
| `make prep-data` | `scripts/prep-instruct-data.py` | Extract instruct pairs from ground truth corpora |
| `make eval-humaneval` | `scripts/eval-pass-at-k.sh` | Generate completions → sandbox execute → pass@k |
| `make eval-mbpp` | `scripts/eval-pass-at-k.sh` | Same pipeline, MBPP dataset |
| `make eval-bigcodebench` | `scripts/eval-pass-at-k.sh` | Same pipeline, BigCodeBench dataset |
| `make eval-all` | `scripts/eval-pass-at-k.sh` × 3 | All benchmarks sequentially |
| `make eval-perplexity` | `apr eval --dataset wikitext-2` | Perplexity baseline |
| `make finetune-instruct` | `apr finetune --task instruct` | Instruction LoRA fine-tuning (GH-371) |
| `make finetune` | `apr finetune` | Classification LoRA/QLoRA fine-tuning |
| `make distill` | `apr distill` | Knowledge distillation (teacher → student) |
| `make merge` | `apr merge` | Model merging (SLERP, TIES, DARE, linear) |
| `make prune` | `apr prune` | Structured/unstructured pruning |
| `make quantize` | `apr quantize` | Post-training quantization |
| `make compile` | `apr compile` | Compile model to standalone binary |
| `make check` | `apr check` | Validate APR format and integrity |
| `make inspect` | `apr inspect` | Model inspection |
| `make export` | `apr export` | SafeTensors/GGUF export |
| `make publish` | `scripts/submit.sh` | Export + model card + HF Hub upload |
| `make model-card` | `apr eval --generate-card` | Generate model card |
| `make pipeline` | `scripts/pipeline.sh` | Config-driven end-to-end pipeline (12 stages) |
| `make pipeline-plan` | `scripts/pipeline.sh --plan` | Dry-run: validate config, show commands |
| `make verify` | smoke-tests all `apr` subcommands | Validate `apr` CLI installation |
| `make dogfood` | CLI + config validation | End-to-end smoke test |

## 6.2.1 Import

```bash
# Import a HuggingFace model to .apr format
make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

# Import with custom output path
make import MODEL=Qwen/Qwen2.5-Coder-7B CHECKPOINT=checkpoints/qwen7b.apr

# Import via standalone script (with validation)
./scripts/import.sh Qwen/Qwen2.5-Coder-7B checkpoints/qwen7b.apr
```

## 6.2.2 Eval

```bash
# Run HumanEval with defaults (512 tokens, temperature 0.0, 1 sample)
make eval-humaneval CHECKPOINT=checkpoints/qwen-7b.apr

# Full benchmark suite
make eval-all CHECKPOINT=checkpoints/qwen-7b.apr

# Custom parameters
make eval-humaneval CHECKPOINT=checkpoints/qwen-7b.apr \
    MAX_TOKENS=1024 TEMPERATURE=0.2 NUM_SAMPLES=10

# Perplexity baseline
make eval-perplexity CHECKPOINT=checkpoints/qwen-7b.apr
```

The eval script (`scripts/eval-pass-at-k.sh`) handles the full pipeline:
1. Downloads benchmark data (HumanEval, MBPP, BigCodeBench) if not cached
2. For each problem: generates completion via `apr run`
3. Combines completion + test cases
4. Executes in sandbox with `timeout 10`
5. Computes pass@k and writes result JSON

## 6.2.3 Data Preparation

```bash
# Extract instruction pairs from all 4 ground truth corpora
make prep-data

# View corpus statistics
make prep-data-stats
```

The data prep script (`scripts/prep-instruct-data.py`) extracts function/class definitions with docstrings from:
- **depyler** (~11.8K pairs): Python algorithms, data structures, CLI examples
- **hf-gtc** (~3.5K pairs): HuggingFace production recipes
- **jax-gtc** (~58 pairs): JAX numerical computing patterns
- **vllm-gtc** (~81 pairs): vLLM inference optimization patterns

Total: **~15.5K instruction/response pairs** in JSONL format.

## 6.2.4 Finetune

```bash
# Instruction fine-tuning with data from ground truth corpora (GH-371)
make prep-data                    # generate data/instruct-corpus.jsonl
make finetune-instruct            # defaults: model_size=7B, rank=16, lr=0.0002, 3 epochs

# Custom instruction fine-tuning config
make finetune-instruct MODEL_SIZE=7B RANK=32 LR=0.001 EPOCHS=5

# Classification LoRA fine-tune (original path)
make finetune CHECKPOINT=checkpoints/qwen-7b.apr DATA=data/code-instruct.jsonl

# QLoRA with custom config
make finetune CHECKPOINT=checkpoints/qwen-7b.apr DATA=data/code-instruct.jsonl \
    METHOD=qlora RANK=32 LR=0.001 EPOCHS=5
```

**Tasks:** `instruct` (generative, GH-371), `classify` (classification).
**Methods:** `lora` (default), `qlora` (quantized LoRA), `full` (all parameters).

| Variable | Default | Description |
|---|---|---|
| `METHOD` | `lora` | Fine-tuning method |
| `RANK` | `16` | LoRA rank |
| `LR` | `0.0002` | Learning rate |
| `EPOCHS` | `3` | Number of epochs |
| `DATA` | `data/instruct-corpus.jsonl` | Training dataset |
| `MODEL_SIZE` | `7B` | Model size for instruct task (tiny/0.5B/7B/9B) |

## 6.2.4 Distill

```bash
# Progressive distillation (recommended for code models)
make distill TEACHER=checkpoints/teacher-32b.apr STUDENT=checkpoints/student-7b.apr \
    DIST_STRATEGY=progressive DIST_TEMP=3.0 DIST_ALPHA=0.7
```

**Strategies:** `standard` (KL divergence), `progressive` (curriculum learning), `ensemble` (multi-teacher).

| Variable | Default | Description |
|---|---|---|
| `DIST_STRATEGY` | `standard` | Distillation strategy |
| `DIST_TEMP` | `3.0` | Softmax temperature |
| `DIST_ALPHA` | `0.7` | Mixing coefficient (0=student, 1=teacher) |

## 6.2.5 Merge

```bash
# SLERP merge of two models
make merge MODELS="checkpoints/a.apr checkpoints/b.apr" STRATEGY=slerp

# TIES merge (set via recipe TOML for full control)
make pipeline RECIPE=recipe-b-merge-alchemist
```

**Strategies:** `slerp`, `ties` (TIES-Merging), `dare` (DARE-TIES), `linear` (linear average).

## 6.2.6 Prune

```bash
# Wanda pruning with 50% sparsity (default)
make prune CHECKPOINT=checkpoints/tuned.apr PRUNE_METHOD=wanda SPARSITY=0.5

# Magnitude pruning
make prune CHECKPOINT=checkpoints/tuned.apr PRUNE_METHOD=magnitude SPARSITY=0.3
```

**Methods:** `wanda` (default), `magnitude`, `sparsegpt`. Sparsity: 0.0–1.0.

## 6.2.7 Quantize

```bash
# INT4 quantization
make quantize CHECKPOINT=checkpoints/pruned.apr SCHEME=int4

# Q6K quantization
make quantize CHECKPOINT=checkpoints/pruned.apr SCHEME=q6k
```

**Schemes:** `int4`, `int8`, `q4k`, `q5k`, `q6k`.

## 6.2.8 Pipeline (config-driven)

```bash
# Run entire pipeline from a recipe TOML config
make pipeline RECIPE=recipe-a-quick-lora

# Dry-run: show commands without executing
make pipeline-plan RECIPE=recipe-c-full-pipeline
```

The pipeline script (`scripts/pipeline.sh`) reads a recipe TOML and runs each stage in order:

```
import → [distill] → [finetune] → [align] → [merge] → [prune] → [quantize] → eval → [submit] → [compile]
```

Stages in brackets are optional — only included if the corresponding TOML section exists.

## 6.2.9 Submit

```bash
# Export and publish to HuggingFace Hub
make publish CHECKPOINT=checkpoints/model.apr HF_REPO=paiml/qwen-coder-7b-apr

# Export only (SafeTensors)
make export CHECKPOINT=checkpoints/model.apr EXPORT_FORMAT=safetensors
```

The submit script (`scripts/submit.sh`):
1. Exports model to SafeTensors via `apr export`
2. Generates model card with benchmark results table
3. Dry-run preview via `apr publish --dry-run`
4. Prompts for confirmation before actual upload

## 6.2.10 Verification

```bash
# Verify apr CLI and all subcommands
make verify

# End-to-end smoke test (CLI + configs)
make dogfood
```

## 6.3 Orchestration Surface Mapping

The full mapping between Makefile targets and `apr` CLI operations:

```
make pipeline RECIPE=recipe-c-full-pipeline.toml
    │
    │  scripts/pipeline.sh reads TOML, runs stages:
    │
    ├── [import]    ──► apr import hf://... -o checkpoints/base.apr
    ├── [distill]   ──► apr distill teacher.apr --student base.apr -o distilled.apr
    ├── [finetune]  ──► apr finetune distilled.apr --method lora -o tuned.apr
    ├── [align]     ──► apr finetune tuned.apr --method dpo -o aligned.apr
    ├── [merge]     ──► apr merge aligned.apr variant.apr --strategy slerp -o merged.apr
    ├── [prune]     ──► apr prune merged.apr --method wanda -o pruned.apr
    ├── [quantize]  ──► apr quantize pruned.apr --scheme int4 -o quantized.apr
    ├── [eval]      ──► scripts/eval-pass-at-k.sh humaneval quantized.apr
    ├── [submit]    ──► scripts/submit.sh quantized.apr org/model
    └── [compile]   ──► apr compile quantized.apr --release --lto --strip
```
