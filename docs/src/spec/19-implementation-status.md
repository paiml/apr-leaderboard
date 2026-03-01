# Implementation Status

Tracking table mapping spec sections to apr-leaderboard implementation. Updated as code lands.

## 19.1 Orchestration Targets (§6.2)

apr-leaderboard is a thin orchestrator — a Makefile + shell scripts — that calls `apr` CLI subcommands. There is no Rust source code; all ML operations are delegated to aprender.

| Make Target | Script/Command | Status | Notes |
|---|---|---|---|
| `make import` | `apr import hf://$(MODEL) -o $(CHECKPOINT)` | ✅ Working | Real HF download, GGUF and SafeTensors paths |
| `make finetune` | `apr finetune $(CHECKPOINT) --method lora ...` | ✅ Wired | LoRA/QLoRA via entrenar |
| `make merge` | `apr merge $(MODELS) --strategy slerp ...` | ✅ Wired | SLERP/TIES/DARE/Linear |
| `make prune` | `apr prune $(CHECKPOINT) --method wanda ...` | ✅ Wired | Wanda/magnitude pruning |
| `make quantize` | `apr quantize $(CHECKPOINT) --scheme int4 ...` | ✅ Wired | INT4/INT8/Q4K/Q5K/Q6K |
| `make distill` | `apr distill $(TEACHER) --student $(STUDENT) ...` | ✅ Wired | Standard/progressive/ensemble |
| `make compile` | `apr compile $(CHECKPOINT) --release --lto` | ✅ Wired | Standalone binary compilation |
| `make eval-humaneval` | `scripts/eval-pass-at-k.sh humaneval $(CHECKPOINT)` | ✅ Working | Generate + sandbox execute + pass@k |
| `make eval-mbpp` | `scripts/eval-pass-at-k.sh mbpp $(CHECKPOINT)` | ✅ Working | Same pipeline, MBPP dataset |
| `make eval-bigcodebench` | `scripts/eval-pass-at-k.sh bigcodebench $(CHECKPOINT)` | ✅ Working | Same pipeline, BigCodeBench dataset |
| `make eval-all` | Loops over all benchmarks | ✅ Working | Runs humaneval + mbpp + bigcodebench |
| `make eval-perplexity` | `apr eval $(CHECKPOINT) --dataset wikitext-2 --json` | ✅ Working | Perplexity baseline |
| `make export` | `apr export $(CHECKPOINT) --format safetensors` | ✅ Wired | SafeTensors/GGUF/MLX/ONNX |
| `make publish` | `scripts/submit.sh $(CHECKPOINT) $(HF_REPO)` | ✅ Working | Dry-run + confirm + HF Hub upload |
| `make model-card` | `apr eval $(CHECKPOINT) --generate-card --json` | ✅ Wired | Model card generation |
| `make pipeline` | `scripts/pipeline.sh configs/recipes/$(RECIPE).toml` | ✅ Working | Config-driven multi-stage pipeline |
| `make pipeline-plan` | `scripts/pipeline.sh --plan ...` | ✅ Working | Dry-run: validate config, show commands |
| `make check` | `apr check $(CHECKPOINT) --json` | ✅ Working | APR file integrity validation |
| `make inspect` | `apr inspect $(CHECKPOINT)` | ✅ Working | Model inspection |
| `make verify` | Smoke-tests all `apr` subcommands | ✅ Working | 16 subcommands verified |
| `make dogfood` | End-to-end smoke test | ✅ Working | CLI + configs validated |

## 19.2 Shell Scripts

| Script | Purpose | Status |
|---|---|---|
| `scripts/eval-pass-at-k.sh` | Download benchmark → generate completions via `apr run` → sandbox execute → compute pass@k → write JSON | ✅ Working |
| `scripts/pipeline.sh` | Parse recipe TOML → determine stages → execute sequentially (or `--plan` dry-run) | ✅ Working |
| `scripts/submit.sh` | Export to SafeTensors → generate model card → dry-run → publish to HF Hub | ✅ Working |
| `scripts/import.sh` | Wrapper around `apr import` with HF Hub reachability check + `apr check` validation | ✅ Working |

## 19.3 Quality Metrics

| Metric | Current | Target | Gate |
|---|---|---|---|
| `apr` CLI version | 0.4.10 | ≥ 0.4.10 | `apr --version` |
| Subcommand smoke test | 16/16 OK | 16/16 | `make verify` |
| Pipeline configs | 9 | — | `configs/models/*.toml` (5) + `configs/recipes/*.toml` (4) |
| Shell scripts | 4 | — | All executable, pass `bashrs lint` |
| Makefile targets | 21 | — | `make verify` + `make dogfood` |
| TOML config validity | 9/9 | 9/9 | `python3 -c "import tomllib; ..."` in `make dogfood` |
| Pipeline stages | 12 | — | import → distill → finetune → align → merge → prune → quantize → eval → submit → compile |

## 19.4 Config Templates (§4)

| Config | Location | Model | Strategy | Status |
|---|---|---|---|---|
| `qwen-coder-7b.toml` | `configs/models/` | Qwen2.5-Coder-7B | LoRA finetune → eval | ✅ Complete |
| `qwen-coder-32b.toml` | `configs/models/` | Qwen2.5-Coder-32B | Eval only (q8) | ✅ Complete |
| `qwen-coder-1.5b.toml` | `configs/models/` | Qwen2.5-Coder-1.5B | QLoRA → prune → INT4 → compile | ✅ Complete |
| `deepseek-r1-distill-7b.toml` | `configs/models/` | DeepSeek-R1-Distill-Qwen-7B | DPO align → prune → INT4 | ✅ Complete |
| `phi-4.toml` | `configs/models/` | Phi-4 | LoRA finetune → INT8 | ✅ Complete |
| `recipe-a-quick-lora.toml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | Quick LoRA (§9.1) | ✅ Complete |
| `recipe-b-merge-alchemist.toml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | Zero-training merge (§9.2) | ✅ Complete |
| `recipe-c-full-pipeline.toml` | `configs/recipes/` | Qwen2.5-Coder-7B | Full pipeline (§9.3) | ✅ Complete |
| `recipe-d-sovereign-binary.toml` | `configs/recipes/` | Qwen2.5-Coder-1.5B | Sovereign binary (§9.4) | ✅ Complete |

## 19.5 `apr` CLI Subcommand Availability

All ML operations are provided by `apr` CLI v0.4.10. Verified via `make verify`:

| `apr` Subcommand | Status | Used By |
|---|---|---|
| `apr import` | ✅ OK | `make import`, `scripts/import.sh`, `scripts/pipeline.sh` |
| `apr run` | ✅ OK | `scripts/eval-pass-at-k.sh` (generate completions) |
| `apr serve` | ✅ OK | (HTTP API — partial: doesn't bind for .apr files) |
| `apr chat` | ✅ OK | (interactive — not used by pipeline) |
| `apr finetune` | ✅ OK | `make finetune`, `scripts/pipeline.sh` |
| `apr merge` | ✅ OK | `make merge`, `scripts/pipeline.sh` |
| `apr prune` | ✅ OK | `make prune`, `scripts/pipeline.sh` |
| `apr quantize` | ✅ OK | `make quantize`, `scripts/pipeline.sh` |
| `apr distill` | ✅ OK | `make distill`, `scripts/pipeline.sh` |
| `apr eval` | ✅ OK | `make eval-perplexity`, `make model-card` |
| `apr export` | ✅ OK | `make export`, `scripts/submit.sh` |
| `apr publish` | ✅ OK | `scripts/submit.sh` |
| `apr check` | ✅ OK | `make check`, `scripts/import.sh` |
| `apr compile` | ✅ OK | `make compile`, `scripts/pipeline.sh` |
| `apr bench` | ✅ OK | (latency benchmarks — not used by pipeline) |
| `apr inspect` | ✅ OK | `make inspect` |

## 19.6 Dogfooding Findings

End-to-end dogfooding with real model import and inference. See also §22 for detailed findings.

### 19.6.1 GGUF vs SafeTensors Import Path

SafeTensors imports produce F16/BF16 tensors that realizar cannot run inference on (fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K). **GGUF import (pre-quantized Q4_K_M) is the working path** — produces runnable models with embedded tokenizer.

| Import Path | `apr check` Score | Inference | Notes |
|---|---|---|---|
| SafeTensors (F16) | F (3/100) | Fails | "Fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K, got type 30" |
| GGUF (Q4_K_M) | B+ (85/100) | Works | 10/10 validation stages, real code generation |

### 19.6.2 GPU Inference Gap

realizar CUDA path panics with shape mismatch on Qwen2.5-Coder-1.5B: `range end index 1536 out of range for slice of length 1024`. CPU inference with `--no-gpu` works. This is an upstream realizar issue.

### 19.6.3 `apr serve` for .apr Files

`apr serve` loads .apr models but the HTTP server doesn't bind. Serve may only be implemented for raw GGUF files. `apr run` works correctly for single-prompt inference.

### 19.6.4 Pipeline Ordering Validation

Recipe B (merge-alchemist) correctly emits a warning:
```
WARNING: Merge without finetune: merging untrained variants is suboptimal.
```

The §10 golden ordering enforcement works. The pipeline allows violation but warns.

### 19.6.5 Real Inference Verified

`apr run checkpoints/qwen2.5-coder-1.5b-q4k.apr "def fibonacci(n):" --max-tokens 128 --no-gpu` generates real Python code (Fibonacci implementation) in ~20s on CPU.
