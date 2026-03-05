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
| `make pipeline` | `scripts/pipeline.sh configs/recipes/$(RECIPE).yaml` | ✅ Working | Config-driven multi-stage pipeline (YAML-first) |
| `make pipeline-plan` | `scripts/pipeline.sh --plan ...` | ✅ Working | Dry-run: validate config, show commands |
| `make validate` | `bashrs config lint` + `bashrs lint` + `bashrs make lint` | ✅ Working | Sovereign stack config validation (zero Python) |
| `make check` | `apr check $(CHECKPOINT) --json` | ✅ Working | APR file integrity validation |
| `make inspect` | `apr inspect $(CHECKPOINT)` | ✅ Working | Model inspection |
| `make verify` | Smoke-tests all `apr` subcommands | ✅ Working | 16 subcommands verified |
| `make dogfood` | End-to-end smoke test | ✅ Working | CLI + configs validated |
| `make prove-wgpu` | `scripts/prove-wgpu.sh` | ✅ Working | wgpu training proof (§22.14) |
| `make align` | `apr finetune --method dpo/orpo` | ✅ Wired | DPO/ORPO alignment (GH-8) |
| `make book` | `mdbook build` | ✅ Working | Build specification book |
| `make docs` | `mdbook build` | ✅ Working | Alias for book |
| `make docs-serve` | `mdbook serve` | ✅ Working | Local book preview |
| `make prep-data` | `apr data prep` | ✅ Wired | Extract instruction/response pairs (GH-7) |
| `make prep-data-audit` | `apr data audit --verbose` | ✅ Working | Detailed corpus audit |
| `make finetune-instruct` | `apr finetune --task instruct` | ✅ Wired | Instruction LoRA fine-tuning |
| `make import-plan` | HF Hub check + dry-run | ✅ Working | Import plan preview |
| `make clean` | `rm -rf checkpoints/ results/` | ✅ Working | Remove build artifacts |
| `make decontaminate` | `apr data decontaminate` | ✅ Wired | N-gram overlap gate (AC-016) |
| `make qa` | `apr qa $(CHECKPOINT) --verbose` | ✅ Wired | Full model QA gate |
| `make compare-hf` | `apr compare-hf $(CHECKPOINT) --json` | ✅ Wired | HF parity check |
| `make benchmark-download` | `scripts/download-benchmarks.sh` | ✅ Working | Download HumanEval/MBPP data |
| `make results-history` | `scripts/results-history.sh` | ✅ Working | View and compare eval results |

## 19.2 Shell Scripts

| Script | Purpose | Status |
|---|---|---|
| `scripts/eval-pass-at-k.sh` | Download benchmark → generate completions via `apr run` → strip markdown fences → sandbox execute (python3/Docker) → Chen et al. unbiased pass@k estimator → write JSON | ✅ Working |
| `scripts/pipeline.sh` | Parse recipe YAML (bash-native) → determine stages → execute sequentially (or `--plan` dry-run) | ✅ Working |
| `scripts/submit.sh` | Export to SafeTensors → generate model card → dry-run → publish to HF Hub | ✅ Working |
| `scripts/import.sh` | Wrapper around `apr import` with HF Hub reachability check + `apr check` validation | ✅ Working |
| `scripts/prove-wgpu.sh` | End-to-end wgpu training proof: import → train (QLoRA) → verify → report | ✅ Working |
| `scripts/download-benchmarks.sh` | Download HumanEval/MBPP benchmark data for eval + decontamination | ✅ Working |
| `scripts/results-history.sh` | View and compare evaluation results with filtering by benchmark/model | ✅ Working |

## 19.3 Quality Metrics

| Metric | Current | Target | Gate |
|---|---|---|---|
| `apr` CLI version | 0.4.10 | ≥ 0.4.10 | `apr --version` |
| Subcommand smoke test | 19/19 OK | 19/19 | `make verify` |
| YAML configs | 17 | — | models (6) + recipes (7) + eval (1) + pipeline (2) + data catalog (1) |
| Shell scripts | 7 | — | All executable, pass `bashrs lint` |
| Makefile targets | 37 | — | `make verify` + `make validate` + `make dogfood` |
| Config validity | 19/19 | 19/19 | `bashrs config lint` in `make validate` (zero Python) |
| Pipeline stages | 12 | — | import → distill → finetune → align → merge → prune → quantize → eval → submit → compile |

## 19.4 Config Templates (§4)

| Config | Location | Model | Strategy | Status |
|---|---|---|---|---|
| `qwen-coder-7b.yaml` | `configs/models/` | Qwen2.5-Coder-7B | LoRA finetune → eval | ✅ Complete |
| `qwen-coder-32b.yaml` | `configs/models/` | Qwen2.5-Coder-32B | Eval only (q8) | ✅ Complete |
| `qwen-coder-1.5b.yaml` | `configs/models/` | Qwen2.5-Coder-1.5B | QLoRA → prune → INT4 → compile | ✅ Complete |
| `deepseek-r1-distill-7b.yaml` | `configs/models/` | DeepSeek-R1-Distill-Qwen-7B | DPO align → prune → INT4 | ✅ Complete |
| `phi-4.yaml` | `configs/models/` | Phi-4 | LoRA finetune → INT8 | ✅ Complete |
| `qwen3-8b.yaml` | `configs/models/` | Qwen3-8B | QLoRA instruct + eval | ✅ Complete |
| `recipe-a-quick-lora.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | Quick LoRA (§9.1) | ✅ Complete |
| `recipe-b-merge-alchemist.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | Zero-training merge (§9.2) | ✅ Complete |
| `recipe-c-full-pipeline.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B | Full pipeline (§9.3) | ✅ Complete |
| `recipe-d-sovereign-binary.yaml` | `configs/recipes/` | Qwen2.5-Coder-1.5B | Sovereign binary (§9.4) | ✅ Complete |
| `recipe-e-instruct-finetune.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | Instruct fine-tune (§9.5) | ✅ Complete |
| `recipe-f-qwen3-qlora.yaml` | `configs/recipes/` | Qwen3-8B | QLoRA instruct pipeline (§9.6) | ✅ Complete |
| `recipe-g-wgpu-proof.yaml` | `configs/recipes/` | Qwen2.5-Coder-1.5B | wgpu training proof (§22.14) | ✅ Complete |
| `coding-benchmarks.yaml` | `configs/eval/` | — | Benchmark suite definitions + targets + baselines | ✅ Complete |
| `leaderboard.yaml` | `configs/pipeline/` | — | Forjar infrastructure manifest | ✅ Complete |
| `leaderboard-playbook.yaml` | `configs/pipeline/` | — | Batuta playbook DAG | ✅ Complete |
| `data_catalog.yaml` | root | — | Data governance, lineage, classification | ✅ Complete |

## 19.4.1 GPU Sharing Infrastructure (entrenar)

The GPU-SHARE specification is fully implemented in entrenar with 143 tests across all modules.

| Component | Module | Status | Tests |
|---|---|---|---|
| VRAM guard | `entrenar::gpu::guard` | ✅ Complete | 12 |
| VRAM ledger (flock + JSON) | `entrenar::gpu::ledger` | ✅ Complete | 15 |
| Wait-for-VRAM queue | `entrenar::gpu::wait` | ✅ Complete | 8 |
| GPU profiler | `entrenar::gpu::profiler` | ✅ Complete | 6 |
| MPS (experimental) | `entrenar::gpu::mps` | ✅ Complete | 11 |
| Cluster config | `entrenar::gpu::cluster` | ✅ Complete | 12 |
| Job placement | `entrenar::gpu::placement` | ✅ Complete | 10 |
| Checkpoint coordinator | `entrenar::gpu::coordinator` | ✅ Complete | 16 |
| Multi-adapter pipeline | `entrenar::finetune::multi_adapter_pipeline` | ✅ Complete | 18 |

CLI flags: `--wait-gpu`, `--vram`, `--experimental-mps`, `--gpu-share`, `--adapters`, `--adapters-config`

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
| `apr data` | ✅ OK | `make prep-data`, `make decontaminate`, `make prep-data-audit` |
| `apr qa` | ✅ OK | `make qa` |
| `apr compare-hf` | ✅ OK | `make compare-hf` |

## 19.6 Dogfooding Findings

End-to-end dogfooding with real model import and inference. See also §22 for detailed findings.

### 19.6.1 GGUF vs SafeTensors Import Path

SafeTensors imports produce F16/BF16 tensors that realizar cannot run inference on (fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K). **GGUF import (pre-quantized Q4_K_M) is the working path** — produces runnable models with embedded tokenizer.

| Import Path | `apr check` Score | Inference | Notes |
|---|---|---|---|
| SafeTensors (F16) | F (3/100) | Fails | "Fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K, got type 30" |
| GGUF (Q4_K_M) | B+ (85/100) | Works | 10/10 validation stages, real code generation |

### 19.6.2 GPU Inference Status

GPU inference uses wgpu (Vulkan/Metal/DX12) for vendor-agnostic compute. Historical CUDA path has been replaced. CPU inference with `--no-gpu` is always available as a fallback.

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

### 19.6.6 GPU Sharing Spec Complete

All three phases of the GPU-SHARE specification implemented and tested:

- **Phase 1:** VRAM guard prevents OOM crashes. Ledger uses flock + atomic JSON write for crash safety. Wait queue polls until VRAM budget is available. MPS available as `--experimental-mps` opt-in.
- **Phase 2:** Multi-adapter pipeline loads base model once, trains N LoRA adapters concurrently (3x VRAM savings for 3 adapters). Round-robin and priority scheduling. TOML config via `--adapters-config`.
- **Phase 3:** Cluster config (YAML), job placement (VRAM-aware scoring), SSH transport (real `std::process::Command`, not stubs), checkpoint coordination with leaderboard, health check via SSH.

143 GPU tests pass. Zero SATD. Examples: `gpu_ledger`, `multi_adapter_training`, `cluster_training`.
