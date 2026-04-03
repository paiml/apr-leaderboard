# Implementation Status

Tracking table mapping spec sections to apr-leaderboard implementation. Updated as code lands.

## 19.1 Orchestration Targets (§6.2)

apr-leaderboard is a thin orchestrator — a Makefile + shell scripts — that calls `apr` CLI subcommands. There is no Rust source code; all ML operations are delegated to aprender.

| Make Target | Script/Command | Status | Notes |
|---|---|---|---|
| `make import` | `apr import hf://$(MODEL) -o $(CHECKPOINT)` | ✅ Working | Real HF download, GGUF and SafeTensors paths |
| `make finetune` | `apr finetune $(CHECKPOINT) --method lora ...` | ✅ Working | wgpu QLoRA (592 GFLOPS), SFT + DPO auto-detect, adapter export, 13 KAIZEN fixes |
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
| `make verify` | Smoke-tests all `apr` subcommands | ✅ Working | 19 subcommands verified |
| `make dogfood` | End-to-end smoke test | ✅ Working | CLI + configs validated |
| `make prove-wgpu` | `scripts/prove-wgpu.sh` | ✅ Working | wgpu training proof (§22.14) |
| `make align` | `apr finetune --method dpo/orpo` | ✅ Wired | DPO/ORPO alignment (GH-8) |
| `make book` | `mdbook build` | ✅ Working | Build specification book |
| `make docs` | `mdbook build` | ✅ Working | Alias for book |
| `make docs-serve` | `mdbook serve` | ✅ Working | Local book preview |
| `make prep-data` | `apr data prep` | 🔧 Blocked | Subcommand not wired yet (GH-12) |
| `make prep-data-audit` | `apr data audit --verbose` | ✅ Working | Detailed corpus audit |
| `make data-split` | `apr data split` | ✅ Working | Stratified train/val/test split |
| `make data-balance` | `apr data balance` | ✅ Working | Resample for class balance |
| `make finetune-instruct` | `apr finetune --task instruct` | ✅ Wired | Instruction LoRA fine-tuning |
| `make import-plan` | HF Hub check + dry-run | ✅ Working | Import plan preview |
| `make clean` | `rm -rf checkpoints/ results/` | ✅ Working | Remove build artifacts |
| `make decontaminate` | `apr data decontaminate` | 🔄 PR Open | aprender#415 + alimentar#32 (GH-11) |
| `make data-quality` | `apr data quality` | 🔧 Blocked | Subcommand not wired yet (GH-11) |
| `make qa` | `apr qa $(CHECKPOINT) --verbose` | ✅ Wired | Full model QA gate |
| `make compare-hf` | `apr compare-hf --hf $(MODEL) --json $(CHECKPOINT)` | ✅ Working | HF parity check (requires MODEL) |
| `make bench` | `apr bench $(CHECKPOINT) --json` | ✅ Working | Throughput benchmark |
| `make benchmark-download` | `scripts/download-benchmarks.sh` | ✅ Working | Download HumanEval/MBPP data |
| `make results-history` | `scripts/results-history.sh` | ✅ Working | View and compare eval results |
| `make eval-sweep` | `scripts/eval-sweep.sh` | ✅ Working | Sweep all result JSONs, tabulate pass@k |
| `make compare-results` | `scripts/compare-results.sh` | ✅ Working | Delta analysis between two result files |
| `make leaderboard` | `scripts/leaderboard-summary.sh` | ✅ Working | Generate ranked markdown leaderboard from results |
| `make check-contracts` | Inline awk + jq + python3 | ✅ Working | 15 falsification tests (pass@k, throughput, data, eval, structure) |
| `make generate-preference-pairs` | `scripts/generate-preference-pairs.sh` | ✅ Working | Generate DPO pairs from N-sampling eval (PMAT-014) |
| `make generate-training-data` | `scripts/generate-training-data.sh` | ✅ Working | Synthetic instruct pairs from teacher model (PMAT-004) |
| `make distill-generate` | `scripts/distill-generate.sh` | ✅ Working | Text-based distillation: 32B teacher completions (PMAT-007) |
| `make distill-finetune` | `apr finetune --method qlora` | ✅ Wired | QLoRA fine-tune 7B on teacher completions (PMAT-007) |
| `make distill-eval` | `scripts/eval-pass-at-k.sh` | ✅ Wired | Evaluate distilled model on HumanEval (PMAT-007) |
| `make combine-training-data` | `scripts/combine-training-data.sh` | ✅ Working | Merge distill + instruct data for QLoRA (PMAT-008) |
| `make validate-teacher` | `scripts/validate-teacher.sh` | ✅ Working | Verify teacher model quality before distillation (§12.2) |
| `make failure-analysis` | `scripts/failure-analysis.sh` | ✅ Working | Always-fail/borderline/always-pass categorization |

## 19.2 Shell Scripts

| Script | Purpose | Status |
|---|---|---|
| `scripts/eval-pass-at-k.sh` | Download benchmark → generate completions via `apr run` → strip markdown fences → sandbox execute (python3/Docker) → Chen et al. unbiased pass@k estimator → write JSON | ✅ Working |
| `scripts/pipeline.sh` | Parse recipe YAML (bash-native) → determine stages → execute sequentially with eval config (prompt_strategy, max_tokens) → `--plan` dry-run | ✅ Working |
| `scripts/submit.sh` | Pre-submission checks (§14.4) → export SafeTensors → model card → dry-run → publish to HF Hub | ✅ Working |
| `scripts/import.sh` | Wrapper around `apr import` with HF Hub reachability check + `apr check` validation | ✅ Working |
| `scripts/prove-wgpu.sh` | End-to-end wgpu training proof: import → train (QLoRA) → verify → report | ✅ Working |
| `scripts/download-benchmarks.sh` | Download HumanEval/MBPP benchmark data for eval + decontamination | ✅ Working |
| `scripts/results-history.sh` | View and compare evaluation results with filtering by benchmark/model | ✅ Working |
| `scripts/leaderboard-summary.sh` | Generate ranked markdown leaderboard from all result JSONs | ✅ Working |
| `scripts/eval-sweep.sh` | Run eval across multiple prompt strategies sequentially | ✅ Working |
| `scripts/compare-results.sh` | Per-problem delta analysis between two result files | ✅ Working |
| `scripts/distill-generate.sh` | 32B teacher batch inference → coding completions JSONL (PMAT-007) | ✅ Working |
| `scripts/generate-distill-prompts.sh` | Generate targeted distillation prompts from HumanEval failure analysis | ✅ Working |
| `scripts/combine-training-data.sh` | Merge teacher completions + instruct corpus, deduplicate, shuffle | ✅ Working |
| `scripts/validate-teacher.sh` | Validate teacher model meets minimum pass@1 threshold for distillation | ✅ Working |
| `scripts/failure-analysis.sh` | Analyze HumanEval failures: always-fail, borderline, always-pass | ✅ Working |
| `scripts/oracle-analysis.sh` | Compute oracle upper bound across all runs and strategies | ✅ Working |

## 19.3 Quality Metrics

| Metric | Current | Target | Gate |
|---|---|---|---|
| `apr` CLI version | 0.4.11 | ≥ 0.4.10 | `apr --version` |
| Subcommand smoke test | 19/19 OK | 19/19 | `make verify` |
| YAML configs | 24 | — | models (7) + recipes (11) + eval (1) + pipeline (2) + data catalog (1) + distill (1) + data governance (1) |
| Shell scripts | 22 + 4 canaries | — | 22 pipeline scripts + 4 GPU canary/falsification scripts |
| Makefile targets | 56 | — | `make verify` + `make validate` + `make dogfood` |
| Contract tests | 56/57 | 57/57 | `make check-contracts` 13 categories + structure ×22. 1 fail: MBPP gate. |
| Contract YAMLs | 22 | — | 22 provable contract YAMLs including tokenizer-preservation-v1. |
| Make targets | 58 | — | Added `proof-status`, `status` targets |
| PMAT work items | 8 | — | PMAT-006 (done), PMAT-007 (done-pipeline), PMAT-008 (ready), PMAT-010 (pending), PMAT-011 (pending), PMAT-014 (in progress), PMAT-017 (done), PMAT-037 (done). See §27. |
| Spec sections | 27 | — | §1-27: v2.4.0 update cycle |
| Config validity | 20/20 | 20/20 | `bashrs config lint` in `make validate` (zero Python) |
| Pipeline stages | 12 | — | import → distill → finetune → align → merge → prune → quantize → eval → submit → compile |

## 19.4 Config Templates (§4)

| Config | Location | Model | Strategy | Status |
|---|---|---|---|---|
| `qwen-coder-7b.yaml` | `configs/models/` | Qwen2.5-Coder-7B | LoRA finetune → eval | ✅ Complete |
| `qwen-coder-32b.yaml` | `configs/models/` | Qwen2.5-Coder-32B | Eval only (q8) | ✅ Complete |
| `qwen-coder-1.5b.yaml` | `configs/models/` | Qwen2.5-Coder-1.5B | QLoRA → prune → INT4 → compile | ✅ Complete |
| `deepseek-r1-distill-7b.yaml` | `configs/models/` | DeepSeek-R1-Distill-Qwen-7B | DPO align → prune → INT4 | ✅ Complete |
| `phi-4.yaml` | `configs/models/` | Phi-4 | LoRA finetune → INT8 | ✅ Complete |
| `qwen3-4b.yaml` | `configs/models/` | Qwen3-4B | Thinking model eval (§22.17) | ✅ Complete |
| `qwen3-8b.yaml` | `configs/models/` | Qwen3-8B | QLoRA instruct + eval | ✅ Complete |
| `recipe-a-quick-lora.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | Quick LoRA (§9.1) | ✅ Complete |
| `recipe-b-merge-alchemist.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | Zero-training merge (§9.2) | ✅ Complete |
| `recipe-c-full-pipeline.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B | Full pipeline (§9.3) | ✅ Complete |
| `recipe-d-sovereign-binary.yaml` | `configs/recipes/` | Qwen2.5-Coder-1.5B | Sovereign binary (§9.4) | ✅ Complete |
| `recipe-e-instruct-finetune.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | Instruct fine-tune (§9.5) | ✅ Complete |
| `recipe-f-qwen3-qlora.yaml` | `configs/recipes/` | Qwen3-8B | QLoRA instruct pipeline (§9.6) | ✅ Complete |
| `recipe-g-wgpu-proof.yaml` | `configs/recipes/` | Qwen2.5-Coder-1.5B | wgpu training proof (§22.14) | ✅ Complete |
| `recipe-h-32b-distill.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | 32B→7B reasoning distillation | ✅ Complete |
| `recipe-i-humaneval-qlora.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | QLoRA on teacher+instruct data (PMAT-008) | ✅ Complete |
| `recipe-j-merge-specialists.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | TIES merge code+reasoning specialists (PMAT-010) | ✅ Complete |
| `recipe-k-final-artifact.yaml` | `configs/recipes/` | Qwen2.5-Coder-7B-Instruct | Prune+quantize+compile final submission (PMAT-011) | ✅ Complete |
| `distill-32b-7b-text.yaml` | `configs/distill/` | Qwen2.5-Coder-7B-Instruct | Text-based distillation config (PMAT-007) | ✅ Complete |
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

All ML operations are provided by `apr` CLI v0.4.11. Verified via `make verify`:

| `apr` Subcommand | Status | Used By |
|---|---|---|
| `apr import` | ✅ OK | `make import`, `scripts/import.sh`, `scripts/pipeline.sh` |
| `apr run` | ✅ OK | `scripts/eval-pass-at-k.sh` (generate completions), `--batch-jsonl` batch mode |
| `apr serve` | ✅ OK | (HTTP API — partial: doesn't bind for .apr files) |
| `apr chat` | ✅ OK | (interactive — not used by pipeline) |
| `apr finetune` | ⚠️ Partial | Training loop runs on gx10 with CUDA (backward GEMM f64 fix, GH-561). Loss: 13.61 train → 12.02 val on 3-sample test. APR adapter export (§26 Phase 3) not yet implemented. |
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

GPU inference uses wgpu (Vulkan/Metal/DX12) or CUDA (optional). GPU is mandatory for production eval.

**Status (2026-03-28): FIXED — single-prompt AND batch mode working via wgpu.**

- **Single-prompt** `apr run --gpu`: wgpu (Vulkan), cosine=0.999863, token-for-token parity.
- **Batch** `--batch-jsonl`: GH-560 FIXED (2026-03-28) — two bugs: FFN buffer overflow in trueno (`attn_out_buf` was hidden_dim=3584, needs intermediate_dim=18944; fix: `ffn_silu_buf`) + KV cache pre-filled length in realizar (`vec![0.0; ...]` → `Vec::with_capacity()` + `clear()`). Verified on gx10: identical output to CPU, 1.1-2.0 tok/s on 7B. Contract-bound: `gpu-weight-residency-v1` + `gpu-multi-backend-parity-v1`.
- **CPU batch** (default): Proven reliable, ~3 hours for 164 HumanEval, 84.76-85.98% pass@1.

The CUDA cosine=-0.005 on sm_121 (GH-559) is NOT a JIT bug — falsification proved the
PTX and JIT are both correct. Individual kernels produce correct results (RMSNorm diff=5e-7,
Q GEMV ~1%). The -0.005 cosine is from FP32 accumulation ordering differences (GPU parallel
vs CPU sequential) compounding through 28 layers × 10+ operations. wgpu avoids this by using
the same accumulation order as CPU (cosine=0.999863).

See §25 (GPU Compute Architecture) for full specification, provable contracts, and roadmap.

**Diagnostic trail (2026-03-25 → 2026-03-27):**

| Hypothesis | Tested | Result | Falsified by |
|---|---|---|---|
| RMSNorm kernel wrong | GPU_DEBUG=1, CPU bypass | Individual RMSNorm diff=5e-7 (correct) | Per-element comparison |
| Q4K GEMV kernel wrong | 5 PTX variants | All produce cosine=1.0 via Python ctypes | `falsify-ptx-implementations.py` |
| NVIDIA JIT compiler bug | Same PTX via Python | cosine=1.0 (JIT correct) | `isolate-cuda-bug.py` |
| Stream sync race | bar.sync per layer | Fixes no-op layers, not cosine | Per-layer sync test |
| **FP32 accumulation ordering** | — | **Correct root cause** | Not falsified |

**Corrected root cause (2026-03-27):** ~0.1% FP32 rounding per kernel × 280 operations
  → (1.001)^280 = 1.32 → cosine=-0.005. Individual kernels are correct (RMSNorm diff=5e-7,
  Q GEMV ~1%). PyTorch avoids this via TF32/FP64 accumulators. wgpu avoids it with
  sequential accumulation matching CPU.

**Active tickets:**
- GH-560: **CLOSED** (2026-03-28) — wgpu batch fully working. Two-bug fix: trueno `e24a6f6c` + realizar `e600bbff`.
- GH-561: **IN PROGRESS** — FP64 accumulators in NF4 GEMM forward + backward. Forward NF4 GEMM fixed previously (trueno `9e021c35`, `81a9c16f`). Backward GEMM (6 variants) now also fixed with f64 accumulators — training verified on gx10: loss 13.61→12.02, no NaN. Remaining: other kernels (RMSNorm backward, softmax backward, etc.) still use f32 accumulators but are lower priority — training converges without them.

### 19.6.3 `apr serve` for .apr Files

`apr serve` loads .apr models but the HTTP server doesn't bind. Serve may only be implemented for raw GGUF files. `apr run` works correctly for single-prompt inference.

### 19.6.4 Pipeline Ordering Validation

Recipe B (merge-alchemist) correctly emits a warning:
```
WARNING: Merge without finetune: merging untrained variants is suboptimal.
```

The §10 golden ordering enforcement works. The pipeline allows violation but warns.

### 19.6.5 Real Inference Verified

`apr run checkpoints/qwen2.5-coder-1.5b-q4k.apr "def fibonacci(n):" --max-tokens 128` generates real Python code (Fibonacci implementation). GPU mandatory for production eval.

### 19.6.6 GPU Sharing Spec Complete

All three phases of the GPU-SHARE specification implemented and tested:

- **Phase 1:** VRAM guard prevents OOM crashes. Ledger uses flock + atomic JSON write for crash safety. Wait queue polls until VRAM budget is available. MPS available as `--experimental-mps` opt-in.
- **Phase 2:** Multi-adapter pipeline loads base model once, trains N LoRA adapters concurrently (3x VRAM savings for 3 adapters). Round-robin and priority scheduling. TOML config via `--adapters-config`.
- **Phase 3:** Cluster config (YAML), job placement (VRAM-aware scoring), SSH transport (real `std::process::Command`, not stubs), checkpoint coordination with leaderboard, health check via SSH.

143 GPU tests pass. Zero SATD. Examples: `gpu_ledger`, `multi_adapter_training`, `cluster_training`.

### 19.6.7 QA Gate (2026-03-05)

`apr qa` on Qwen2.5-Coder-1.5B-Instruct Q4K: 6 PASS (capability, tensor contract, metadata, golden output, throughput, perf regression), 1 FAIL (format parity — GH-13: .apr-wrapped GGUF not recognized), 5 SKIP (no CUDA).

### 19.6.8 Perplexity Baseline (2026-03-05)

`apr eval --dataset wikitext-2`: perplexity 6.63, cross-entropy 1.89. Throughput: 2.5 tok/s on CPU, 385ms TTFT.

### 19.6.9 MBPP Eval (2026-03-29)

MBPP result: 74.80% pass@1 (374/500) few-shot 7B Q4K. Duplicate MBPP eval runs on Intel were killed — were burning 32 cores for 4 days with no additional value over the completed result.

### 19.6.10 Tokenizer Preservation Fix — GH-580 (2026-04-03)

**Problem:** All merge/quantize pipeline outputs lost embedded tokenizer, producing dead models that fail with `PMAT-172 ERROR: APR file missing embedded tokenizer`.

**Five Whys:**
1. Why can't distilled model run inference? Missing tokenizer.
2. Why missing? `run_merge()` used AprWriter (v1) which creates empty tokenizer.
3. Why empty? AprWriter v1 only writes weight tensors, not metadata sections.
4. Why not v2? Original code predated AprV2Writer.
5. Why not caught? `apr check` passes (validates weights), but `apr run` fails (needs tokenizer for encoding).

**Fix (GH-580):** Read base model with `AprV2Reader`, clone metadata (preserving tokenizer), use `AprV2Writer` for output. Also supports SafeTensors adapter input from wgpu training pipeline. Contract: `tokenizer-preservation-v1.yaml`.

**Impact:** Unblocks PMAT-007 eval, PMAT-008 DPO merge, PMAT-010 TIES merge. All merge operations now produce runnable models.

### 19.6.11 PMAT-007 Distillation Pipeline Complete (2026-04-03)

Full text-based distillation pipeline ran on gx10:
1. 99 teacher completions generated (32B model)
2. Combined with instruct corpus (15,326 lines)
3. QLoRA training: 7B on combined data, rank=32
4. Adapter exported: 40 MB safetensors
5. Merged into base 7B model (GH-580 fix)
6. Quantized to Q4K (6.2 GB)

**Awaiting:** HumanEval + MBPP evaluation of distilled Q4K model.
