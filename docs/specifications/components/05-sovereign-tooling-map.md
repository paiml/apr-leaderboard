# Sovereign Tooling Map: World-Class or Wire It In

Every leaderboard-winning technique maps to a sovereign stack component. When a component doesn't support a technique at world-class level, we don't skip it — we find or build the capability and wire it into `apr` CLI commands.

## 5.1 Tooling Coverage Matrix

| Technique | Required Capability | Sovereign Component | Status | Gap Action |
|-----------|-------------------|-------------------|--------|------------|
| Import HF models | SafeTensors/GGUF → .apr | **aprender** 0.27 | ✅ Complete | `apr import` — 14+ architectures supported |
| Inference (decode) | Transformer forward pass | **realizar** 0.8 | ✅ Complete | `apr run` — 8-21% faster than llama.cpp |
| Inference (serve) | HTTP API, batching, streaming | **realizar** 0.8 | ✅ Complete | `apr serve` — OpenAI-compatible, PagedAttention |
| LoRA/QLoRA training | Low-rank adaptation, autograd | **entrenar** 0.7 | ✅ Complete | `apr finetune` — AdamW, cosine LR, checkpointing |
| Checkpoint management | Atomic save, resume, NaN scan, filtered load | **aprender** 0.27 | ✅ Complete | `AprWriter::write()` atomic (F-CKPT-009), `AprReader::open_filtered()` (F-CKPT-016), `read_tensor_f32_checked()` (F-CKPT-013), `validate_tensor_shape()` (F-CKPT-014) — 18/18 contracts |
| Knowledge distillation | KL-divergence, progressive | **entrenar** 0.7 | ✅ Complete | `apr distill` — standard, progressive, ensemble |
| Model merging | SLERP, TIES, DARE | **aprender** 0.27 | ✅ Complete | `apr merge` — 5 strategies |
| Pruning | Wanda, SparseGPT, structured | **aprender** 0.27 | ✅ Complete | `apr prune` — 6 methods |
| Quantization | INT4, INT8, Q4K, Q6K | **aprender** 0.27 | ✅ Complete | `apr quantize` — 4 formats |
| SIMD tensor ops | AVX2, AVX-512, NEON matmul | **trueno** 0.16 | ✅ Complete | 6% faster than NumPy at 256×256 |
| GPU compute | wgpu (Vulkan/Metal/DX12), CUDA PTX JIT | **trueno** 0.16 + **trueno-gpu** 0.4 | ✅ Complete | Pure Rust, any GPU vendor. CUDA via PTX JIT (no nvcc). See §5.10. |
| Speculative decoding | Draft model + verification | **realizar** 0.8 | ⚠️ Planned | GH-10: `apr run --speculative` not yet implemented |
| KV cache management | PagedAttention, CoW | **realizar** 0.8 | ✅ Complete | vLLM-style paged KV |
| Data loading | Parquet, JSONL, Arrow, HF Hub | **alimentar** 0.2 | ✅ Complete | Zero-copy Arrow RecordBatches |
| Data quality | Null/outlier/drift detection | **alimentar** 0.2 | ✅ Complete | 100-point quality scoring |
| Data decontamination | N-gram overlap detection | **alimentar** 0.2 | ✅ **Wired** | `apr data decontaminate` — n-gram overlap vs benchmarks (alimentar#30, aprender#415) |
| HPO | TPE, Hyperband, ASHA | **entrenar** 0.7 | ✅ Complete | `apr tune --strategy tpe` |
| Compile to binary | Model + runtime → executable | **aprender** 0.27 | ✅ Complete | `apr compile` |
| Correctness proofs | Kani bounded model checking | **provable-contracts** | ✅ Complete | 262 proof obligations |
| Quality gates | Compliance enforcement | **pmat** | ✅ Complete | 30+ automated checks |
| **DPO/ORPO alignment** | Preference optimization | **entrenar** 0.7 | ✅ **Wired** | `make align` → `apr finetune --method dpo` (GH-8: dedicated `apr align` planned) |
| **Execution sandbox** | Run generated code safely | — | ❌ **Missing** | **External harness** (see §5.3) |
| **N-sampling + rerank** | Batched generation, voting | **aprender** 0.27 | ⚠️ **Partial** | N-sampling via `NUM_SAMPLES` in eval script; `--temperature` + `--top-k` wired through batch mode. Reranking not yet implemented. |
| **Prompt templates** | SCoT, few-shot strategies | **eval script** | ✅ **Working** | 5 strategies in `build_instruction()`: standard, scot, few-shot, cgo, default. Few-shot best for HumanEval (+1.83pp). MBPP test assertions = +25.4pp. |
| **Synthetic data gen** | Teacher → training corpus | **alimentar** 0.2 + **aprender** | ⚠️ Partial | Generation via `apr chat --batch`; curation pipeline needed |
| **Continued pretraining** | Full-weight code corpus training | **entrenar** 0.7 | ⚠️ Partial | Full finetune works; needs large-corpus streaming |
| **Flash Attention** | Online softmax, tiled attention | **trueno** 0.16 | 🔧 In Progress | Phase 12 planned; tiling infra ready (wgpu compute shaders) |

## 5.2 Gap 1: DPO/ORPO Preference Optimization (CRITICAL)

**Why world-class:** DPO is the single most impactful post-training technique for leaderboards. Merged + DPO models "completely dominate" HF leaderboard rankings. Without DPO, we compete with one hand tied.

**Current state:** `make align` routes through `apr finetune --method dpo`
which connects to entrenar's loss functions. A dedicated `apr align`
subcommand is planned (GH-8).

**Current implementation:**

```bash
# DPO alignment via make align (routes through apr finetune)
make align CHECKPOINT=model.apr PREFS_DATA=prefs.jsonl ALIGN_METHOD=dpo

# Equivalent direct command
apr finetune model.apr --method dpo --data prefs.jsonl \
    --output aligned.apr --verbose
```

**Remaining wire-in plan:**

```
Component: entrenar
  Add: src/dpo/mod.rs — DPO loss (β-scaled log-ratio of policy vs reference)
  Add: src/dpo/data.rs — preference pair loader (chosen/rejected format)
  Add: src/dpo/orpo.rs — ORPO variant (no reference model needed)

Component: alimentar
  Add: Preference pair generation from execution feedback
    alimentar generate-preferences \
      --model model.apr \
      --problems humaneval.jsonl \
      --n-samples 10 \
      --judge execution \
      -o preference-pairs.jsonl

Component: Ground truth corpus
  Use: hf-ground-truth-corpus, algorithm-competition-corpus
    → Source of verified correct/incorrect code pairs for DPO training
```

**Acceptance criterion:** `apr align --method dpo` produces a model with ≥2% higher HumanEval+ than the input model after 3 epochs.

## 5.3 Gap 2: Code Execution Sandbox (CRITICAL)

**Why world-class:** HumanEval and MBPP require executing generated code against test cases. Without execution, we can't compute pass@k — we can only measure perplexity, which doesn't correlate well with code correctness.

**Current state:** aprender has no sandboxed code execution. Generated completions must be evaluated externally.

**Wire-in plan (two options):**

```
Option A: External EvalPlus harness (short-term, pragmatic)
  apr eval model.apr --data humaneval.jsonl --n-samples 10 \
    --output-completions completions/ --json
  # Then externally: evalplus.evaluate --samples completions/
  # This is what everyone does — even Google and Meta use external harnesses

Option B: WASM sandbox (long-term, sovereign)
  Component: realizar or new crate
  Add: Embedded WASM runtime (wasmtime) for safe code execution
    apr eval model.apr --data humaneval.jsonl \
      --sandbox wasm --timeout 10s --json
  Advantage: Fully sovereign, no Python dependency even for eval
  Risk: Python test cases require Python-in-WASM (CPython compiled to WASM)
```

**Decision:** Option A for v1.0 (get on the leaderboard), Option B as stretch goal. Neither compromises the "zero Python" claim for the model pipeline — eval is a separate concern.

## 5.4 Gap 3: N-Sampling + Reranking Pipeline

**Why world-class:** Generating N=10-50 completions and selecting the best one boosts effective pass@1 by 10-30%. This is the single most impactful inference-time technique.

**Current state:** aprender can generate multiple completions via temperature sampling. Missing: batched generation, reranking logic, majority voting.

**Wire-in plan:**

```
Component: aprender (apr-cli)
  Extend: `apr eval --n-samples N --rerank strategy`
    Strategies: logprob (sum of log-probabilities), majority (output voting),
                execution (run and pick passing code — requires sandbox)

Component: realizar
  Already supports: batched generation, concurrent requests
  Need: expose batch generation for N completions per prompt efficiently

Component: alimentar
  Add: Result aggregation and voting logic for N-sample outputs
```

## 5.5 Gap 4: Synthetic Training Data Pipeline

**Why world-class:** Qwen2.5-Coder, Phi-4, and NVIDIA OCR-Nemotron all credit large-scale synthetic data as core to their success. Without high-quality synthetic training data, fine-tuning is limited to existing datasets.

**Current state:** `apr chat --batch` can generate completions. alimentar handles data loading and quality scoring. Ground-truth corpora exist (hf-ground-truth-corpus, algorithm-competition-corpus). Missing: end-to-end curation pipeline.

**Wire-in plan:**

```
Component: alimentar
  CLI pipeline:
    # 1. Generate raw synthetic code from teacher
    apr chat teacher.apr --batch problems.txt --n-samples 5 \
      --temperature 0.8 --json > raw-synthetic.jsonl

    # 2. Quality-filter with alimentar
    alimentar quality raw-synthetic.jsonl --min-score 80 \
      -o filtered-synthetic.jsonl

    # 3. Decontaminate against eval benchmarks
    alimentar drift raw-synthetic.jsonl \
      --reference humaneval.jsonl mbpp.jsonl \
      --overlap-threshold 0.01 \
      -o clean-synthetic.jsonl

    # 4. Balance and split
    alimentar convert clean-synthetic.jsonl \
      -o training-data.parquet

Component: Ground truth corpora
  hf-ground-truth-corpus → HuggingFace API patterns, transformer implementations
  algorithm-competition-corpus → Algorithm problems with verified solutions
  → Both feed into fine-tuning data mix
```

## 5.6 Gap 5: Prompt Strategy Engine

**Why world-class:** SCoT prompting improves HumanEval pass@1 by up to 13.79%. Few-shot exemplars add 3-8%. The prompt template matters as much as the model weights.

**Current state:** `PROMPT_STRATEGY` is implemented in `scripts/eval-pass-at-k.sh` with 4 built-in strategies. The upstream `apr run --chat` provides raw chat template support.

**Implemented in eval pipeline:**

```bash
# All 5 strategies work via Makefile targets (best: few-shot 87.20%):
make eval-humaneval CHECKPOINT=m.apr PROMPT_STRATEGY=standard
make eval-humaneval CHECKPOINT=m.apr PROMPT_STRATEGY=scot
make eval-humaneval CHECKPOINT=m.apr PROMPT_STRATEGY=few-shot
make eval-humaneval CHECKPOINT=m.apr PROMPT_STRATEGY=cgo
```

**Built-in strategies (with aliases):**

| Strategy | Aliases | Description |
|---|---|---|
| `standard` | `default` | Raw problem → code (baseline) |
| `scot` | `structured-cot` | Structured chain-of-thought → code (+5-14%) |
| `few-shot` | `fewshot` | N exemplars + problem → code (+3-8%) |
| `cgo` | `code-gen-opt` | Chain of grounded objectives → code (+5-10%) |
| `reflexion` | `reflect` | Generate → test → reflect → regenerate (multi-turn) |

**Remaining wire-in for upstream apr:**

```
Component: realizar
  Already supports: chat templates (ChatML, LLaMA2, Mistral, Phi, Alpaca)
  Need: expose template composition for eval pipeline
```

## 5.7 Sovereign Stack Version Requirements

All gap closures must use published crates from crates.io. No git dependencies.

| Crate | Current | Required For Gaps | Minimum Version |
|-------|---------|-------------------|-----------------|
| aprender | 0.27.2 | `apr align`, `--n-samples --rerank`, checkpoint contracts (18/18 done in 0.27.2) | **0.28** |
| entrenar | 0.7.5 | DPO loss, preference pair loader, ORPO | **0.8** |
| trueno | 0.16.1 | Flash attention (Phase 12) | **0.17** |
| realizar | 0.8.0 | Batch N-sampling, prompt template composition | **0.9** |
| alimentar | 0.2.6 | Decontamination pipeline, preference pair generation, quality filtering | **0.3** |
| provable-contracts | 0.1 | DPO kernel contracts | **0.2** |

## 5.8 The Decision Rule

When we find a gap:

1. **Can an existing sovereign crate do it?** → Wire it in via `apr` CLI. No new crates.
2. **Does a sovereign crate need a new module?** → Add it to that crate, publish to crates.io, bump apr-leaderboard's dependency.
3. **Is it fundamentally outside the stack's scope?** → Use an external tool (e.g., EvalPlus for code execution) and document the boundary explicitly.
4. **Is it a research problem with no clear solution?** → Add to §21 Open Questions. Don't block the pipeline.

**Hard rule:** We never add a Python dependency. We never add a C/C++ FFI dependency. GPU compute is wgpu (primary, any vendor, pure Rust) with optional CUDA backend for hardware where wgpu support lags (e.g., Blackwell sm_121). No GPU vendor lock-in. If the sovereign stack can't do it in pure Rust, we either build it or scope it out with an explicit boundary.

## 5.9 Parity Check: Ludwig Feature Coverage

Ludwig (ludwig.ai) is the state-of-the-art declarative ML framework. Every feature Ludwig ships, the sovereign stack must match or exceed — in pure Rust, with zero Python. This is the parity bar.

### 5.9.1 Feature-by-Feature Parity Matrix

**Training & Fine-tuning:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Full fine-tuning | PyTorch, trainable=true | **entrenar** `apr finetune --method full` | ✅ Parity |
| LoRA adapters | PEFT library, configurable rank/dropout/targets | **entrenar** `apr finetune --method lora` | ✅ Parity |
| QLoRA (4-bit base + LoRA) | bitsandbytes + PEFT | **entrenar** `apr finetune --method qlora` | ✅ Parity |
| AdaLoRA (dynamic rank allocation) | PEFT AdaLoRA | **entrenar** — not yet | ❌ **Gap** |
| IA3 (inhibiting/amplifying activations) | PEFT IA3 | **entrenar** — not yet | ❌ **Gap** |
| DoRA (weight-decomposed LoRA) | PEFT DoRA variant | **entrenar** — not yet | ❌ **Gap** |
| NEFTune (embedding noise) | noise injection during fine-tune | **entrenar** — not yet | ❌ **Gap** |
| Gradient accumulation | PyTorch native | **entrenar** gradient accumulation | ✅ Parity |
| Mixed precision (fp16/bf16) | PyTorch AMP | **entrenar** GradScaler, bf16/fp16 | ✅ Parity |
| Early stopping | callback-based | **entrenar** EarlyStopping callback | ✅ Parity |
| Checkpointing | periodic save, atomic write, resume | **aprender** `AprWriter::write()` (atomic) + **entrenar** `CheckpointCallback` | ✅ **Exceeds** (18 contracts: atomic writes, NaN scan, filtered load, round-trip determinism, provenance) |
| Learning rate warmup + cosine decay | scheduler | **entrenar** WarmupCosineDecayLR | ✅ Parity |

**Optimizers:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| AdamW | PyTorch AdamW | **entrenar** AdamW (SIMD-accelerated) | ✅ Exceeds |
| Adam | PyTorch Adam | **entrenar** Adam | ✅ Parity |
| SGD with momentum | PyTorch SGD | **entrenar** SGD with momentum | ✅ Parity |
| 8-bit optimizers | bitsandbytes 8-bit Adam | — not yet | ❌ **Gap** |
| Paged optimizers | bitsandbytes paged | — not yet | ❌ **Gap** |

**Distributed Training:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Multi-GPU DDP | PyTorch DDP via Ray | — not yet (single-GPU via wgpu) | ❌ **Gap** |
| DeepSpeed ZeRO | Microsoft DeepSpeed | — not yet | ❌ **Gap** |
| Multi-node training | Ray cluster | **entrenar** GPU-SHARE Phase 3 (SSH cluster, job placement) | ✅ **Exceeds** (heterogeneous: 4090 + Jetson + CPU nodes) |
| Automatic batch size selection | binary search on GPU OOM | **aprender** `--vram` planning + **entrenar** VRAM guard | ✅ Parity |
| GPU sharing (multi-adapter) | not supported | **entrenar** GPU-SHARE (multi-adapter single-process, 3x VRAM savings) | ✅ **Exceeds** |

**Quantization:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| 4-bit quantization (nf4/fp4) | bitsandbytes | **aprender** INT4, Q4K | ✅ Parity |
| 8-bit quantization | bitsandbytes | **aprender** INT8, Q8_0 | ✅ Parity |
| Double quantization | bitsandbytes nested | — not yet | ⚠️ Partial |
| GPTQ | auto-gptq | — not yet | ❌ **Gap** |
| AWQ | autoawq | — not yet | ❌ **Gap** |

**Inference & Generation:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Greedy decoding | HF generate | **realizar** greedy | ✅ Parity |
| Temperature sampling | HF generate | **realizar** temperature | ✅ Parity |
| Top-k sampling | HF generate | **realizar** top-k | ✅ Parity |
| Nucleus (top-p) sampling | HF generate | **realizar** top-p | ✅ Parity |
| Beam search | HF generate | **aprender** num_beams | ✅ Parity |
| Contrastive search | HF generate | — not yet | ❌ **Gap** |
| Diverse beam search | HF generate | — not yet | ❌ **Gap** |
| Repetition penalty | HF generate | **aprender** repetition_penalty | ✅ Parity |
| Speculative decoding | not supported | **realizar** speculative | ✅ **Exceeds** |
| Streaming generation | not documented | **realizar** SSE streaming | ✅ **Exceeds** |
| OpenAI-compatible API | not supported | **realizar** /v1/chat/completions | ✅ **Exceeds** |
| PagedAttention KV cache | not supported | **realizar** paged KV | ✅ **Exceeds** |
| Continuous batching | not supported | **realizar** batch scheduling | ✅ **Exceeds** |

**Serving & Deployment:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| REST API serving | `ludwig serve` (Flask) | **realizar** `apr serve` (Axum) | ✅ Parity |
| Docker containers | prebuilt images | — user-provided | ⚠️ Partial |
| TorchScript export | PyTorch jit.trace | — not applicable (native binary) | N/A |
| Triton Inference Server | export format | — not applicable | N/A |
| HuggingFace Hub upload | `ludwig upload` | **aprender** `apr publish` | ✅ Parity |
| Compile to standalone binary | not supported | **aprender** `apr compile` | ✅ **Exceeds** |
| ONNX/CoreML/OpenVINO export | not supported | **aprender** `apr export` | ✅ **Exceeds** |

**Data Processing:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| CSV/JSON/Parquet/HDF5 loading | pandas | **alimentar** Arrow-native | ✅ Exceeds (zero-copy) |
| Auto preprocessing per feature type | Ludwig preprocessors | **alimentar** transforms | ✅ Parity |
| Train/val/test splitting | Ludwig split | **alimentar** DatasetSplit (stratified) | ✅ Parity |
| Larger-than-memory datasets | Ray datasets | **alimentar** MmapDataset, streaming | ✅ Parity |
| Data quality scoring | not built-in | **alimentar** 100-point quality scoring | ✅ **Exceeds** |
| Drift detection | not built-in | **alimentar** KS/Chi-sq/PSI/JSD | ✅ **Exceeds** |
| Imbalance detection + resampling | not built-in | **alimentar** SMOTE, oversample | ✅ **Exceeds** |

**Hyperparameter Optimization:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Random search | Ray Tune | **entrenar** RandomSearch | ✅ Parity |
| Grid search | Ray Tune | **entrenar** GridSearch | ✅ Parity |
| Bayesian (TPE) | Ray Tune Optuna | **entrenar** TPEOptimizer | ✅ Parity |
| ASHA scheduler | Ray Tune ASHA | **entrenar** HyperbandScheduler | ✅ Parity |
| Distributed HPO | Ray cluster | — not yet (local only) | ❌ **Gap** |

**Model Architecture:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| ECD (Encoder-Combiner-Decoder) | Ludwig native | — different architecture | N/A (not needed) |
| GBM (LightGBM) | LightGBM wrapper | — not in scope | N/A |
| LLM causal models | HF Transformers | **aprender** + **realizar** | ✅ Parity |
| Multi-modal (text+image+audio) | ECD combiner | — LLM-only for leaderboard | N/A (future) |
| Multi-task learning | multiple output heads | — not yet | ⚠️ Partial |
| Custom PyTorch modules | register API | — Rust modules via entrenar | ✅ Parity |

**Experiment Tracking:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| TensorBoard | callback | — not yet | ❌ **Gap** |
| Weights & Biases | callback | — not yet | ❌ **Gap** |
| MLflow | callback | — not yet | ❌ **Gap** |
| Comet ML | callback | — not yet | ❌ **Gap** |
| Built-in TUI monitoring | not supported | **entrenar** monitor + TUI | ✅ **Exceeds** |
| Prometheus metrics | not supported | **realizar** /metrics | ✅ **Exceeds** |

**Explainability & Visualization:**

| Ludwig Feature | Ludwig Implementation | Sovereign Stack | Status |
|---|---|---|---|
| Feature importance | built-in | **entrenar** ExplainabilityCallback | ✅ Parity |
| Learning curves | matplotlib | **entrenar** MonitorCallback | ⚠️ Partial |
| Confusion matrices | built-in | **entrenar** eval metrics | ⚠️ Partial |
| Model architecture visualization | built-in | **aprender** `apr tree`, `apr flow` | ✅ Parity |

**Correctness & Quality (sovereign stack advantages):**

| Feature | Ludwig | Sovereign Stack | Advantage |
|---|---|---|---|
| Provable kernel correctness | none | **provable-contracts** Kani L4 | ✅ **Unique** |
| 262 proof obligations | none | **provable-contracts** | ✅ **Unique** |
| Compliance enforcement | none | **pmat comply** 30+ checks | ✅ **Unique** |
| Deterministic builds | pip/conda chaos | Cargo.lock | ✅ **Unique** |
| wgpu GPU compute (any vendor) | requires CUDA toolkit | **trueno** wgpu (Vulkan/Metal/DX12) | ✅ **Unique** |
| Format-agnostic conversion | not supported | **aprender** `apr rosetta` | ✅ **Unique** |
| Model diff/forensics | not supported | **aprender** `apr diff`, `apr hex` | ✅ **Unique** |
| 10-stage integrity check | not supported | **aprender** `apr check` | ✅ **Unique** |

### 5.9.2 Summary: Where We Exceed, Where We Must Close Gaps

**We have parity in 24+ areas:** LoRA, QLoRA, full fine-tuning, AdamW/Adam/SGD, gradient accumulation, mixed precision, early stopping, LR scheduling, all sampling strategies, beam search, REST serving, HF upload, data loading, preprocessing, train/val/test splits, HPO (grid/random/TPE/ASHA), feature importance.

**We exceed Ludwig in 16+ areas** (updated): speculative decoding, PagedAttention, continuous batching, streaming API, OpenAI-compatible serving, compile-to-binary, multi-format export (ONNX/CoreML/OpenVINO), data quality scoring, drift detection, imbalance detection, Prometheus metrics, TUI monitoring, provable contracts, deterministic builds, format forensics, **checkpointing** (18 verified contracts: atomic writes, NaN scan, filtered loading, round-trip determinism, provenance chain — vs Ludwig's basic callback).

**Gaps to close (9 items):**

| Gap | Priority | Wire-in Target |
|---|---|---|
| AdaLoRA (dynamic rank) | Medium | **entrenar** 0.8 |
| IA3 adapter | Low | **entrenar** 0.8 |
| DoRA (weight-decomposed LoRA) | Medium | **entrenar** 0.8 |
| NEFTune (embedding noise) | Low | **entrenar** 0.8 |
| 8-bit optimizers | Low | **entrenar** 0.8 |
| Contrastive search decoding | Low | **aprender** 0.28 |
| Diverse beam search | Low | **aprender** 0.28 |
| Multi-GPU DDP | High | **entrenar** 0.9 |
| GPTQ quantization | Medium | **aprender** 0.28 |

**Recently closed gaps:**
- ~~Multi-node training~~ → GPU-SHARE Phase 3: SSH cluster config, job placement, checkpoint coordination (143 tests)
- ~~Automatic batch size selection~~ → VRAM guard + ledger prevents OOM, `--vram` planning
- ~~Experiment tracking~~ → `entrenar` TUI monitor + JSONL event logging + checkpoint metadata

**Out of scope (not needed for leaderboard):** ECD architecture, GBM/LightGBM, multi-modal (text+image+audio), Triton export, TorchScript. These serve Ludwig's "general ML framework" positioning. We are a purpose-built leaderboard pipeline, not a general framework.

## 5.10 GPU Compute Architecture: PTX JIT vs Pre-compiled Kernels

### 5.10.1 Why PTX JIT (Not nvcc)

PyTorch ships **fat binaries** — pre-compiled SASS (GPU machine code) for every supported architecture (sm_70, sm_80, sm_86, sm_89, sm_90). At runtime, the CUDA driver selects the matching SASS — zero JIT, instant startup. This requires `nvcc` (NVIDIA's proprietary compiler) and the CUDA toolkit (~2+ GB) at build time.

trueno-gpu takes a fundamentally different approach: **PTX string templates embedded in Rust**. PTX (Parallel Thread Execution) is NVIDIA's stable intermediate assembly language. trueno-gpu writes CUDA kernels directly as PTX strings in Rust source code, compiled into the `apr` binary by `cargo build` — no nvcc, no CUDA toolkit, no C/C++ FFI.

At runtime, the CUDA driver JIT-compiles PTX to device-specific SASS for whatever GPU is present. This is the same mechanism PyTorch uses as a **fallback** for unsupported architectures — trueno-gpu uses it as the **primary** path.

### 5.10.2 Trade-offs

| Aspect | PyTorch (pre-compiled SASS) | trueno-gpu (PTX JIT) |
|--------|---------------------------|---------------------|
| Build deps | nvcc + CUDA toolkit (2+ GB) | `cargo build` only |
| New GPU support | Requires new release with SASS | Automatic (PTX forward-compatible) |
| Startup time | Instant | 20-80s JIT (amortized by `--batch-jsonl`) |
| Binary size | ~500 MB (fat binaries) | ~10 MB (PTX strings) |
| Vendor lock-in | CUDA toolkit version | None (PTX is stable ISA) |
| Reproducibility | Tied to CUDA/cuDNN version | Same binary, any NVIDIA GPU |

### 5.10.3 Amortization via Batch Mode

The `--batch-jsonl` flag is the architectural answer to JIT overhead. For a 164-problem HumanEval eval:

- **Without batch:** 80s JIT × 164 invocations = **3.6 hours** of JIT alone
- **With batch:** 80s JIT × 1 load = **80s** total JIT, then pure inference

Amortized JIT cost per problem: <0.5s. The sovereignty benefit (zero external toolchain, forward GPU compatibility) far outweighs the one-time startup cost.

### 5.10.4 Blackwell sm_121 and the Try 1/Try 2 Pattern

On Blackwell (sm_121), the CUDA 13.0 driver has a JIT bug: it rejects PTX with `.target sm_121` (error 300, CUDA_ERROR_INVALID_SOURCE). The GH-480 fix implements a defensive fallback:

1. **Try 1:** Compile PTX with explicit `.target sm_121` — fails (error 300)
2. **Try 2:** Compile with `cuModuleLoadData` (no explicit target) — succeeds

This Try 1 → Try 2 pattern is a driver workaround, not a design choice. When NVIDIA fixes the sm_121 JIT in a future driver, Try 1 will succeed and the fallback becomes dead code. The PTX post-processor (GH-480) also patches backward `bra LABEL` instructions to `@%p_jw bra LABEL` for sm_121 compatibility.

### 5.10.5 FP8 Architecture Guard (GH-542)

FP8 E4M3 GEMM kernels (Ada/Hopper-specific) cause `CUDA_ERROR_ILLEGAL_ADDRESS` on Blackwell, poisoning the CUDA context. Fix: `detect_fp8_prefill()` uses `cc >= 89 && cc < 100` to auto-disable FP8 on Blackwell. Provable contract: `gpu-context-health-v1.yaml` (3 proof obligations, 3 falsification tests).

**Five-whys:** (1) Why crash? FP8 warmup writes invalid memory on sm_121. (2) Why invalid? FP8 E4M3 cuBLASLt kernels are Ada/Hopper-specific. (3) Why enabled? `cc >= 89` without upper bound. (4) Why no bound? Blackwell didn't exist when written. (5) Fix: `cc < 100` guard in 3 files (commit `a4bcd908`).
