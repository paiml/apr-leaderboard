# Sovereign Tooling Map: World-Class or Wire It In

Every leaderboard-winning technique maps to a sovereign stack component. When a component doesn't support a technique at world-class level, we don't skip it — we find or build the capability and wire it into `apr` CLI commands.

## 1. Tooling Coverage Matrix

| Technique | Required Capability | Sovereign Component | Status | Gap Action |
|-----------|-------------------|-------------------|--------|------------|
| Import HF models | SafeTensors/GGUF → .apr | **aprender** 0.27 | ✅ Complete | `apr import` — 14+ architectures supported |
| Inference (decode) | Transformer forward pass | **realizar** 0.8 | ✅ Complete | `apr run` — 8-21% faster than llama.cpp |
| Inference (serve) | HTTP API, batching, streaming | **realizar** 0.8 | ✅ Complete | `apr serve` — OpenAI-compatible, PagedAttention |
| LoRA/QLoRA training | Low-rank adaptation, autograd | **entrenar** 0.7 | ✅ Complete | `apr finetune` — AdamW, cosine LR, checkpointing |
| Knowledge distillation | KL-divergence, progressive | **entrenar** 0.7 | ✅ Complete | `apr distill` — standard, progressive, ensemble |
| Model merging | SLERP, TIES, DARE | **aprender** 0.27 | ✅ Complete | `apr merge` — 5 strategies |
| Pruning | Wanda, SparseGPT, structured | **aprender** 0.27 | ✅ Complete | `apr prune` — 6 methods |
| Quantization | INT4, INT8, Q4K, Q6K | **aprender** 0.27 | ✅ Complete | `apr quantize` — 4 formats |
| SIMD tensor ops | AVX2, AVX-512, NEON matmul | **trueno** 0.16 | ✅ Complete | 6% faster than NumPy at 256×256 |
| GPU compute | PTX generation, wgpu | **trueno** 0.16 | ✅ Complete | Pure Rust, no nvcc |
| Speculative decoding | Draft model + verification | **realizar** 0.8 | ✅ Complete | `apr run --speculative` |
| KV cache management | PagedAttention, CoW | **realizar** 0.8 | ✅ Complete | vLLM-style paged KV |
| Data loading | Parquet, JSONL, Arrow, HF Hub | **alimentar** 0.2 | ✅ Complete | Zero-copy Arrow RecordBatches |
| Data quality | Null/outlier/drift detection | **alimentar** 0.2 | ✅ Complete | 100-point quality scoring |
| Data decontamination | N-gram overlap detection | **alimentar** 0.2 | ⚠️ Partial | Doctest extraction exists; need benchmark-specific decontamination |
| HPO | TPE, Hyperband, ASHA | **entrenar** 0.7 | ✅ Complete | `apr tune --strategy tpe` |
| Compile to binary | Model + runtime → executable | **aprender** 0.27 | ✅ Complete | `apr compile` |
| Correctness proofs | Kani bounded model checking | **provable-contracts** | ✅ Complete | 262 proof obligations |
| Quality gates | Compliance enforcement | **pmat** | ✅ Complete | 30+ automated checks |
| **DPO/ORPO alignment** | Preference optimization | **entrenar** 0.7 | ❌ **Missing** | **Must build** (see §5.2) |
| **Execution sandbox** | Run generated code safely | — | ❌ **Missing** | **External harness** (see §5.3) |
| **N-sampling + rerank** | Batched generation, voting | **aprender** 0.27 | ⚠️ Partial | Generation works; reranking logic needed |
| **Prompt templates** | SCoT, few-shot strategies | **aprender** 0.27 | ⚠️ Partial | `--system` exists; `--prompt-strategy` needed |
| **Synthetic data gen** | Teacher → training corpus | **alimentar** 0.2 + **aprender** | ⚠️ Partial | Generation via `apr chat --batch`; curation pipeline needed |
| **Continued pretraining** | Full-weight code corpus training | **entrenar** 0.7 | ⚠️ Partial | Full finetune works; needs large-corpus streaming |
| **Flash Attention** | Online softmax, tiled attention | **trueno** 0.16 | 🔧 In Progress | Phase 12 planned; tiling infra ready |

## 2. Gap 1: DPO/ORPO Preference Optimization (CRITICAL)

**Why world-class:** DPO is the single most impactful post-training technique for leaderboards. Merged + DPO models "completely dominate" HF leaderboard rankings. Without DPO, we compete with one hand tied.

**Current state:** entrenar has the training infrastructure (autograd, AdamW, LoRA) but no DPO loss function or preference pair data loader.

**Wire-in plan:**

```
Component: entrenar
  Add: src/dpo/mod.rs — DPO loss (β-scaled log-ratio of policy vs reference)
  Add: src/dpo/data.rs — preference pair loader (chosen/rejected format)
  Add: src/dpo/orpo.rs — ORPO variant (no reference model needed)

Component: aprender (apr-cli)
  Add: `apr align` subcommand
    apr align model.apr --method dpo \
      --reference base.apr \
      --data preference-pairs.jsonl \
      --beta 0.1 --epochs 3 \
      -o aligned.apr

    apr align model.apr --method orpo \
      --data preference-pairs.jsonl \
      --lambda 0.1 --epochs 3 \
      -o aligned.apr

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

## 3. Gap 2: Code Execution Sandbox (CRITICAL)

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

## 4. Gap 3: N-Sampling + Reranking Pipeline

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

## 5. Gap 4: Synthetic Training Data Pipeline

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

## 6. Gap 5: Prompt Strategy Engine

**Why world-class:** SCoT prompting improves HumanEval pass@1 by up to 13.79%. Few-shot exemplars add 3-8%. The prompt template matters as much as the model weights.

**Current state:** `apr chat --system` and `apr run --chat` exist. Missing: `--prompt-strategy` flag with built-in templates.

**Wire-in plan:**

```
Component: aprender (apr-cli)
  Add: Prompt strategy registry
    apr eval model.apr --data humaneval.jsonl \
      --prompt-strategy scot --json

    apr eval model.apr --data humaneval.jsonl \
      --prompt-strategy few-shot \
      --exemplars exemplars.jsonl --json

  Built-in strategies:
    standard  — raw problem → code (baseline)
    scot      — structured chain-of-thought → code (+5-14%)
    few-shot  — N exemplars + problem → code (+3-8%)
    cgo       — chain of grounded objectives → code (+5-10%)
    reflexion — generate → test → reflect → regenerate (multi-turn)

Component: realizar
  Already supports: chat templates (ChatML, LLaMA2, Mistral, Phi, Alpaca)
  Need: expose template composition for eval pipeline
```

## 7. Sovereign Stack Version Requirements

All gap closures must use published crates from crates.io. No git dependencies.

| Crate | Current | Required For Gaps | Minimum Version |
|-------|---------|-------------------|-----------------|
| aprender | 0.27.2 | `apr align`, `--prompt-strategy`, `--n-samples --rerank` | **0.28** |
| entrenar | 0.7.5 | DPO loss, preference pair loader, ORPO | **0.8** |
| trueno | 0.16.1 | Flash attention (Phase 12) | **0.17** |
| realizar | 0.8.0 | Batch N-sampling, prompt template composition | **0.9** |
| alimentar | 0.2.6 | Decontamination pipeline, preference pair generation, quality filtering | **0.3** |
| provable-contracts | 0.1 | DPO kernel contracts | **0.2** |

## 8. The Decision Rule

When we find a gap:

1. **Can an existing sovereign crate do it?** → Wire it in via `apr` CLI. No new crates.
2. **Does a sovereign crate need a new module?** → Add it to that crate, publish to crates.io, bump apr-leaderboard's dependency.
3. **Is it fundamentally outside the stack's scope?** → Use an external tool (e.g., EvalPlus for code execution) and document the boundary explicitly.
4. **Is it a research problem with no clear solution?** → Add to §20 Open Questions. Don't block the pipeline.

**Hard rule:** We never add a Python dependency. We never add a C/C++ FFI dependency. If the sovereign stack can't do it in pure Rust, we either build it or scope it out with an explicit boundary.
