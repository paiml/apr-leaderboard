# Competitive Advantage: Why `apr` Wins

## 11.1 Head-to-Head Comparison

| Aspect | Python Ecosystem | `apr` CLI |
|--------|-----------------|-----------|
| Dependencies | transformers, torch, accelerate, bitsandbytes, peft, trl, vllm | Single binary |
| Setup time | 30-60 min (CUDA toolkit, conda, pip conflicts) | 0 min (`cargo install apr-cli`, wgpu auto-detects any GPU) |
| Merge | 50-line Python script | `apr merge --strategy slerp` |
| Prune | 100+ lines, custom hooks | `apr prune --method wanda` |
| LoRA | peft + trl + custom training loop | `apr finetune --method lora` |
| Distill | Custom training loop, 200+ lines | `apr distill --strategy progressive` |
| Quantize | bitsandbytes or GPTQ, GPU required | `apr quantize --scheme int4` |
| Reproducibility | requirements.txt + CUDA version + random seeds | Deterministic Rust binary |
| Deployment | Docker + CUDA runtime + Python | `apr compile → single binary` (runs on any GPU) |
| CI/CD | Complex, flaky GPU runners | `cargo test` on any machine |
| Auditability | Opaque Python state | `apr check` — 10-stage integrity pipeline |
| Correctness | pytest + hope | `pv proof-status` — Kani bounded model checking |
| Quality gates | Ad-hoc linting | `pmat comply check --strict` — 30+ checks |
| Contracts | None | `#[contract]` macro — compile-time mathematical spec binding |
| Speculative decoding | vLLM config | `apr run --speculative` — native, no runtime |
| N-sampling + rerank | Custom scripts | `apr eval --n-samples 50 --rerank` — single command |
| Preference optimization | trl + custom scripts | `apr align --method dpo/orpo` — integrated |

## 11.2 Why This Matters for Leaderboards

**Speed of iteration.** Leaderboard competition is a feedback loop: optimize →
evaluate → iterate. The faster the loop, the more experiments you can run. `apr`
eliminates setup overhead: no conda environments, no CUDA version conflicts, no
Docker builds. `make pipeline RECIPE=recipe-a-quick-lora` runs the full loop.

**Reproducibility.** Python's dependency hell means two researchers running the
same training script may get different results depending on PyTorch version,
CUDA version, and random seed handling. `apr` is a deterministic Rust binary —
same input, same output, every time.

**Any GPU vendor.** The Python ecosystem is NVIDIA-locked via CUDA. `apr` runs
on AMD (Vulkan), Intel Arc (Vulkan), Apple Silicon (Metal), and NVIDIA (Vulkan
or DX12) via wgpu. This means cheaper hardware, more accessible competition.

## 11.3 What `apr` Does Not Win On (Yet)

Honesty about current limitations:

| Aspect | Python Ecosystem | `apr` CLI | Gap |
|--------|-----------------|-----------|-----|
| Ecosystem maturity | 10+ years, millions of users | New, small community | Large |
| Flash Attention | Native CUDA kernel | Planned (§21) | Medium |
| Model zoo | 500K+ HF models | GGUF/SafeTensors import | Small (import path works) |
| Distributed training | DeepSpeed, FSDP, Megatron | SSH-based cluster (§19.4.1) | Medium |
| Community support | StackOverflow, forums | Spec + dogfooding | Large |

These gaps are real but none are blockers for the leaderboard thesis. The import
path works for every model we target. Flash Attention is a throughput optimization,
not a correctness requirement. Distributed training is not needed for 7B models
on 32 GB VRAM.

## 11.4 The Sovereign Stack Advantage

The deepest competitive advantage is **sovereignty** — zero external runtime
dependencies in production:

```
Python ecosystem:      apr ecosystem:
  Python 3.x             (nothing)
  + PyTorch
  + CUDA toolkit
  + cuDNN
  + transformers
  + tokenizers
  + safetensors
  + ...

  Total: ~6 GB runtime    Total: ~671 KiB binary + model weights
```

A compiled `apr` model is a single file. No Docker. No Python runtime. No CUDA
toolkit. Ship a binary, run it anywhere. This matters for edge deployment,
air-gapped environments, and anywhere dependency management is a cost center.
