# Dogfooding Findings

Real end-to-end dogfooding with Qwen2.5-Coder-1.5B model import, validation, and inference. These findings inform spec updates and upstream `apr` CLI improvements.

## 22.1 Model Import: GGUF vs SafeTensors

Two import paths were tested. Only GGUF produces runnable models today.

### 22.1.1 SafeTensors Import Path (Broken for Inference)

```bash
apr import hf://Qwen/Qwen2.5-Coder-1.5B -o checkpoints/qwen-1.5b.apr
```

**Result:** Import succeeds but inference fails.

- `apr check` score: **F (3/100)** — fails most validation stages
- Produces F16/BF16 tensors
- realizar's fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K (not F16/BF16)
- Error: `Operation 'owned_fused_matmul' not supported: Fused matmul only supports Q4_0/Q8_0/Q4_K/Q5_K/Q6_K, got type 30`
- `apr quantize` also fails: `Failed to dequantize tensor 'model.embed_tokens.weight'` (BF16 embedding)

**Root cause:** SafeTensors import preserves original tensor dtype (BF16). realizar expects quantized tensors for inference. There is no working SafeTensors → quantized pipeline today.

### 22.1.2 GGUF Import Path (Working)

```bash
apr import Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf -o checkpoints/qwen-1.5b-q4k.apr
```

**Result:** Full success.

- `apr check` score: **B+ (85/100)** — 10/10 validation stages pass
- Embedded tokenizer included automatically
- Quantized tensors (Q4_K_M) work with realizar
- File size: 1.1 GB

### 22.1.3 Recommendation

Use pre-quantized GGUF files from HuggingFace for the import step. The SafeTensors path needs upstream work in realizar to support F16/BF16 inference or in `apr import` to auto-quantize on ingest.

## 22.2 Inference Testing

### 22.2.1 CPU Inference (Working)

```bash
apr run checkpoints/qwen2.5-coder-1.5b-q4k.apr \
    "def fibonacci(n):" --max-tokens 128 --no-gpu
```

**Result:** Generates real Python code (correct Fibonacci implementation) in ~20 seconds.

### 22.2.2 GPU Inference (Broken)

```bash
apr run checkpoints/qwen2.5-coder-1.5b-q4k.apr \
    "def fibonacci(n):" --max-tokens 128
```

**Result:** Panic in realizar CUDA path:
```
thread 'main' panicked at 'range end index 1536 out of range for slice of length 1024'
```

Location: `realizar/src/weights_preload_gpu.rs`

**Root cause:** Shape mismatch in CUDA weight preloading for Qwen2.5-Coder-1.5B architecture. CPU inference works fine with `--no-gpu`.

### 22.2.3 `apr serve` (Partial)

`apr serve` loads .apr models but the HTTP server does not bind to a port. This may be an unimplemented feature for the .apr format — serve may only work with raw GGUF files. `apr run` is the reliable path for batch inference in eval scripts.

## 22.3 Validation (`apr check`)

The 10 validation stages for GGUF-imported models:

| Stage | Status | Notes |
|---|---|---|
| Tokenizer | ✅ Pass | Embedded in GGUF import |
| Embedding | ✅ Pass | Q4_K_M quantized |
| RoPE | ✅ Pass | Rotary position embeddings |
| Q/K/V | ✅ Pass | Attention projections |
| Attention | ✅ Pass | Multi-head attention |
| MLP | ✅ Pass | Feed-forward network |
| LayerNorm | ✅ Pass | Layer normalization |
| LM Head | ✅ Pass | Language model head |
| Logits | ✅ Pass | Output logits |
| Sampler | ✅ Pass | Token sampling |

## 22.4 Import Prerequisites

`apr import` for SafeTensors models requires these files in the HF cache:
- `config.json` — model architecture config
- `tokenizer.json` — tokenizer vocabulary

These may not download automatically for all model formats. If missing:
```bash
# Manual download to HF cache
curl -L "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B/resolve/main/config.json" \
    -o ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B/snapshots/<hash>/config.json
```

GGUF imports do not have this issue — all metadata is embedded in the GGUF file.

## 22.5 Pipeline Integration

### 22.5.1 `make verify` Output

All 16 `apr` subcommands respond to `--help`:

```
import       OK      run          OK      serve        OK
chat         OK      finetune     OK      merge        OK
prune        OK      quantize     OK      distill      OK
eval         OK      export       OK      publish      OK
check        OK      compile      OK      bench        OK
inspect      OK
```

### 22.5.2 `make dogfood` Output

All 9 TOML configs validated:
- 5 model configs in `configs/models/`
- 4 recipe configs in `configs/recipes/`

### 22.5.3 `make pipeline-plan` Output

Dry-run correctly shows all stages and commands for each recipe. Example for recipe-a-quick-lora:

```
Pipeline stages: import finetune eval
[import]   apr import hf://Qwen/Qwen2.5-Coder-7B-Instruct -o checkpoints/...
[finetune] apr finetune ... --method lora --rank 16 --learning-rate 0.0002 --epochs 3
[eval]     ./scripts/eval-pass-at-k.sh <benchmark> checkpoints/...
```

## 22.6 Upstream Issues Identified

| Issue | Component | Severity | Workaround |
|---|---|---|---|
| F16/BF16 inference unsupported | realizar | High | Use GGUF import path |
| GPU shape mismatch panic | realizar CUDA | High | `--no-gpu` flag |
| `apr serve` doesn't bind HTTP for .apr | aprender | Medium | Use `apr run` for batch inference |
| SafeTensors import missing auto-quantize | aprender | Medium | Import GGUF instead |
| `apr quantize` fails on BF16 tensors | aprender | Medium | Use pre-quantized GGUF |
| `apr import` missing auto-download of config.json | aprender | Low | Manual download to HF cache |
