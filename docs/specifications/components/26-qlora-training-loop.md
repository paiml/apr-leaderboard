# 26. QLoRA Training Loop Specification

## 26.1 Problem Statement

`apr finetune --method qlora` creates LoRA adapter tensors with random initialization and writes an APR file, but does not execute training. The `execute_training()` function in `aprender/crates/apr-cli/src/commands/finetune.rs` is a stub — it calls `create_lora_tensors()` (Kaiming init) and exits.

**Root cause:** aprender has no training loop. The training loop exists in entrenar (`InstructPipeline::train_step`) but is not wired to the `apr finetune` CLI.

## 26.2 Existing Infrastructure Audit

### 26.2.1 What EXISTS (entrenar)

| Component | Location | Status |
|-----------|----------|--------|
| Autograd engine | `entrenar/src/autograd/` | Tape-based, backward ops for matmul, attention, activations, normalize |
| AdamW optimizer | `entrenar/src/optim/adamw.rs` | Full implementation with decoupled weight decay |
| LR schedulers | `entrenar/src/optim/scheduler/` | Cosine decay, linear warmup, step decay |
| Cross-entropy loss | `entrenar/src/finetune/classification.rs:577` | With autograd backward |
| Causal LM loss | `entrenar/src/finetune/instruct_pipeline.rs` | Response-only masking |
| LoRA layers | `entrenar/src/finetune/instruct_pipeline.rs` | `LoraLinear` with trainable A/B |
| Training loop | `entrenar/src/finetune/instruct_trainer.rs:156` | Epoch management, validation, checkpointing, early stopping |
| `train_step` | `entrenar/src/finetune/instruct_pipeline.rs:574` | Forward → loss → backward → optimizer, CPU + CUDA paths |
| Gradient clipping | `entrenar/src/finetune/instruct_pipeline.rs` | Max-norm clipping |
| CUDA training | `entrenar/src/autograd/cuda_training.rs` | NF4 QLoRA on GPU |
| Memory planner | `entrenar-lora/src/memory.rs` | VRAM estimation for QLoRA configs |
| Merge engine | `entrenar-lora/src/merge.rs` | Adapter merge into base model |

### 26.2.2 What EXISTS (aprender)

| Component | Location | Status |
|-----------|----------|--------|
| CLI `finetune` command | `apr-cli/src/commands/finetune.rs` | Parses args, plans config, creates adapter APR — **no training** |
| LoRA tensor creation | `apr-cli/src/commands/finetune.rs:create_lora_tensors` | Kaiming init A, zero B |
| APR writer | `aprender/src/serialization/apr.rs` | Writes .apr with metadata + tensors |
| Model loading | `realizar/src/gguf/` | `OwnedQuantizedModel` from .apr files |

### 26.2.3 What is MISSING

| Component | Gap | Required For |
|-----------|-----|-------------|
| Wiring `InstructPipeline` into `apr finetune` | `execute_training()` creates tensors but doesn't call entrenar | Training execution |
| APR model → entrenar model bridge | `OwnedQuantizedModel` → entrenar's model trait | Forward pass in training |
| Data loader for JSONL | Parse `{"instruction": ..., "response": ...}` → tokenized pairs | Training data |
| Checkpoint-to-APR export | Save trained LoRA weights back to .apr format | Output |
| Tokenizer integration | APR sibling tokenizer → entrenar tokenizer interface | Tokenization |

## 26.3 Architecture: Bridge Pattern

The fix is NOT reimplementing training in aprender. The fix is **bridging** aprender's model loading + CLI with entrenar's training loop.

```
apr finetune model.apr --method qlora --data train.jsonl --output distilled.apr
    │
    ├── 1. Load model: realizar::OwnedQuantizedModel::from_apr(path)
    ├── 2. Load tokenizer: sibling tokenizer.json
    ├── 3. Load data: parse JSONL → Vec<(instruction, response)>
    ├── 4. Create InstructPipeline with model + tokenizer + LoRA config
    ├── 5. Create InstructTrainer with pipeline + training config
    ├── 6. trainer.train() → epoch loop with loss/backward/optimizer
    ├── 7. Export trained LoRA weights → APR file
    └── 8. Optionally merge: base + adapter → merged APR
```

## 26.4 Mathematical Specification

### 26.4.1 QLoRA Forward Pass (per Dettmers et al. 2023)

For each linear layer `W ∈ ℝ^{m×n}` in the transformer:

```
W_nf4 = NF4Quantize(W)              # 4-bit NormalFloat quantization (frozen)
h_base = Dequant(W_nf4) @ x         # Base model forward (frozen)
h_lora = (x @ A) @ B * (α/r)        # LoRA adapter forward (trainable)
h = h_base + h_lora                  # Combined output
```

Where:
- `A ∈ ℝ^{n×r}` — LoRA down-projection (Kaiming init)
- `B ∈ ℝ^{r×m}` — LoRA up-projection (zero init)
- `r` — LoRA rank (e.g., 32)
- `α` — LoRA alpha scaling (e.g., 64)
- `x ∈ ℝ^{1×n}` — input hidden state

### 26.4.2 Causal Language Model Loss

For a sequence `[t₁, t₂, ..., t_T]` with prompt length `P`:

```
logits = model([t₁, ..., t_T])              # [T, V] logit matrix
loss = -(1/R) Σᵢ₌ₚᵀ⁻¹ log softmax(logitsᵢ)[tᵢ₊₁]   # NLL on response tokens only
```

Where `R = T - P` is the number of response tokens.

### 26.4.3 Backward Pass (LoRA only)

Gradients flow only through LoRA A and B matrices:

```
∂L/∂B = (α/r) · (A^T @ x)^T @ (∂L/∂h)     # Gradient w.r.t. B
∂L/∂A = (α/r) · x^T @ ((∂L/∂h) @ B^T)      # Gradient w.r.t. A
```

Base weights `W_nf4` receive no gradient (frozen).

### 26.4.4 AdamW Update (per Loshchilov & Hutter 2017)

For each LoRA parameter `θ ∈ {A, B}`:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          # First moment
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          # Second moment
m̂_t = m_t / (1 - β₁ᵗ)                         # Bias-corrected first moment
v̂_t = v_t / (1 - β₂ᵗ)                         # Bias-corrected second moment
θ_t = θ_{t-1} - lr · (m̂_t / (√v̂_t + ε) + λ · θ_{t-1})  # Decoupled weight decay
```

Default hyperparameters: `β₁=0.9, β₂=0.999, ε=1e-8, λ=0.01`.

### 26.4.5 Learning Rate Schedule (Cosine with Warmup)

```
if step < warmup_steps:
    lr = lr_base * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + cos(π * progress))
```

## 26.5 Memory Model

For a model with `P` parameters, LoRA rank `r`, and `L` adapted layers:

```
Trainable params:    T = 2 · r · d · L        (A and B per layer, d = layer dim)
Base model:          P_bytes / 2               (NF4 = 0.5 bytes/param)
LoRA adapters:       T × 4 bytes              (FP32)
Optimizer states:    T × 8 bytes              (m + v, both FP32)
Activations:         S × d × 4 bytes          (seq_len × hidden, FP32)
Gradients:           T × 4 bytes              (one gradient per trainable param)

Total ≈ P/2 + 16·T + S·d·4
```

For 7B Q4K, rank 32, 28 layers:
- Base model: 3.75 GB (Q4K)
- LoRA: 2 × 32 × 3584 × 28 × 7 projections ≈ 140M params × 4 = 0.56 GB
- Optimizer: 140M × 8 = 1.12 GB
- Activations: 512 × 3584 × 4 ≈ 7 MB (single sample, no batching)
- **Total: ~5.5 GB** (fits easily on gx10 119 GB)

## 26.6 Provable Contracts

### 26.6.1 Required Contracts (from `../provable-contracts`)

| Contract | File | Equations Used |
|----------|------|---------------|
| `lora-algebra-v1` | `lora-algebra-v1.yaml` | `lora_shape`, `task_vector` |
| `adamw-kernel-v1` | `adamw-kernel-v1.yaml` | `adam_moments`, `adam_variance`, `bias_correction`, `weight_update` |
| `loss-functions-v1` | `loss-functions-v1.yaml` | `nll` (causal LM loss = NLL on response tokens) |
| `classification-finetune-v1` | `classification-finetune-v1.yaml` | `softmax_sum`, `label_bounds` |

### 26.6.2 New Contract: `qlora-training-loop-v1`

```yaml
metadata:
  version: 1.0.0
  description: QLoRA training loop — frozen base + trainable LoRA adapters
  depends_on:
    - lora-algebra-v1
    - adamw-kernel-v1
    - loss-functions-v1
equations:
  frozen_base:
    formula: ∂L/∂W_base = 0 (no gradient flows to base weights)
    invariants:
      - Base weights unchanged after training step
      - Only LoRA A/B receive gradients
  lora_forward:
    formula: h = dequant(W_nf4) @ x + (x @ A) @ B * (α/r)
    invariants:
      - Output shape matches base layer output shape
      - LoRA contribution is zero when B is zero-initialized
  response_only_loss:
    formula: loss computed only on response tokens (positions P..T-1)
    invariants:
      - Prompt tokens do not contribute to loss
      - Loss is NLL (non-negative)
  loss_decreasing:
    formula: E[L(θ_{t+1})] < E[L(θ_t)] for sufficiently small lr
    invariants:
      - Training makes progress (loss decreasing in expectation)
```

### 26.6.3 Contract Annotations on Functions

```rust
#[provable_contracts_macros::contract("qlora-training-loop-v1", equation = "frozen_base")]
fn train_step(/* ... */) { /* ... */ }

#[provable_contracts_macros::contract("adamw-kernel-v1", equation = "weight_update")]
fn optimizer_step(/* ... */) { /* ... */ }

#[provable_contracts_macros::contract("loss-functions-v1", equation = "nll")]
fn compute_causal_lm_loss(/* ... */) { /* ... */ }

#[provable_contracts_macros::contract("lora-algebra-v1", equation = "lora_shape")]
fn create_lora_layer(/* ... */) { /* ... */ }
```

### 26.6.4 Falsification Tests

| ID | Rule | Prediction | Test |
|----|------|-----------|------|
| FT-001 | Frozen base | Base weights identical before/after `train_step` | Hash base weights, compare after N steps |
| FT-002 | LoRA zero init | First forward pass without training = base model output | Compare logits: model vs model+LoRA(B=0) |
| FT-003 | Response-only loss | Changing prompt tokens doesn't change loss gradient | Perturb prompt, verify same gradient on LoRA |
| FT-004 | Loss non-negative | NLL loss >= 0 for all inputs | proptest with random logits and labels |
| FT-005 | Loss decreasing | Loss at step N < loss at step 0 (averaged over 10 runs) | Train 100 steps, compare first vs last loss |
| FT-006 | AdamW decoupled | Weight decay applied to θ, not gradient | Compare with L2-regularized Adam |
| FT-007 | Shape preservation | LoRA output shape = base layer output shape | proptest with random dimensions |
| FT-008 | Gradient flow | ∂L/∂A ≠ 0 and ∂L/∂B ≠ 0 after first step (B no longer zero) | Check gradient norms after step 1 |

## 26.7 Implementation Plan

### Phase 1: Bridge `apr finetune` → entrenar (aprender change)

**File:** `aprender/crates/apr-cli/src/commands/finetune.rs`

Replace the stub `execute_training()` with:

```rust
fn execute_training(
    model_path: &Path,
    config: &OptimalConfig,
    data_path: &Path,
    output_path: &Path,
    epochs: u32,
    learning_rate: f64,
    json_output: bool,
) -> Result<()> {
    // 1. Load Q4K model via realizar
    let mapped = realizar::apr::MappedAprModel::from_path(model_path)?;
    let model = realizar::gguf::OwnedQuantizedModel::from_apr(&mapped)?;

    // 2. Load tokenizer (sibling .tokenizer.json)
    let tokenizer = load_sibling_tokenizer(model_path)?;

    // 3. Load JSONL training data
    let samples = load_instruct_jsonl(data_path)?;

    // 4. Create InstructPipeline (entrenar)
    let pipeline_config = InstructPipelineConfig {
        rank: config.rank,
        alpha: config.alpha,
        learning_rate: learning_rate as f32,
        max_seq_len: 512,
        gradient_clip_norm: Some(1.0),
        ..Default::default()
    };
    let pipeline = InstructPipeline::from_quantized_model(model, tokenizer, pipeline_config)?;

    // 5. Create InstructTrainer
    let train_config = InstructTrainingConfig {
        epochs: epochs as usize,
        val_split: 0.1,
        early_stopping_patience: 5,
        checkpoint_dir: output_path.parent().unwrap().join("checkpoints"),
        ..Default::default()
    };
    let mut trainer = InstructTrainer::new(pipeline, samples, train_config);

    // 6. Train
    let result = trainer.train();

    // 7. Export trained LoRA weights to APR
    export_lora_to_apr(trainer.pipeline(), output_path, model_path)?;

    // 8. Report
    report_training_result(&result, json_output);
    Ok(())
}
```

### Phase 2: Model Bridge (`InstructPipeline::from_quantized_model`)

**File:** `entrenar/src/finetune/instruct_pipeline.rs`

New constructor that accepts `OwnedQuantizedModel` instead of requiring SafeTensors:

```rust
/// Create InstructPipeline from a quantized APR/GGUF model.
/// Base weights stay in Q4K form (frozen). LoRA adapters are FP32 (trainable).
/// Forward: dequant(Q4K) @ x + (x @ A) @ B * (α/r)
#[provable_contracts_macros::contract("qlora-training-loop-v1", equation = "lora_forward")]
pub fn from_quantized_model(
    model: OwnedQuantizedModel,
    tokenizer: Tokenizer,
    config: InstructPipelineConfig,
) -> Result<Self> {
    // Wrap Q4K model in trait object that implements forward()
    // LoRA layers inject at q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    // Base weights frozen (no gradient). Only LoRA A/B are trainable.
    // ...
}
```

### Phase 3: APR Export

**File:** `aprender/crates/apr-cli/src/commands/finetune.rs`

```rust
/// Export trained LoRA A/B weights from pipeline to APR format.
#[provable_contracts_macros::contract("lora-algebra-v1", equation = "lora_shape")]
fn export_lora_to_apr(
    pipeline: &InstructPipeline,
    output_path: &Path,
    base_model_path: &Path,
) -> Result<()> {
    let mut writer = AprWriter::new();
    // Write metadata (base model, rank, alpha, training config)
    // Write LoRA A/B tensors (trained weights, not random init)
    // Copy tokenizer from base model
    // ...
}
```

### Phase 4: Merge Support

```bash
# Train adapter
apr finetune model.apr --method qlora --data train.jsonl --output adapter.apr

# Merge adapter into base
apr finetune model.apr --adapter adapter.apr --merge --output merged.apr

# Evaluate merged model
make eval-humaneval CHECKPOINT=checkpoints/merged.apr
```

## 26.8 Test Plan

| Test | Type | Validates |
|------|------|-----------|
| `test_train_step_decreases_loss` | Integration | Loss at step 10 < loss at step 0 |
| `test_base_weights_frozen` | Unit | Base model weights unchanged after training |
| `test_lora_zero_init` | Unit | B=0 init → LoRA contribution = 0 |
| `test_response_only_loss` | Unit | Prompt tokens don't contribute to gradient |
| `test_adamw_decoupled` | Unit | AdamW ≠ L2-regularized Adam |
| `test_export_reimport` | Integration | Export → import → same adapter weights |
| `test_merged_model_inference` | Integration | Merged model produces valid completions |
| `test_99_completions_training` | E2E | Train on teacher completions, verify loss decrease |

## 26.9 Acceptance Criteria

- **AC-FT-001:** `apr finetune model.apr --method qlora --data train.jsonl` trains for N epochs with decreasing loss
- **AC-FT-002:** Training produces an APR file with trained LoRA weights (not random init)
- **AC-FT-003:** Merged model passes `apr check` and produces valid inference output
- **AC-FT-004:** All 8 falsification tests from §26.6.4 pass
- **AC-FT-005:** All 4 provable contracts annotated and verified
- **AC-FT-006:** 7B QLoRA on 99 teacher completions completes in < 1 hour on gx10
- **AC-FT-007:** Distilled 7B model achieves ≥ 85% pass@1 on HumanEval (no regression from baseline)

## 26.10 References

- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" arXiv:2106.09685
- Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs" arXiv:2305.14314
- Loshchilov & Hutter (2017) "Decoupled Weight Decay Regularization" arXiv:1711.05101
- Eckart-Young-Mirsky theorem (1936) — optimal low-rank approximation
