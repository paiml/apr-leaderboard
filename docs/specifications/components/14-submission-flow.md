# Submission Flow

## 14.1 Leaderboard Targets

The `apr-leaderboard submit` command formats results for the target leaderboard's submission API:

| Leaderboard | Flag value | Submission method |
|---|---|---|
| Open LLM Leaderboard | `open-llm-leaderboard` (default) | HF Hub model upload → leaderboard evaluation queue |
| BigCodeBench | `bigcode` / `bigcodebench` | Direct result JSON submission |
| EvalPlus | `evalplus` | HF Hub model upload + EvalPlus-format results |

## 14.2 Submission Pipeline

```bash
# 1. Generate HuggingFace model card
apr eval final.apr --generate-card

# 2. Export to HuggingFace-compatible format
apr export final.apr --format safetensors -o submission/

# 3. Publish to HuggingFace Hub
apr publish submission/ --repo paiml/qwen-coder-7b-apr --private

# 4. Submit results via apr-leaderboard
apr-leaderboard submit --results results/humaneval_20260228.json \
    --model-id paiml/qwen-coder-7b-apr --leaderboard open-llm-leaderboard

# 5. Submit to leaderboard evaluation queue (via HF)
# The leaderboard pulls from your HF repo and runs its own evaluation
```

## 14.3 Model Card Template

The model card (`README.md` in the HF repo) MUST include:

- **Base model:** Qwen2.5-Coder-7B (with HF link)
- **Pipeline stages applied:** distill/finetune/merge/prune/quantize (which ones, in order)
- **Training data:** Summary with decontamination attestation
- **Evaluation results:** pass@1/pass@10 on HumanEval, MBPP, BigCodeBench
- **Infrastructure:** "Built with aprender (Rust, no Python dependencies)"
- **Quantization:** Scheme used, size reduction, quality impact
- **Reproducibility:** Link to pipeline config YAML

## 14.4 Pre-Submission Checklist

Before `apr-leaderboard submit`:

- [ ] `apr check model.apr` passes (format validation via `aprender::format::v2::AprV2Reader` — **wired**)
- [ ] `apr compare-hf model.apr` shows <5% parity gap
- [ ] `pmat comply check --strict` passes
- [ ] Decontamination report shows <1% n-gram overlap
- [ ] Model card generated and reviewed
- [ ] Results JSON includes all required benchmarks

**Implementation note:** The `--pre-submit-check` flag in `apr-leaderboard submit` runs 5 automated checks: APR format validation (via `AprV2Reader::from_reader()`), results JSON parsing, required benchmark presence, model ID format, and model card existence. See §19.6 for wiring status.
