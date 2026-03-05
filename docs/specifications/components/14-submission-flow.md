# Submission Flow

## 14.1 Leaderboard Targets

The submission script (`scripts/submit.sh`) exports and publishes to HuggingFace Hub:

| Leaderboard | Flag value | Submission method |
|---|---|---|
| Open LLM Leaderboard | `open-llm-leaderboard` (default) | HF Hub model upload → leaderboard evaluation queue |
| BigCodeBench | `bigcode` / `bigcodebench` | Direct result JSON submission |
| EvalPlus | `evalplus` | HF Hub model upload + EvalPlus-format results |

## 14.2 Submission Pipeline

```bash
# One-command submission (preflight checks → export → model card → dry-run → publish)
make publish CHECKPOINT=checkpoints/final.apr HF_REPO=paiml/qwen-coder-7b-apr

# Or manually:
./scripts/submit.sh checkpoints/final.apr paiml/qwen-coder-7b-apr results/

# The script:
# 1. Runs 4 preflight checks (apr check, pmat comply, results present, repo format)
# 2. Exports to SafeTensors via apr export
# 3. Generates model card with benchmark results table
# 4. Dry-run via apr publish --dry-run
# 5. Prompts for confirmation → apr publish
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

Automated by `scripts/submit.sh` (4 gates that block on failure):

- [x] `apr check model.apr` passes (format validation)
- [x] `pmat comply check --strict` passes
- [x] Evaluation results present in `results/` directory
- [x] HF repo ID matches `org/model` format
- [ ] `apr compare-hf model.apr` shows <5% parity gap (manual)
- [ ] Decontamination report shows <1% n-gram overlap (manual)
- [ ] Model card reviewed (generated automatically, review is manual)
