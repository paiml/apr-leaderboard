# Success Criteria

| Metric | Target | Stretch | Notes |
|--------|--------|---------|-------|
| HumanEval pass@1 | ≥ apr baseline | ≥ HF reference | Relative to Step 0 baseline |
| MBPP pass@1 | ≥ apr baseline | ≥ HF reference | Relative to Step 0 baseline |
| Inference parity | <5% gap vs HF | <2% gap vs HF | `apr compare-hf` gate |
| Pipeline commands | <= 10 | <= 6 | |
| Total binary size (compiled, 7B INT4) | < 5GB | < 4GB | 3.5GB weights + runtime |
| Wall-clock (import → submit) | < 24h (GPU) | < 8h (GPU) | CPU-only: much longer |
| Python dependencies | 0 | 0 | External sandbox for eval only |
| CUDA toolkit | Not required | Not required | trueno PTX generation |
| GPU hardware | Recommended | Optional (≤7B) | Required for distill/finetune 32B |
