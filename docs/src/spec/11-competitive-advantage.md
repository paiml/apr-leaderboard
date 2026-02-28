#  Competitive Advantage: Why `apr` Wins

| Aspect | Python Ecosystem | `apr` CLI |
|--------|-----------------|-----------|
| Dependencies | transformers, torch, accelerate, bitsandbytes, peft, trl, vllm | Single binary |
| Setup time | 30-60 min (CUDA toolkit, conda, pip conflicts) | 0 min (`cargo install apr-cli`, trueno generates PTX natively) |
| Merge | 50-line Python script | `apr merge --strategy slerp` |
| Prune | 100+ lines, custom hooks | `apr prune --method wanda` |
| LoRA | peft + trl + custom training loop | `apr finetune --method lora` |
| Distill | Custom training loop, 200+ lines | `apr distill --strategy progressive` |
| Quantize | bitsandbytes or GPTQ, GPU required | `apr quantize --scheme int4` |
| Reproducibility | requirements.txt + CUDA version + random seeds | Deterministic Rust binary |
| Deployment | Docker + CUDA runtime + Python | `apr compile → single binary` |
| CI/CD | Complex, flaky GPU runners | `cargo test` on any machine |
| Auditability | Opaque Python state | `apr check` — 10-stage integrity pipeline |
| Correctness | pytest + hope | `pv proof-status` — Kani bounded model checking, 262 proof obligations |
| Quality gates | Ad-hoc linting | `pmat comply check --strict` — 30+ automated compliance checks |
| Contracts | None | `#[contract]` macro — compile-time binding to mathematical spec |
| Speculative decoding | vLLM config | `apr run --speculative` — native, no external runtime |
| N-sampling + rerank | Custom scripts | `apr eval --n-samples 50 --rerank` — single command |
| Preference optimization | trl + custom scripts | `apr align --method dpo/orpo` — integrated |
