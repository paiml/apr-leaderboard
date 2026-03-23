# Scientific Foundation (References)

Every technique in this spec has a peer-reviewed or widely-cited basis. References
are grouped by the pipeline stage they support.

## 20.1 Training Techniques

[1] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
*Basis for `apr finetune --method lora`. Rank-16 to rank-64 adapters on Q/V projections.*

[2] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models", NeurIPS 2023.
*Basis for `apr finetune --method qlora`. NF4 base weights + FP16 adapters. 4-8 GB VRAM.*

[3] Hinton et al., "Distilling the Knowledge in a Neural Network", arXiv:1503.02531, 2015.
*Basis for `apr distill`. KL-divergence soft-target transfer from teacher to student.*

[4] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023.
*Basis for `apr align --method dpo`. Preference optimization without reward model.*

[5] Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model", EMNLP 2024.
*Basis for `apr align --method orpo`. No reference model needed — simpler than DPO.*

## 20.2 Model Compression

[6] Sun et al., "A Simple and Effective Pruning Approach for Large Language Models" (Wanda), ICLR 2024.
*Basis for `apr prune --method wanda`. Activation-aware pruning in one shot.*

[7] Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot", ICML 2023.
*Alternative pruning approach. Basis for structured pruning comparisons.*

[8] Yadav et al., "TIES-Merging: Resolving Interference When Merging Models", NeurIPS 2023.
*Basis for `apr merge --strategy ties`. Trim, elect sign, disjoint merge.*

[9] Yu et al., "Language Model is Sometimes a Knowledge Base" (DARE), arXiv:2311.03099, 2023.
*Basis for `apr merge --strategy dare`. Drop and rescale for sparse merging.*

[10] Goddard et al., "Arcee's MergeKit: A Toolkit for Merging Large Language Models", arXiv:2403.13257, 2024.
*Reference implementation for SLERP, TIES, DARE merge strategies.*

## 20.3 GPU Architecture

[20] NVIDIA, "Parallel Thread Execution ISA Version 8.5", 2024.
*PTX is NVIDIA's stable intermediate representation. trueno-gpu writes kernels as PTX string templates in Rust — no nvcc, no CUDA toolkit. JIT-compiled to SASS at runtime by the CUDA driver. This is the same fallback mechanism PyTorch uses for unsupported architectures; trueno-gpu uses it as the primary path (§5.10).*

## 20.4 Inference Optimization

[11] Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
*Basis for `apr run --speculative`. Draft model proposes, main model verifies.*

[12] Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ICLR 2023.
*Basis for N-sampling + majority voting reranking in `apr eval --n-samples --rerank majority`.*

[13] Li et al., "Structured Chain-of-Thought Prompting for Code Generation", ACM TOSEM 2025.
*Basis for `--prompt-strategy scot`. Structure reasoning before code output. **Dogfooding note:** SCoT hurts ≤7B Q4K models (-3.05pp on HumanEval, §22.0). Reasoning overhead consumes token budget. Simple few-shot prompting (+1.83pp) is superior at this scale.*

## 20.4 Benchmarks and Evaluation

[14] Hui et al., "Qwen2.5-Coder Technical Report", arXiv:2409.12186, 2024.
*Primary target model architecture. Baseline scores for HumanEval/MBPP.*

[15] Jain et al., "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code", arXiv:2403.07974, 2024.
*Continuously refreshed benchmark. Contamination-resistant evaluation.*

[16] Zhuo et al., "BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions", arXiv:2406.15877, 2024.
*Practical coding tasks with library usage. Not yet saturated (GPT-4o ~61%).*

[17] NVIDIA, "OpenCodeReasoning: Advancing Data Distillation for Competitive Coding", arXiv:2504.01943, 2025.
*OCR-Nemotron reasoning distillation results. LiveCodeBench SOTA.*

## 20.5 Code Generation Foundations

[18] Rozière et al., "Code Llama: Open Foundation Models for Code", arXiv:2308.12950, 2023.
*Fill-in-middle (FIM) training methodology. Infilling objective for code completion.*

[19] Chen et al., "Evaluating Large Language Models Trained on Code" (Codex/HumanEval), arXiv:2107.03374, 2021.
*Defines pass@k metric and unbiased estimator. The benchmark that started it all.*
