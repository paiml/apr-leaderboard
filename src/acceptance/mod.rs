//! Acceptance criteria test harness (§18).
//!
//! Defines the 27 falsifiable acceptance criteria from the leaderboard spec.
//! Each AC has an ID, description, measurement command, and pass/fail gate.

use serde::{Deserialize, Serialize};

/// Category of acceptance criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum AcCategory {
    FormatParity,
    Technique,
    Pipeline,
    Performance,
    Tooling,
}

impl std::fmt::Display for AcCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FormatParity => write!(f, "Format & Parity"),
            Self::Technique => write!(f, "Technique Validation"),
            Self::Pipeline => write!(f, "Pipeline & Quality"),
            Self::Performance => write!(f, "Model Performance"),
            Self::Tooling => write!(f, "Tooling Completeness"),
        }
    }
}

/// A single acceptance criterion definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AcceptanceCriterion {
    pub id: String,
    pub description: String,
    pub category: AcCategory,
    pub measurement: String,
    pub gate: String,
}

impl std::fmt::Display for AcceptanceCriterion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} [{}]", self.id, self.description, self.category)
    }
}

/// Build the full list of 27 acceptance criteria per §18.
pub(crate) fn all_criteria() -> Vec<AcceptanceCriterion> {
    vec![
        // Format & Parity (AC-001, AC-002, AC-014)
        ac("AC-001", "Import produces valid .apr", AcCategory::FormatParity,
           "apr import hf://Qwen/Qwen2.5-Coder-7B && apr check model.apr",
           "apr check passes with valid APR header"),
        ac("AC-002", "Eval produces non-zero perplexity within 10% of HF", AcCategory::FormatParity,
           "apr eval --model model.apr --benchmark wikitext2",
           "Perplexity > 0 and within 10% of reference"),
        ac("AC-014", "HF parity gap <5% on perplexity", AcCategory::FormatParity,
           "apr compare-hf model.apr",
           "Parity gap < 5% on WikiText-2 perplexity"),
        // Technique Validation (AC-003 to AC-010)
        ac("AC-003", "Progressive distillation outperforms untrained student", AcCategory::Technique,
           "apr distill --teacher t.apr --student s.apr --strategy progressive",
           "Student pass@1 > baseline student pass@1"),
        ac("AC-004", "LoRA finetune completes with decreasing loss", AcCategory::Technique,
           "apr finetune --model m.apr --dataset d.jsonl --method lora",
           "Final loss < initial loss"),
        ac("AC-005", "QLoRA uses <50% VRAM vs LoRA", AcCategory::Technique,
           "apr finetune --method qlora vs --method lora",
           "Peak VRAM (qlora) < 50% × peak VRAM (lora)"),
        ac("AC-006", "SLERP merge preserves L2 norms within 5%", AcCategory::Technique,
           "apr merge a.apr b.apr --strategy slerp",
           "Output L2 norm within 5% of input average"),
        ac("AC-007", "TIES merge resolves sign conflicts", AcCategory::Technique,
           "apr merge a.apr b.apr --strategy ties --base-model base.apr",
           "No sign-conflict warnings in output"),
        ac("AC-008", "Wanda prune degrades perplexity <5%", AcCategory::Technique,
           "apr prune --model m.apr --method wanda --target-ratio 0.2",
           "Perplexity increase < 5% at conservative ratio"),
        ac("AC-009", "INT4 quantization produces <50% of FP16 size", AcCategory::Technique,
           "apr quantize --model m.apr --scheme int4",
           "Output size < 50% of input size"),
        ac("AC-010", "Compiled binary has zero external dependencies", AcCategory::Technique,
           "apr compile --model m.apr --release --lto --strip && ldd output",
           "ldd reports statically linked or no dependencies"),
        // Pipeline & Quality (AC-011 to AC-020)
        ac("AC-011", "Full pipeline completes end-to-end", AcCategory::Pipeline,
           "apr pipeline --config recipe-c.toml",
           "Exit code 0, all stages complete"),
        ac("AC-012", "Contract binding coverage >= 95%", AcCategory::Pipeline,
           "pv proof-status",
           ">= 95% of contracts have bindings"),
        ac("AC-013", "pmat comply passes strict mode", AcCategory::Pipeline,
           "pmat comply check --strict",
           "Exit code 0"),
        ac("AC-015", "Provable contract tests pass for Kernel Class E", AcCategory::Pipeline,
           "cargo test -p provable-contracts --features qwen",
           "All tests pass"),
        ac("AC-016", "Training data <1% n-gram overlap", AcCategory::Pipeline,
           "apr validate --data train.jsonl --benchmarks humaneval mbpp --threshold 0.01",
           "Overlap < 1% for all benchmarks"),
        ac("AC-017", "N-sampling generates distinct completions", AcCategory::Pipeline,
           "apr eval --model m.apr --benchmark humaneval --n-samples 20",
           "20 unique completions (no duplicates)"),
        ac("AC-018", "Speculative decoding achieves >= 1.5x throughput", AcCategory::Pipeline,
           "apr run --model m.apr --prompt test --speculative --draft-model draft.apr",
           "Tokens/sec >= 1.5x baseline"),
        ac("AC-019", "SCoT produces structured reasoning before code", AcCategory::Pipeline,
           "apr eval --model m.apr --benchmark humaneval --prompt-strategy scot",
           "Output contains reasoning section before code block"),
        ac("AC-020", "DPO alignment reduces loss over epochs", AcCategory::Pipeline,
           "apr align --model m.apr --data prefs.jsonl --method dpo --epochs 3",
           "Epoch 3 loss < epoch 1 loss"),
        // Model Performance (AC-021 to AC-026)
        ac("AC-021", "Qwen-7B-Instruct baseline >= 85% HumanEval", AcCategory::Performance,
           "apr eval --model qwen-7b-instruct.apr --benchmark humaneval",
           "pass@1 >= 85%"),
        ac("AC-022", "Full pipeline Qwen-7B >= 85% HE, 82% HE+, 80% MBPP", AcCategory::Performance,
           "apr eval --model pipeline-qwen-7b.apr --benchmark humaneval humaneval-plus mbpp",
           "HE >= 85%, HE+ >= 82%, MBPP >= 80%"),
        ac("AC-023", "INT4 quantized loses <2% pass@1 vs FP16", AcCategory::Performance,
           "apr eval int4.apr vs fp16.apr --benchmark humaneval",
           "pass@1 difference < 2%"),
        ac("AC-024", "TIES merged model >= best input on >= 1 benchmark", AcCategory::Performance,
           "apr eval merged.apr --benchmark humaneval mbpp",
           "Score >= max(input scores) on at least one benchmark"),
        ac("AC-025", "Training data quality >= 80/100", AcCategory::Performance,
           "alimentar quality --data train.jsonl",
           "All samples score >= 80/100"),
        ac("AC-026", "Compiled 1.5B INT4 binary <1GB with valid output", AcCategory::Performance,
           "ls -la qwen-coder-1.5b && echo 'print(42)' | ./qwen-coder-1.5b",
           "Binary < 1GB and produces valid Python"),
        // Tooling Completeness (AC-027)
        ac("AC-027", "All tooling gaps have implementation or boundary", AcCategory::Tooling,
           "pmat comply check --gaps",
           "Every §5 gap has wire-in or documented boundary"),
    ]
}

fn ac(id: &str, desc: &str, cat: AcCategory, measurement: &str, gate: &str) -> AcceptanceCriterion {
    AcceptanceCriterion {
        id: id.into(),
        description: desc.into(),
        category: cat,
        measurement: measurement.into(),
        gate: gate.into(),
    }
}

/// List all acceptance criteria, optionally filtered by category.
pub(crate) fn list(category: Option<&str>) -> anyhow::Result<()> {
    let criteria = all_criteria();
    let filter = category.map(str::to_lowercase);

    println!("Acceptance Criteria (§18) — {} total\n", criteria.len());

    let mut current_cat: Option<AcCategory> = None;
    for ac in &criteria {
        if let Some(ref f) = filter {
            let cat_str = ac.category.to_string().to_lowercase();
            if !cat_str.contains(f) {
                continue;
            }
        }
        if current_cat != Some(ac.category) {
            current_cat = Some(ac.category);
            println!("## {}\n", ac.category);
        }
        println!("  {} — {}", ac.id, ac.description);
        println!("    Measure: {}", ac.measurement);
        println!("    Gate:    {}", ac.gate);
        println!();
    }

    Ok(())
}

/// Run scaffold verification for acceptance criteria that can be checked locally.
pub(crate) fn verify_scaffold() -> anyhow::Result<AcReport> {
    let criteria = all_criteria();
    let mut results = Vec::new();

    for ac in &criteria {
        let status = match ac.id.as_str() {
            // These are verifiable by checking CLI subcommands exist
            "AC-001" | "AC-004" | "AC-006" | "AC-007" | "AC-008"
            | "AC-009" | "AC-010" | "AC-011" | "AC-016" | "AC-017"
            | "AC-019" | "AC-020" => AcStatus::Scaffolded,
            // These require external tools
            "AC-012" | "AC-013" | "AC-015" | "AC-027" => AcStatus::External,
            // Everything else requires real model execution
            _ => AcStatus::Pending,
        };
        results.push(AcResult {
            id: ac.id.clone(),
            status,
        });
    }

    let scaffolded = results.iter().filter(|r| r.status == AcStatus::Scaffolded).count();
    let pending = results.iter().filter(|r| r.status == AcStatus::Pending).count();
    let external = results.iter().filter(|r| r.status == AcStatus::External).count();

    println!("Acceptance Criteria Scaffold Verification:");
    println!("  Scaffolded: {scaffolded}/{}",  results.len());
    println!("  Pending (needs real models): {pending}");
    println!("  External (needs tooling): {external}");

    Ok(AcReport { results })
}

/// Status of an acceptance criterion check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum AcStatus {
    Scaffolded,
    Pending,
    External,
}

/// Result of checking a single acceptance criterion.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AcResult {
    pub id: String,
    pub status: AcStatus,
}

/// Report from running acceptance criteria verification.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AcReport {
    pub results: Vec<AcResult>,
}

/// Validate all provable-contracts YAML files in the contracts/ directory.
pub(crate) fn validate_contracts() -> anyhow::Result<usize> {
    let contracts_dir = std::path::Path::new("contracts");
    if !contracts_dir.exists() {
        println!("  No contracts/ directory found");
        return Ok(0);
    }

    let mut count = 0;
    let mut errors = 0;
    for entry in std::fs::read_dir(contracts_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
            count += 1;
            match provable_contracts::schema::parse_contract(&path) {
                Ok(contract) => {
                    let violations = provable_contracts::schema::validate_contract(&contract);
                    let errs: Vec<_> = violations.iter()
                        .filter(|v| v.severity == provable_contracts::error::Severity::Error)
                        .collect();
                    if errs.is_empty() {
                        let eq_count = contract.equations.len();
                        let ob_count = contract.proof_obligations.len();
                        println!("  {} — {} equations, {} obligations",
                            path.display(), eq_count, ob_count);
                    } else {
                        errors += errs.len();
                        println!("  {} — {} validation errors", path.display(), errs.len());
                        for v in &errs {
                            println!("    ERROR: {}", v.message);
                        }
                    }
                }
                Err(e) => {
                    errors += 1;
                    println!("  {} — parse error: {e}", path.display());
                }
            }
        }
    }

    if errors > 0 {
        anyhow::bail!("{errors} contract validation error(s)");
    }
    Ok(count)
}

#[cfg(test)]
mod tests;
