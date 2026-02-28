//! Pipeline optimization operations: distill, merge, prune, quantize, compare.
//!
//! These map to `apr` CLI subcommands defined in §6 of the spec.
//! Currently scaffolded — will wire to real aprender API per PMAT-017.

use anyhow::{bail, Result};

/// Distill knowledge from a teacher model to a student model.
///
/// Maps to: `apr distill teacher.apr --student student.apr --strategy progressive`
pub(crate) fn distill(
    teacher: &str,
    student: &str,
    strategy: &str,
    temperature: f64,
    alpha: f64,
    output: &str,
) -> Result<()> {
    validate_model_path(teacher)?;
    validate_model_path(student)?;
    let strategy = DistillStrategy::from_str(strategy)?;

    println!("Distilling knowledge:");
    println!("  Teacher: {teacher}");
    println!("  Student: {student}");
    println!("  Strategy: {strategy:?}");
    println!("  Temperature: {temperature}");
    println!("  Alpha: {alpha}");

    // Scaffold: in production, calls `apr distill`
    println!("  [scaffold] Would run: apr distill {teacher} --student {student} \\");
    println!("    --strategy {strategy:?} --temperature {temperature} --alpha {alpha} -o {output}");
    println!("  Output: {output}");
    Ok(())
}

/// Merge two or more models using a merge strategy.
///
/// Maps to: `apr merge model-a.apr model-b.apr --strategy slerp`
pub(crate) fn merge(
    models: &[String],
    strategy: &str,
    output: &str,
) -> Result<()> {
    if models.len() < 2 {
        bail!("merge requires at least 2 models");
    }
    for m in models {
        validate_model_path(m)?;
    }
    let strategy = MergeStrategy::from_str(strategy)?;

    println!("Merging {} models:", models.len());
    for m in models {
        println!("  - {m}");
    }
    println!("  Strategy: {strategy:?}");

    // Scaffold: in production, calls `apr merge`
    let model_args: Vec<&str> = models.iter().map(String::as_str).collect();
    println!(
        "  [scaffold] Would run: apr merge {} --strategy {strategy:?} -o {output}",
        model_args.join(" ")
    );
    println!("  Output: {output}");
    Ok(())
}

/// Prune a model to reduce size while preserving quality.
///
/// Maps to: `apr prune model.apr --method wanda --target-ratio 0.2`
pub(crate) fn prune(
    model: &str,
    method: &str,
    target_ratio: f64,
    output: &str,
) -> Result<()> {
    validate_model_path(model)?;
    let method = PruneMethod::from_str(method)?;

    if !(0.0..1.0).contains(&target_ratio) {
        bail!("target-ratio must be between 0.0 and 1.0, got {target_ratio}");
    }

    println!("Pruning model:");
    println!("  Model: {model}");
    println!("  Method: {method:?}");
    println!("  Target ratio: {:.0}%", target_ratio * 100.0);

    // Scaffold: in production, calls `apr prune`
    println!("  [scaffold] Would run: apr prune {model} --method {method:?} --target-ratio {target_ratio} -o {output}");
    println!("  Output: {output}");
    Ok(())
}

/// Quantize a model to a lower precision.
///
/// Maps to: `apr quantize model.apr --scheme int4`
pub(crate) fn quantize(
    model: &str,
    scheme: &str,
    output: &str,
) -> Result<()> {
    validate_model_path(model)?;
    let scheme = QuantScheme::from_str(scheme)?;

    println!("Quantizing model:");
    println!("  Model: {model}");
    println!("  Scheme: {scheme:?}");

    // Scaffold: in production, calls `apr quantize`
    println!("  [scaffold] Would run: apr quantize {model} --scheme {scheme:?} -o {output}");
    println!("  Output: {output}");
    Ok(())
}

/// Compare an .apr model against HuggingFace reference implementation.
///
/// Maps to: `apr compare-hf model.apr --json`
pub(crate) fn compare(model: &str) -> Result<()> {
    validate_model_path(model)?;

    println!("Comparing against HuggingFace reference:");
    println!("  Model: {model}");

    // Scaffold: in production, calls `apr compare-hf`
    println!("  [scaffold] Would run: apr compare-hf {model} --json");
    println!("  Parity gap: [scaffold — requires real inference]");
    Ok(())
}

fn validate_model_path(path: &str) -> Result<()> {
    if path.is_empty() {
        bail!("model path cannot be empty");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum DistillStrategy {
    Standard,
    Progressive,
    Ensemble,
}

impl DistillStrategy {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "standard" | "kl" => Ok(Self::Standard),
            "progressive" | "curriculum" => Ok(Self::Progressive),
            "ensemble" => Ok(Self::Ensemble),
            _ => bail!("Unknown distill strategy: {s}. Use standard, progressive, or ensemble"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MergeStrategy {
    Slerp,
    Ties,
    Dare,
    LinearAvg,
}

impl MergeStrategy {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "slerp" => Ok(Self::Slerp),
            "ties" | "ties-merging" => Ok(Self::Ties),
            "dare" | "dare-ties" => Ok(Self::Dare),
            "linear" | "avg" | "linear-avg" => Ok(Self::LinearAvg),
            _ => bail!("Unknown merge strategy: {s}. Use slerp, ties, dare, or linear"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PruneMethod {
    Wanda,
    Magnitude,
    SparseGpt,
}

impl PruneMethod {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "wanda" => Ok(Self::Wanda),
            "magnitude" | "mag" => Ok(Self::Magnitude),
            "sparsegpt" | "sparse-gpt" => Ok(Self::SparseGpt),
            _ => bail!("Unknown prune method: {s}. Use wanda, magnitude, or sparsegpt"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum QuantScheme {
    Int4,
    Int8,
    Q4K,
    Q5K,
    Q6K,
}

impl QuantScheme {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "int4" | "q4" => Ok(Self::Int4),
            "int8" | "q8" => Ok(Self::Int8),
            "q4k" | "q4_k" => Ok(Self::Q4K),
            "q5k" | "q5_k" => Ok(Self::Q5K),
            "q6k" | "q6_k" => Ok(Self::Q6K),
            _ => bail!("Unknown quant scheme: {s}. Use int4, int8, q4k, q5k, or q6k"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DistillStrategy ---
    #[test]
    fn test_distill_strategy_parsing() {
        assert!(matches!(DistillStrategy::from_str("standard").unwrap(), DistillStrategy::Standard));
        assert!(matches!(DistillStrategy::from_str("kl").unwrap(), DistillStrategy::Standard));
        assert!(matches!(DistillStrategy::from_str("progressive").unwrap(), DistillStrategy::Progressive));
        assert!(matches!(DistillStrategy::from_str("curriculum").unwrap(), DistillStrategy::Progressive));
        assert!(matches!(DistillStrategy::from_str("ensemble").unwrap(), DistillStrategy::Ensemble));
        assert!(DistillStrategy::from_str("invalid").is_err());
    }

    // --- MergeStrategy ---
    #[test]
    fn test_merge_strategy_parsing() {
        assert!(matches!(MergeStrategy::from_str("slerp").unwrap(), MergeStrategy::Slerp));
        assert!(matches!(MergeStrategy::from_str("ties").unwrap(), MergeStrategy::Ties));
        assert!(matches!(MergeStrategy::from_str("dare").unwrap(), MergeStrategy::Dare));
        assert!(matches!(MergeStrategy::from_str("linear").unwrap(), MergeStrategy::LinearAvg));
        assert!(MergeStrategy::from_str("invalid").is_err());
    }

    // --- PruneMethod ---
    #[test]
    fn test_prune_method_parsing() {
        assert!(matches!(PruneMethod::from_str("wanda").unwrap(), PruneMethod::Wanda));
        assert!(matches!(PruneMethod::from_str("magnitude").unwrap(), PruneMethod::Magnitude));
        assert!(matches!(PruneMethod::from_str("sparsegpt").unwrap(), PruneMethod::SparseGpt));
        assert!(PruneMethod::from_str("invalid").is_err());
    }

    // --- QuantScheme ---
    #[test]
    fn test_quant_scheme_parsing() {
        assert!(matches!(QuantScheme::from_str("int4").unwrap(), QuantScheme::Int4));
        assert!(matches!(QuantScheme::from_str("int8").unwrap(), QuantScheme::Int8));
        assert!(matches!(QuantScheme::from_str("q4k").unwrap(), QuantScheme::Q4K));
        assert!(matches!(QuantScheme::from_str("q5k").unwrap(), QuantScheme::Q5K));
        assert!(matches!(QuantScheme::from_str("q6k").unwrap(), QuantScheme::Q6K));
        assert!(QuantScheme::from_str("invalid").is_err());
    }

    // --- distill ---
    #[test]
    fn test_distill_runs() {
        let result = distill("teacher.apr", "student.apr", "standard", 3.0, 0.7, "out.apr");
        assert!(result.is_ok());
    }

    #[test]
    fn test_distill_empty_teacher() {
        assert!(distill("", "student.apr", "standard", 3.0, 0.7, "out.apr").is_err());
    }

    #[test]
    fn test_distill_empty_student() {
        assert!(distill("teacher.apr", "", "standard", 3.0, 0.7, "out.apr").is_err());
    }

    #[test]
    fn test_distill_invalid_strategy() {
        assert!(distill("t.apr", "s.apr", "invalid", 3.0, 0.7, "o.apr").is_err());
    }

    #[test]
    fn test_distill_all_strategies() {
        for s in &["standard", "progressive", "ensemble"] {
            assert!(distill("t.apr", "s.apr", s, 3.0, 0.7, "o.apr").is_ok());
        }
    }

    // --- merge ---
    #[test]
    fn test_merge_runs() {
        let models = vec!["a.apr".into(), "b.apr".into()];
        assert!(merge(&models, "slerp", "out.apr").is_ok());
    }

    #[test]
    fn test_merge_too_few_models() {
        let models = vec!["a.apr".into()];
        assert!(merge(&models, "slerp", "out.apr").is_err());
    }

    #[test]
    fn test_merge_empty_model_path() {
        let models = vec!["a.apr".into(), String::new()];
        assert!(merge(&models, "slerp", "out.apr").is_err());
    }

    #[test]
    fn test_merge_three_models() {
        let models = vec!["a.apr".into(), "b.apr".into(), "c.apr".into()];
        assert!(merge(&models, "ties", "out.apr").is_ok());
    }

    #[test]
    fn test_merge_all_strategies() {
        let models = vec!["a.apr".into(), "b.apr".into()];
        for s in &["slerp", "ties", "dare", "linear"] {
            assert!(merge(&models, s, "o.apr").is_ok(), "Failed for {s}");
        }
    }

    // --- prune ---
    #[test]
    fn test_prune_runs() {
        assert!(prune("model.apr", "wanda", 0.2, "out.apr").is_ok());
    }

    #[test]
    fn test_prune_invalid_ratio_high() {
        assert!(prune("model.apr", "wanda", 1.0, "out.apr").is_err());
    }

    #[test]
    fn test_prune_invalid_ratio_negative() {
        assert!(prune("model.apr", "wanda", -0.1, "out.apr").is_err());
    }

    #[test]
    fn test_prune_empty_model() {
        assert!(prune("", "wanda", 0.2, "out.apr").is_err());
    }

    #[test]
    fn test_prune_all_methods() {
        for m in &["wanda", "magnitude", "sparsegpt"] {
            assert!(prune("m.apr", m, 0.2, "o.apr").is_ok(), "Failed for {m}");
        }
    }

    // --- quantize ---
    #[test]
    fn test_quantize_runs() {
        assert!(quantize("model.apr", "int4", "out.apr").is_ok());
    }

    #[test]
    fn test_quantize_empty_model() {
        assert!(quantize("", "int4", "out.apr").is_err());
    }

    #[test]
    fn test_quantize_all_schemes() {
        for s in &["int4", "int8", "q4k", "q5k", "q6k"] {
            assert!(quantize("m.apr", s, "o.apr").is_ok(), "Failed for {s}");
        }
    }

    // --- compare ---
    #[test]
    fn test_compare_runs() {
        assert!(compare("model.apr").is_ok());
    }

    #[test]
    fn test_compare_empty_model() {
        assert!(compare("").is_err());
    }

    // --- validate_model_path ---
    #[test]
    fn test_validate_model_path() {
        assert!(validate_model_path("model.apr").is_ok());
        assert!(validate_model_path("").is_err());
    }
}
