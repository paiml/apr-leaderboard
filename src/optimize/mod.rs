//! Pipeline optimization operations: distill, merge, prune, quantize, compare.
//!
//! These map to `apr` CLI subcommands defined in §6 of the spec.
//! Currently scaffolded — will wire to real aprender API per PMAT-017.

use anyhow::{bail, Result};

/// Configuration for knowledge distillation.
pub(crate) struct DistillOpts<'a> {
    pub teacher: &'a str,
    pub student: &'a str,
    pub strategy: &'a str,
    pub temperature: f64,
    pub alpha: f64,
    pub epochs: usize,
    pub data: Option<&'a str>,
    pub output: &'a str,
}

/// Distill knowledge from a teacher model to a student model.
///
/// Maps to: `apr distill teacher.apr --student student.apr --strategy progressive`
pub(crate) fn distill(opts: &DistillOpts<'_>) -> Result<()> {
    validate_model_path(opts.teacher)?;
    validate_model_path(opts.student)?;
    let strategy = DistillStrategy::from_str(opts.strategy)?;

    println!("Distilling knowledge:");
    println!("  Teacher: {}", opts.teacher);
    println!("  Student: {}", opts.student);
    println!("  Strategy: {strategy:?}");
    println!("  Temperature: {}", opts.temperature);
    println!("  Alpha: {}", opts.alpha);
    println!("  Epochs: {}", opts.epochs);
    if let Some(data) = opts.data {
        println!("  Data: {data}");
    }

    // Scaffold: in production, calls `apr distill`
    println!("  [scaffold] Would run: apr distill {} --student {} \\", opts.teacher, opts.student);
    println!("    --strategy {strategy:?} --temperature {} --alpha {} -o {}", opts.temperature, opts.alpha, opts.output);
    println!("  Output: {}", opts.output);
    Ok(())
}

/// Configuration for model merging.
pub(crate) struct MergeOpts<'a> {
    pub models: &'a [String],
    pub strategy: &'a str,
    pub weights: Option<&'a str>,
    pub base_model: Option<&'a str>,
    pub density: Option<f64>,
    pub drop_rate: Option<f64>,
    pub output: &'a str,
}

/// Merge two or more models using a merge strategy.
///
/// Maps to: `apr merge model-a.apr model-b.apr --strategy slerp`
pub(crate) fn merge(opts: &MergeOpts<'_>) -> Result<()> {
    if opts.models.len() < 2 {
        bail!("merge requires at least 2 models");
    }
    for m in opts.models {
        validate_model_path(m)?;
    }
    let strategy = MergeStrategy::from_str(opts.strategy)?;

    println!("Merging {} models:", opts.models.len());
    for m in opts.models {
        println!("  - {m}");
    }
    println!("  Strategy: {strategy:?}");
    if let Some(w) = opts.weights {
        println!("  Weights: {w}");
    }
    if let Some(base) = opts.base_model {
        println!("  Base model: {base}");
    }
    if let Some(d) = opts.density {
        println!("  Density: {d}");
    }
    if let Some(dr) = opts.drop_rate {
        println!("  Drop rate: {dr}");
    }

    // Scaffold: in production, calls `apr merge`
    let model_args: Vec<&str> = opts.models.iter().map(String::as_str).collect();
    println!(
        "  [scaffold] Would run: apr merge {} --strategy {strategy:?} -o {}",
        model_args.join(" "), opts.output
    );
    println!("  Output: {}", opts.output);
    Ok(())
}

/// Prune a model to reduce size while preserving quality.
///
/// Maps to: `apr prune model.apr --method wanda --target-ratio 0.2`
pub(crate) fn prune(
    model: &str,
    method: &str,
    target_ratio: f64,
    calibration: Option<&str>,
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
    if let Some(cal) = calibration {
        println!("  Calibration: {cal}");
    }

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
    calibration: Option<&str>,
    output: &str,
) -> Result<()> {
    validate_model_path(model)?;
    let scheme = QuantScheme::from_str(scheme)?;

    println!("Quantizing model:");
    println!("  Model: {model}");
    println!("  Scheme: {scheme:?}");
    if let Some(cal) = calibration {
        println!("  Calibration: {cal}");
    }

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
    fn distill_opts<'a>(teacher: &'a str, student: &'a str, strategy: &'a str) -> DistillOpts<'a> {
        DistillOpts { teacher, student, strategy, temperature: 3.0, alpha: 0.7, epochs: 5, data: None, output: "o.apr" }
    }

    #[test]
    fn test_distill_runs() {
        assert!(distill(&distill_opts("teacher.apr", "student.apr", "standard")).is_ok());
    }

    #[test]
    fn test_distill_empty_teacher() {
        assert!(distill(&distill_opts("", "student.apr", "standard")).is_err());
    }

    #[test]
    fn test_distill_empty_student() {
        assert!(distill(&distill_opts("teacher.apr", "", "standard")).is_err());
    }

    #[test]
    fn test_distill_invalid_strategy() {
        assert!(distill(&distill_opts("t.apr", "s.apr", "invalid")).is_err());
    }

    #[test]
    fn test_distill_all_strategies() {
        for s in &["standard", "progressive", "ensemble"] {
            assert!(distill(&distill_opts("t.apr", "s.apr", s)).is_ok());
        }
    }

    #[test]
    fn test_distill_with_data() {
        let mut opts = distill_opts("t.apr", "s.apr", "progressive");
        opts.epochs = 10;
        opts.data = Some("corpus.jsonl");
        assert!(distill(&opts).is_ok());
    }

    // --- merge ---
    fn merge_opts<'a>(models: &'a [String], strategy: &'a str) -> MergeOpts<'a> {
        MergeOpts { models, strategy, weights: None, base_model: None, density: None, drop_rate: None, output: "o.apr" }
    }

    #[test]
    fn test_merge_runs() {
        let models = vec!["a.apr".into(), "b.apr".into()];
        assert!(merge(&merge_opts(&models, "slerp")).is_ok());
    }

    #[test]
    fn test_merge_too_few_models() {
        let models = vec!["a.apr".into()];
        assert!(merge(&merge_opts(&models, "slerp")).is_err());
    }

    #[test]
    fn test_merge_empty_model_path() {
        let models = vec!["a.apr".into(), String::new()];
        assert!(merge(&merge_opts(&models, "slerp")).is_err());
    }

    #[test]
    fn test_merge_three_models() {
        let models = vec!["a.apr".into(), "b.apr".into(), "c.apr".into()];
        assert!(merge(&merge_opts(&models, "ties")).is_ok());
    }

    #[test]
    fn test_merge_all_strategies() {
        let models = vec!["a.apr".into(), "b.apr".into()];
        for s in &["slerp", "ties", "dare", "linear"] {
            assert!(merge(&merge_opts(&models, s)).is_ok(), "Failed for {s}");
        }
    }

    #[test]
    fn test_merge_with_weights() {
        let models = vec!["a.apr".into(), "b.apr".into()];
        let mut opts = merge_opts(&models, "slerp");
        opts.weights = Some("0.7,0.3");
        assert!(merge(&opts).is_ok());
    }

    #[test]
    fn test_merge_with_base_model_and_density() {
        let models = vec!["a.apr".into(), "b.apr".into()];
        let mut opts = merge_opts(&models, "ties");
        opts.base_model = Some("base.apr");
        opts.density = Some(0.2);
        assert!(merge(&opts).is_ok());
    }

    #[test]
    fn test_merge_dare_with_drop_rate() {
        let models = vec!["a.apr".into(), "b.apr".into()];
        let mut opts = merge_opts(&models, "dare");
        opts.base_model = Some("base.apr");
        opts.drop_rate = Some(0.3);
        assert!(merge(&opts).is_ok());
    }

    // --- prune ---
    #[test]
    fn test_prune_runs() {
        assert!(prune("model.apr", "wanda", 0.2, None, "out.apr").is_ok());
    }

    #[test]
    fn test_prune_invalid_ratio_high() {
        assert!(prune("model.apr", "wanda", 1.0, None, "out.apr").is_err());
    }

    #[test]
    fn test_prune_invalid_ratio_negative() {
        assert!(prune("model.apr", "wanda", -0.1, None, "out.apr").is_err());
    }

    #[test]
    fn test_prune_empty_model() {
        assert!(prune("", "wanda", 0.2, None, "out.apr").is_err());
    }

    #[test]
    fn test_prune_all_methods() {
        for m in &["wanda", "magnitude", "sparsegpt"] {
            assert!(prune("m.apr", m, 0.2, None, "o.apr").is_ok(), "Failed for {m}");
        }
    }

    #[test]
    fn test_prune_with_calibration() {
        assert!(prune("m.apr", "wanda", 0.3, Some("calib.jsonl"), "o.apr").is_ok());
    }

    // --- quantize ---
    #[test]
    fn test_quantize_runs() {
        assert!(quantize("model.apr", "int4", None, "out.apr").is_ok());
    }

    #[test]
    fn test_quantize_empty_model() {
        assert!(quantize("", "int4", None, "out.apr").is_err());
    }

    #[test]
    fn test_quantize_all_schemes() {
        for s in &["int4", "int8", "q4k", "q5k", "q6k"] {
            assert!(quantize("m.apr", s, None, "o.apr").is_ok(), "Failed for {s}");
        }
    }

    #[test]
    fn test_quantize_with_calibration() {
        assert!(quantize("m.apr", "int4", Some("calib.jsonl"), "o.apr").is_ok());
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
