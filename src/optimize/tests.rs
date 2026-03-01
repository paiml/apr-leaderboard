#![allow(unused_variables)] // TempDir bindings held for RAII cleanup
use super::*;

/// Write a valid APR v2 file at the given path for testing.
fn write_test_apr(path: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let bytes = apr_bridge::create_minimal_apr_bytes().unwrap();
    std::fs::write(path, &bytes).unwrap();
}

// --- DistillStrategy ---
#[test]
fn test_distill_strategy_parsing_and_display() {
    assert!(matches!(DistillStrategy::from_str("standard").unwrap(), DistillStrategy::Standard));
    assert!(matches!(DistillStrategy::from_str("kl").unwrap(), DistillStrategy::Standard));
    assert!(matches!(DistillStrategy::from_str("progressive").unwrap(), DistillStrategy::Progressive));
    assert!(matches!(DistillStrategy::from_str("curriculum").unwrap(), DistillStrategy::Progressive));
    assert!(matches!(DistillStrategy::from_str("ensemble").unwrap(), DistillStrategy::Ensemble));
    assert!(DistillStrategy::from_str("invalid").is_err());
    for s in &["standard", "progressive", "ensemble"] {
        assert_eq!(DistillStrategy::from_str(s).unwrap().to_string(), *s);
    }
}

// --- MergeStrategy ---
#[test]
fn test_merge_strategy_parsing_and_display() {
    assert!(matches!(MergeStrategy::from_str("slerp").unwrap(), MergeStrategy::Slerp));
    assert!(matches!(MergeStrategy::from_str("ties").unwrap(), MergeStrategy::Ties));
    assert!(matches!(MergeStrategy::from_str("dare").unwrap(), MergeStrategy::Dare));
    assert!(matches!(MergeStrategy::from_str("linear").unwrap(), MergeStrategy::LinearAvg));
    assert!(matches!(MergeStrategy::from_str("average").unwrap(), MergeStrategy::LinearAvg));
    assert!(matches!(MergeStrategy::from_str("avg").unwrap(), MergeStrategy::LinearAvg));
    assert!(MergeStrategy::from_str("invalid").is_err());
    for s in &["slerp", "ties", "dare", "linear"] {
        assert_eq!(MergeStrategy::from_str(s).unwrap().to_string(), *s);
    }
}

// --- PruneMethod ---
#[test]
fn test_prune_method_parsing_and_display() {
    assert!(matches!(PruneMethod::from_str("wanda").unwrap(), PruneMethod::Wanda));
    assert!(matches!(PruneMethod::from_str("magnitude").unwrap(), PruneMethod::Magnitude));
    assert!(matches!(PruneMethod::from_str("sparsegpt").unwrap(), PruneMethod::SparseGpt));
    assert!(matches!(PruneMethod::from_str("structured").unwrap(), PruneMethod::Structured));
    assert!(matches!(PruneMethod::from_str("depth").unwrap(), PruneMethod::Depth));
    assert!(matches!(PruneMethod::from_str("width").unwrap(), PruneMethod::Width));
    assert!(PruneMethod::from_str("invalid").is_err());
    for s in &["wanda", "magnitude", "sparsegpt", "structured", "depth", "width"] {
        assert_eq!(PruneMethod::from_str(s).unwrap().to_string(), *s);
    }
}

// --- QuantScheme ---
#[test]
fn test_quant_scheme_parsing_and_display() {
    assert!(matches!(QuantScheme::from_str("int4").unwrap(), QuantScheme::Int4));
    assert!(matches!(QuantScheme::from_str("int8").unwrap(), QuantScheme::Int8));
    assert!(matches!(QuantScheme::from_str("q4k").unwrap(), QuantScheme::Q4K));
    assert!(matches!(QuantScheme::from_str("q5k").unwrap(), QuantScheme::Q5K));
    assert!(matches!(QuantScheme::from_str("q6k").unwrap(), QuantScheme::Q6K));
    assert!(QuantScheme::from_str("invalid").is_err());
    for s in &["int4", "int8", "q4k", "q5k", "q6k"] {
        assert_eq!(QuantScheme::from_str(s).unwrap().to_string(), *s);
    }
}

// --- distill ---
fn distill_opts_simple<'a>(teacher: &'a str, student: &'a str, strategy: &'a str) -> DistillOpts<'a> {
    DistillOpts { teacher, student, strategy, temperature: 3.0, alpha: 0.7, epochs: 5, data: None, output: "o.apr" }
}

/// Helper: create distill opts with APR v2 fixtures. Returns (tmp, teacher, student, output).
fn distill_fixture() -> (tempfile::TempDir, String, String, String) {
    let tmp = tempfile::TempDir::new().unwrap();
    let t = tmp.path().join("teacher.apr");
    let s = tmp.path().join("student.apr");
    let o = tmp.path().join("out.apr");
    write_test_apr(t.to_str().unwrap());
    write_test_apr(s.to_str().unwrap());
    (tmp, t.to_str().unwrap().into(), s.to_str().unwrap().into(), o.to_str().unwrap().into())
}

#[test]
fn test_distill_runs() {
    let (tmp, t, s, o) = distill_fixture();
    let opts = DistillOpts { teacher: &t, student: &s, strategy: "standard", temperature: 3.0, alpha: 0.7, epochs: 5, data: None, output: &o };
    assert!(distill(&opts).is_ok());
}

#[test]
fn test_distill_empty_teacher() {
    assert!(distill(&distill_opts_simple("", "student.apr", "standard")).is_err());
}

#[test]
fn test_distill_empty_student() {
    assert!(distill(&distill_opts_simple("teacher.apr", "", "standard")).is_err());
}

#[test]
fn test_distill_invalid_strategy() {
    assert!(distill(&distill_opts_simple("t.apr", "s.apr", "invalid")).is_err());
}

#[test]
fn test_distill_all_strategies() {
    for strat in &["standard", "progressive", "ensemble"] {
        let (tmp, t, s, o) = distill_fixture();
        let opts = DistillOpts { teacher: &t, student: &s, strategy: strat, temperature: 3.0, alpha: 0.7, epochs: 5, data: None, output: &o };
        assert!(distill(&opts).is_ok(), "Failed for {strat}");
    }
}

#[test]
fn test_distill_invalid_temperature() {
    let mut opts = distill_opts_simple("t.apr", "s.apr", "standard");
    opts.temperature = 0.0;
    assert!(distill(&opts).is_err());
    opts.temperature = -1.0;
    assert!(distill(&opts).is_err());
}

#[test]
fn test_distill_invalid_alpha() {
    let mut opts = distill_opts_simple("t.apr", "s.apr", "standard");
    opts.alpha = -0.1;
    assert!(distill(&opts).is_err());
    opts.alpha = 1.1;
    assert!(distill(&opts).is_err());
}

#[test]
fn test_distill_alpha_boundary() {
    let (tmp, t, s, o) = distill_fixture();
    let mut opts = DistillOpts { teacher: &t, student: &s, strategy: "standard", temperature: 3.0, alpha: 0.0, epochs: 5, data: None, output: &o };
    assert!(distill(&opts).is_ok());
    opts.alpha = 1.0;
    assert!(distill(&opts).is_ok());
}

#[test]
fn test_distill_with_data() {
    let (tmp, t, s, o) = distill_fixture();
    let opts = DistillOpts { teacher: &t, student: &s, strategy: "progressive", temperature: 3.0, alpha: 0.7, epochs: 10, data: Some("corpus.jsonl"), output: &o };
    assert!(distill(&opts).is_ok());
}

// --- merge ---
/// Create N valid APR v2 fixtures and an output path. Returns (tmp, model_paths, output).
fn merge_fixture(n: usize) -> (tempfile::TempDir, Vec<String>, String) {
    let tmp = tempfile::TempDir::new().unwrap();
    let models: Vec<String> = (0..n).map(|i| {
        let p = tmp.path().join(format!("model_{i}.apr"));
        write_test_apr(p.to_str().unwrap());
        p.to_str().unwrap().to_string()
    }).collect();
    let output = tmp.path().join("merged.apr").to_str().unwrap().to_string();
    (tmp, models, output)
}

fn merge_opts_simple<'a>(models: &'a [String], strategy: &'a str) -> MergeOpts<'a> {
    MergeOpts { models, strategy, weights: None, base_model: None, density: None, drop_rate: None, output: "o.apr" }
}

#[test]
fn test_merge_runs() {
    let (tmp, models, out) = merge_fixture(2);
    let opts = MergeOpts { models: &models, strategy: "slerp", weights: None, base_model: None, density: None, drop_rate: None, output: &out };
    assert!(merge(&opts).is_ok());
}

#[test]
fn test_merge_too_few_models() {
    let models = vec!["a.apr".into()];
    assert!(merge(&merge_opts_simple(&models, "slerp")).is_err());
}

#[test]
fn test_merge_empty_model_path() {
    let models = vec!["a.apr".into(), String::new()];
    assert!(merge(&merge_opts_simple(&models, "slerp")).is_err());
}

#[test]
fn test_merge_three_models() {
    let (tmp, models, out) = merge_fixture(3);
    let base = tmp.path().join("base.apr");
    write_test_apr(base.to_str().unwrap());
    let base_s = base.to_str().unwrap().to_string();
    let opts = MergeOpts { models: &models, strategy: "ties", weights: None, base_model: Some(&base_s), density: None, drop_rate: None, output: &out };
    assert!(merge(&opts).is_ok());
}

#[test]
fn test_merge_all_strategies() {
    for s in &["slerp", "linear"] {
        let (tmp, models, out) = merge_fixture(2);
        let opts = MergeOpts { models: &models, strategy: s, weights: None, base_model: None, density: None, drop_rate: None, output: &out };
        assert!(merge(&opts).is_ok(), "Failed for {s}");
    }
    for s in &["ties", "dare"] {
        let (tmp, models, out) = merge_fixture(2);
        let base = tmp.path().join("base.apr");
        write_test_apr(base.to_str().unwrap());
        let base_s = base.to_str().unwrap().to_string();
        let opts = MergeOpts { models: &models, strategy: s, weights: None, base_model: Some(&base_s), density: None, drop_rate: None, output: &out };
        assert!(merge(&opts).is_ok(), "Failed for {s}");
    }
}

#[test]
fn test_merge_with_weights() {
    let (tmp, models, out) = merge_fixture(2);
    let opts = MergeOpts { models: &models, strategy: "slerp", weights: Some("0.7,0.3"), base_model: None, density: None, drop_rate: None, output: &out };
    assert!(merge(&opts).is_ok());
}

#[test]
fn test_merge_with_base_model_and_density() {
    let (tmp, models, out) = merge_fixture(2);
    let base = tmp.path().join("base.apr");
    write_test_apr(base.to_str().unwrap());
    let base_s = base.to_str().unwrap().to_string();
    let opts = MergeOpts { models: &models, strategy: "ties", weights: None, base_model: Some(&base_s), density: Some(0.2), drop_rate: None, output: &out };
    assert!(merge(&opts).is_ok());
}

#[test]
fn test_merge_dare_with_drop_rate() {
    let (tmp, models, out) = merge_fixture(2);
    let base = tmp.path().join("base.apr");
    write_test_apr(base.to_str().unwrap());
    let base_s = base.to_str().unwrap().to_string();
    let opts = MergeOpts { models: &models, strategy: "dare", weights: None, base_model: Some(&base_s), density: None, drop_rate: Some(0.3), output: &out };
    assert!(merge(&opts).is_ok());
}

#[test]
fn test_merge_ties_requires_base_model() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let result = merge(&merge_opts_simple(&models, "ties"));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("base-model"));
}

#[test]
fn test_merge_dare_requires_base_model() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    assert!(merge(&merge_opts_simple(&models, "dare")).is_err());
}

#[test]
fn test_merge_weights_must_sum_to_one() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts_simple(&models, "slerp");
    opts.weights = Some("0.5,0.3");
    assert!(merge(&opts).is_err());
}

#[test]
fn test_merge_weights_valid() {
    let (tmp, models, out) = merge_fixture(2);
    let opts = MergeOpts { models: &models, strategy: "slerp", weights: Some("0.6,0.4"), base_model: None, density: None, drop_rate: None, output: &out };
    assert!(merge(&opts).is_ok());
}

#[test]
fn test_merge_weights_count_mismatch() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts_simple(&models, "slerp");
    opts.weights = Some("0.3,0.3,0.4");
    assert!(merge(&opts).is_err());
}

#[test]
fn test_merge_weights_invalid_parse() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts_simple(&models, "slerp");
    opts.weights = Some("abc,def");
    assert!(merge(&opts).is_err());
}

#[test]
fn test_merge_density_out_of_range() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts_simple(&models, "slerp");
    opts.density = Some(1.5);
    assert!(merge(&opts).is_err());
    opts.density = Some(-0.1);
    assert!(merge(&opts).is_err());
}

#[test]
fn test_merge_drop_rate_out_of_range() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts_simple(&models, "slerp");
    opts.drop_rate = Some(1.1);
    assert!(merge(&opts).is_err());
}

// --- prune ---
/// Helper: create a fixture model and output path for prune/quantize tests.
fn prune_quant_fixture() -> (tempfile::TempDir, String, String) {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("model.apr");
    write_test_apr(model_path.to_str().unwrap());
    let output_path = tmp.path().join("out.apr");
    (tmp, model_path.to_str().unwrap().into(), output_path.to_str().unwrap().into())
}

#[test]
fn test_prune_runs() {
    let (_tmp, model, output) = prune_quant_fixture();
    assert!(prune(&model, "wanda", 0.2, None, &output).is_ok());
}

#[test]
fn test_prune_invalid_ratio_high() {
    let (_tmp, model, output) = prune_quant_fixture();
    assert!(prune(&model, "wanda", 1.0, None, &output).is_err());
}

#[test]
fn test_prune_invalid_ratio_negative() {
    let (_tmp, model, output) = prune_quant_fixture();
    assert!(prune(&model, "wanda", -0.1, None, &output).is_err());
}

#[test]
fn test_prune_empty_model() {
    assert!(prune("", "wanda", 0.2, None, "out.apr").is_err());
}

#[test]
fn test_prune_all_methods() {
    for m in &["wanda", "magnitude", "sparsegpt", "structured", "depth", "width"] {
        let (_tmp, model, output) = prune_quant_fixture();
        assert!(prune(&model, m, 0.2, None, &output).is_ok(), "Failed for {m}");
    }
}

#[test]
fn test_prune_with_calibration() {
    let (_tmp, model, output) = prune_quant_fixture();
    assert!(prune(&model, "wanda", 0.3, Some("calib.jsonl"), &output).is_ok());
}

#[test]
fn test_prune_output_is_valid_apr() {
    let (_tmp, model, output) = prune_quant_fixture();
    prune(&model, "magnitude", 0.5, None, &output).unwrap();
    let pruned = apr_bridge::load_apr_as_merge_model(&output);
    assert!(pruned.is_ok(), "Pruned output is not valid APR v2");
    assert!(!pruned.unwrap().is_empty());
}

// --- quantize ---
#[test]
fn test_quantize_runs() {
    let (_tmp, model, output) = prune_quant_fixture();
    assert!(quantize(&model, "int4", None, &output).is_ok());
}

#[test]
fn test_quantize_empty_model() {
    assert!(quantize("", "int4", None, "out.apr").is_err());
}

#[test]
fn test_quantize_all_schemes() {
    for s in &["int4", "int8", "q4k", "q5k", "q6k"] {
        let (_tmp, model, output) = prune_quant_fixture();
        assert!(quantize(&model, s, None, &output).is_ok(), "Failed for {s}");
    }
}

#[test]
fn test_quantize_with_calibration() {
    let (_tmp, model, output) = prune_quant_fixture();
    assert!(quantize(&model, "int4", Some("calib.jsonl"), &output).is_ok());
}

#[test]
fn test_quantize_output_is_valid_apr() {
    let (_tmp, model, output) = prune_quant_fixture();
    quantize(&model, "int8", None, &output).unwrap();
    let quantized = apr_bridge::load_apr_as_merge_model(&output);
    assert!(quantized.is_ok(), "Quantized output is not valid APR v2");
    assert!(!quantized.unwrap().is_empty());
}

// --- compare ---
#[test]
fn test_compare_runs() {
    let (tmp, model, _output) = prune_quant_fixture();
    assert!(compare(&model).is_ok());
    drop(tmp);
}

#[test]
fn test_compare_empty_model() {
    assert!(compare("").is_err());
}

// --- tune ---
#[test]
fn test_tune_strategy_parsing_and_display() {
    assert!(matches!(TuneStrategy::from_str("tpe").unwrap(), TuneStrategy::Tpe));
    assert!(matches!(TuneStrategy::from_str("grid").unwrap(), TuneStrategy::Grid));
    assert!(matches!(TuneStrategy::from_str("random").unwrap(), TuneStrategy::Random));
    assert!(matches!(TuneStrategy::from_str("bayesian").unwrap(), TuneStrategy::Tpe));
    assert!(TuneStrategy::from_str("invalid").is_err());
    for s in &["tpe", "grid", "random"] {
        assert_eq!(TuneStrategy::from_str(s).unwrap().to_string(), *s);
    }
}

#[test]
fn test_tune_runs() {
    let (tmp, model, _output) = prune_quant_fixture();
    assert!(tune(&model, "data.jsonl", "tpe", 3, 3).is_ok());
    drop(tmp);
}

#[test]
fn test_tune_all_strategies() {
    let (tmp, model, _output) = prune_quant_fixture();
    for s in &["tpe", "grid", "random"] {
        assert!(tune(&model, "d.jsonl", s, 2, 1).is_ok(), "Failed for {s}");
    }
    drop(tmp);
}

#[test]
fn test_tune_empty_model() {
    assert!(tune("", "data.jsonl", "tpe", 20, 3).is_err());
}

#[test]
fn test_tune_zero_budget() {
    assert!(tune("model.apr", "data.jsonl", "tpe", 0, 3).is_err());
}

// --- validate_model_path ---
#[test]
fn test_validate_model_path() {
    assert!(validate_model_path("model.apr").is_ok());
    assert!(validate_model_path("").is_err());
}
