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

#[test]
fn test_distill_strategy_display() {
    assert_eq!(DistillStrategy::Standard.to_string(), "standard");
    assert_eq!(DistillStrategy::Progressive.to_string(), "progressive");
    assert_eq!(DistillStrategy::Ensemble.to_string(), "ensemble");
}

#[test]
fn test_distill_strategy_roundtrip() {
    for s in &["standard", "progressive", "ensemble"] {
        let parsed = DistillStrategy::from_str(s).unwrap();
        assert_eq!(parsed.to_string(), *s);
    }
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

#[test]
fn test_merge_strategy_display() {
    assert_eq!(MergeStrategy::Slerp.to_string(), "slerp");
    assert_eq!(MergeStrategy::Ties.to_string(), "ties");
    assert_eq!(MergeStrategy::Dare.to_string(), "dare");
    assert_eq!(MergeStrategy::LinearAvg.to_string(), "linear");
}

#[test]
fn test_merge_strategy_roundtrip() {
    for s in &["slerp", "ties", "dare", "linear"] {
        let parsed = MergeStrategy::from_str(s).unwrap();
        assert_eq!(parsed.to_string(), *s);
    }
}

// --- PruneMethod ---
#[test]
fn test_prune_method_parsing() {
    assert!(matches!(PruneMethod::from_str("wanda").unwrap(), PruneMethod::Wanda));
    assert!(matches!(PruneMethod::from_str("magnitude").unwrap(), PruneMethod::Magnitude));
    assert!(matches!(PruneMethod::from_str("sparsegpt").unwrap(), PruneMethod::SparseGpt));
    assert!(PruneMethod::from_str("invalid").is_err());
}

#[test]
fn test_prune_method_display() {
    assert_eq!(PruneMethod::Wanda.to_string(), "wanda");
    assert_eq!(PruneMethod::Magnitude.to_string(), "magnitude");
    assert_eq!(PruneMethod::SparseGpt.to_string(), "sparsegpt");
}

#[test]
fn test_prune_method_roundtrip() {
    for s in &["wanda", "magnitude", "sparsegpt"] {
        let parsed = PruneMethod::from_str(s).unwrap();
        assert_eq!(parsed.to_string(), *s);
    }
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

#[test]
fn test_quant_scheme_display() {
    assert_eq!(QuantScheme::Int4.to_string(), "int4");
    assert_eq!(QuantScheme::Int8.to_string(), "int8");
    assert_eq!(QuantScheme::Q4K.to_string(), "q4k");
    assert_eq!(QuantScheme::Q5K.to_string(), "q5k");
    assert_eq!(QuantScheme::Q6K.to_string(), "q6k");
}

#[test]
fn test_quant_scheme_roundtrip() {
    for s in &["int4", "int8", "q4k", "q5k", "q6k"] {
        let parsed = QuantScheme::from_str(s).unwrap();
        assert_eq!(parsed.to_string(), *s);
    }
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
fn test_distill_invalid_temperature() {
    let mut opts = distill_opts("t.apr", "s.apr", "standard");
    opts.temperature = 0.0;
    assert!(distill(&opts).is_err());
    opts.temperature = -1.0;
    assert!(distill(&opts).is_err());
}

#[test]
fn test_distill_invalid_alpha() {
    let mut opts = distill_opts("t.apr", "s.apr", "standard");
    opts.alpha = -0.1;
    assert!(distill(&opts).is_err());
    opts.alpha = 1.1;
    assert!(distill(&opts).is_err());
}

#[test]
fn test_distill_alpha_boundary() {
    let mut opts = distill_opts("t.apr", "s.apr", "standard");
    opts.alpha = 0.0;
    assert!(distill(&opts).is_ok());
    opts.alpha = 1.0;
    assert!(distill(&opts).is_ok());
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
    let mut opts = merge_opts(&models, "ties");
    opts.base_model = Some("base.apr");
    assert!(merge(&opts).is_ok());
}

#[test]
fn test_merge_all_strategies() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    for s in &["slerp", "linear"] {
        assert!(merge(&merge_opts(&models, s)).is_ok(), "Failed for {s}");
    }
    // TIES and DARE require base_model
    for s in &["ties", "dare"] {
        let mut opts = merge_opts(&models, s);
        opts.base_model = Some("base.apr");
        assert!(merge(&opts).is_ok(), "Failed for {s}");
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

#[test]
fn test_merge_ties_requires_base_model() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let result = merge(&merge_opts(&models, "ties"));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("base-model"));
}

#[test]
fn test_merge_dare_requires_base_model() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let result = merge(&merge_opts(&models, "dare"));
    assert!(result.is_err());
}

#[test]
fn test_merge_weights_must_sum_to_one() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts(&models, "slerp");
    opts.weights = Some("0.5,0.3");
    assert!(merge(&opts).is_err());
}

#[test]
fn test_merge_weights_valid() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts(&models, "slerp");
    opts.weights = Some("0.6,0.4");
    assert!(merge(&opts).is_ok());
}

#[test]
fn test_merge_weights_count_mismatch() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts(&models, "slerp");
    opts.weights = Some("0.3,0.3,0.4");
    assert!(merge(&opts).is_err());
}

#[test]
fn test_merge_weights_invalid_parse() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts(&models, "slerp");
    opts.weights = Some("abc,def");
    assert!(merge(&opts).is_err());
}

#[test]
fn test_merge_density_out_of_range() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts(&models, "slerp");
    opts.density = Some(1.5);
    assert!(merge(&opts).is_err());
    opts.density = Some(-0.1);
    assert!(merge(&opts).is_err());
}

#[test]
fn test_merge_drop_rate_out_of_range() {
    let models = vec!["a.apr".into(), "b.apr".into()];
    let mut opts = merge_opts(&models, "slerp");
    opts.drop_rate = Some(1.1);
    assert!(merge(&opts).is_err());
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
