use super::*;

fn make_test_result() -> EvalResult {
    EvalResult {
        model: "test.apr".into(),
        benchmark: "humaneval".into(),
        metric: "pass@1".into(),
        score: 0.75,
        samples_evaluated: 164,
        samples_total: 164,
        timestamp: "2026-01-01T00:00:00Z".into(),
        prompt_strategy: "standard".into(),
        n_samples: 1,
        details: EvalDetails {
            pass_at_1: 0.75,
            pass_at_10: Some(0.85),
            pass_at_100: None,
            avg_tokens_generated: 150.0,
            avg_latency_ms: 42.5,
            category_scores: vec![CategoryScore {
                category: "arrays".into(),
                score: 0.8,
                count: 20,
            }],
        },
    }
}

#[test]
fn test_eval_result_serialization() {
    let result = make_test_result();
    let json = serde_json::to_string(&result).unwrap();
    let parsed: EvalResult = serde_json::from_str(&json).unwrap();
    assert!((parsed.details.pass_at_1 - 0.75).abs() < f64::EPSILON);
}

#[test]
fn test_eval_result_json_roundtrip() {
    let result = make_test_result();
    let json = serde_json::to_string_pretty(&result).unwrap();
    let parsed: EvalResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model, "test.apr");
    assert_eq!(parsed.benchmark, "humaneval");
    assert_eq!(parsed.metric, "pass@1");
    assert_eq!(parsed.samples_evaluated, 164);
    assert_eq!(parsed.samples_total, 164);
    assert_eq!(parsed.details.pass_at_10, Some(0.85));
    assert_eq!(parsed.details.pass_at_100, None);
    assert_eq!(parsed.details.category_scores.len(), 1);
    assert_eq!(parsed.details.category_scores[0].category, "arrays");
}

#[test]
fn test_eval_result_no_pass_at_10() {
    let result = EvalResult {
        model: "test.apr".into(),
        benchmark: "bigcodebench".into(),
        metric: "pass@1".into(),
        score: 0.5,
        samples_evaluated: 100,
        samples_total: 1140,
        timestamp: "2026-01-01T00:00:00Z".into(),
        prompt_strategy: "standard".into(),
        n_samples: 1,
        details: EvalDetails {
            pass_at_1: 0.5,
            pass_at_10: None,
            pass_at_100: None,
            avg_tokens_generated: 200.0,
            avg_latency_ms: 100.0,
            category_scores: vec![],
        },
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: EvalResult = serde_json::from_str(&json).unwrap();
    assert!(parsed.details.pass_at_10.is_none());
}

#[test]
fn test_run_benchmark_humaneval() {
    let spec = harness::get_benchmark("humaneval").unwrap();
    let config = EvalConfig::default();
    let result = run_benchmark(&spec, "test.apr", 10, &config).unwrap();
    assert_eq!(result.benchmark, "humaneval");
    assert_eq!(result.samples_evaluated, 10);
    assert_eq!(result.samples_total, 164);
    assert!(result.details.pass_at_10.is_some());
    assert_eq!(result.prompt_strategy, "standard");
    assert_eq!(result.n_samples, 1);
}

#[test]
fn test_run_benchmark_bigcodebench() {
    let spec = harness::get_benchmark("bigcodebench").unwrap();
    let config = EvalConfig::default();
    let result = run_benchmark(&spec, "test.apr", 50, &config).unwrap();
    assert_eq!(result.benchmark, "bigcodebench");
    assert!(result.details.pass_at_10.is_none());
}

#[test]
fn test_run_benchmark_all_samples() {
    let spec = harness::get_benchmark("humaneval").unwrap();
    let config = EvalConfig::default();
    let result = run_benchmark(&spec, "test.apr", 164, &config).unwrap();
    assert_eq!(result.samples_evaluated, 164);
}

#[test]
fn test_run_benchmark_with_scot() {
    let spec = harness::get_benchmark("humaneval").unwrap();
    let config = EvalConfig {
        prompt_strategy: PromptStrategy::SCoT,
        n_samples: 20,
        ..EvalConfig::default()
    };
    let result = run_benchmark(&spec, "test.apr", 10, &config).unwrap();
    assert_eq!(result.prompt_strategy, "scot");
    assert_eq!(result.n_samples, 20);
}

#[test]
fn test_prompt_strategy_parsing() {
    assert!(matches!(PromptStrategy::from_str("standard").unwrap(), PromptStrategy::Standard));
    assert!(matches!(PromptStrategy::from_str("scot").unwrap(), PromptStrategy::SCoT));
    assert!(matches!(PromptStrategy::from_str("few-shot").unwrap(), PromptStrategy::FewShot));
    assert!(matches!(PromptStrategy::from_str("cgo").unwrap(), PromptStrategy::Cgo));
    assert!(matches!(PromptStrategy::from_str("reflexion").unwrap(), PromptStrategy::Reflexion));
    assert!(PromptStrategy::from_str("invalid").is_err());
}

#[test]
fn test_prompt_strategy_aliases() {
    assert!(matches!(PromptStrategy::from_str("default").unwrap(), PromptStrategy::Standard));
    assert!(matches!(PromptStrategy::from_str("structured-cot").unwrap(), PromptStrategy::SCoT));
    assert!(matches!(PromptStrategy::from_str("fewshot").unwrap(), PromptStrategy::FewShot));
    assert!(matches!(PromptStrategy::from_str("code-gen-opt").unwrap(), PromptStrategy::Cgo));
    assert!(matches!(PromptStrategy::from_str("reflect").unwrap(), PromptStrategy::Reflexion));
}

#[test]
fn test_prompt_strategy_display() {
    assert_eq!(PromptStrategy::Standard.to_string(), "standard");
    assert_eq!(PromptStrategy::SCoT.to_string(), "scot");
    assert_eq!(PromptStrategy::FewShot.to_string(), "few-shot");
    assert_eq!(PromptStrategy::Cgo.to_string(), "cgo");
    assert_eq!(PromptStrategy::Reflexion.to_string(), "reflexion");
}

#[test]
fn test_rerank_strategy_display() {
    assert_eq!(RerankStrategy::None.to_string(), "none");
    assert_eq!(RerankStrategy::LogProb.to_string(), "logprob");
    assert_eq!(RerankStrategy::Majority.to_string(), "majority");
}

#[test]
fn test_prompt_strategy_roundtrip() {
    for s in &["standard", "scot", "few-shot", "cgo", "reflexion"] {
        let parsed = PromptStrategy::from_str(s).unwrap();
        assert_eq!(parsed.to_string(), *s);
    }
}

#[test]
fn test_eval_config_default() {
    let config = EvalConfig::default();
    assert!(matches!(config.prompt_strategy, PromptStrategy::Standard));
    assert_eq!(config.n_samples, 1);
    assert!((config.temperature - 0.0).abs() < f64::EPSILON);
    assert!((config.top_p - 0.95).abs() < f64::EPSILON);
    assert!(matches!(config.rerank, RerankStrategy::None));
}

#[test]
fn test_rerank_strategy_parsing() {
    assert!(matches!(RerankStrategy::from_str("none").unwrap(), RerankStrategy::None));
    assert!(matches!(RerankStrategy::from_str("logprob").unwrap(), RerankStrategy::LogProb));
    assert!(matches!(RerankStrategy::from_str("log-prob").unwrap(), RerankStrategy::LogProb));
    assert!(matches!(RerankStrategy::from_str("majority").unwrap(), RerankStrategy::Majority));
    assert!(matches!(RerankStrategy::from_str("voting").unwrap(), RerankStrategy::Majority));
    assert!(RerankStrategy::from_str("invalid").is_err());
}

#[test]
fn test_run_with_config() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2test").unwrap();

    let results_dir = tmp.path().join("results");
    let config = EvalConfig {
        prompt_strategy: PromptStrategy::SCoT,
        n_samples: 5,
        ..EvalConfig::default()
    };
    run_with_config(
        model_path.to_str().unwrap(),
        "humaneval",
        10,
        results_dir.to_str().unwrap(),
        &config,
    )
    .unwrap();

    let entries: Vec<_> = std::fs::read_dir(&results_dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .collect();
    let content = std::fs::read_to_string(entries[0].path()).unwrap();
    let result: EvalResult = serde_json::from_str(&content).unwrap();
    assert_eq!(result.prompt_strategy, "scot");
    assert_eq!(result.n_samples, 5);
}

#[test]
fn test_run_creates_result_file() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_dir = tmp.path().join("models");
    std::fs::create_dir_all(&model_dir).unwrap();
    let model_path = model_dir.join("test.apr");
    std::fs::write(&model_path, b"APR2test").unwrap();

    let results_dir = tmp.path().join("results");
    run(
        model_path.to_str().unwrap(),
        "humaneval",
        0,
        results_dir.to_str().unwrap(),
    )
    .unwrap();

    assert!(results_dir.exists());
    let entries: Vec<_> = std::fs::read_dir(&results_dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .collect();
    assert_eq!(entries.len(), 1);
    let result_file = &entries[0].path();
    assert!(result_file.extension().unwrap() == "json");
}

#[test]
fn test_run_with_sample_limit() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2test").unwrap();

    let results_dir = tmp.path().join("results");
    run(
        model_path.to_str().unwrap(),
        "humaneval",
        10,
        results_dir.to_str().unwrap(),
    )
    .unwrap();

    let entries: Vec<_> = std::fs::read_dir(&results_dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .collect();
    let content = std::fs::read_to_string(entries[0].path()).unwrap();
    let result: EvalResult = serde_json::from_str(&content).unwrap();
    assert_eq!(result.samples_evaluated, 10);
}

#[test]
fn test_run_model_not_found() {
    let result = run("/nonexistent/model.apr", "humaneval", 0, "results/");
    assert!(result.is_err());
}

#[test]
fn test_run_invalid_benchmark() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2test").unwrap();

    let result = run(model_path.to_str().unwrap(), "nonexistent", 0, "results/");
    assert!(result.is_err());
}

#[test]
fn test_print_summary_with_pass_at_10() {
    let result = make_test_result();
    print_summary(&result);
}

#[test]
fn test_print_summary_without_pass_at_10() {
    let mut result = make_test_result();
    result.details.pass_at_10 = None;
    print_summary(&result);
}

#[test]
fn test_show_history_no_results_dir() {
    let result = show_history(None);
    let _ = result;
}

#[test]
fn test_show_history_with_results() {
    let tmp = tempfile::TempDir::new().unwrap();
    let results_dir = tmp.path().join("results");
    std::fs::create_dir_all(&results_dir).unwrap();

    let result = make_test_result();
    let json = serde_json::to_string_pretty(&result).unwrap();
    std::fs::write(results_dir.join("test_20260101.json"), &json).unwrap();

    let content = std::fs::read_to_string(results_dir.join("test_20260101.json")).unwrap();
    let parsed: EvalResult = serde_json::from_str(&content).unwrap();
    assert_eq!(parsed.model, "test.apr");
}

#[test]
fn test_category_score_serialization() {
    let cs = CategoryScore {
        category: "math".into(),
        score: 0.9,
        count: 15,
    };
    let json = serde_json::to_string(&cs).unwrap();
    let parsed: CategoryScore = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.category, "math");
    assert!((parsed.score - 0.9).abs() < f64::EPSILON);
    assert_eq!(parsed.count, 15);
}
