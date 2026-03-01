use super::*;

fn make_test_submission() -> Submission {
    Submission {
        model_id: "org/model".into(),
        results: serde_json::json!({"pass@1": 0.85}),
        model_type: "pretrained".into(),
        precision: "float16".into(),
        weight_type: "Original".into(),
        leaderboard: "test".into(),
        submitted_at: "2026-01-01T00:00:00Z".into(),
    }
}

// Clone for test helper only
impl Clone for Submission {
    fn clone(&self) -> Self {
        Self {
            model_id: self.model_id.clone(),
            results: self.results.clone(),
            model_type: self.model_type.clone(),
            precision: self.precision.clone(),
            weight_type: self.weight_type.clone(),
            leaderboard: self.leaderboard.clone(),
            submitted_at: self.submitted_at.clone(),
        }
    }
}

#[test]
fn test_leaderboard_parsing() {
    assert!(matches!(Leaderboard::from_str("open-llm-leaderboard"), Leaderboard::OpenLlm));
    assert!(matches!(Leaderboard::from_str("bigcode"), Leaderboard::BigCode));
    assert!(matches!(Leaderboard::from_str("evalplus"), Leaderboard::EvalPlus));
    assert!(matches!(Leaderboard::from_str("custom-board"), Leaderboard::Custom(_)));
}

#[test]
fn test_leaderboard_aliases() {
    assert!(matches!(Leaderboard::from_str("open-llm"), Leaderboard::OpenLlm));
    assert!(matches!(Leaderboard::from_str("bigcode-leaderboard"), Leaderboard::BigCode));
}

#[test]
fn test_leaderboard_submission_repos() {
    assert_eq!(Leaderboard::OpenLlm.submission_repo(), "open-llm-leaderboard/requests");
    assert_eq!(Leaderboard::BigCode.submission_repo(), "bigcode/bigcode-models-leaderboard");
    assert_eq!(Leaderboard::EvalPlus.submission_repo(), "evalplus/evalplus-results");
    assert_eq!(Leaderboard::Custom("my/board".into()).submission_repo(), "my/board");
}

#[test]
fn test_leaderboard_debug() {
    assert_eq!(format!("{:?}", Leaderboard::OpenLlm), "OpenLlm");
    assert_eq!(format!("{:?}", Leaderboard::BigCode), "BigCode");
    assert_eq!(format!("{:?}", Leaderboard::EvalPlus), "EvalPlus");
}

#[test]
fn test_validate_submission() {
    let valid = make_test_submission();
    assert!(validate_submission(&valid).is_ok());
    let no_slash = Submission { model_id: "model-without-org".into(), ..valid.clone() };
    assert!(validate_submission(&no_slash).is_err());
    let empty = Submission { model_id: String::new(), ..valid };
    assert!(validate_submission(&empty).is_err());
}

#[test]
fn test_validate_submission_error_messages() {
    let empty = Submission { model_id: String::new(), ..make_test_submission() };
    let err = validate_submission(&empty).unwrap_err();
    assert!(err.to_string().contains("empty"));
    let no_slash = Submission { model_id: "no-org".into(), ..make_test_submission() };
    let err = validate_submission(&no_slash).unwrap_err();
    assert!(err.to_string().contains("org/name"));
}

#[test]
fn test_submission_serialization() {
    let sub = make_test_submission();
    let json = serde_json::to_string(&sub).unwrap();
    let parsed: Submission = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model_id, "org/model");
    assert_eq!(parsed.model_type, "pretrained");
}

#[test]
fn test_submission_json_roundtrip() {
    let sub = make_test_submission();
    let json = serde_json::to_string_pretty(&sub).unwrap();
    let parsed: Submission = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.results, serde_json::json!({"pass@1": 0.85}));
}

#[test]
fn test_leaderboard_display() {
    assert_eq!(Leaderboard::OpenLlm.to_string(), "open-llm-leaderboard");
    assert_eq!(Leaderboard::BigCode.to_string(), "bigcode");
    assert_eq!(Leaderboard::EvalPlus.to_string(), "evalplus");
    assert_eq!(Leaderboard::Custom("my-board".into()).to_string(), "my-board");
}

#[test]
fn test_leaderboard_roundtrip() {
    for s in &["open-llm-leaderboard", "bigcode", "evalplus"] {
        let parsed = Leaderboard::from_str(s);
        assert_eq!(parsed.to_string(), *s);
    }
}

#[test]
fn test_run_creates_submission_file() {
    let tmp = tempfile::TempDir::new().unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"pass@1": 0.85}"#).unwrap();
    std::fs::create_dir_all("results").ok();
    let result = run(results_path.to_str().unwrap(), "org/model", "bigcode");
    assert!(result.is_ok());
}

#[test]
fn test_run_missing_results_file() {
    let result = run("/nonexistent/results.json", "org/model", "bigcode");
    assert!(result.is_err());
}

#[test]
fn test_run_invalid_json() {
    let tmp = tempfile::TempDir::new().unwrap();
    let results_path = tmp.path().join("bad.json");
    std::fs::write(&results_path, "not json").unwrap();
    let result = run(results_path.to_str().unwrap(), "org/model", "bigcode");
    assert!(result.is_err());
}

#[test]
fn test_run_all_leaderboards() {
    let tmp = tempfile::TempDir::new().unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"score": 0.9}"#).unwrap();
    std::fs::create_dir_all("results").ok();
    for lb in &["open-llm-leaderboard", "bigcode", "evalplus", "custom"] {
        let result = run(results_path.to_str().unwrap(), "org/model", lb);
        assert!(result.is_ok(), "Failed for leaderboard: {lb}");
    }
}

#[test]
fn test_generate_model_card() {
    let tmp = tempfile::TempDir::new().unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"score": 0.75, "benchmark": "humaneval"}"#).unwrap();
    std::fs::create_dir_all("results").ok();
    let card = generate_model_card("org/model", results_path.to_str().unwrap()).unwrap();
    assert!(card.contains("org/model"));
    assert!(card.contains("humaneval"));
    assert!(card.contains("pass@1"));
    assert!(card.contains("aprender"));
}

#[test]
fn test_generate_model_card_yaml_header() {
    let tmp = tempfile::TempDir::new().unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"score": 0.5, "benchmark": "mbpp"}"#).unwrap();
    std::fs::create_dir_all("results").ok();
    let card = generate_model_card("org/test", results_path.to_str().unwrap()).unwrap();
    assert!(card.starts_with("---\n"));
    assert!(card.contains("license: mit"));
}

#[test]
fn test_generate_model_card_missing_file() {
    assert!(generate_model_card("org/model", "/nonexistent/results.json").is_err());
}

#[test]
fn test_export_model_basic() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2test").unwrap();
    let out = tmp.path().join("export");
    assert!(export_model(model_path.to_str().unwrap(), "safetensors", out.to_str().unwrap(), None).is_ok());
    assert!(out.join("metadata.json").exists());
}

#[test]
fn test_export_model_with_results() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2test").unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"pass@1": 0.85}"#).unwrap();
    let out = tmp.path().join("export");
    export_model(
        model_path.to_str().unwrap(), "safetensors",
        out.to_str().unwrap(), Some(results_path.to_str().unwrap()),
    ).unwrap();
    let meta = std::fs::read_to_string(out.join("metadata.json")).unwrap();
    let parsed: ExportMetadata = serde_json::from_str(&meta).unwrap();
    assert!(parsed.results.is_some());
}

#[test]
fn test_export_model_invalid_format() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2test").unwrap();
    let result = export_model(model_path.to_str().unwrap(), "invalid", "out/", None);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Unknown export format"));
}

#[test]
fn test_export_model_missing_model() {
    assert!(export_model("/nonexistent/model.apr", "safetensors", "out/", None).is_err());
}

#[test]
fn test_export_metadata_serialization() {
    let meta = ExportMetadata {
        model_path: "test.apr".into(),
        format: "safetensors".into(),
        exported_at: "2026-01-01T00:00:00Z".into(),
        results: Some(serde_json::json!({"score": 0.9})),
    };
    let json = serde_json::to_string(&meta).unwrap();
    let parsed: ExportMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.format, "safetensors");
}

// --- pre-submit check tests ---

#[test]
fn test_pre_submit_check_all_pass() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2test-data").unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"benchmark": "humaneval", "benchmarks": ["humaneval", "mbpp"]}"#).unwrap();
    // Create model card
    std::fs::create_dir_all("results").ok();
    std::fs::write("results/org_model_README.md", "card").ok();
    let report = pre_submit_check(
        model_path.to_str().unwrap(),
        results_path.to_str().unwrap(),
        "org/model",
    ).unwrap();
    assert!(report.checks.iter().any(|c| c.name == "apr-format" && c.passed));
    assert!(report.checks.iter().any(|c| c.name == "results-json" && c.passed));
    assert!(report.checks.iter().any(|c| c.name == "model-id" && c.passed));
}

#[test]
fn test_pre_submit_check_invalid_model() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("bad.bin");
    std::fs::write(&model_path, b"NOT_APR").unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"benchmarks": ["humaneval", "mbpp"]}"#).unwrap();
    let report = pre_submit_check(
        model_path.to_str().unwrap(),
        results_path.to_str().unwrap(),
        "org/model",
    ).unwrap();
    assert!(!report.all_passed);
    let apr_check = report.checks.iter().find(|c| c.name == "apr-format").unwrap();
    assert!(!apr_check.passed);
    assert!(apr_check.detail.contains("magic bytes"));
}

#[test]
fn test_pre_submit_check_missing_model() {
    let tmp = tempfile::TempDir::new().unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"benchmarks": ["humaneval", "mbpp"]}"#).unwrap();
    let report = pre_submit_check(
        "/nonexistent.apr",
        results_path.to_str().unwrap(),
        "org/model",
    ).unwrap();
    assert!(!report.all_passed);
}

#[test]
fn test_pre_submit_check_invalid_results() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2data").unwrap();
    let results_path = tmp.path().join("bad.json");
    std::fs::write(&results_path, "not json").unwrap();
    let report = pre_submit_check(
        model_path.to_str().unwrap(),
        results_path.to_str().unwrap(),
        "org/model",
    ).unwrap();
    let json_check = report.checks.iter().find(|c| c.name == "results-json").unwrap();
    assert!(!json_check.passed);
}

#[test]
fn test_pre_submit_check_missing_benchmarks() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2data").unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"score": 0.9}"#).unwrap();
    let report = pre_submit_check(
        model_path.to_str().unwrap(),
        results_path.to_str().unwrap(),
        "org/model",
    ).unwrap();
    let bench_check = report.checks.iter().find(|c| c.name == "required-benchmarks").unwrap();
    assert!(!bench_check.passed);
    assert!(bench_check.detail.contains("humaneval"));
}

#[test]
fn test_pre_submit_check_bad_model_id() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.apr");
    std::fs::write(&model_path, b"APR2data").unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"benchmarks": ["humaneval", "mbpp"]}"#).unwrap();
    let report = pre_submit_check(
        model_path.to_str().unwrap(),
        results_path.to_str().unwrap(),
        "no-org",
    ).unwrap();
    let id_check = report.checks.iter().find(|c| c.name == "model-id").unwrap();
    assert!(!id_check.passed);
}

#[test]
fn test_pre_submit_report_serialization() {
    let report = PreSubmitReport {
        model_path: "test.apr".into(),
        results_path: "results.json".into(),
        checks: vec![
            CheckResult { name: "test".into(), passed: true, detail: "ok".into() },
        ],
        all_passed: true,
    };
    let json = serde_json::to_string(&report).unwrap();
    let parsed: PreSubmitReport = serde_json::from_str(&json).unwrap();
    assert!(parsed.all_passed);
    assert_eq!(parsed.checks.len(), 1);
}

#[test]
fn test_check_apr_format_too_small() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("tiny.bin");
    std::fs::write(&model_path, b"AP").unwrap();
    let result = check_apr_format(model_path.to_str().unwrap());
    assert!(!result.passed);
    assert!(result.detail.contains("too small"));
}

#[test]
fn test_check_model_id_empty() {
    let result = check_model_id("");
    assert!(!result.passed);
    assert!(result.detail.contains("empty"));
}

#[test]
fn test_check_required_benchmarks_single() {
    let tmp = tempfile::TempDir::new().unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"benchmark": "humaneval"}"#).unwrap();
    let result = check_required_benchmarks(results_path.to_str().unwrap());
    // Only humaneval found, mbpp missing
    assert!(!result.passed);
    assert!(result.detail.contains("mbpp"));
}
