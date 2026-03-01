use super::*;

#[test]
fn test_all_criteria_count() {
    let criteria = all_criteria();
    assert_eq!(criteria.len(), 27);
}

#[test]
fn test_all_criteria_unique_ids() {
    let criteria = all_criteria();
    let mut ids: Vec<&str> = criteria.iter().map(|c| c.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 27);
}

#[test]
fn test_criteria_id_format() {
    let criteria = all_criteria();
    for ac in &criteria {
        assert!(ac.id.starts_with("AC-"), "Bad ID: {}", ac.id);
        let num: usize = ac.id[3..].parse().unwrap_or(0);
        assert!((1..=27).contains(&num), "ID out of range: {}", ac.id);
    }
}

#[test]
fn test_criteria_categories() {
    let criteria = all_criteria();
    let format_count = criteria.iter().filter(|c| c.category == AcCategory::FormatParity).count();
    let technique_count = criteria.iter().filter(|c| c.category == AcCategory::Technique).count();
    let pipeline_count = criteria.iter().filter(|c| c.category == AcCategory::Pipeline).count();
    let perf_count = criteria.iter().filter(|c| c.category == AcCategory::Performance).count();
    let tooling_count = criteria.iter().filter(|c| c.category == AcCategory::Tooling).count();
    assert_eq!(format_count, 3);
    assert_eq!(technique_count, 8);
    assert_eq!(pipeline_count, 9);
    assert_eq!(perf_count, 6);
    assert_eq!(tooling_count, 1);
    assert_eq!(format_count + technique_count + pipeline_count + perf_count + tooling_count, 27);
}

#[test]
fn test_ac_display() {
    let criteria = all_criteria();
    let ac001 = &criteria[0];
    let display = format!("{ac001}");
    assert!(display.contains("AC-001"));
    assert!(display.contains("Import produces valid .apr"));
    assert!(display.contains("Format & Parity"));
}

#[test]
fn test_category_display() {
    assert_eq!(AcCategory::FormatParity.to_string(), "Format & Parity");
    assert_eq!(AcCategory::Technique.to_string(), "Technique Validation");
    assert_eq!(AcCategory::Pipeline.to_string(), "Pipeline & Quality");
    assert_eq!(AcCategory::Performance.to_string(), "Model Performance");
    assert_eq!(AcCategory::Tooling.to_string(), "Tooling Completeness");
}

#[test]
fn test_verify_scaffold() {
    let report = verify_scaffold().unwrap();
    assert_eq!(report.results.len(), 27);
    let scaffolded = report.results.iter().filter(|r| r.status == AcStatus::Scaffolded).count();
    assert!(scaffolded >= 12, "Expected at least 12 scaffolded, got {scaffolded}");
}

#[test]
fn test_ac_report_serialization() {
    let report = AcReport {
        results: vec![
            AcResult { id: "AC-001".into(), status: AcStatus::Scaffolded },
            AcResult { id: "AC-002".into(), status: AcStatus::Pending },
        ],
    };
    let json = serde_json::to_string(&report).unwrap();
    let parsed: AcReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.results.len(), 2);
    assert_eq!(parsed.results[0].status, AcStatus::Scaffolded);
}

#[test]
fn test_ac_criterion_serialization() {
    let criteria = all_criteria();
    let json = serde_json::to_string(&criteria[0]).unwrap();
    let parsed: AcceptanceCriterion = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, "AC-001");
}

#[test]
fn test_list_no_filter() {
    assert!(list(None).is_ok());
}

#[test]
fn test_list_with_filter() {
    assert!(list(Some("technique")).is_ok());
    assert!(list(Some("format")).is_ok());
    assert!(list(Some("pipeline")).is_ok());
    assert!(list(Some("performance")).is_ok());
    assert!(list(Some("tooling")).is_ok());
}

#[test]
fn test_list_nonexistent_filter() {
    assert!(list(Some("nonexistent")).is_ok());
}

#[test]
fn test_criteria_have_measurements() {
    let criteria = all_criteria();
    for ac in &criteria {
        assert!(!ac.measurement.is_empty(), "{} has empty measurement", ac.id);
        assert!(!ac.gate.is_empty(), "{} has empty gate", ac.id);
    }
}
