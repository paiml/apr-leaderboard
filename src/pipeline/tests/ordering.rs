use super::super::*;
use super::base_config;

// --- pipeline ordering validation (§10) ---

#[test]
fn test_validate_order_no_warnings_minimal() {
    let config = base_config("out");
    assert!(validate_pipeline_order(&config).is_empty());
}

#[test]
fn test_validate_order_no_warnings_golden() {
    let mut config = base_config("out");
    config.distill = Some(DistillConfig {
        teacher: "t.apr".into(), strategy: "standard".into(),
        temperature: 3.0, alpha: 0.7, epochs: None, data: None,
    });
    config.finetune = Some(FinetuneConfig {
        dataset: "d.jsonl".into(), method: None, rank: 16,
        lr: 1e-4, epochs: 3, output: None,
    });
    config.merge = Some(MergeConfig {
        models: vec!["v.apr".into()], strategy: "slerp".into(),
        weights: None, base_model: None, density: None, drop_rate: None,
    });
    config.prune = Some(PruneConfig {
        method: "wanda".into(), target_ratio: 0.2, calibration: None,
    });
    config.quantize = Some(QuantizeConfig {
        scheme: "int4".into(), calibration: None,
    });
    assert!(validate_pipeline_order(&config).is_empty());
}

#[test]
fn test_validate_order_merge_without_finetune() {
    let mut config = base_config("out");
    config.merge = Some(MergeConfig {
        models: vec!["v.apr".into()], strategy: "slerp".into(),
        weights: None, base_model: None, density: None, drop_rate: None,
    });
    let warnings = validate_pipeline_order(&config);
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("Merge without finetune"));
}

#[test]
fn test_validate_order_prune_without_quantize() {
    let mut config = base_config("out");
    config.finetune = Some(FinetuneConfig {
        dataset: "d.jsonl".into(), method: None, rank: 16,
        lr: 1e-4, epochs: 3, output: None,
    });
    config.prune = Some(PruneConfig {
        method: "wanda".into(), target_ratio: 0.2, calibration: None,
    });
    let warnings = validate_pipeline_order(&config);
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("Prune without quantize"));
}

#[test]
fn test_validate_order_distill_without_finetune() {
    let mut config = base_config("out");
    config.distill = Some(DistillConfig {
        teacher: "t.apr".into(), strategy: "standard".into(),
        temperature: 3.0, alpha: 0.7, epochs: None, data: None,
    });
    let warnings = validate_pipeline_order(&config);
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("Distill without finetune"));
}

#[test]
fn test_validate_order_align_without_finetune() {
    let mut config = base_config("out");
    config.align = Some(AlignConfig {
        data: "pairs.jsonl".into(), method: None,
        beta: None, epochs: None, ref_model: None,
    });
    let warnings = validate_pipeline_order(&config);
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("Align without finetune"));
}

#[test]
fn test_validate_order_multiple_warnings() {
    let mut config = base_config("out");
    config.distill = Some(DistillConfig {
        teacher: "t.apr".into(), strategy: "standard".into(),
        temperature: 3.0, alpha: 0.7, epochs: None, data: None,
    });
    config.prune = Some(PruneConfig {
        method: "wanda".into(), target_ratio: 0.2, calibration: None,
    });
    config.align = Some(AlignConfig {
        data: "pairs.jsonl".into(), method: None,
        beta: None, epochs: None, ref_model: None,
    });
    // distill w/o finetune + prune w/o quantize + align w/o finetune = 3 warnings
    let warnings = validate_pipeline_order(&config);
    assert_eq!(warnings.len(), 3);
}

// --- config hashing (§11) ---

#[test]
fn test_config_hash_deterministic() {
    let config = base_config("out");
    let hash1 = config_hash(&config).unwrap();
    let hash2 = config_hash(&config).unwrap();
    assert_eq!(hash1, hash2);
    assert_eq!(hash1.len(), 64); // BLAKE3 hex = 64 chars
}

#[test]
fn test_config_hash_changes_with_config() {
    let config1 = base_config("out1");
    let mut config2 = base_config("out2");
    config2.quantization = "q8_0".into();
    let hash1 = config_hash(&config1).unwrap();
    let hash2 = config_hash(&config2).unwrap();
    assert_ne!(hash1, hash2);
}

#[test]
fn test_config_hash_sensitive_to_optional_sections() {
    let config1 = base_config("out");
    let mut config2 = base_config("out");
    config2.finetune = Some(FinetuneConfig {
        dataset: "data.jsonl".into(), method: None,
        rank: 16, lr: 1e-4, epochs: 3, output: None,
    });
    let hash1 = config_hash(&config1).unwrap();
    let hash2 = config_hash(&config2).unwrap();
    assert_ne!(hash1, hash2);
}
