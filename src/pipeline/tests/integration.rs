use super::super::*;

fn base_config(output_dir: &str) -> PipelineConfig {
    PipelineConfig {
        model_id: "test/model".into(),
        output_dir: output_dir.into(),
        quantization: "fp16".into(),
        benchmarks: vec![],
        submit: false,
        leaderboard: "bigcode".into(),
        validate: None,
        distill: None,
        finetune: None,
        align: None,
        merge: None,
        tune: None,
        prune: None,
        quantize: None,
        eval: None,
        compile: None,
    }
}

#[test]
fn test_run_pipeline_with_validate() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.validate = Some(ValidateConfig {
        data: "train.jsonl".into(),
        benchmarks: vec!["humaneval".into()],
        threshold: Some(0.02),
        decontaminate: Some(false),
    });
    let result = run_pipeline(&config);
    assert!(result.is_ok());
}

#[test]
fn test_run_pipeline_with_align() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.align = Some(AlignConfig {
        data: "pairs.jsonl".into(),
        method: Some("dpo".into()),
        beta: Some(0.1),
        epochs: Some(3),
        ref_model: None,
    });
    let result = run_pipeline(&config);
    assert!(result.is_ok());
}

#[test]
fn test_run_pipeline_with_align_orpo() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.align = Some(AlignConfig {
        data: "pairs.jsonl".into(),
        method: Some("orpo".into()),
        beta: None,
        epochs: None,
        ref_model: Some("ref.apr".into()),
    });
    let result = run_pipeline(&config);
    assert!(result.is_ok());
}

#[test]
fn test_run_pipeline_with_tune() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.tune = Some(TuneConfig {
        data: "data.jsonl".into(),
        strategy: Some("tpe".into()),
        budget: Some(10),
        max_epochs: Some(3),
    });
    let result = run_pipeline(&config);
    assert!(result.is_ok());
}

#[test]
fn test_run_pipeline_with_compile() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.compile = Some(CompileConfig {
        release: Some(true),
        lto: Some(true),
        strip: Some(false),
        output: Some("model-binary".into()),
    });
    let result = run_pipeline(&config);
    assert!(result.is_ok());
}

#[test]
fn test_run_pipeline_validate_then_finetune_then_align() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.validate = Some(ValidateConfig {
        data: "train.jsonl".into(),
        benchmarks: vec!["humaneval".into(), "mbpp".into()],
        threshold: None,
        decontaminate: Some(true),
    });
    config.finetune = Some(FinetuneConfig {
        dataset: "data.jsonl".into(),
        method: Some("lora".into()),
        rank: 16,
        lr: 1e-4,
        epochs: 2,
        output: None,
    });
    config.align = Some(AlignConfig {
        data: "pairs.jsonl".into(),
        method: None,
        beta: None,
        epochs: None,
        ref_model: None,
    });
    let result = run_pipeline(&config);
    assert!(result.is_ok());
    assert_eq!(count_steps(&config), 4); // validate + convert + finetune + align
}

#[test]
fn test_run_pipeline_full_11_steps() {
    let tmp = tempfile::TempDir::new().unwrap();
    let out = tmp.path().to_str().unwrap();
    // Create APR v2 fixtures with matching tensor names for merge/distill steps.
    // Convert produces "model.embed_tokens.weight", so fixtures must match.
    let fixture_bytes = {
        let metadata = aprender::format::v2::AprV2Metadata::default();
        let mut writer = aprender::format::v2::AprV2Writer::new(metadata);
        writer.add_f32_tensor("model.embed_tokens.weight", vec![1, 256], &vec![0.5_f32; 256]);
        writer.write().unwrap()
    };
    let other_apr = tmp.path().join("other.apr");
    std::fs::write(&other_apr, &fixture_bytes).unwrap();
    let other_path = other_apr.to_str().unwrap().to_string();
    let teacher_apr = tmp.path().join("teacher.apr");
    std::fs::write(&teacher_apr, &fixture_bytes).unwrap();
    let teacher_path = teacher_apr.to_str().unwrap().to_string();
    let mut config = base_config(out);
    config.validate = Some(ValidateConfig {
        data: "d.jsonl".into(),
        benchmarks: vec!["humaneval".into()],
        threshold: None,
        decontaminate: None,
    });
    config.distill = Some(DistillConfig {
        teacher: teacher_path,
        strategy: "standard".into(),
        temperature: 2.0,
        alpha: 0.5,
        epochs: Some(1),
        data: None,
    });
    config.finetune = Some(FinetuneConfig {
        dataset: "d.jsonl".into(),
        method: None,
        rank: 8,
        lr: 1e-4,
        epochs: 1,
        output: None,
    });
    config.align = Some(AlignConfig {
        data: "pairs.jsonl".into(),
        method: Some("dpo".into()),
        beta: Some(0.1),
        epochs: Some(1),
        ref_model: None,
    });
    config.merge = Some(MergeConfig {
        models: vec![other_path],
        strategy: "slerp".into(),
        weights: None,
        base_model: None,
        density: None,
        drop_rate: None,
    });
    config.tune = Some(TuneConfig {
        data: "d.jsonl".into(),
        strategy: None,
        budget: Some(5),
        max_epochs: None,
    });
    config.prune = Some(PruneConfig {
        method: "magnitude".into(),
        target_ratio: 0.1,
        calibration: None,
    });
    config.quantize = Some(QuantizeConfig {
        scheme: "int8".into(),
        calibration: None,
    });
    config.benchmarks = vec!["humaneval".into()];
    config.compile = Some(CompileConfig {
        release: Some(true),
        lto: None,
        strip: None,
        output: None,
    });
    config.submit = false;
    // All 11 non-submit steps (submit tested separately via test_run_pipeline_with_submit)
    assert_eq!(count_steps(&config), 11);
    let result = run_pipeline(&config);
    assert!(result.is_ok(), "pipeline failed: {}", result.unwrap_err());
}
