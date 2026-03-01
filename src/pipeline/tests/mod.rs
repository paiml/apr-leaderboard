use super::*;

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
fn test_pipeline_config_deserialization() {
    let toml_str = r#"
model_id = "Qwen/Qwen2.5-Coder-7B"
output_dir = "models"
quantization = "fp16"
submit = false
leaderboard = "bigcode"
benchmarks = ["humaneval", "mbpp"]
"#;
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.model_id, "Qwen/Qwen2.5-Coder-7B");
    assert_eq!(config.output_dir, "models");
    assert_eq!(config.quantization, "fp16");
    assert!(!config.submit);
    assert_eq!(config.leaderboard, "bigcode");
    assert_eq!(config.benchmarks, vec!["humaneval", "mbpp"]);
    assert!(config.finetune.is_none());
}

#[test]
fn test_pipeline_config_with_finetune() {
    let toml_str = r#"
model_id = "Qwen/Qwen2.5-Coder-7B"
output_dir = "models"
quantization = "q4"
submit = true
leaderboard = "open-llm-leaderboard"
benchmarks = ["humaneval"]

[finetune]
dataset = "bigcode/starcoderdata"
rank = 16
lr = 1e-4
epochs = 3
"#;
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    assert!(config.submit);
    let ft = config.finetune.unwrap();
    assert_eq!(ft.dataset, "bigcode/starcoderdata");
    assert_eq!(ft.rank, 16);
    assert!((ft.lr - 1e-4).abs() < f64::EPSILON);
    assert_eq!(ft.epochs, 3);
}

#[test]
fn test_pipeline_config_roundtrip() {
    let mut config = base_config("out");
    config.model_id = "test/model".into();
    config.benchmarks = vec!["humaneval".into()];
    let serialized = toml::to_string(&config).unwrap();
    let deserialized: PipelineConfig = toml::from_str(&serialized).unwrap();
    assert_eq!(deserialized.model_id, config.model_id);
    assert_eq!(deserialized.benchmarks.len(), 1);
}

#[test]
fn test_pipeline_config_real_file() {
    let content = std::fs::read_to_string("configs/qwen-coder-7b.toml").unwrap();
    let config: PipelineConfig = toml::from_str(&content).unwrap();
    assert_eq!(config.model_id, "Qwen/Qwen2.5-Coder-7B");
    assert_eq!(config.benchmarks.len(), 6);
    assert!(config.finetune.is_some());
}

#[test]
fn test_recipe_a_config() {
    let content = std::fs::read_to_string("configs/recipe-a-quick-lora.toml").unwrap();
    let config: PipelineConfig = toml::from_str(&content).unwrap();
    assert!(config.finetune.is_some());
    assert!(config.distill.is_none());
    assert!(config.merge.is_none());
    assert!(config.prune.is_none());
    assert!(config.quantize.is_none());
    assert_eq!(config.benchmarks.len(), 2);
}

#[test]
fn test_recipe_c_config() {
    let content = std::fs::read_to_string("configs/recipe-c-full-pipeline.toml").unwrap();
    let config: PipelineConfig = toml::from_str(&content).unwrap();
    assert!(config.distill.is_some());
    assert!(config.finetune.is_some());
    assert!(config.merge.is_some());
    assert!(config.prune.is_some());
    assert!(config.quantize.is_some());
    assert_eq!(config.benchmarks.len(), 5);
    let dist = config.distill.unwrap();
    assert_eq!(dist.strategy, "progressive");
    let mg = config.merge.unwrap();
    assert_eq!(mg.strategy, "ties");
}

#[test]
fn test_run_pipeline_with_convert() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output_dir = tmp.path().to_str().unwrap();
    let config = base_config(output_dir);
    let result = run_pipeline(&config);
    assert!(result.is_ok());
    let apr_path = format!("{}/test_model.apr", output_dir);
    assert!(std::path::Path::new(&apr_path).exists());
}

#[test]
fn test_run_pipeline_with_benchmarks() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.benchmarks = vec!["humaneval".into()];
    let result = run_pipeline(&config);
    assert!(result.is_ok());
}

#[test]
fn test_run_pipeline_with_finetune() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.finetune = Some(FinetuneConfig {
        dataset: "test-data".into(),
        method: None,
        rank: 8,
        lr: 1e-4,
        epochs: 1,
        output: None,
    });
    let result = run_pipeline(&config);
    assert!(result.is_ok());
}

#[test]
fn test_run_pipeline_with_submit() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.benchmarks = vec!["humaneval".into()];
    config.submit = true;
    let _result = run_pipeline(&config);
}

#[test]
fn test_run_pipeline_invalid_quantization() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.quantization = "invalid".into();
    let result = run_pipeline(&config);
    assert!(result.is_err());
}

#[test]
fn test_run_pipeline_with_distill() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.distill = Some(DistillConfig {
        teacher: "teacher.apr".into(),
        strategy: "progressive".into(),
        temperature: 3.0,
        alpha: 0.7,
        epochs: None,
        data: None,
    });
    let result = run_pipeline(&config);
    assert!(result.is_ok());
}

#[test]
fn test_run_pipeline_with_prune_and_quantize() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = base_config(tmp.path().to_str().unwrap());
    config.prune = Some(PruneConfig {
        method: "wanda".into(),
        target_ratio: 0.2,
        calibration: None,
    });
    config.quantize = Some(QuantizeConfig {
        scheme: "int4".into(),
        calibration: None,
    });
    let result = run_pipeline(&config);
    assert!(result.is_ok());
}

#[test]
fn test_count_steps_full_kitchen_sink() {
    let mut config = base_config("/tmp/test");
    config.distill = Some(DistillConfig {
        teacher: "teacher.apr".into(),
        strategy: "progressive".into(),
        temperature: 3.0,
        alpha: 0.7,
        epochs: None,
        data: None,
    });
    config.finetune = Some(FinetuneConfig {
        dataset: "data.jsonl".into(),
        method: None,
        rank: 32,
        lr: 2e-4,
        epochs: 5,
        output: None,
    });
    config.merge = Some(MergeConfig {
        models: vec!["variant.apr".into()],
        strategy: "slerp".into(),
        weights: None,
        base_model: None,
        density: None,
        drop_rate: None,
    });
    config.prune = Some(PruneConfig {
        method: "wanda".into(),
        target_ratio: 0.2,
        calibration: None,
    });
    config.quantize = Some(QuantizeConfig {
        scheme: "int4".into(),
        calibration: None,
    });
    config.benchmarks = vec!["humaneval".into()];
    config.submit = true;
    // convert + distill + finetune + merge + prune + quantize + eval + submit = 8
    assert_eq!(count_steps(&config), 8);
}

#[test]
fn test_count_steps_all_sections() {
    let mut config = base_config("/tmp/test");
    config.validate = Some(ValidateConfig {
        data: "d.jsonl".into(),
        benchmarks: vec!["humaneval".into()],
        threshold: None,
        decontaminate: None,
    });
    config.distill = Some(DistillConfig {
        teacher: "t.apr".into(),
        strategy: "progressive".into(),
        temperature: 3.0,
        alpha: 0.7,
        epochs: None,
        data: None,
    });
    config.finetune = Some(FinetuneConfig {
        dataset: "d.jsonl".into(),
        method: None,
        rank: 16,
        lr: 1e-4,
        epochs: 3,
        output: None,
    });
    config.align = Some(AlignConfig {
        data: "pairs.jsonl".into(),
        method: None,
        beta: None,
        epochs: None,
        ref_model: None,
    });
    config.merge = Some(MergeConfig {
        models: vec!["v.apr".into()],
        strategy: "slerp".into(),
        weights: None,
        base_model: None,
        density: None,
        drop_rate: None,
    });
    config.tune = Some(TuneConfig {
        data: "d.jsonl".into(),
        strategy: None,
        budget: None,
        max_epochs: None,
    });
    config.prune = Some(PruneConfig {
        method: "wanda".into(),
        target_ratio: 0.2,
        calibration: None,
    });
    config.quantize = Some(QuantizeConfig {
        scheme: "int4".into(),
        calibration: None,
    });
    config.benchmarks = vec!["humaneval".into()];
    config.compile = Some(CompileConfig {
        release: Some(true),
        lto: None,
        strip: None,
        output: None,
    });
    config.submit = true;
    // validate + convert + distill + finetune + align + merge + tune + prune + quantize + eval + compile + submit = 12
    assert_eq!(count_steps(&config), 12);
}

#[test]
fn test_count_steps() {
    let config = base_config("out");
    assert_eq!(count_steps(&config), 1); // just convert

    let mut config2 = base_config("out");
    config2.benchmarks = vec!["humaneval".into()];
    config2.submit = true;
    assert_eq!(count_steps(&config2), 3); // convert + eval + submit
}

#[test]
fn test_finetune_config_deserialization() {
    let toml_str = r#"
dataset = "data.jsonl"
rank = 32
lr = 0.001
epochs = 5
"#;
    let config: FinetuneConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.dataset, "data.jsonl");
    assert_eq!(config.rank, 32);
    assert!((config.lr - 0.001).abs() < f64::EPSILON);
    assert_eq!(config.epochs, 5);
}

#[test]
fn test_full_pipeline_toml() {
    let toml_str = r#"
model_id = "Qwen/Qwen2.5-Coder-7B"
output_dir = "models"
quantization = "fp16"
submit = false
leaderboard = "bigcode"
benchmarks = ["humaneval", "mbpp"]

[distill]
teacher = "teacher-32b.apr"
strategy = "progressive"
temperature = 3.0
alpha = 0.7

[finetune]
dataset = "code-instruct.jsonl"
rank = 32
lr = 2e-4
epochs = 5

[merge]
models = ["variant-b.apr"]
strategy = "slerp"

[prune]
method = "wanda"
target_ratio = 0.2

[quantize]
scheme = "int4"
"#;
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    assert!(config.distill.is_some());
    assert!(config.finetune.is_some());
    assert!(config.merge.is_some());
    assert!(config.prune.is_some());
    assert!(config.quantize.is_some());
    let dist = config.distill.unwrap();
    assert_eq!(dist.teacher, "teacher-32b.apr");
    assert_eq!(dist.strategy, "progressive");
    let mg = config.merge.unwrap();
    assert_eq!(mg.models, vec!["variant-b.apr"]);
    assert_eq!(mg.strategy, "slerp");
}

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

mod configs;
mod integration;
