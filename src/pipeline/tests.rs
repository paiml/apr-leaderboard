use super::*;

fn base_config(output_dir: &str) -> PipelineConfig {
    PipelineConfig {
        model_id: "test/model".into(),
        output_dir: output_dir.into(),
        quantization: "fp16".into(),
        benchmarks: vec![],
        submit: false,
        leaderboard: "bigcode".into(),
        finetune: None,
        distill: None,
        merge: None,
        prune: None,
        quantize: None,
        eval: None,
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

#[test]
fn test_eval_config_toml_parsing() {
    let toml_str = r#"
model_id = "test/model"
output_dir = "models"
quantization = "fp16"
submit = false
leaderboard = "bigcode"
benchmarks = ["humaneval"]

[eval]
samples = 50
prompt_strategy = "scot"
n_samples = 20
temperature = 0.8
top_p = 0.9
rerank = "majority"
"#;
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    let ec = config.eval.unwrap();
    assert_eq!(ec.samples, Some(50));
    assert_eq!(ec.prompt_strategy.as_deref(), Some("scot"));
    assert_eq!(ec.n_samples, Some(20));
    assert!((ec.temperature.unwrap() - 0.8).abs() < f64::EPSILON);
    assert!((ec.top_p.unwrap() - 0.9).abs() < f64::EPSILON);
    assert_eq!(ec.rerank.as_deref(), Some("majority"));
}

#[test]
fn test_eval_config_toml_defaults() {
    let toml_str = r#"
model_id = "test/model"
output_dir = "models"
quantization = "fp16"
submit = false
leaderboard = "bigcode"
benchmarks = ["humaneval"]
"#;
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    assert!(config.eval.is_none());
}

#[test]
fn test_build_eval_config_none() {
    let config = build_eval_config(None).unwrap();
    assert!(matches!(config.prompt_strategy, eval::PromptStrategy::Standard));
    assert_eq!(config.n_samples, 1);
}

#[test]
fn test_build_eval_config_with_values() {
    let toml_config = EvalConfigToml {
        samples: Some(100),
        prompt_strategy: Some("scot".into()),
        n_samples: Some(10),
        temperature: Some(0.5),
        top_p: Some(0.8),
        rerank: Some("logprob".into()),
    };
    let config = build_eval_config(Some(&toml_config)).unwrap();
    assert!(matches!(config.prompt_strategy, eval::PromptStrategy::SCoT));
    assert_eq!(config.n_samples, 10);
    assert!((config.temperature - 0.5).abs() < f64::EPSILON);
    assert!((config.top_p - 0.8).abs() < f64::EPSILON);
    assert!(matches!(config.rerank, eval::RerankStrategy::LogProb));
}

#[test]
fn test_recipe_b_toml_parses() {
    let toml_str = include_str!("../../configs/recipe-b-merge-alchemist.toml");
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.model_id, "Qwen/Qwen2.5-Coder-7B-Instruct");
    assert!(config.finetune.is_none());
    assert!(config.distill.is_none());
    let merge = config.merge.unwrap();
    assert_eq!(merge.strategy, "ties");
    assert!(merge.base_model.is_some());
    assert!((merge.density.unwrap() - 0.2).abs() < f64::EPSILON);
    let prune = config.prune.unwrap();
    assert_eq!(prune.method, "structured");
    assert!((prune.target_ratio - 0.15).abs() < f64::EPSILON);
    let quant = config.quantize.unwrap();
    assert_eq!(quant.scheme, "q4k");
}

#[test]
fn test_recipe_d_toml_parses() {
    let toml_str = include_str!("../../configs/recipe-d-sovereign-binary.toml");
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.model_id, "Qwen/Qwen2.5-Coder-1.5B");
    assert!(config.distill.is_none());
    assert!(config.merge.is_none());
    let ft = config.finetune.unwrap();
    assert_eq!(ft.method, Some("qlora".into()));
    assert_eq!(ft.rank, 16);
    let prune = config.prune.unwrap();
    assert_eq!(prune.method, "magnitude");
    assert!((prune.target_ratio - 0.4).abs() < f64::EPSILON);
    let quant = config.quantize.unwrap();
    assert_eq!(quant.scheme, "int4");
}
