use super::super::*;

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
    let toml_str = include_str!("../../../configs/recipe-b-merge-alchemist.toml");
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
    let toml_str = include_str!("../../../configs/recipe-d-sovereign-binary.toml");
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
    let comp = config.compile.unwrap();
    assert!(comp.release.unwrap());
    assert!(comp.lto.unwrap());
    assert_eq!(comp.output, Some("qwen-coder".into()));
}

#[test]
fn test_align_config_toml_parsing() {
    let toml_str = r#"
model_id = "test/model"
output_dir = "models"
quantization = "fp16"
benchmarks = ["humaneval"]
submit = false
leaderboard = "bigcode"

[align]
data = "pairs.jsonl"
method = "orpo"
beta = 0.5
epochs = 5
"#;
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    let al = config.align.unwrap();
    assert_eq!(al.data, "pairs.jsonl");
    assert_eq!(al.method, Some("orpo".into()));
    assert!((al.beta.unwrap() - 0.5).abs() < f64::EPSILON);
    assert_eq!(al.epochs, Some(5));
    assert_eq!(al.ref_model, None);
}

#[test]
fn test_validate_config_toml_parsing() {
    let toml_str = r#"
model_id = "test/model"
output_dir = "models"
quantization = "fp16"
benchmarks = ["humaneval"]
submit = false
leaderboard = "bigcode"

[validate]
data = "train.jsonl"
benchmarks = ["humaneval", "mbpp"]
threshold = 0.02
decontaminate = true
"#;
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    let val = config.validate.unwrap();
    assert_eq!(val.data, "train.jsonl");
    assert_eq!(val.benchmarks, vec!["humaneval", "mbpp"]);
    assert!((val.threshold.unwrap() - 0.02).abs() < f64::EPSILON);
    assert!(val.decontaminate.unwrap());
}

#[test]
fn test_tune_config_toml_parsing() {
    let toml_str = r#"
model_id = "test/model"
output_dir = "models"
quantization = "fp16"
benchmarks = ["humaneval"]
submit = false
leaderboard = "bigcode"

[tune]
data = "data.jsonl"
strategy = "grid"
budget = 10
max_epochs = 5
"#;
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    let tu = config.tune.unwrap();
    assert_eq!(tu.data, "data.jsonl");
    assert_eq!(tu.strategy, Some("grid".into()));
    assert_eq!(tu.budget, Some(10));
    assert_eq!(tu.max_epochs, Some(5));
}

#[test]
fn test_compile_config_toml_parsing() {
    let toml_str = r#"
model_id = "test/model"
output_dir = "models"
quantization = "fp16"
benchmarks = []
submit = false
leaderboard = "bigcode"

[compile]
release = true
lto = true
output = "qwen-coder"
"#;
    let config: PipelineConfig = toml::from_str(toml_str).unwrap();
    let comp = config.compile.unwrap();
    assert!(comp.release.unwrap());
    assert!(comp.lto.unwrap());
    assert_eq!(comp.output, Some("qwen-coder".into()));
}
