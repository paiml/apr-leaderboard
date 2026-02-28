//! Pipeline orchestration: convert → finetune → eval → submit.
//!
//! Reads a TOML config and runs the full leaderboard pipeline.

use anyhow::Result;
use serde::Deserialize;

use crate::{convert, eval, finetune, optimize, submit};

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct PipelineConfig {
    pub model_id: String,
    pub output_dir: String,
    pub quantization: String,
    pub benchmarks: Vec<String>,
    pub submit: bool,
    pub leaderboard: String,
    pub finetune: Option<FinetuneConfig>,
    pub distill: Option<DistillConfig>,
    pub merge: Option<MergeConfig>,
    pub prune: Option<PruneConfig>,
    pub quantize: Option<QuantizeConfig>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct FinetuneConfig {
    pub dataset: String,
    pub rank: usize,
    pub lr: f64,
    pub epochs: usize,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct DistillConfig {
    pub teacher: String,
    pub strategy: String,
    pub temperature: f64,
    pub alpha: f64,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct MergeConfig {
    pub models: Vec<String>,
    pub strategy: String,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct PruneConfig {
    pub method: String,
    pub target_ratio: f64,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct QuantizeConfig {
    pub scheme: String,
}

pub(crate) fn run_pipeline(config: &PipelineConfig) -> Result<()> {
    println!("=== APR Leaderboard Pipeline ===\n");

    let total_steps = count_steps(config);
    let mut step = 0;

    // Step: Convert
    step += 1;
    println!("[{step}/{total_steps}] Converting {} to .apr...", config.model_id);
    convert::run(&config.model_id, &config.output_dir, &config.quantization)?;

    let mut model_path = format!(
        "{}/{}.apr",
        config.output_dir,
        config.model_id.replace('/', "_")
    );

    // Step: Optional distill
    if let Some(dist) = &config.distill {
        step += 1;
        println!("[{step}/{total_steps}] Distilling from {}...", dist.teacher);
        let output = format!("{}/distilled.apr", config.output_dir);
        optimize::distill(&dist.teacher, &model_path, &dist.strategy, dist.temperature, dist.alpha, &output)?;
        model_path = output;
    }

    // Step: Optional fine-tune
    if let Some(ft) = &config.finetune {
        step += 1;
        println!("[{step}/{total_steps}] Fine-tuning with LoRA (rank={})...", ft.rank);
        finetune::run(&model_path, &ft.dataset, ft.rank, ft.lr, ft.epochs)?;
    }

    // Step: Optional merge
    if let Some(mg) = &config.merge {
        step += 1;
        println!("[{step}/{total_steps}] Merging {} models...", mg.models.len() + 1);
        let mut all_models = vec![model_path.clone()];
        all_models.extend(mg.models.iter().cloned());
        let output = format!("{}/merged.apr", config.output_dir);
        optimize::merge(&all_models, &mg.strategy, &output)?;
        model_path = output;
    }

    // Step: Optional prune
    if let Some(pr) = &config.prune {
        step += 1;
        println!("[{step}/{total_steps}] Pruning ({}, {:.0}%)...", pr.method, pr.target_ratio * 100.0);
        let output = format!("{}/pruned.apr", config.output_dir);
        optimize::prune(&model_path, &pr.method, pr.target_ratio, &output)?;
        model_path = output;
    }

    // Step: Optional quantize
    if let Some(qt) = &config.quantize {
        step += 1;
        println!("[{step}/{total_steps}] Quantizing ({})...", qt.scheme);
        let output = format!("{}/quantized.apr", config.output_dir);
        optimize::quantize(&model_path, &qt.scheme, &output)?;
        model_path = output;
    }

    // Step: Evaluate
    if !config.benchmarks.is_empty() {
        step += 1;
        println!("[{step}/{total_steps}] Running {} benchmark(s)...", config.benchmarks.len());
        for benchmark in &config.benchmarks {
            eval::run(&model_path, benchmark, 0, "results/")?;
        }
    }

    // Step: Submit
    if config.submit {
        step += 1;
        println!("[{step}/{total_steps}] Submitting to {}...", config.leaderboard);
        submit::run(
            &format!("results/{}.json", config.model_id.replace('/', "_")),
            &config.model_id,
            &config.leaderboard,
        )?;
    }

    println!("\n=== Pipeline complete ({step} steps) ===");
    Ok(())
}

fn count_steps(config: &PipelineConfig) -> usize {
    let mut n = 1; // convert always runs
    if config.distill.is_some() { n += 1; }
    if config.finetune.is_some() { n += 1; }
    if config.merge.is_some() { n += 1; }
    if config.prune.is_some() { n += 1; }
    if config.quantize.is_some() { n += 1; }
    if !config.benchmarks.is_empty() { n += 1; }
    if config.submit { n += 1; }
    n
}

#[cfg(test)]
mod tests {
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
            rank: 8,
            lr: 1e-4,
            epochs: 1,
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
        });
        config.quantize = Some(QuantizeConfig {
            scheme: "int4".into(),
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
        });
        config.finetune = Some(FinetuneConfig {
            dataset: "data.jsonl".into(),
            rank: 32,
            lr: 2e-4,
            epochs: 5,
        });
        config.merge = Some(MergeConfig {
            models: vec!["variant.apr".into()],
            strategy: "slerp".into(),
        });
        config.prune = Some(PruneConfig {
            method: "wanda".into(),
            target_ratio: 0.2,
        });
        config.quantize = Some(QuantizeConfig {
            scheme: "int4".into(),
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
}
