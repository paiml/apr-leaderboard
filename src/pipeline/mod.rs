//! Pipeline orchestration: convert → finetune → eval → submit.
//!
//! Reads a TOML config and runs the full leaderboard pipeline.

use anyhow::Result;
use serde::Deserialize;

use crate::{convert, eval, finetune, submit};

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
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct FinetuneConfig {
    pub dataset: String,
    pub rank: usize,
    pub lr: f64,
    pub epochs: usize,
}

pub(crate) fn run_pipeline(config: &PipelineConfig) -> Result<()> {
    println!("=== APR Leaderboard Pipeline ===\n");

    // Step 1: Convert
    println!("[1/4] Converting {} to .apr...", config.model_id);
    convert::run(&config.model_id, &config.output_dir, &config.quantization)?;

    let model_path = format!(
        "{}/{}.apr",
        config.output_dir,
        config.model_id.replace('/', "_")
    );

    // Step 2: Optional fine-tune
    if let Some(ft) = &config.finetune {
        println!("[2/4] Fine-tuning with LoRA (rank={})...", ft.rank);
        finetune::run(&model_path, &ft.dataset, ft.rank, ft.lr, ft.epochs)?;
    } else {
        println!("[2/4] Skipping fine-tune (not configured)");
    }

    // Step 3: Evaluate
    println!("[3/4] Running benchmarks...");
    for benchmark in &config.benchmarks {
        eval::run(&model_path, benchmark, 0, "results/")?;
    }

    // Step 4: Submit
    if config.submit {
        println!("[4/4] Submitting to {}...", config.leaderboard);
        submit::run(
            &format!("results/{}.json", config.model_id.replace('/', "_")),
            &config.model_id,
            &config.leaderboard,
        )?;
    } else {
        println!("[4/4] Skipping submission (submit = false)");
    }

    println!("\n=== Pipeline complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let config = PipelineConfig {
            model_id: "test/model".into(),
            output_dir: "out".into(),
            quantization: "fp16".into(),
            benchmarks: vec!["humaneval".into()],
            submit: false,
            leaderboard: "bigcode".into(),
            finetune: None,
        };
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
    fn test_run_pipeline_with_convert() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output_dir = tmp.path().to_str().unwrap();

        let config = PipelineConfig {
            model_id: "test/model".into(),
            output_dir: output_dir.into(),
            quantization: "fp16".into(),
            benchmarks: vec![],
            submit: false,
            leaderboard: "bigcode".into(),
            finetune: None,
        };
        let result = run_pipeline(&config);
        assert!(result.is_ok());
        let apr_path = format!("{}/test_model.apr", output_dir);
        assert!(std::path::Path::new(&apr_path).exists());
    }

    #[test]
    fn test_run_pipeline_with_benchmarks() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output_dir = tmp.path().to_str().unwrap();

        let config = PipelineConfig {
            model_id: "test/model".into(),
            output_dir: output_dir.into(),
            quantization: "fp16".into(),
            benchmarks: vec!["humaneval".into()],
            submit: false,
            leaderboard: "bigcode".into(),
            finetune: None,
        };
        let result = run_pipeline(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_pipeline_with_finetune() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output_dir = tmp.path().to_str().unwrap();

        let config = PipelineConfig {
            model_id: "test/model".into(),
            output_dir: output_dir.into(),
            quantization: "fp16".into(),
            benchmarks: vec![],
            submit: false,
            leaderboard: "bigcode".into(),
            finetune: Some(FinetuneConfig {
                dataset: "test-data".into(),
                rank: 8,
                lr: 1e-4,
                epochs: 1,
            }),
        };
        let result = run_pipeline(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_pipeline_with_submit() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output_dir = tmp.path().to_str().unwrap();

        let config = PipelineConfig {
            model_id: "test/model".into(),
            output_dir: output_dir.into(),
            quantization: "fp16".into(),
            benchmarks: vec!["humaneval".into()],
            submit: true,
            leaderboard: "bigcode".into(),
            finetune: None,
        };
        let _result = run_pipeline(&config);
    }

    #[test]
    fn test_run_pipeline_invalid_quantization() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = PipelineConfig {
            model_id: "test/model".into(),
            output_dir: tmp.path().to_str().unwrap().into(),
            quantization: "invalid".into(),
            benchmarks: vec![],
            submit: false,
            leaderboard: "bigcode".into(),
            finetune: None,
        };
        let result = run_pipeline(&config);
        assert!(result.is_err());
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
}
