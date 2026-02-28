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
    pub eval: Option<EvalConfigToml>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct FinetuneConfig {
    pub dataset: String,
    pub method: Option<String>,
    pub rank: usize,
    pub lr: f64,
    pub epochs: usize,
    pub output: Option<String>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct DistillConfig {
    pub teacher: String,
    pub strategy: String,
    pub temperature: f64,
    pub alpha: f64,
    pub epochs: Option<usize>,
    pub data: Option<String>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct MergeConfig {
    pub models: Vec<String>,
    pub strategy: String,
    pub weights: Option<String>,
    pub base_model: Option<String>,
    pub density: Option<f64>,
    pub drop_rate: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct PruneConfig {
    pub method: String,
    pub target_ratio: f64,
    pub calibration: Option<String>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct QuantizeConfig {
    pub scheme: String,
    pub calibration: Option<String>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(crate) struct EvalConfigToml {
    pub samples: Option<usize>,
    pub prompt_strategy: Option<String>,
    pub n_samples: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub rerank: Option<String>,
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
        optimize::distill(&optimize::DistillOpts {
            teacher: &dist.teacher,
            student: &model_path,
            strategy: &dist.strategy,
            temperature: dist.temperature,
            alpha: dist.alpha,
            epochs: dist.epochs.unwrap_or(5),
            data: dist.data.as_deref(),
            output: &output,
        })?;
        model_path = output;
    }

    // Step: Optional fine-tune
    if let Some(ft) = &config.finetune {
        step += 1;
        println!("[{step}/{total_steps}] Fine-tuning with LoRA (rank={})...", ft.rank);
        finetune::run(&model_path, &ft.dataset, ft.method.as_deref().unwrap_or("lora"), ft.rank, ft.lr, ft.epochs, ft.output.as_deref())?;
    }

    // Step: Optional merge
    if let Some(mg) = &config.merge {
        step += 1;
        println!("[{step}/{total_steps}] Merging {} models...", mg.models.len() + 1);
        let mut all_models = vec![model_path.clone()];
        all_models.extend(mg.models.iter().cloned());
        let output = format!("{}/merged.apr", config.output_dir);
        optimize::merge(&optimize::MergeOpts {
            models: &all_models,
            strategy: &mg.strategy,
            weights: mg.weights.as_deref(),
            base_model: mg.base_model.as_deref(),
            density: mg.density,
            drop_rate: mg.drop_rate,
            output: &output,
        })?;
        model_path = output;
    }

    // Step: Optional prune
    if let Some(pr) = &config.prune {
        step += 1;
        println!("[{step}/{total_steps}] Pruning ({}, {:.0}%)...", pr.method, pr.target_ratio * 100.0);
        let output = format!("{}/pruned.apr", config.output_dir);
        optimize::prune(&model_path, &pr.method, pr.target_ratio, pr.calibration.as_deref(), &output)?;
        model_path = output;
    }

    // Step: Optional quantize
    if let Some(qt) = &config.quantize {
        step += 1;
        println!("[{step}/{total_steps}] Quantizing ({})...", qt.scheme);
        let output = format!("{}/quantized.apr", config.output_dir);
        optimize::quantize(&model_path, &qt.scheme, qt.calibration.as_deref(), &output)?;
        model_path = output;
    }

    // Step: Evaluate
    if !config.benchmarks.is_empty() {
        step += 1;
        println!("[{step}/{total_steps}] Running {} benchmark(s)...", config.benchmarks.len());
        let eval_config = build_eval_config(config.eval.as_ref())?;
        let samples = config.eval.as_ref().and_then(|e| e.samples).unwrap_or(0);
        for benchmark in &config.benchmarks {
            eval::run_with_config(&model_path, benchmark, samples, "results/", &eval_config)?;
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

fn build_eval_config(toml: Option<&EvalConfigToml>) -> Result<eval::EvalConfig> {
    let mut config = eval::EvalConfig::default();
    if let Some(t) = toml {
        if let Some(ps) = &t.prompt_strategy {
            config.prompt_strategy = eval::PromptStrategy::from_str(ps)?;
        }
        if let Some(n) = t.n_samples {
            config.n_samples = n;
        }
        if let Some(temp) = t.temperature {
            config.temperature = temp;
        }
        if let Some(tp) = t.top_p {
            config.top_p = tp;
        }
        if let Some(r) = &t.rerank {
            config.rerank = eval::RerankStrategy::from_str(r)?;
        }
    }
    Ok(config)
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
mod tests;
