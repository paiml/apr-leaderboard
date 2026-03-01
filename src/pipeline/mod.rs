//! Pipeline orchestration: validate → convert → distill → finetune → align → merge → prune → quantize → eval → compile → submit.
//!
//! Reads a TOML config and runs the full leaderboard pipeline.

use anyhow::Result;
use serde::Deserialize;

use crate::{align, compile, convert, eval, finetune, optimize, submit, validate};

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct PipelineConfig {
    pub model_id: String,
    pub output_dir: String,
    pub quantization: String,
    pub benchmarks: Vec<String>,
    pub submit: bool,
    pub leaderboard: String,
    pub validate: Option<ValidateConfig>,
    pub distill: Option<DistillConfig>,
    pub finetune: Option<FinetuneConfig>,
    pub align: Option<AlignConfig>,
    pub merge: Option<MergeConfig>,
    pub prune: Option<PruneConfig>,
    pub quantize: Option<QuantizeConfig>,
    pub tune: Option<TuneConfig>,
    pub eval: Option<EvalConfigToml>,
    pub compile: Option<CompileConfig>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct FinetuneConfig {
    pub dataset: String,
    pub method: Option<String>,
    pub rank: usize,
    pub lr: f64,
    pub epochs: usize,
    pub output: Option<String>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct DistillConfig {
    pub teacher: String,
    pub strategy: String,
    pub temperature: f64,
    pub alpha: f64,
    pub epochs: Option<usize>,
    pub data: Option<String>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct MergeConfig {
    pub models: Vec<String>,
    pub strategy: String,
    pub weights: Option<String>,
    pub base_model: Option<String>,
    pub density: Option<f64>,
    pub drop_rate: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct PruneConfig {
    pub method: String,
    pub target_ratio: f64,
    pub calibration: Option<String>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct QuantizeConfig {
    pub scheme: String,
    pub calibration: Option<String>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct AlignConfig {
    pub data: String,
    pub method: Option<String>,
    pub beta: Option<f64>,
    pub epochs: Option<usize>,
    pub ref_model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct ValidateConfig {
    pub data: String,
    pub benchmarks: Vec<String>,
    pub threshold: Option<f64>,
    pub decontaminate: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct TuneConfig {
    pub data: String,
    pub strategy: Option<String>,
    pub budget: Option<usize>,
    pub max_epochs: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct CompileConfig {
    pub release: Option<bool>,
    pub lto: Option<bool>,
    pub strip: Option<bool>,
    pub output: Option<String>,
}

#[derive(Debug, Deserialize)]
#[derive(serde::Serialize)]
pub(crate) struct EvalConfigToml {
    pub samples: Option<usize>,
    pub prompt_strategy: Option<String>,
    pub n_samples: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub rerank: Option<String>,
    pub exemplars: Option<String>,
    pub system: Option<String>,
}

/// Validate pipeline ordering against the technique interaction matrix (§10).
///
/// Golden ordering: distill → finetune → merge → prune → quantize
/// Warns about known anti-patterns that degrade quality.
pub(crate) fn validate_pipeline_order(config: &PipelineConfig) -> Vec<String> {
    let mut warnings = Vec::new();

    // Anti-pattern 1: prune → finetune (LoRA can't recover pruned knowledge)
    if config.prune.is_some() && config.finetune.is_some() {
        // In our golden ordering, finetune comes before prune — this is correct.
        // But if someone had prune without finetune, that's fine too.
        // The real anti-pattern would be if ordering were reversed,
        // but our pipeline enforces golden ordering already.
    }

    // Anti-pattern 2: finetune → distill (overwrites fine-tuned specialization)
    if config.finetune.is_some() && config.distill.is_some() {
        // Our pipeline runs distill before finetune (golden ordering).
        // This is correct — warn only if distill is absent but finetune is present
        // AND the user might be feeding an already-finetuned model.
    }

    // Anti-pattern 3: quantize → anything else (quality loss compounds)
    // Our pipeline always runs quantize last — this is enforced by ordering.

    // Warn about suboptimal combos
    if config.merge.is_some() && config.finetune.is_none() {
        warnings.push(
            "Merge without finetune: merging untrained variants is suboptimal. \
             Consider adding [finetune] before [merge] (§10 golden ordering).".into()
        );
    }

    if config.prune.is_some() && config.quantize.is_none() {
        warnings.push(
            "Prune without quantize: pruning alone may not reduce model size effectively. \
             Consider adding [quantize] after [prune].".into()
        );
    }

    if config.distill.is_some() && config.finetune.is_none() && config.merge.is_none() {
        warnings.push(
            "Distill without finetune or merge: distilled knowledge benefits from \
             task-specific adaptation. Consider adding [finetune] after [distill].".into()
        );
    }

    if config.align.is_some() && config.finetune.is_none() {
        warnings.push(
            "Align without finetune: preference optimization (DPO/ORPO) works best \
             after supervised fine-tuning. Consider adding [finetune] before [align].".into()
        );
    }

    warnings
}

/// Compute a deterministic BLAKE3 hash of the pipeline config for reproducibility (§11).
///
/// Serializes the config to canonical TOML and returns the hex-encoded hash.
/// Two identical configs always produce the same fingerprint.
pub(crate) fn config_hash(config: &PipelineConfig) -> Result<String> {
    let toml_str = toml::to_string(config)
        .map_err(|e| anyhow::anyhow!("Failed to serialize config for hashing: {e}"))?;
    Ok(blake3::hash(toml_str.as_bytes()).to_hex().to_string())
}

pub(crate) fn run_pipeline(config: &PipelineConfig) -> Result<()> {
    println!("=== APR Leaderboard Pipeline ===\n");

    // Print deterministic config fingerprint for reproducibility (§11)
    let hash = config_hash(config)?;
    println!("  Config fingerprint: {hash}");

    // Check pipeline ordering for anti-patterns (§10)
    let warnings = validate_pipeline_order(config);
    for warning in &warnings {
        println!("  WARNING: {warning}");
    }
    if !warnings.is_empty() {
        println!();
    }

    let total_steps = count_steps(config);
    let mut step = 0;

    // Step: Optional validate (decontamination check before training)
    if let Some(val) = &config.validate {
        step += 1;
        println!("[{step}/{total_steps}] Validating training data for contamination...");
        validate::run(
            &val.data,
            &val.benchmarks,
            val.threshold.unwrap_or(0.01),
            val.decontaminate.unwrap_or(false),
            None,
        )?;
    }

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

    // Step: Optional align (DPO/ORPO after finetune)
    if let Some(al) = &config.align {
        step += 1;
        let method = al.method.as_deref().unwrap_or("dpo");
        println!("[{step}/{total_steps}] Aligning with {method}...");
        let output_path = format!("{}/aligned.apr", config.output_dir);
        align::run(
            &model_path,
            &al.data,
            method,
            al.beta.unwrap_or(0.1),
            al.epochs.unwrap_or(3),
            al.ref_model.as_deref(),
            Some(&output_path),
        )?;
        model_path = output_path;
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

    // Step: Optional tune (HPO before prune/quantize)
    if let Some(tu) = &config.tune {
        step += 1;
        let strategy = tu.strategy.as_deref().unwrap_or("tpe");
        println!("[{step}/{total_steps}] Running HPO ({strategy}, {} trials)...", tu.budget.unwrap_or(20));
        optimize::tune(
            &model_path,
            &tu.data,
            strategy,
            tu.budget.unwrap_or(20),
            tu.max_epochs.unwrap_or(3),
        )?;
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

    // Step: Optional compile (binary output)
    if let Some(comp) = &config.compile {
        step += 1;
        println!("[{step}/{total_steps}] Compiling to standalone binary...");
        compile::run(
            &model_path,
            comp.release.unwrap_or(false),
            comp.lto.unwrap_or(false),
            comp.strip.unwrap_or(false),
            comp.output.as_deref(),
        )?;
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
        if t.exemplars.is_some() {
            config.exemplars.clone_from(&t.exemplars);
        }
        if t.system.is_some() {
            config.system.clone_from(&t.system);
        }
    }
    Ok(config)
}

fn count_steps(config: &PipelineConfig) -> usize {
    let mut n = 1; // convert always runs
    if config.validate.is_some() { n += 1; }
    if config.distill.is_some() { n += 1; }
    if config.finetune.is_some() { n += 1; }
    if config.align.is_some() { n += 1; }
    if config.merge.is_some() { n += 1; }
    if config.tune.is_some() { n += 1; }
    if config.prune.is_some() { n += 1; }
    if config.quantize.is_some() { n += 1; }
    if !config.benchmarks.is_empty() { n += 1; }
    if config.compile.is_some() { n += 1; }
    if config.submit { n += 1; }
    n
}

#[cfg(test)]
mod tests;
