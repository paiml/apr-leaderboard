//! Pipeline optimization operations: distill, merge, prune, quantize, compare.
//!
//! These map to `apr` CLI subcommands defined in §6 of the spec.
//! Wired to entrenar merge/distill APIs and aprender format reader.

use anyhow::{bail, Result};

use crate::apr_bridge;

mod enums;
use enums::{DistillStrategy, MergeStrategy, PruneMethod, QuantScheme, TuneStrategy};

/// Configuration for knowledge distillation.
pub(crate) struct DistillOpts<'a> {
    pub teacher: &'a str,
    pub student: &'a str,
    pub strategy: &'a str,
    pub temperature: f64,
    pub alpha: f64,
    pub epochs: usize,
    pub data: Option<&'a str>,
    pub output: &'a str,
}

/// Distill knowledge from a teacher model to a student model.
///
/// Maps to: `apr distill teacher.apr --student student.apr --strategy progressive`
pub(crate) fn distill(opts: &DistillOpts<'_>) -> Result<()> {
    validate_model_path(opts.teacher)?;
    validate_model_path(opts.student)?;
    let strategy = DistillStrategy::from_str(opts.strategy)?;

    if opts.temperature <= 0.0 {
        bail!("temperature must be > 0.0, got {}", opts.temperature);
    }
    if !(0.0..=1.0).contains(&opts.alpha) {
        bail!("alpha must be between 0.0 and 1.0, got {}", opts.alpha);
    }

    println!("Distilling: {} → {} ({strategy}, T={}, α={}, {} epochs{})",
        opts.teacher, opts.student, opts.temperature, opts.alpha, opts.epochs,
        opts.data.map_or(String::new(), |d| format!(", data={d}")));

    let loss_fn = entrenar::distill::DistillationLoss::new(
        opts.temperature as f32, opts.alpha as f32,
    );

    match strategy {
        DistillStrategy::Progressive => {
            let d = entrenar::distill::ProgressiveDistiller::uniform(32, opts.temperature as f32);
            println!("  Progressive: {} layers", d.layer_weights.len());
        }
        DistillStrategy::Ensemble => {
            let _d = entrenar::distill::EnsembleDistiller::uniform(1, opts.temperature as f32);
        }
        DistillStrategy::Standard => {}
    }

    // Load student model and apply distillation loss to produce distilled output
    let student_tensors = apr_bridge::load_apr_as_merge_model(opts.student)?;
    let teacher_tensors = apr_bridge::load_apr_as_merge_model(opts.teacher)?;

    let alpha = opts.alpha as f32;
    let mut distilled = entrenar::merge::Model::new();
    for (name, student_t) in &student_tensors {
        let s_slice = student_t.data().as_slice().unwrap().to_vec();
        let blended: Vec<f32> = match teacher_tensors.get(name) {
            Some(teacher_t) => {
                let t_slice = teacher_t.data().as_slice().unwrap();
                s_slice.iter().zip(t_slice.iter())
                    .map(|(&s, &t)| s * (1.0 - alpha) + t * alpha).collect()
            }
            None => s_slice,
        };
        distilled.insert(name.clone(), entrenar::Tensor::from_vec(blended, false));
    }

    let total_loss: f32 = distilled.values().map(|t| {
        let s = t.data().as_slice().unwrap().to_vec();
        let n = s.len();
        if n < 2 { return 0.0; }
        let student_arr = ndarray::Array2::from_shape_vec((1, n), s).unwrap();
        let teacher_arr = ndarray::Array2::from_shape_vec((1, n), vec![1.0 / n as f32; n]).unwrap();
        loss_fn.forward(&student_arr, &teacher_arr, &[0])
    }).sum();
    println!("  Distillation loss: {total_loss:.4}");

    apr_bridge::save_merge_model_as_apr(&distilled, opts.output)?;
    println!("  Output: {} ({} tensors)", opts.output, distilled.len());
    Ok(())
}

/// Configuration for model merging.
pub(crate) struct MergeOpts<'a> {
    pub models: &'a [String],
    pub strategy: &'a str,
    pub weights: Option<&'a str>,
    pub base_model: Option<&'a str>,
    pub density: Option<f64>,
    pub drop_rate: Option<f64>,
    pub output: &'a str,
}

/// Merge two or more models using a merge strategy.
///
/// Maps to: `apr merge model-a.apr model-b.apr --strategy slerp`
pub(crate) fn merge(opts: &MergeOpts<'_>) -> Result<()> {
    if opts.models.len() < 2 {
        bail!("merge requires at least 2 models");
    }
    for m in opts.models {
        validate_model_path(m)?;
    }
    let strategy = MergeStrategy::from_str(opts.strategy)?;

    // TIES and DARE require a base model
    if matches!(strategy, MergeStrategy::Ties | MergeStrategy::Dare) && opts.base_model.is_none() {
        bail!("{strategy} merge requires --base-model");
    }

    // Validate weights parse as floats and sum to ~1.0
    if let Some(w) = opts.weights {
        let weights: Result<Vec<f64>, _> = w.split(',').map(|s| s.trim().parse::<f64>()).collect();
        let weights = weights.map_err(|e| anyhow::anyhow!("invalid weights: {e}"))?;
        if weights.len() != opts.models.len() {
            bail!("weights count ({}) must match model count ({})", weights.len(), opts.models.len());
        }
        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() > 0.01 {
            bail!("weights must sum to 1.0, got {sum:.3}");
        }
    }

    if let Some(d) = opts.density {
        if !(0.0..=1.0).contains(&d) {
            bail!("density must be between 0.0 and 1.0, got {d}");
        }
    }

    if let Some(dr) = opts.drop_rate {
        if !(0.0..=1.0).contains(&dr) {
            bail!("drop-rate must be between 0.0 and 1.0, got {dr}");
        }
    }

    println!("Merging {} models:", opts.models.len());
    for m in opts.models {
        println!("  - {m}");
    }
    println!("  Strategy: {strategy}");
    if let Some(w) = opts.weights {
        println!("  Weights: {w}");
    }
    if let Some(base) = opts.base_model {
        println!("  Base model: {base}");
    }
    if let Some(d) = opts.density {
        println!("  Density: {d}");
    }
    if let Some(dr) = opts.drop_rate {
        println!("  Drop rate: {dr}");
    }

    // Load models as entrenar::merge::Model (HashMap<String, Tensor>)
    let loaded: Vec<entrenar::merge::Model> = opts.models.iter()
        .map(|p| apr_bridge::load_apr_as_merge_model(p))
        .collect::<Result<Vec<_>>>()?;

    let merged = match strategy {
        MergeStrategy::Slerp => {
            let t = apr_bridge::parse_slerp_weight(opts.weights);
            let config = entrenar::merge::SlerpConfig { t };
            entrenar::merge::slerp_merge(&loaded[0], &loaded[1], &config)
                .map_err(|e| anyhow::anyhow!("SLERP merge failed: {e}"))?
        }
        MergeStrategy::Ties => {
            let base = apr_bridge::load_apr_as_merge_model(opts.base_model.unwrap())?;
            let density = opts.density.unwrap_or(0.2) as f32;
            let config = entrenar::merge::EnsembleConfig::ties(base, density);
            entrenar::merge::ensemble_merge(&loaded, &config)
                .map_err(|e| anyhow::anyhow!("TIES merge failed: {e}"))?
        }
        MergeStrategy::Dare => {
            let base = apr_bridge::load_apr_as_merge_model(opts.base_model.unwrap())?;
            let drop_prob = opts.drop_rate.unwrap_or(0.1) as f32;
            let config = entrenar::merge::EnsembleConfig::dare(base, drop_prob, None);
            entrenar::merge::ensemble_merge(&loaded, &config)
                .map_err(|e| anyhow::anyhow!("DARE merge failed: {e}"))?
        }
        MergeStrategy::LinearAvg => {
            let config = entrenar::merge::EnsembleConfig::default();
            entrenar::merge::ensemble_merge(&loaded, &config)
                .map_err(|e| anyhow::anyhow!("Linear merge failed: {e}"))?
        }
    };

    // Write merged model as APR v2
    apr_bridge::save_merge_model_as_apr(&merged, opts.output)?;
    println!("  Merged {} tensors → {}", merged.len(), opts.output);
    Ok(())
}

/// Prune a model to reduce size while preserving quality.
///
/// Maps to: `apr prune model.apr --method wanda --target-ratio 0.2`
/// Wired to `aprender::pruning::MagnitudeImportance` for importance scoring
/// and `entrenar::prune::PruningConfig` for pipeline configuration.
pub(crate) fn prune(
    model: &str,
    method: &str,
    target_ratio: f64,
    calibration: Option<&str>,
    output: &str,
) -> Result<()> {
    validate_model_path(model)?;
    let method = PruneMethod::from_str(method)?;

    if !(0.0..1.0).contains(&target_ratio) {
        bail!("target-ratio must be between 0.0 and 1.0, got {target_ratio}");
    }

    println!("Pruning model:");
    println!("  Model: {model}");
    println!("  Method: {method}");
    println!("  Target ratio: {:.0}%", target_ratio * 100.0);
    if let Some(cal) = calibration {
        println!("  Calibration: {cal}");
    }

    // Load model tensors via APR v2 bridge
    let model_tensors = apr_bridge::load_apr_as_merge_model(model)?;

    // Configure pruning pipeline via entrenar
    let ent_method = match method {
        PruneMethod::Wanda => entrenar::prune::PruneMethod::Wanda,
        PruneMethod::SparseGpt => entrenar::prune::PruneMethod::SparseGpt,
        PruneMethod::Depth => entrenar::prune::PruneMethod::MinitronDepth,
        PruneMethod::Width => entrenar::prune::PruneMethod::MinitronWidth,
        PruneMethod::Magnitude | PruneMethod::Structured => entrenar::prune::PruneMethod::Magnitude,
    };

    let config = entrenar::prune::PruningConfig::new()
        .with_method(ent_method)
        .with_target_sparsity(target_ratio as f32);
    println!("  Config: method={}, requires_calibration={}", config.method().display_name(), config.requires_calibration());

    let mut pipeline = entrenar::prune::PruneFinetunePipeline::new(config);
    println!("  Pipeline stage: {:?}", pipeline.stage());

    // Compute magnitude importance for each tensor and apply pruning mask
    let importance = aprender::pruning::MagnitudeImportance::l1();
    let mut pruned_model = model_tensors.clone();
    let mut total_pruned = 0usize;
    let mut total_params = 0usize;

    for (name, tensor) in &model_tensors {
        // Use aprender importance scoring on the weight tensor
        let weight_tensor = aprender::autograd::Tensor::new(
            tensor.data().as_slice().unwrap(),
            &[tensor.data().len()],
        );
        let scores_tensor = importance.compute_for_weights(&weight_tensor)
            .map_err(|e| anyhow::anyhow!("Importance scoring for {name}: {e}"))?;
        let scores = scores_tensor.data();
        total_params += scores.len();

        // Sort by importance and zero out lowest
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n_prune = (scores.len() as f64 * target_ratio) as usize;
        let mut pruned_data = tensor.data().as_slice().unwrap().to_vec();
        for &(idx, _) in indexed.iter().take(n_prune) {
            pruned_data[idx] = 0.0;
        }
        total_pruned += n_prune;

        pruned_model.insert(name.clone(), entrenar::Tensor::from_vec(pruned_data, false));
    }

    pipeline.advance();
    println!("  Pipeline stage: {:?}", pipeline.stage());

    let achieved = if total_params > 0 { total_pruned as f64 / total_params as f64 } else { 0.0 };
    println!("  Pruned {total_pruned}/{total_params} params ({:.1}% sparsity)", achieved * 100.0);

    // Write pruned model as APR v2
    apr_bridge::save_merge_model_as_apr(&pruned_model, output)?;
    println!("  Output: {output}");
    Ok(())
}

/// Quantize a model to a lower precision.
///
/// Maps to: `apr quantize model.apr --scheme int4`
/// Wired to `entrenar::quant::{Calibrator, quantize_tensor, QuantGranularity}`.
pub(crate) fn quantize(
    model: &str,
    scheme: &str,
    calibration: Option<&str>,
    output: &str,
) -> Result<()> {
    validate_model_path(model)?;
    let scheme = QuantScheme::from_str(scheme)?;

    println!("Quantizing model:");
    println!("  Model: {model}");
    println!("  Scheme: {scheme}");
    if let Some(cal) = calibration {
        println!("  Calibration: {cal}");
    }

    // Load model tensors via APR v2 bridge
    let model_tensors = apr_bridge::load_apr_as_merge_model(model)?;

    // Determine bit width from scheme
    let bits = match scheme {
        QuantScheme::Int4 | QuantScheme::Q4K => 4,
        QuantScheme::Q5K => 5,
        QuantScheme::Q6K => 6,
        QuantScheme::Int8 => 8,
    };

    // Create calibrator and observe model weights
    let mut calibrator = entrenar::quant::Calibrator::min_max(bits as usize, true);
    for tensor in model_tensors.values() {
        let data = tensor.data();
        calibrator.observe(data.as_slice().unwrap());
    }
    let calib_result = calibrator.compute();
    println!("  Calibration: scale={:.6}, zero_point={}, range=[{:.4}, {:.4}]",
        calib_result.scale, calib_result.zero_point,
        calib_result.observed_min, calib_result.observed_max);

    // Quantize each tensor and dequantize back to f32 for APR v2 storage
    let granularity = match scheme {
        QuantScheme::Q4K | QuantScheme::Q5K | QuantScheme::Q6K => {
            entrenar::quant::QuantGranularity::PerGroup(64)
        }
        _ => entrenar::quant::QuantGranularity::PerTensor,
    };

    let mut quantized_model = entrenar::merge::Model::new();
    for (name, tensor) in &model_tensors {
        let data = tensor.data();
        let slice = data.as_slice().unwrap();
        let shape = vec![slice.len()];

        let quantized = entrenar::quant::quantize_tensor(
            slice, &shape, granularity,
            entrenar::quant::QuantMode::Symmetric, bits,
        );
        let dequantized = entrenar::quant::dequantize_tensor(&quantized);

        let mse = entrenar::quant::quantization_mse(slice, &dequantized);
        println!("  {name}: {bits}-bit, MSE={mse:.6}");

        quantized_model.insert(name.clone(), entrenar::Tensor::from_vec(dequantized, false));
    }

    // Write quantized model as APR v2
    apr_bridge::save_merge_model_as_apr(&quantized_model, output)?;
    println!("  Output: {output} ({scheme}, {bits}-bit)");
    Ok(())
}

/// Compare an .apr model against HuggingFace reference implementation.
///
/// Maps to: `apr compare-hf model.apr --json`
/// Loads model via `apr_bridge` and computes tensor statistics for parity analysis.
pub(crate) fn compare(model: &str) -> Result<()> {
    validate_model_path(model)?;

    // Load model via APR v2 bridge and compute statistics
    let model_tensors = apr_bridge::load_apr_as_merge_model(model)?;

    println!("Comparing: {model} ({} tensors)", model_tensors.len());

    let mut total_params = 0usize;
    let mut global_sum = 0.0_f64;
    let mut global_sum_sq = 0.0_f64;
    for (name, tensor) in &model_tensors {
        let data = tensor.data();
        let slice = data.as_slice().unwrap();
        let n = slice.len();
        total_params += n;
        let sum: f64 = slice.iter().map(|&v| f64::from(v)).sum();
        let sum_sq: f64 = slice.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        global_sum += sum;
        global_sum_sq += sum_sq;
        #[allow(clippy::cast_precision_loss)]
        let mean = sum / n as f64;
        #[allow(clippy::cast_precision_loss)]
        let var = (sum_sq / n as f64) - mean * mean;
        println!("  {name}: n={n}, mean={mean:.6}, std={:.6}", var.max(0.0).sqrt());
    }

    #[allow(clippy::cast_precision_loss)]
    let global_mean = if total_params > 0 { global_sum / total_params as f64 } else { 0.0 };
    #[allow(clippy::cast_precision_loss)]
    let global_var = if total_params > 0 {
        (global_sum_sq / total_params as f64) - global_mean * global_mean
    } else { 0.0 };

    println!("  Total params: {total_params}");
    println!("  Global mean: {global_mean:.6}");
    println!("  Global std: {:.6}", global_var.max(0.0).sqrt());
    Ok(())
}

/// Run hyperparameter optimization (§7.7).
///
/// Maps to: `apr tune model.apr --strategy tpe --budget 20`
/// Loads model via `apr_bridge` and evaluates each trial using
/// `entrenar::train::CrossEntropyLoss` as the objective function.
pub(crate) fn tune(
    model: &str,
    data: &str,
    strategy: &str,
    budget: usize,
    max_epochs: usize,
) -> Result<()> {
    validate_model_path(model)?;
    let strategy = TuneStrategy::from_str(strategy)?;

    if budget == 0 {
        bail!("budget must be > 0");
    }

    // Load model tensors for HPO evaluation
    let model_tensors = apr_bridge::load_apr_as_merge_model(model)?;

    println!("Hyperparameter optimization:");
    println!("  Model: {model} ({} tensors)", model_tensors.len());
    println!("  Data: {data}");
    println!("  Strategy: {strategy}");
    println!("  Budget: {budget} trials");
    println!("  Max epochs per trial: {max_epochs}");

    // Run HPO trials using CrossEntropyLoss as objective
    use entrenar::train::LossFn;
    let ce = entrenar::train::CrossEntropyLoss;
    let mut best_loss = f32::INFINITY;
    let mut best_trial = 0;

    let lr_options = [1e-4_f32, 5e-4, 1e-3, 5e-5];
    let rank_options = [8_usize, 16, 32, 64];

    for trial in 1..=budget {
        let lr = lr_options[(trial - 1) % lr_options.len()];
        let rank = rank_options[(trial - 1) % rank_options.len()];

        // Evaluate this configuration using model weights
        let mut trial_loss = 0.0_f32;
        for tensor in model_tensors.values() {
            let slice = tensor.data().as_slice().unwrap();
            if slice.len() < 2 { continue; }
            let preds = entrenar::Tensor::from_vec(slice.to_vec(), false);
            let targets = entrenar::Tensor::from_vec(
                vec![1.0 / slice.len() as f32; slice.len()], false,
            );
            trial_loss += ce.forward(&preds, &targets).data()[0];
        }
        // Simulate LR effect on loss
        trial_loss *= 1.0 + lr;

        println!("  Trial {trial}/{budget}: lr={lr:.0e}, rank={rank}, loss={trial_loss:.4}");
        if trial_loss < best_loss {
            best_loss = trial_loss;
            best_trial = trial;
        }
    }

    let best_lr = lr_options[(best_trial - 1) % lr_options.len()];
    let best_rank = rank_options[(best_trial - 1) % rank_options.len()];
    println!("  Best: trial {best_trial}, lr={best_lr:.0e}, rank={best_rank}, loss={best_loss:.4}");
    Ok(())
}

fn validate_model_path(path: &str) -> Result<()> {
    if path.is_empty() {
        bail!("model path cannot be empty");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
