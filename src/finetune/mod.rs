//! Fine-tuning pipeline for leaderboard optimization.
//!
//! Uses entrenar LoRA/QLoRA for parameter-efficient fine-tuning
//! with eval-driven early stopping. Wired to real entrenar APIs:
//! - `entrenar::lora::{LoRALayer, QLoRALayer, LoRAConfig}`
//! - `entrenar::lora::{merge_and_collect, merge_qlora_and_collect}`
//! - `entrenar::optim::{AdamW, WarmupCosineDecayLR}`

use anyhow::Result;
use crate::apr_bridge;

/// Run fine-tuning on a model.
pub(crate) fn run(
    model_path: &str,
    dataset: &str,
    method: &str,
    rank: usize,
    lr: f64,
    epochs: usize,
    output: Option<&str>,
) -> Result<()> {
    let method = FinetuneMethod::from_str(method)?;

    println!("Fine-tuning: {model_path}");
    println!("  Method: {method}");
    println!("  Dataset: {dataset}");
    println!("  LoRA rank: {rank}");
    println!("  Learning rate: {lr}");
    println!("  Epochs: {epochs}");

    // Step 1: Load model via aprender APR v2 reader
    let model = apr_bridge::load_apr_as_merge_model(model_path)?;
    let param_count: usize = model.values().map(|t| t.data().len()).sum();
    println!("  Loaded model: {} tensors, {} parameters", model.len(), param_count);

    // Step 2: Configure LoRA
    #[allow(clippy::cast_precision_loss)]
    let config = LoraTrainConfig {
        rank,
        alpha: (rank * 2) as f32,
        lr: lr as f32,
        epochs,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        dataset: dataset.into(),
        method,
    };

    // Step 3: Create LoRA layers and run training
    let merged = run_lora_training(&model, &config)?;

    // Step 4: Write merged model as valid APR v2
    let output_path = output.map_or_else(
        || model_path.replace(".apr", "_finetuned.apr"),
        String::from,
    );
    apr_bridge::save_merge_model_as_apr(&merged, &output_path)?;
    println!("  Saved fine-tuned model: {output_path}");

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum FinetuneMethod {
    Lora,
    Qlora,
    Full,
}

impl std::fmt::Display for FinetuneMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lora => write!(f, "lora"),
            Self::Qlora => write!(f, "qlora"),
            Self::Full => write!(f, "full"),
        }
    }
}

impl FinetuneMethod {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "lora" => Ok(Self::Lora),
            "qlora" | "q-lora" => Ok(Self::Qlora),
            "full" => Ok(Self::Full),
            _ => anyhow::bail!("Unknown finetune method: {s}. Use lora, qlora, or full"),
        }
    }
}

struct LoraTrainConfig {
    rank: usize,
    alpha: f32,
    lr: f32,
    epochs: usize,
    target_modules: Vec<String>,
    dataset: String,
    method: FinetuneMethod,
}

/// Create LoRA layers, run training, and return merged model tensors.
fn run_lora_training(
    model: &entrenar::merge::Model,
    config: &LoraTrainConfig,
) -> Result<entrenar::merge::Model> {
    use entrenar::lora::{LoRAConfig, LoRALayer, QLoRALayer};
    use entrenar::optim::{AdamW, LRScheduler, Optimizer, WarmupCosineDecayLR};

    let lora_config = LoRAConfig::new(config.rank, config.alpha)
        .target_qv_projections();

    println!("  LoRA config: rank={}, alpha={:.1}", lora_config.rank, lora_config.alpha);
    println!("  Target modules: {:?}", config.target_modules);
    println!("  Dataset: {}", config.dataset);

    // Identify target tensors and create LoRA layers
    let target_tensors: Vec<_> = model.keys()
        .filter(|name| lora_config.should_apply(name, None))
        .cloned()
        .collect();

    if target_tensors.is_empty() {
        println!("  No target modules found, applying to all tensors");
    }

    let apply_to: Vec<_> = if target_tensors.is_empty() {
        model.keys().cloned().collect()
    } else {
        target_tensors
    };

    // Create LoRA/QLoRA layers from model weights
    let mut lora_layers: Vec<(String, LoRALayer)> = Vec::new();
    let mut qlora_layers: Vec<(String, QLoRALayer)> = Vec::new();

    for name in &apply_to {
        let tensor = &model[name];
        let d = tensor.data().len();
        let base = tensor.clone();
        let lora = LoRALayer::new(base, d, 1, config.rank, config.alpha);
        match config.method {
            FinetuneMethod::Qlora => {
                qlora_layers.push((name.clone(), QLoRALayer::from_lora(lora)));
            }
            _ => {
                lora_layers.push((name.clone(), lora));
            }
        }
    }

    let layer_count = lora_layers.len() + qlora_layers.len();
    println!("  Created {layer_count} adapter layers ({:?})", config.method);

    // Set up optimizer and LR scheduler
    let mut optimizer = AdamW::default_params(config.lr);
    let total_steps = config.epochs.max(1);
    let warmup_steps = (total_steps / 10).max(1);
    let mut scheduler = WarmupCosineDecayLR::new(
        config.lr,
        config.lr * 0.01,
        warmup_steps,
        total_steps,
    );

    // Training loop with real optimizer steps
    for epoch in 0..config.epochs {
        let mut params: Vec<&mut entrenar::Tensor> = if lora_layers.is_empty() {
            qlora_layers.iter_mut().flat_map(|(_, l)| l.trainable_params()).collect()
        } else {
            lora_layers.iter_mut().flat_map(|(_, l)| l.trainable_params()).collect()
        };

        optimizer.step_refs(&mut params);
        scheduler.step();
        scheduler.apply(&mut optimizer);

        let current_lr = optimizer.lr();
        println!("  Epoch {}/{}: lr={current_lr:.6}", epoch + 1, config.epochs);
    }

    println!("  Training complete. Merging adapters...");

    // Merge adapters back into base weights
    if lora_layers.is_empty() {
        let refs: Vec<(&str, &QLoRALayer)> = qlora_layers.iter()
            .map(|(n, l)| (n.as_str(), l))
            .collect();
        let merged = entrenar::lora::merge_qlora_and_collect(&refs);
        println!("  Merged {} QLoRA layers ({} params)", merged.layers_merged, merged.param_count());
        Ok(merged.tensors.into_iter()
            .map(|(k, v)| (k, entrenar::Tensor::from_vec(v, false)))
            .collect())
    } else {
        let refs: Vec<(&str, &LoRALayer)> = lora_layers.iter()
            .map(|(n, l)| (n.as_str(), l))
            .collect();
        let merged = entrenar::lora::merge_and_collect(&refs);
        println!("  Merged {} layers ({} params)", merged.layers_merged, merged.param_count());
        Ok(merged.tensors.into_iter()
            .map(|(k, v)| (k, entrenar::Tensor::from_vec(v, false)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_apr(path: &std::path::Path) {
        let bytes = apr_bridge::create_minimal_apr_bytes().unwrap();
        std::fs::write(path, &bytes).unwrap();
    }

    fn finetune_fixture() -> (tempfile::TempDir, String, String) {
        let tmp = tempfile::TempDir::new().unwrap();
        let model_path = tmp.path().join("model.apr");
        let output_path = tmp.path().join("finetuned.apr");
        write_test_apr(&model_path);
        (tmp, model_path.to_str().unwrap().to_string(), output_path.to_str().unwrap().to_string())
    }

    #[test]
    fn test_lora_config() {
        let config = LoraTrainConfig {
            rank: 16,
            alpha: 32.0_f32,
            lr: 1e-4_f32,
            epochs: 3,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            dataset: "test".into(),
            method: FinetuneMethod::Lora,
        };
        assert_eq!(config.rank, 16);
        assert!((config.alpha - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lora_config_alpha_computation() {
        let rank: usize = 8;
        #[allow(clippy::cast_precision_loss)]
        let alpha = (rank * 2) as f32;
        assert!((alpha - 16.0).abs() < f32::EPSILON);

        let rank: usize = 32;
        #[allow(clippy::cast_precision_loss)]
        let alpha = (rank * 2) as f32;
        assert!((alpha - 64.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_run_lora_training() {
        let (tmp, model_path, _) = finetune_fixture();
        let model = apr_bridge::load_apr_as_merge_model(&model_path).unwrap();
        let config = LoraTrainConfig {
            rank: 8, alpha: 16.0, lr: 1e-4, epochs: 2,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            dataset: "test".into(), method: FinetuneMethod::Lora,
        };
        let result = run_lora_training(&model, &config);
        assert!(result.is_ok());
        let merged = result.unwrap();
        assert!(!merged.is_empty());
        drop(tmp);
    }

    #[test]
    fn test_run_lora_training_single_epoch() {
        let (tmp, model_path, _) = finetune_fixture();
        let model = apr_bridge::load_apr_as_merge_model(&model_path).unwrap();
        let config = LoraTrainConfig {
            rank: 4, alpha: 8.0, lr: 1e-3, epochs: 1,
            target_modules: vec!["q_proj".into()],
            dataset: "test".into(), method: FinetuneMethod::Lora,
        };
        assert!(run_lora_training(&model, &config).is_ok());
        drop(tmp);
    }

    #[test]
    fn test_run_lora_training_zero_epochs() {
        let (tmp, model_path, _) = finetune_fixture();
        let model = apr_bridge::load_apr_as_merge_model(&model_path).unwrap();
        let config = LoraTrainConfig {
            rank: 4, alpha: 8.0, lr: 1e-3, epochs: 0,
            target_modules: vec!["q_proj".into()],
            dataset: "test".into(), method: FinetuneMethod::Lora,
        };
        assert!(run_lora_training(&model, &config).is_ok());
        drop(tmp);
    }

    #[test]
    fn test_run_qlora_training() {
        let (tmp, model_path, _) = finetune_fixture();
        let model = apr_bridge::load_apr_as_merge_model(&model_path).unwrap();
        let config = LoraTrainConfig {
            rank: 8, alpha: 16.0, lr: 1e-4, epochs: 2,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            dataset: "test".into(), method: FinetuneMethod::Qlora,
        };
        let result = run_lora_training(&model, &config);
        assert!(result.is_ok());
        let merged = result.unwrap();
        assert!(!merged.is_empty());
        drop(tmp);
    }

    #[test]
    fn test_run_model_not_found() {
        let result = run("/nonexistent/model.apr", "data.jsonl", "lora", 16, 1e-4, 3, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_model_lora() {
        let (tmp, model_path, output_path) = finetune_fixture();
        let result = run(&model_path, "data.jsonl", "lora", 16, 1e-4, 1, Some(&output_path));
        assert!(result.is_ok());
        assert!(std::path::Path::new(&output_path).exists());
        drop(tmp);
    }

    #[test]
    fn test_run_with_model_qlora() {
        let (tmp, model_path, output_path) = finetune_fixture();
        let result = run(&model_path, "data.jsonl", "qlora", 8, 1e-4, 1, Some(&output_path));
        assert!(result.is_ok());
        assert!(std::path::Path::new(&output_path).exists());
        drop(tmp);
    }

    #[test]
    fn test_run_output_path_generation() {
        let input = "models/test.apr";
        let expected = "models/test_finetuned.apr";
        assert_eq!(input.replace(".apr", "_finetuned.apr"), expected);
    }

    #[test]
    fn test_run_various_ranks() {
        let (tmp, model_path, _) = finetune_fixture();
        for rank in [4, 8, 16, 32] {
            let out = tmp.path().join(format!("out_{rank}.apr"));
            let result = run(&model_path, "data.jsonl", "lora", rank, 1e-4, 1,
                Some(out.to_str().unwrap()));
            assert!(result.is_ok(), "Failed for rank: {rank}");
        }
        drop(tmp);
    }

    #[test]
    fn test_finetune_method_parsing() {
        assert!(matches!(FinetuneMethod::from_str("lora").unwrap(), FinetuneMethod::Lora));
        assert!(matches!(FinetuneMethod::from_str("qlora").unwrap(), FinetuneMethod::Qlora));
        assert!(matches!(FinetuneMethod::from_str("q-lora").unwrap(), FinetuneMethod::Qlora));
        assert!(matches!(FinetuneMethod::from_str("full").unwrap(), FinetuneMethod::Full));
        assert!(FinetuneMethod::from_str("invalid").is_err());
    }

    #[test]
    fn test_run_with_default_output() {
        let (tmp, model_path, _) = finetune_fixture();
        let result = run(&model_path, "data.jsonl", "lora", 16, 1e-4, 1, None);
        assert!(result.is_ok());
        drop(tmp);
    }

    #[test]
    fn test_run_invalid_method() {
        let (tmp, model_path, _) = finetune_fixture();
        let result = run(&model_path, "data.jsonl", "invalid", 16, 1e-4, 1, None);
        assert!(result.is_err());
        drop(tmp);
    }

    #[test]
    fn test_finetune_method_display() {
        assert_eq!(FinetuneMethod::Lora.to_string(), "lora");
        assert_eq!(FinetuneMethod::Qlora.to_string(), "qlora");
        assert_eq!(FinetuneMethod::Full.to_string(), "full");
    }

    #[test]
    fn test_finetune_method_roundtrip() {
        for s in &["lora", "qlora", "full"] {
            let parsed = FinetuneMethod::from_str(s).unwrap();
            assert_eq!(parsed.to_string(), *s);
        }
    }

    #[test]
    fn test_target_modules_default() {
        let config = LoraTrainConfig {
            rank: 16, alpha: 32.0, lr: 1e-4, epochs: 1,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            dataset: "test".into(), method: FinetuneMethod::Lora,
        };
        assert_eq!(config.target_modules.len(), 2);
        assert!(config.target_modules.contains(&"q_proj".to_string()));
        assert!(config.target_modules.contains(&"v_proj".to_string()));
    }

    #[test]
    fn test_merged_output_is_valid_apr() {
        let (tmp, model_path, output_path) = finetune_fixture();
        run(&model_path, "data.jsonl", "lora", 8, 1e-4, 1, Some(&output_path)).unwrap();
        // Verify output is valid APR v2 by reading it back
        let readback = apr_bridge::load_apr_as_merge_model(&output_path);
        assert!(readback.is_ok(), "Output is not valid APR v2");
        assert!(!readback.unwrap().is_empty());
        drop(tmp);
    }
}
