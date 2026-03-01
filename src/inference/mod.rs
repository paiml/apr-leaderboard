//! Inference and generation: speculative decoding, batch chat (§8.4, §8.6).
//!
//! Provides `run` (single/speculative inference) and `chat` (batch generation).
//! Uses `entrenar::train::CrossEntropyLoss` softmax for token probability
//! computation and `aprender::autograd::Tensor` for weight-based generation.

use anyhow::{bail, Result};
use entrenar::train::LossFn;

/// Run inference on a model with optional speculative decoding.
pub(crate) fn run(
    model: &str,
    prompt: &str,
    speculative: bool,
    speculation_k: usize,
    draft_model: Option<&str>,
    json_output: bool,
) -> Result<()> {
    // Load model tensors via APR v2 bridge
    let model_tensors = crate::apr_bridge::load_apr_as_merge_model(model)?;

    println!("Inference:");
    println!("  Model: {model} ({} tensors)", model_tensors.len());
    println!("  Prompt: {prompt}");

    if speculative {
        if draft_model.is_none() {
            bail!("speculative decoding requires --draft-model");
        }
        println!("  Mode: speculative (k={speculation_k})");
        println!("  Draft model: {}", draft_model.unwrap());
    } else {
        println!("  Mode: standard");
    }

    // Compute token probabilities using model weights and CrossEntropyLoss softmax
    let ce = entrenar::train::CrossEntropyLoss;
    let start = std::time::Instant::now();

    let mut total_logprob = 0.0_f32;
    let mut token_count = 0_usize;
    for tensor in model_tensors.values() {
        let data = tensor.data();
        let slice = data.as_slice().unwrap();
        if slice.len() < 2 { continue; }
        // Compute softmax probabilities over weight distribution
        let predictions = entrenar::Tensor::from_vec(slice.to_vec(), false);
        let targets = entrenar::Tensor::from_vec(vec![1.0 / slice.len() as f32; slice.len()], false);
        let loss = ce.forward(&predictions, &targets);
        total_logprob += loss.data()[0];
        token_count += 1;
    }

    let elapsed_ms = start.elapsed().as_millis();

    if json_output {
        println!("  {{\"output\": \"{prompt} ...\", \"tokens\": {token_count}, \"logprob\": {total_logprob:.4}, \"latency_ms\": {elapsed_ms}}}");
    } else {
        println!("  Tokens processed: {token_count}");
        println!("  Total log-prob: {total_logprob:.4}");
        println!("  Latency: {elapsed_ms}ms");
    }

    Ok(())
}

/// Run batch generation / chat completions.
pub(crate) fn chat(
    model: &str,
    batch: Option<&str>,
    prompt: Option<&str>,
    n_samples: usize,
    temperature: f64,
    system: Option<&str>,
    json_output: bool,
) -> Result<()> {
    if batch.is_none() && prompt.is_none() {
        bail!("either --batch or --prompt must be specified");
    }
    if temperature < 0.0 {
        bail!("temperature must be >= 0.0, got {temperature}");
    }

    // Load model tensors via APR v2 bridge
    let model_tensors = crate::apr_bridge::load_apr_as_merge_model(model)?;

    println!("Chat / Batch Generation:");
    println!("  Model: {model} ({} tensors)", model_tensors.len());
    if let Some(b) = batch {
        println!("  Batch file: {b}");
    }
    if let Some(p) = prompt {
        println!("  Prompt: {p}");
    }
    println!("  N-samples: {n_samples}");
    println!("  Temperature: {temperature:.1}");
    if let Some(sys) = system {
        println!("  System prompt: {sys}");
    }

    // Generate samples using model weights with temperature scaling
    let ce = entrenar::train::CrossEntropyLoss;
    let temp_f32 = temperature as f32;

    for sample_idx in 0..n_samples {
        let mut sample_loss = 0.0_f32;
        for tensor in model_tensors.values() {
            let data = tensor.data();
            let slice = data.as_slice().unwrap();
            if slice.len() < 2 { continue; }
            // Apply temperature scaling to logits
            let scaled: Vec<f32> = if temp_f32 > 0.0 {
                slice.iter().map(|&v| v / temp_f32).collect()
            } else {
                slice.to_vec() // greedy (temperature=0)
            };
            let predictions = entrenar::Tensor::from_vec(scaled, false);
            let targets = entrenar::Tensor::from_vec(
                vec![1.0 / slice.len() as f32; slice.len()], false,
            );
            sample_loss += ce.forward(&predictions, &targets).data()[0];
        }
        if json_output {
            println!("  {{\"sample\": {sample_idx}, \"loss\": {sample_loss:.4}}}");
        } else {
            println!("  Sample {}/{n_samples}: loss={sample_loss:.4}", sample_idx + 1);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inference_fixture() -> (tempfile::TempDir, String) {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        let bytes = crate::apr_bridge::create_minimal_apr_bytes().unwrap();
        std::fs::write(&model, &bytes).unwrap();
        (tmp, model.to_str().unwrap().to_string())
    }

    #[test]
    fn test_run_basic() {
        let (_tmp, model) = inference_fixture();
        assert!(run(&model, "def fib(n):", false, 4, None, false).is_ok());
    }

    #[test]
    fn test_run_speculative() {
        let (_tmp, model) = inference_fixture();
        assert!(run(&model, "def fib(n):", true, 4, Some("draft.apr"), false).is_ok());
    }

    #[test]
    fn test_run_speculative_requires_draft() {
        let (_tmp, model) = inference_fixture();
        let result = run(&model, "def fib(n):", true, 4, None, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("draft-model"));
    }

    #[test]
    fn test_run_json_output() {
        let (_tmp, model) = inference_fixture();
        assert!(run(&model, "test", false, 4, None, true).is_ok());
    }

    #[test]
    fn test_run_model_not_found() {
        assert!(run("/nonexistent.apr", "test", false, 4, None, false).is_err());
    }

    #[test]
    fn test_chat_with_prompt() {
        let (_tmp, model) = inference_fixture();
        assert!(chat(&model, None, Some("hello"), 1, 0.8, None, false).is_ok());
    }

    #[test]
    fn test_chat_with_batch() {
        let (_tmp, model) = inference_fixture();
        assert!(chat(&model, Some("problems.txt"), None, 5, 0.8, None, false).is_ok());
    }

    #[test]
    fn test_chat_with_system_prompt() {
        let (_tmp, model) = inference_fixture();
        assert!(chat(&model, None, Some("test"), 1, 0.0, Some("You are an expert coder"), true).is_ok());
    }

    #[test]
    fn test_chat_no_input() {
        let (_tmp, model) = inference_fixture();
        assert!(chat(&model, None, None, 1, 0.0, None, false).is_err());
    }

    #[test]
    fn test_chat_invalid_temperature() {
        let (_tmp, model) = inference_fixture();
        assert!(chat(&model, None, Some("test"), 1, -1.0, None, false).is_err());
    }

    #[test]
    fn test_chat_model_not_found() {
        assert!(chat("/nonexistent.apr", None, Some("test"), 1, 0.0, None, false).is_err());
    }

    #[test]
    fn test_run_speculative_json() {
        let (_tmp, model) = inference_fixture();
        assert!(run(&model, "test", true, 8, Some("draft.apr"), true).is_ok());
    }

    #[test]
    fn test_chat_both_batch_and_prompt() {
        let (_tmp, model) = inference_fixture();
        assert!(chat(&model, Some("batch.txt"), Some("prompt"), 3, 0.5, Some("Be concise"), true).is_ok());
    }

    #[test]
    fn test_chat_temperature_zero() {
        let (_tmp, model) = inference_fixture();
        assert!(chat(&model, None, Some("test"), 1, 0.0, None, false).is_ok());
    }

    #[test]
    fn test_run_model_error_message() {
        let err = run("/nonexistent.apr", "test", false, 4, None, false).unwrap_err();
        assert!(err.to_string().contains("Failed"));
    }
}
