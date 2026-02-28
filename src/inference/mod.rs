//! Inference and generation: speculative decoding, batch chat (§8.4, §8.6).
//!
//! Provides `run` (single/speculative inference) and `chat` (batch generation).

use anyhow::{bail, Result};

/// Run inference on a model with optional speculative decoding.
pub(crate) fn run(
    model: &str,
    prompt: &str,
    speculative: bool,
    speculation_k: usize,
    draft_model: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let model_data = std::fs::read(model)
        .map_err(|e| anyhow::anyhow!("Failed to load model {model}: {e}"))?;

    println!("Inference:");
    println!("  Model: {model} ({} bytes)", model_data.len());
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

    // Scaffold: in production, runs inference through apr runtime
    println!("  [scaffold] Would run: apr run {model} --prompt \"{prompt}\"");
    if speculative {
        println!("    --speculative --speculation-k {speculation_k} --draft-model {}", draft_model.unwrap());
    }

    if json_output {
        println!("  {{\"output\": \"[scaffold — requires real inference]\", \"tokens\": 0, \"latency_ms\": 0}}");
    } else {
        println!("  Output: [scaffold — requires real inference]");
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

    let model_data = std::fs::read(model)
        .map_err(|e| anyhow::anyhow!("Failed to load model {model}: {e}"))?;

    println!("Chat / Batch Generation:");
    println!("  Model: {model} ({} bytes)", model_data.len());
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

    // Scaffold: in production, runs batch generation
    println!("  [scaffold] Would run: apr chat {model}");
    if let Some(b) = batch {
        println!("    --batch {b}");
    }
    println!("    --n-samples {n_samples} --temperature {temperature}");
    if json_output {
        println!("    --json");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- run ---
    #[test]
    fn test_run_basic() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        assert!(run(model.to_str().unwrap(), "def fib(n):", false, 4, None, false).is_ok());
    }

    #[test]
    fn test_run_speculative() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        assert!(run(model.to_str().unwrap(), "def fib(n):", true, 4, Some("draft.apr"), false).is_ok());
    }

    #[test]
    fn test_run_speculative_requires_draft() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let result = run(model.to_str().unwrap(), "def fib(n):", true, 4, None, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("draft-model"));
    }

    #[test]
    fn test_run_json_output() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        assert!(run(model.to_str().unwrap(), "test", false, 4, None, true).is_ok());
    }

    #[test]
    fn test_run_model_not_found() {
        assert!(run("/nonexistent.apr", "test", false, 4, None, false).is_err());
    }

    // --- chat ---
    #[test]
    fn test_chat_with_prompt() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        assert!(chat(model.to_str().unwrap(), None, Some("hello"), 1, 0.8, None, false).is_ok());
    }

    #[test]
    fn test_chat_with_batch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        assert!(chat(model.to_str().unwrap(), Some("problems.txt"), None, 5, 0.8, None, false).is_ok());
    }

    #[test]
    fn test_chat_with_system_prompt() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        assert!(chat(
            model.to_str().unwrap(), None, Some("test"), 1, 0.0,
            Some("You are an expert coder"), true
        ).is_ok());
    }

    #[test]
    fn test_chat_no_input() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let result = chat(model.to_str().unwrap(), None, None, 1, 0.0, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_chat_invalid_temperature() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let result = chat(model.to_str().unwrap(), None, Some("test"), 1, -1.0, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_chat_model_not_found() {
        assert!(chat("/nonexistent.apr", None, Some("test"), 1, 0.0, None, false).is_err());
    }
}
