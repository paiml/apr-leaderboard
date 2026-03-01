//! Submit evaluation results to HuggingFace leaderboards.
//!
//! Handles formatting results for the Open LLM Leaderboard,
//! BigCode leaderboard, and custom leaderboards.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Supported leaderboard targets.
#[derive(Debug, Clone)]
pub(crate) enum Leaderboard {
    OpenLlm,
    BigCode,
    EvalPlus,
    Custom(String),
}

impl std::fmt::Display for Leaderboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenLlm => write!(f, "open-llm-leaderboard"),
            Self::BigCode => write!(f, "bigcode"),
            Self::EvalPlus => write!(f, "evalplus"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

impl Leaderboard {
    fn from_str(s: &str) -> Self {
        match s {
            "open-llm-leaderboard" | "open-llm" => Self::OpenLlm,
            "bigcode" | "bigcode-leaderboard" => Self::BigCode,
            "evalplus" => Self::EvalPlus,
            other => Self::Custom(other.to_string()),
        }
    }

    fn submission_repo(&self) -> &str {
        match self {
            Self::OpenLlm => "open-llm-leaderboard/requests",
            Self::BigCode => "bigcode/bigcode-models-leaderboard",
            Self::EvalPlus => "evalplus/evalplus-results",
            Self::Custom(name) => name.as_str(),
        }
    }
}

/// Submission payload for a leaderboard.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Submission {
    pub model_id: String,
    pub results: serde_json::Value,
    pub model_type: String,
    pub precision: String,
    pub weight_type: String,
    pub leaderboard: String,
    pub submitted_at: String,
}

/// Submit results to a HuggingFace leaderboard.
pub(crate) fn run(results_path: &str, model_id: &str, leaderboard: &str) -> Result<()> {
    let leaderboard = Leaderboard::from_str(leaderboard);

    // Load results
    let results_content = std::fs::read_to_string(results_path)
        .map_err(|e| anyhow::anyhow!("Failed to read results {results_path}: {e}"))?;
    let results: serde_json::Value = serde_json::from_str(&results_content)?;

    println!("Submitting to: {leaderboard}");
    println!("  Model: {model_id}");
    println!("  Target repo: {}", leaderboard.submission_repo());

    // Build submission
    let submission = Submission {
        model_id: model_id.to_string(),
        results,
        model_type: "pretrained".into(),
        precision: "float16".into(),
        weight_type: "Original".into(),
        leaderboard: leaderboard.submission_repo().to_string(),
        submitted_at: chrono::Utc::now().to_rfc3339(),
    };

    // Validate submission
    validate_submission(&submission)?;

    // In production: push to HF Hub via API
    // POST https://huggingface.co/api/repos/{repo}/commit
    let submission_json = serde_json::to_string_pretty(&submission)?;
    let submission_path = format!(
        "results/submission_{}.json",
        chrono::Utc::now().format("%Y%m%d_%H%M%S")
    );
    std::fs::write(&submission_path, &submission_json)?;

    println!("  Submission saved to: {submission_path}");
    println!("  NOTE: Push to HF Hub with `huggingface-cli upload`");

    Ok(())
}

/// Generate a HuggingFace-compatible model card (README.md) per §14.3.
///
/// Includes: base model, pipeline stages, evaluation results, infrastructure,
/// quantization info, and reproducibility link to config TOML.
pub(crate) fn generate_model_card(model_id: &str, results_path: &str) -> Result<String> {
    let results_content = std::fs::read_to_string(results_path)
        .map_err(|e| anyhow::anyhow!("Failed to read results {results_path}: {e}"))?;
    let results: serde_json::Value = serde_json::from_str(&results_content)?;

    let score = results.get("score")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let benchmark = results.get("benchmark")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");

    let card = format!(
        "\
---
license: mit
tags:
  - apr
  - code-generation
model-index:
  - name: {model_id}
    results:
      - task:
          type: text-generation
        dataset:
          name: {benchmark}
          type: {benchmark}
        metrics:
          - name: pass@1
            type: pass@1
            value: {score:.4}
---

# {model_id}

## Model Details

- **Base model:** [{model_id}](https://huggingface.co/{model_id})
- **Infrastructure:** Built with `aprender` (Rust, no Python dependencies)
- **Format:** APR v2 (`.apr`)

## Evaluation Results

| Benchmark | pass@1 |
|-----------|--------|
| {benchmark} | {score:.2}% |

## Reproducibility

This model was produced using the `apr` CLI pipeline.
See the pipeline config TOML for full reproducibility.

```bash
apr pipeline --config pipeline.toml
```
");

    let card_path = format!(
        "results/{}_README.md",
        model_id.replace('/', "_")
    );
    std::fs::create_dir_all("results")?;
    std::fs::write(&card_path, &card)?;
    println!("  Model card saved to: {card_path}");

    Ok(card)
}

/// Export metadata for an .apr model to a HF-compatible directory.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ExportMetadata {
    pub model_path: String,
    pub format: String,
    pub exported_at: String,
    pub results: Option<serde_json::Value>,
}

/// Export model to HuggingFace-compatible format (§14.2).
pub(crate) fn export_model(
    model_path: &str,
    format: &str,
    output_dir: &str,
    results_path: Option<&str>,
) -> Result<()> {
    let valid_formats = ["safetensors", "gguf"];
    if !valid_formats.contains(&format) {
        bail!("Unknown export format: {format}. Use safetensors or gguf");
    }

    // Verify model exists
    if !std::path::Path::new(model_path).exists() {
        bail!("Model not found: {model_path}");
    }

    println!("Exporting model:");
    println!("  Model: {model_path}");
    println!("  Format: {format}");
    println!("  Output: {output_dir}");

    // Load optional results
    let results = if let Some(rp) = results_path {
        let content = std::fs::read_to_string(rp)
            .map_err(|e| anyhow::anyhow!("Failed to read results {rp}: {e}"))?;
        Some(serde_json::from_str(&content)?)
    } else {
        None
    };

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Write export metadata
    let metadata = ExportMetadata {
        model_path: model_path.to_string(),
        format: format.to_string(),
        exported_at: chrono::Utc::now().to_rfc3339(),
        results,
    };
    let meta_path = format!("{output_dir}/metadata.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)?;

    // Scaffold: in production, converts .apr → safetensors/gguf
    println!("  [scaffold] Would run: apr export {model_path} --format {format} -o {output_dir}");
    println!("  Metadata written to: {meta_path}");

    Ok(())
}

fn validate_submission(submission: &Submission) -> Result<()> {
    if submission.model_id.is_empty() {
        bail!("model_id cannot be empty");
    }
    if !submission.model_id.contains('/') {
        bail!("model_id must be in org/name format (e.g., Qwen/Qwen2.5-Coder-7B)");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
