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

    println!("Submitting to: {:?}", leaderboard);
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
mod tests {
    use super::*;

    #[test]
    fn test_leaderboard_parsing() {
        assert!(matches!(Leaderboard::from_str("open-llm-leaderboard"), Leaderboard::OpenLlm));
        assert!(matches!(Leaderboard::from_str("bigcode"), Leaderboard::BigCode));
        assert!(matches!(Leaderboard::from_str("evalplus"), Leaderboard::EvalPlus));
        assert!(matches!(Leaderboard::from_str("custom-board"), Leaderboard::Custom(_)));
    }

    #[test]
    fn test_validate_submission() {
        let valid = Submission {
            model_id: "org/model".into(),
            results: serde_json::json!({}),
            model_type: "pretrained".into(),
            precision: "float16".into(),
            weight_type: "Original".into(),
            leaderboard: "test".into(),
            submitted_at: "2026-01-01T00:00:00Z".into(),
        };
        assert!(validate_submission(&valid).is_ok());

        let no_slash = Submission {
            model_id: "model-without-org".into(),
            ..valid.clone()
        };
        assert!(validate_submission(&no_slash).is_err());

        let empty = Submission {
            model_id: String::new(),
            ..valid
        };
        assert!(validate_submission(&empty).is_err());
    }

    // Clone for test helper only
    impl Clone for Submission {
        fn clone(&self) -> Self {
            Self {
                model_id: self.model_id.clone(),
                results: self.results.clone(),
                model_type: self.model_type.clone(),
                precision: self.precision.clone(),
                weight_type: self.weight_type.clone(),
                leaderboard: self.leaderboard.clone(),
                submitted_at: self.submitted_at.clone(),
            }
        }
    }
}
