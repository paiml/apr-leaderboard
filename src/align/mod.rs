//! Preference optimization: DPO/ORPO alignment.
//!
//! Aligns models to prefer correct, well-structured code
//! over plausible but buggy code (§8.5).

use anyhow::{bail, Result};

/// Alignment method.
#[derive(Debug, Clone, Copy)]
pub(crate) enum AlignMethod {
    Dpo,
    Orpo,
}

impl std::fmt::Display for AlignMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dpo => write!(f, "dpo"),
            Self::Orpo => write!(f, "orpo"),
        }
    }
}

impl AlignMethod {
    pub(crate) fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "dpo" | "direct-preference" => Ok(Self::Dpo),
            "orpo" | "odds-ratio" => Ok(Self::Orpo),
            _ => bail!("Unknown alignment method: {s}. Use dpo or orpo"),
        }
    }
}

/// Run preference alignment on a model.
pub(crate) fn run(
    model: &str,
    data: &str,
    method: &str,
    beta: f64,
    epochs: usize,
    ref_model: Option<&str>,
    output: Option<&str>,
) -> Result<()> {
    let method = AlignMethod::from_str(method)?;

    if beta <= 0.0 {
        bail!("beta must be > 0.0, got {beta}");
    }

    let model_data = std::fs::read(model)
        .map_err(|e| anyhow::anyhow!("Failed to load model {model}: {e}"))?;

    println!("Preference alignment:");
    println!("  Model: {model} ({} bytes)", model_data.len());
    println!("  Method: {method}");
    println!("  Data: {data}");
    println!("  Beta: {beta}");
    println!("  Epochs: {epochs}");
    if let Some(ref_m) = ref_model {
        println!("  Reference model: {ref_m}");
    }

    // DPO requires a reference model; ORPO does not
    if matches!(method, AlignMethod::Dpo) && ref_model.is_none() {
        println!("  NOTE: DPO without --ref-model uses implicit reference (frozen base weights)");
    }

    let output_path = output.map_or_else(
        || model.replace(".apr", "_aligned.apr"),
        String::from,
    );

    // Scaffold: in production, runs DPO/ORPO training via entrenar
    println!("  [scaffold] Would run: apr align {model} --method {method} \\");
    println!("    --data {data} --beta {beta} --epochs {epochs} -o {output_path}");

    for epoch in 1..=epochs {
        println!("  Epoch {epoch}/{epochs}: loss=0.000");
    }

    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output_path, b"APR2scaffold-align")?;
    println!("  Saved aligned model: {output_path}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_method_parsing() {
        assert!(matches!(AlignMethod::from_str("dpo").unwrap(), AlignMethod::Dpo));
        assert!(matches!(AlignMethod::from_str("orpo").unwrap(), AlignMethod::Orpo));
        assert!(matches!(AlignMethod::from_str("direct-preference").unwrap(), AlignMethod::Dpo));
        assert!(matches!(AlignMethod::from_str("odds-ratio").unwrap(), AlignMethod::Orpo));
        assert!(AlignMethod::from_str("invalid").is_err());
    }

    #[test]
    fn test_align_method_display() {
        assert_eq!(AlignMethod::Dpo.to_string(), "dpo");
        assert_eq!(AlignMethod::Orpo.to_string(), "orpo");
    }

    #[test]
    fn test_align_method_roundtrip() {
        for s in &["dpo", "orpo"] {
            let parsed = AlignMethod::from_str(s).unwrap();
            assert_eq!(parsed.to_string(), *s);
        }
    }

    #[test]
    fn test_run_model_not_found() {
        let result = run("/nonexistent.apr", "pairs.jsonl", "dpo", 0.1, 3, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_method() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let result = run(model.to_str().unwrap(), "pairs.jsonl", "invalid", 0.1, 3, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_beta() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let result = run(model.to_str().unwrap(), "pairs.jsonl", "dpo", 0.0, 3, None, None);
        assert!(result.is_err());
        let result = run(model.to_str().unwrap(), "pairs.jsonl", "dpo", -1.0, 3, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_dpo() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let result = run(model.to_str().unwrap(), "pairs.jsonl", "dpo", 0.1, 3, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_orpo() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let result = run(model.to_str().unwrap(), "pairs.jsonl", "orpo", 0.5, 2, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_ref_model() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let result = run(
            model.to_str().unwrap(), "pairs.jsonl", "dpo", 0.1, 3,
            Some("ref.apr"), Some("out.apr"),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_output_path_generation() {
        let input = "models/test.apr";
        assert_eq!(input.replace(".apr", "_aligned.apr"), "models/test_aligned.apr");
    }

    #[test]
    fn test_run_beta_error_message() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let err = run(model.to_str().unwrap(), "pairs.jsonl", "dpo", -0.5, 3, None, None).unwrap_err();
        assert!(err.to_string().contains("beta"));
    }

    #[test]
    fn test_run_with_custom_output() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        let output = tmp.path().join("custom_aligned.apr");
        let result = run(
            model.to_str().unwrap(), "pairs.jsonl", "dpo", 0.1, 2,
            None, Some(output.to_str().unwrap()),
        );
        assert!(result.is_ok());
        assert!(output.exists());
    }
}
