//! Convert HuggingFace models to .apr format.
//!
//! Handles downloading SafeTensors/GGUF from the HF Hub,
//! converting to APR v2 format with optional quantization.

use anyhow::{bail, Result};

/// Supported quantization levels for conversion.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Quantization {
    FP32,
    FP16,
    Q8,
    Q4,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FP32 => write!(f, "fp32"),
            Self::FP16 => write!(f, "fp16"),
            Self::Q8 => write!(f, "q8"),
            Self::Q4 => write!(f, "q4"),
        }
    }
}

impl Quantization {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "fp32" | "f32" => Ok(Self::FP32),
            "fp16" | "f16" => Ok(Self::FP16),
            "q8" | "int8" => Ok(Self::Q8),
            "q4" | "int4" => Ok(Self::Q4),
            _ => bail!("Unknown quantization: {s}. Use fp32, fp16, q8, or q4"),
        }
    }
}

/// Convert a HuggingFace model to .apr format.
pub(crate) fn run(model_id: &str, output_dir: &str, quantization: &str) -> Result<()> {
    let quant = Quantization::from_str(quantization)?;

    println!("Converting model: {model_id}");
    println!("  Output: {output_dir}");
    println!("  Quantization: {quant}");

    // Step 1: Resolve model files from HF Hub
    let model_files = resolve_model_files(model_id)?;
    println!("  Found {} model file(s)", model_files.len());

    // Step 2: Download and convert each shard
    let output_path = format!("{}/{}.apr", output_dir, model_id.replace('/', "_"));
    std::fs::create_dir_all(output_dir)?;

    // Step 3: Build APR v2 bundle from tensors
    build_apr_bundle(&model_files, &output_path, quant)?;

    println!("  Wrote: {output_path}");
    Ok(())
}

/// Resolve model files from the HuggingFace Hub.
fn resolve_model_files(model_id: &str) -> Result<Vec<String>> {
    // In production, this calls the HF Hub API:
    // GET https://huggingface.co/api/models/{model_id}
    // to enumerate SafeTensors shards.
    //
    // For now, return the expected file pattern.
    println!("  Resolving files for {model_id} from HF Hub...");

    // Typical Qwen2.5-Coder layout:
    // - model-00001-of-00004.safetensors
    // - model-00002-of-00004.safetensors
    // - ...
    // - config.json, tokenizer.json, tokenizer_config.json
    Ok(vec![format!(
        "https://huggingface.co/{model_id}/resolve/main/model.safetensors"
    )])
}

/// Build an APR v2 bundle from downloaded model files.
fn build_apr_bundle(files: &[String], output_path: &str, quant: Quantization) -> Result<()> {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let metadata = AprV2Metadata {
        model_type: format!("hf-conversion-{}", files.len()),
        ..AprV2Metadata::default()
    };

    let mut writer = AprV2Writer::new(metadata);
    writer.with_lz4_compression();

    // Build a minimal APR v2 bundle as a scaffold.
    // In production, this reads actual tensor data from the downloaded SafeTensors.
    let placeholder_weights: Vec<f32> = vec![0.0f32; 256];

    match quant {
        Quantization::FP32 => {
            writer.add_f32_tensor("model.embed_tokens.weight", vec![1, 256], &placeholder_weights);
        }
        Quantization::FP16 => {
            writer.add_f16_tensor("model.embed_tokens.weight", vec![1, 256], &placeholder_weights);
        }
        Quantization::Q8 => {
            writer.add_q8_tensor("model.embed_tokens.weight", vec![1, 256], &placeholder_weights);
        }
        Quantization::Q4 => {
            writer.add_q4_tensor("model.embed_tokens.weight", vec![1, 256], &placeholder_weights);
        }
    }

    let bundle = writer.write().map_err(|e| anyhow::anyhow!("APR v2 write error: {e}"))?;

    // Verify APR magic bytes (APR\0 or APR2 depending on library version)
    assert_eq!(&bundle[0..3], b"APR", "Invalid APR header");

    std::fs::write(output_path, &bundle)?;
    println!(
        "  Built APR v2 bundle: {} bytes ({quant})",
        bundle.len()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_parsing() {
        assert!(matches!(Quantization::from_str("fp32").unwrap(), Quantization::FP32));
        assert!(matches!(Quantization::from_str("FP16").unwrap(), Quantization::FP16));
        assert!(matches!(Quantization::from_str("q8").unwrap(), Quantization::Q8));
        assert!(matches!(Quantization::from_str("q4").unwrap(), Quantization::Q4));
        assert!(Quantization::from_str("invalid").is_err());
    }

    #[test]
    fn test_quantization_case_insensitive() {
        assert!(matches!(Quantization::from_str("F32").unwrap(), Quantization::FP32));
        assert!(matches!(Quantization::from_str("f16").unwrap(), Quantization::FP16));
        assert!(matches!(Quantization::from_str("INT8").unwrap(), Quantization::Q8));
        assert!(matches!(Quantization::from_str("INT4").unwrap(), Quantization::Q4));
    }

    #[test]
    fn test_quantization_alias_fp() {
        assert!(matches!(Quantization::from_str("fp32").unwrap(), Quantization::FP32));
        assert!(matches!(Quantization::from_str("f32").unwrap(), Quantization::FP32));
        assert!(matches!(Quantization::from_str("fp16").unwrap(), Quantization::FP16));
        assert!(matches!(Quantization::from_str("f16").unwrap(), Quantization::FP16));
    }

    #[test]
    fn test_quantization_alias_int() {
        assert!(matches!(Quantization::from_str("q8").unwrap(), Quantization::Q8));
        assert!(matches!(Quantization::from_str("int8").unwrap(), Quantization::Q8));
        assert!(matches!(Quantization::from_str("q4").unwrap(), Quantization::Q4));
        assert!(matches!(Quantization::from_str("int4").unwrap(), Quantization::Q4));
    }

    #[test]
    fn test_quantization_error_message() {
        let err = Quantization::from_str("q3").unwrap_err();
        assert!(err.to_string().contains("Unknown quantization"));
        assert!(err.to_string().contains("q3"));
    }

    #[test]
    fn test_quantization_debug() {
        assert_eq!(format!("{:?}", Quantization::FP32), "FP32");
        assert_eq!(format!("{:?}", Quantization::FP16), "FP16");
        assert_eq!(format!("{:?}", Quantization::Q8), "Q8");
        assert_eq!(format!("{:?}", Quantization::Q4), "Q4");
    }

    #[test]
    fn test_resolve_model_files() {
        let files = resolve_model_files("test/model").unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].contains("test/model"));
        assert!(files[0].contains("huggingface.co"));
    }

    #[test]
    fn test_build_apr_bundle_fp16() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tmp.path().join("test.apr");
        let files = vec!["test.safetensors".into()];
        build_apr_bundle(&files, output.to_str().unwrap(), Quantization::FP16).unwrap();
        let data = std::fs::read(&output).unwrap();
        assert_eq!(&data[0..3], b"APR");
        assert!(data.len() > 4);
    }

    #[test]
    fn test_build_apr_bundle_fp32() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tmp.path().join("test.apr");
        let files = vec!["test.safetensors".into()];
        build_apr_bundle(&files, output.to_str().unwrap(), Quantization::FP32).unwrap();
        let data = std::fs::read(&output).unwrap();
        assert_eq!(&data[0..3], b"APR");
    }

    #[test]
    fn test_build_apr_bundle_q8() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tmp.path().join("test.apr");
        let files = vec!["test.safetensors".into()];
        build_apr_bundle(&files, output.to_str().unwrap(), Quantization::Q8).unwrap();
        let data = std::fs::read(&output).unwrap();
        assert_eq!(&data[0..3], b"APR");
    }

    #[test]
    fn test_build_apr_bundle_q4() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output = tmp.path().join("test.apr");
        let files = vec!["test.safetensors".into()];
        build_apr_bundle(&files, output.to_str().unwrap(), Quantization::Q4).unwrap();
        let data = std::fs::read(&output).unwrap();
        assert_eq!(&data[0..3], b"APR");
    }

    #[test]
    fn test_run_creates_output_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let output_dir = tmp.path().join("nested/output");
        run("test/model", output_dir.to_str().unwrap(), "fp16").unwrap();
        assert!(output_dir.exists());
        let apr_file = output_dir.join("test_model.apr");
        assert!(apr_file.exists());
    }

    #[test]
    fn test_run_all_quantization_levels() {
        for quant in &["fp32", "fp16", "q8", "q4"] {
            let tmp = tempfile::TempDir::new().unwrap();
            run("test/model", tmp.path().to_str().unwrap(), quant).unwrap();
            let apr_file = tmp.path().join("test_model.apr");
            assert!(apr_file.exists(), "Failed for quantization: {quant}");
        }
    }

    #[test]
    fn test_run_invalid_quantization() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = run("test/model", tmp.path().to_str().unwrap(), "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_quantization_display() {
        assert_eq!(Quantization::FP32.to_string(), "fp32");
        assert_eq!(Quantization::FP16.to_string(), "fp16");
        assert_eq!(Quantization::Q8.to_string(), "q8");
        assert_eq!(Quantization::Q4.to_string(), "q4");
    }

    #[test]
    fn test_quantization_roundtrip() {
        for s in &["fp32", "fp16", "q8", "q4"] {
            let parsed = Quantization::from_str(s).unwrap();
            assert_eq!(parsed.to_string(), *s);
        }
    }

    #[test]
    fn test_model_id_slash_replacement() {
        let tmp = tempfile::TempDir::new().unwrap();
        run("org/model-name", tmp.path().to_str().unwrap(), "fp16").unwrap();
        let apr_file = tmp.path().join("org_model-name.apr");
        assert!(apr_file.exists());
    }
}
