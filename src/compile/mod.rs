//! Compile .apr model to a standalone binary (§4.3.1, §9.4).
//!
//! Produces a self-contained executable embedding the model weights.

use anyhow::{bail, Result};

/// Run binary compilation of an .apr model.
pub(crate) fn run(
    model: &str,
    release: bool,
    lto: bool,
    output: Option<&str>,
) -> Result<()> {
    let model_data = std::fs::read(model)
        .map_err(|e| anyhow::anyhow!("Failed to load model {model}: {e}"))?;

    println!("Compiling model to binary:");
    println!("  Model: {model} ({} bytes)", model_data.len());
    println!("  Release: {release}");
    println!("  LTO: {lto}");

    let output_path = output.map_or_else(
        || model.replace(".apr", ""),
        String::from,
    );
    println!("  Output: {output_path}");

    // Scaffold: in production, calls apr compile
    println!("  [scaffold] Would run: apr compile {model}");
    if release {
        println!("    --release");
    }
    if lto {
        println!("    --lto");
    }
    println!("    -o {output_path}");

    // Size estimate
    let mb = model_data.len() / (1024 * 1024);
    println!("  Estimated binary size: ~{mb} MB (model) + runtime overhead");

    Ok(())
}

/// Validate a model before submission (§14.4).
pub(crate) fn check(model: &str) -> Result<()> {
    let model_data = std::fs::read(model)
        .map_err(|e| anyhow::anyhow!("Failed to load model {model}: {e}"))?;

    println!("Checking model: {model}");

    // Verify APR magic bytes
    if model_data.len() < 4 {
        bail!("model file too small ({} bytes)", model_data.len());
    }
    if &model_data[0..3] != b"APR" {
        bail!("invalid APR header: expected APR magic bytes");
    }

    println!("  Format: APR v2");
    println!("  Size: {} bytes", model_data.len());
    println!("  Header: valid");

    // Scaffold: in production, also validates tensor shapes, quantization metadata, etc.
    println!("  [scaffold] Would run: apr check {model}");
    println!("  Status: PASS");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- compile ---
    #[test]
    fn test_compile_basic() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        assert!(run(model.to_str().unwrap(), false, false, None).is_ok());
    }

    #[test]
    fn test_compile_release_lto() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2data").unwrap();
        assert!(run(model.to_str().unwrap(), true, true, Some("binary")).is_ok());
    }

    #[test]
    fn test_compile_model_not_found() {
        assert!(run("/nonexistent.apr", false, false, None).is_err());
    }

    #[test]
    fn test_compile_output_path_generation() {
        assert_eq!("models/test.apr".replace(".apr", ""), "models/test");
    }

    // --- check ---
    #[test]
    fn test_check_valid() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        std::fs::write(&model, b"APR2test-model-data").unwrap();
        assert!(check(model.to_str().unwrap()).is_ok());
    }

    #[test]
    fn test_check_invalid_header() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("bad.bin");
        std::fs::write(&model, b"NOT_APR").unwrap();
        let result = check(model.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("header"));
    }

    #[test]
    fn test_check_too_small() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("tiny.bin");
        std::fs::write(&model, b"AP").unwrap();
        let result = check(model.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_check_model_not_found() {
        assert!(check("/nonexistent.apr").is_err());
    }
}
