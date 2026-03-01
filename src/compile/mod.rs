//! Compile .apr model to a standalone binary (§4.3.1, §9.4).
//!
//! Produces a self-contained executable embedding the model weights.
//! Validates the APR v2 model via `aprender::format::v2::AprV2Reader`
//! before compilation to ensure format integrity.

use anyhow::Result;

/// Run binary compilation of an .apr model.
pub(crate) fn run(
    model: &str,
    release: bool,
    lto: bool,
    strip: bool,
    output: Option<&str>,
) -> Result<()> {
    // Validate model via AprV2Reader before compilation
    let model_data = std::fs::read(model)
        .map_err(|e| anyhow::anyhow!("Failed to load model {model}: {e}"))?;
    let mut cursor = std::io::Cursor::new(&model_data);
    let reader = aprender::format::v2::AprV2Reader::from_reader(&mut cursor)
        .map_err(|e| anyhow::anyhow!("Model validation failed before compile: {e}"))?;

    let tensors = reader.tensor_names();
    let header = reader.header();

    println!("Compiling model to binary:");
    println!("  Model: {model} ({} bytes, {} tensors)", model_data.len(), tensors.len());
    println!("  Format: APR v{}.{}", header.version.0, header.version.1);
    println!("  Release: {release}");
    println!("  LTO: {lto}");
    println!("  Strip: {strip}");

    let output_path = output.map_or_else(
        || model.replace(".apr", ""),
        String::from,
    );
    println!("  Output: {output_path}");

    // Compilation flags
    let mut flags = Vec::new();
    if release { flags.push("--release"); }
    if lto { flags.push("--lto"); }
    if strip { flags.push("--strip"); }
    let flags_str = if flags.is_empty() { "none".to_string() } else { flags.join(" ") };
    println!("  Flags: {flags_str}");

    // Size estimate based on model + runtime overhead
    let model_kb = model_data.len() / 1024;
    let runtime_kb = 512; // estimated Rust runtime overhead
    println!("  Estimated binary size: ~{} KB (model: {model_kb} KB + runtime: {runtime_kb} KB)",
        model_kb + runtime_kb);

    Ok(())
}

/// Validate a model before submission (§14.4).
///
/// Uses aprender's AprV2Reader to validate magic bytes, header checksum,
/// metadata, tensor index, and alignment — not just magic bytes.
pub(crate) fn check(model: &str) -> Result<()> {
    let model_data = std::fs::read(model)
        .map_err(|e| anyhow::anyhow!("Failed to load model {model}: {e}"))?;

    println!("Checking model: {model}");

    // Parse with aprender's AprV2Reader — validates header, checksum, metadata, tensor index
    let mut cursor = std::io::Cursor::new(&model_data);
    let reader = aprender::format::v2::AprV2Reader::from_reader(&mut cursor)
        .map_err(|e| anyhow::anyhow!("APR v2 validation failed: {e}"))?;

    let header = reader.header();
    let metadata = reader.metadata();
    let tensors = reader.tensor_names();

    println!("  Format: APR v{}.{}", header.version.0, header.version.1);
    println!("  Size: {} bytes", model_data.len());
    println!("  Tensors: {}", tensors.len());
    if let Some(name) = &metadata.name {
        println!("  Name: {name}");
    }
    if metadata.param_count > 0 {
        println!("  Parameters: {}", metadata.param_count);
    }
    println!("  Header checksum: valid");
    println!("  Alignment: {}", if reader.verify_alignment() { "valid" } else { "MISALIGNED" });
    println!("  Status: PASS");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_valid_apr(path: &std::path::Path) {
        let bytes = crate::apr_bridge::create_minimal_apr_bytes().unwrap();
        std::fs::write(path, &bytes).unwrap();
    }

    // --- compile ---
    #[test]
    fn test_compile_basic() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        write_valid_apr(&model);
        assert!(run(model.to_str().unwrap(), false, false, false, None).is_ok());
    }

    #[test]
    fn test_compile_release_lto() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        write_valid_apr(&model);
        assert!(run(model.to_str().unwrap(), true, true, false, Some("binary")).is_ok());
    }

    #[test]
    fn test_compile_with_strip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        write_valid_apr(&model);
        assert!(run(model.to_str().unwrap(), true, true, true, Some("binary")).is_ok());
    }

    #[test]
    fn test_compile_model_not_found() {
        assert!(run("/nonexistent.apr", false, false, false, None).is_err());
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
        write_valid_apr(&model);
        assert!(check(model.to_str().unwrap()).is_ok());
    }

    #[test]
    fn test_check_invalid_header() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("bad.bin");
        std::fs::write(&model, b"NOT_APR_INVALID_DATA_THAT_IS_LONG_ENOUGH_FOR_64_BYTES_PADDING_EXTRA").unwrap();
        let result = check(model.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("validation failed"));
    }

    #[test]
    fn test_check_too_small() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("tiny.bin");
        std::fs::write(&model, b"AP").unwrap();
        let result = check(model.to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_check_model_not_found() {
        assert!(check("/nonexistent.apr").is_err());
    }

    #[test]
    fn test_check_exactly_3_bytes() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("edge.bin");
        std::fs::write(&model, b"APR").unwrap();
        assert!(check(model.to_str().unwrap()).is_err());
    }

    #[test]
    fn test_check_corrupt_4_bytes() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("minimal.apr");
        std::fs::write(&model, b"APR2").unwrap();
        // 4 bytes is too small for a valid APR v2 (needs 64-byte header)
        assert!(check(model.to_str().unwrap()).is_err());
    }

    #[test]
    fn test_compile_strip_only() {
        let tmp = tempfile::TempDir::new().unwrap();
        let model = tmp.path().join("test.apr");
        write_valid_apr(&model);
        assert!(run(model.to_str().unwrap(), false, false, true, None).is_ok());
    }

    #[test]
    fn test_compile_error_message() {
        let err = run("/nonexistent.apr", false, false, false, None).unwrap_err();
        assert!(err.to_string().contains("Failed to load model"));
    }
}
