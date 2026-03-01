//! Bridge between aprender APR v2 format and entrenar model types.
//!
//! Provides conversion helpers for loading APR files into entrenar::merge::Model
//! (HashMap<String, Tensor>) and writing them back.

use anyhow::Result;
use std::collections::HashMap;

/// Load an APR v2 file into an entrenar merge model (HashMap<String, Tensor>).
pub(crate) fn load_apr_as_merge_model(path: &str) -> Result<entrenar::merge::Model> {
    let data = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("Failed to read model {path}: {e}"))?;
    let mut cursor = std::io::Cursor::new(&data);
    let reader = aprender::format::v2::AprV2Reader::from_reader(&mut cursor)
        .map_err(|e| anyhow::anyhow!("Failed to parse APR model {path}: {e}"))?;

    let mut model = HashMap::new();
    for name in reader.tensor_names() {
        if let Some(f32_data) = reader.get_tensor_as_f32(name) {
            let tensor = entrenar::Tensor::from_vec(f32_data, false);
            model.insert(name.to_string(), tensor);
        }
    }
    Ok(model)
}

/// Write an entrenar merge model back to APR v2 format.
pub(crate) fn save_merge_model_as_apr(model: &entrenar::merge::Model, output: &str) -> Result<()> {
    if let Some(parent) = std::path::Path::new(output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let metadata = aprender::format::v2::AprV2Metadata::default();
    let mut writer = aprender::format::v2::AprV2Writer::new(metadata);
    for (name, tensor) in model {
        let data = tensor.data();
        let shape = vec![data.len()];
        writer.add_f32_tensor(name, shape, data.as_slice().unwrap());
    }
    let bytes = writer.write()
        .map_err(|e| anyhow::anyhow!("Failed to write APR model: {e}"))?;
    std::fs::write(output, &bytes)?;
    Ok(())
}

/// Write a minimal valid APR v2 file to disk (for scaffold outputs).
pub(crate) fn write_scaffold_apr(output: &str) -> Result<()> {
    if let Some(parent) = std::path::Path::new(output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = create_minimal_apr_bytes()?;
    std::fs::write(output, &bytes)?;
    Ok(())
}

/// Create minimal valid APR v2 bytes with a single tensor.
pub(crate) fn create_minimal_apr_bytes() -> Result<Vec<u8>> {
    let metadata = aprender::format::v2::AprV2Metadata::default();
    let mut writer = aprender::format::v2::AprV2Writer::new(metadata);
    writer.add_f32_tensor("weight", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let bytes = writer.write()
        .map_err(|e| anyhow::anyhow!("Failed to create APR v2 bytes: {e}"))?;
    Ok(bytes)
}

/// Parse SLERP weight from comma-separated weights string.
/// Returns the second weight (interpolation parameter t), defaulting to 0.5.
pub(crate) fn parse_slerp_weight(weights: Option<&str>) -> f32 {
    weights
        .and_then(|w| w.split(',').nth(1))
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(0.5)
}
