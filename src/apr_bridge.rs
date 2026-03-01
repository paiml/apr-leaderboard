//! Bridge between aprender APR format and entrenar model types.
//!
//! Uses `aprender::serialization::apr::AprReader` for structured access and
//! `AprV2ReaderRef::get_tensor_as_f32()` for dtype-agnostic tensor loading
//! (handles F32, F16, Q8, Q4 dequantization).
//!
//! Implements checkpoint contract behaviors from APR Checkpoint Spec v1.4.0:
//! - F-CKPT-016: Filtered loading — skips `__training__.*` tensors
//! - F-CKPT-013: NaN/Inf validation on tensor load
//! - F-CKPT-009: Atomic writes via tmp+fsync+rename

use anyhow::Result;
use aprender::serialization::apr::AprWriter;
use std::collections::HashMap;

/// Load an APR file into an entrenar merge model (HashMap<String, Tensor>).
///
/// Applies checkpoint contract behaviors:
/// - F-CKPT-016: Skips `__training__.*` tensors (optimizer moments, grad scaler state)
/// - F-CKPT-013: Rejects tensors containing NaN or Inf values
///
/// Uses `AprV2ReaderRef::get_tensor_as_f32()` for dtype-agnostic loading
/// (auto-dequantizes F16, Q8, Q4 tensors to f32).
pub(crate) fn load_apr_as_merge_model(path: &str) -> Result<entrenar::merge::Model> {
    let data =
        std::fs::read(path).map_err(|e| anyhow::anyhow!("Failed to read model {path}: {e}"))?;
    let reader = aprender::format::v2::AprV2ReaderRef::from_bytes(&data)
        .map_err(|e| anyhow::anyhow!("Failed to parse APR model {path}: {e}"))?;

    let mut model = HashMap::new();
    for name in reader.tensor_names() {
        // F-CKPT-016: skip training-only tensors
        if name.starts_with("__training__.") {
            continue;
        }
        let f32_data = reader
            .get_tensor_as_f32(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found or unsupported dtype: {name}"))?;
        // F-CKPT-013: reject NaN/Inf
        for (i, &v) in f32_data.iter().enumerate() {
            if !v.is_finite() {
                anyhow::bail!(
                    "F-CKPT-013: tensor '{name}' contains non-finite value at index {i}: {v}"
                );
            }
        }
        let tensor = entrenar::Tensor::from_vec(f32_data, false);
        model.insert(name.to_string(), tensor);
    }
    Ok(model)
}

/// Write an entrenar merge model to APR format atomically (F-CKPT-009).
///
/// Writes to a `.tmp` file, fsyncs, then renames — a crash at any point
/// leaves the original file intact.
pub(crate) fn save_merge_model_as_apr(
    model: &entrenar::merge::Model,
    output: &str,
) -> Result<()> {
    use std::io::Write;
    use std::path::Path;

    let path = Path::new(output);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = AprWriter::new();
    for (name, tensor) in model {
        let data = tensor.data();
        let shape = vec![data.len()];
        writer.add_tensor_f32(name, shape, data.as_slice().unwrap());
    }
    let bytes = writer
        .to_bytes()
        .map_err(|e| anyhow::anyhow!("Failed to serialize APR model: {e}"))?;

    // F-CKPT-009: atomic write via tmp+fsync+rename
    let tmp_path = path.with_extension("apr.tmp");
    let mut file = std::fs::File::create(&tmp_path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp_path, path)?;

    Ok(())
}

/// Create minimal valid APR bytes with a single tensor (used by test fixtures).
#[cfg(test)]
pub(crate) fn create_minimal_apr_bytes() -> Result<Vec<u8>> {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("weight", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let bytes = writer
        .to_bytes()
        .map_err(|e| anyhow::anyhow!("Failed to create APR bytes: {e}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_save_load() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("test.apr");
        let path_str = path.to_str().unwrap();

        let mut model = HashMap::new();
        model.insert(
            "layer.weight".to_string(),
            entrenar::Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false),
        );
        save_merge_model_as_apr(&model, path_str).unwrap();
        let loaded = load_apr_as_merge_model(path_str).unwrap();

        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("layer.weight"));
        let data = loaded["layer.weight"].data();
        assert_eq!(data.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// F-CKPT-009: Atomic write leaves no .tmp file on success.
    #[test]
    fn test_atomic_write_no_tmp_residue() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("atomic.apr");
        let path_str = path.to_str().unwrap();

        let mut model = HashMap::new();
        model.insert(
            "w".to_string(),
            entrenar::Tensor::from_vec(vec![1.0], false),
        );
        save_merge_model_as_apr(&model, path_str).unwrap();

        assert!(path.exists(), "output file should exist");
        let tmp_path = path.with_extension("apr.tmp");
        assert!(!tmp_path.exists(), "tmp file should be cleaned up after rename");
    }

    /// F-CKPT-013: Reject tensors with NaN values.
    #[test]
    fn test_nan_rejection() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("nan.apr");

        // Write a file with NaN using AprV2Writer directly (bypasses our bridge)
        let metadata = aprender::format::v2::AprV2Metadata::default();
        let mut writer = aprender::format::v2::AprV2Writer::new(metadata);
        writer.add_f32_tensor("bad", vec![3], &[1.0, f32::NAN, 3.0]);
        let bytes = writer.write().unwrap();
        std::fs::write(&path, &bytes).unwrap();

        let result = load_apr_as_merge_model(path.to_str().unwrap());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("F-CKPT-013"), "error should cite contract: {err}");
        assert!(err.contains("non-finite"), "error should say non-finite: {err}");
    }

    /// F-CKPT-013: Reject tensors with Inf values.
    #[test]
    fn test_inf_rejection() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("inf.apr");

        let metadata = aprender::format::v2::AprV2Metadata::default();
        let mut writer = aprender::format::v2::AprV2Writer::new(metadata);
        writer.add_f32_tensor("bad", vec![2], &[f32::INFINITY, 1.0]);
        let bytes = writer.write().unwrap();
        std::fs::write(&path, &bytes).unwrap();

        let result = load_apr_as_merge_model(path.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("F-CKPT-013"));
    }

    /// F-CKPT-016: Training tensors are filtered out during load.
    #[test]
    fn test_training_tensor_filtering() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("ckpt.apr");

        let metadata = aprender::format::v2::AprV2Metadata::default();
        let mut writer = aprender::format::v2::AprV2Writer::new(metadata);
        writer.add_f32_tensor("model.weight", vec![2], &[1.0, 2.0]);
        writer.add_f32_tensor("__training__.optim.exp_avg", vec![2], &[0.1, 0.2]);
        writer.add_f32_tensor("__training__.optim.exp_avg_sq", vec![2], &[0.01, 0.02]);
        let bytes = writer.write().unwrap();
        std::fs::write(&path, &bytes).unwrap();

        let model = load_apr_as_merge_model(path.to_str().unwrap()).unwrap();
        assert_eq!(model.len(), 1, "should only load model tensors");
        assert!(model.contains_key("model.weight"));
        assert!(!model.contains_key("__training__.optim.exp_avg"));
        assert!(!model.contains_key("__training__.optim.exp_avg_sq"));
    }

    #[test]
    fn test_parse_slerp_weight_basic() {
        assert!((parse_slerp_weight(Some("0.6,0.4")) - 0.4).abs() < 1e-6);
        assert!((parse_slerp_weight(Some("0.3, 0.7")) - 0.7).abs() < 1e-6);
        assert!((parse_slerp_weight(None) - 0.5).abs() < 1e-6);
        assert!((parse_slerp_weight(Some("1.0")) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_create_minimal_apr_bytes_roundtrip() {
        let bytes = create_minimal_apr_bytes().unwrap();
        assert!(!bytes.is_empty());
        // Verify it starts with APR magic
        assert_eq!(&bytes[0..3], b"APR");

        // Write to file and load back
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("minimal.apr");
        std::fs::write(&path, &bytes).unwrap();
        let model = load_apr_as_merge_model(path.to_str().unwrap()).unwrap();
        assert_eq!(model.len(), 1);
        assert!(model.contains_key("weight"));
    }
}
