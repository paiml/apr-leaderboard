//! Data decontamination and validation (§8.7).
//!
//! Ensures training data doesn't overlap with benchmark test sets.

use anyhow::{bail, Result};

/// Run data decontamination check.
pub(crate) fn run(
    data: &str,
    benchmarks: &[String],
    threshold: f64,
    decontaminate: bool,
    output: Option<&str>,
) -> Result<()> {
    if benchmarks.is_empty() {
        bail!("at least one benchmark must be specified for validation");
    }
    if !(0.0..=1.0).contains(&threshold) {
        bail!("threshold must be between 0.0 and 1.0, got {threshold}");
    }

    println!("Data validation:");
    println!("  Data: {data}");
    println!("  Benchmarks: {}", benchmarks.join(", "));
    println!("  N-gram overlap threshold: {:.1}%", threshold * 100.0);
    println!("  Decontaminate: {decontaminate}");

    // Scaffold: in production, loads data and benchmark test sets,
    // computes n-gram overlap, and optionally removes contaminated samples.
    println!("  [scaffold] Would run: apr validate --data {data} \\");
    println!("    --benchmarks {} --threshold {threshold}", benchmarks.join(","));
    if decontaminate {
        println!("    --decontaminate");
    }

    // Mock results
    let contaminated = 0_usize;
    let total = 1000_usize;
    println!("\n  Results:");
    println!("    Total samples: {total}");
    println!("    Contaminated: {contaminated}");
    println!("    Clean: {}", total - contaminated);
    #[allow(clippy::cast_precision_loss)]
    let overlap = contaminated as f64 / total as f64 * 100.0;
    println!("    Overlap: {overlap:.2}%");

    if let Some(out) = output {
        println!("    Clean data written to: {out}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_basic() {
        let result = run("data.jsonl", &["humaneval".into()], 0.01, false, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_multiple_benchmarks() {
        let benchmarks = vec!["humaneval".into(), "mbpp".into(), "bigcodebench".into()];
        let result = run("data.jsonl", &benchmarks, 0.01, false, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_decontaminate() {
        let result = run("data.jsonl", &["humaneval".into()], 0.01, true, Some("clean.jsonl"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_empty_benchmarks() {
        let result = run("data.jsonl", &[], 0.01, false, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("benchmark"));
    }

    #[test]
    fn test_run_invalid_threshold() {
        let result = run("data.jsonl", &["humaneval".into()], 1.5, false, None);
        assert!(result.is_err());
        let result = run("data.jsonl", &["humaneval".into()], -0.1, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_threshold_boundaries() {
        assert!(run("data.jsonl", &["humaneval".into()], 0.0, false, None).is_ok());
        assert!(run("data.jsonl", &["humaneval".into()], 1.0, false, None).is_ok());
    }

    #[test]
    fn test_run_decontaminate_with_output() {
        let result = run("data.jsonl", &["humaneval".into(), "mbpp".into()], 0.05, true, Some("clean.jsonl"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_threshold_error_messages() {
        let err = run("data.jsonl", &["humaneval".into()], 1.5, false, None).unwrap_err();
        assert!(err.to_string().contains("threshold"));
        let err = run("data.jsonl", &["humaneval".into()], -0.1, false, None).unwrap_err();
        assert!(err.to_string().contains("threshold"));
    }
}
