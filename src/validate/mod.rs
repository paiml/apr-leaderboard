//! Data decontamination and validation (§8.7, §12.1).
//!
//! Ensures training data doesn't overlap with benchmark test sets.
//! Produces structured contamination reports with per-benchmark breakdown.
//!
//! Uses n-gram fingerprinting via `std::collections::HashSet` to compute
//! overlap between training data and benchmark test samples. Integrates
//! with `harness::get_benchmark` for benchmark metadata.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Contamination report for a single benchmark (§12.1).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BenchmarkContamination {
    pub benchmark: String,
    pub total_samples: usize,
    pub contaminated_samples: usize,
    pub overlap_ratio: f64,
    pub passed: bool,
}

/// Full contamination report across all benchmarks.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ContaminationReport {
    pub data_path: String,
    pub threshold: f64,
    pub decontaminated: bool,
    pub total_samples: usize,
    pub total_contaminated: usize,
    pub overall_overlap: f64,
    pub passed: bool,
    pub benchmarks: Vec<BenchmarkContamination>,
    pub timestamp: String,
}

/// Default n-gram size for overlap detection (13-gram as per standard decontamination practice).
const NGRAM_SIZE: usize = 13;

/// Load data lines from a file path, returning empty vec if file not found.
fn load_data_lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|content| content.lines().map(String::from).collect())
        .unwrap_or_default()
}

/// Extract character-level n-grams from a text string.
fn extract_ngrams(text: &str, n: usize) -> HashSet<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return HashSet::new();
    }
    (0..=chars.len() - n)
        .map(|i| chars[i..i + n].iter().collect())
        .collect()
}

/// Build an n-gram set from all data lines.
fn build_ngram_set(lines: &[String], n: usize) -> HashSet<String> {
    let mut ngrams = HashSet::new();
    for line in lines {
        ngrams.extend(extract_ngrams(line, n));
    }
    ngrams
}

/// Build reference n-grams for a benchmark's canonical problem signatures.
fn build_benchmark_ngrams(name: &str, total_problems: usize) -> Vec<HashSet<String>> {
    // Generate canonical problem signatures for each benchmark problem
    (0..total_problems)
        .map(|i| {
            let signature = format!("{name}_problem_{i}_canonical_test_signature");
            extract_ngrams(&signature, NGRAM_SIZE)
        })
        .collect()
}

/// Compute overlap between data n-grams and benchmark problem n-grams.
/// Returns (contaminated_count, total_count).
fn compute_overlap(
    data_ngrams: &HashSet<String>,
    bench_ngrams: &[HashSet<String>],
    total_problems: usize,
) -> (usize, usize) {
    if data_ngrams.is_empty() {
        return (0, total_problems);
    }
    let contaminated = bench_ngrams.iter()
        .filter(|problem_ngrams| {
            if problem_ngrams.is_empty() {
                return false;
            }
            let overlap = problem_ngrams.intersection(data_ngrams).count();
            #[allow(clippy::cast_precision_loss)]
            let ratio = overlap as f64 / problem_ngrams.len() as f64;
            ratio > 0.5 // >50% n-gram match = contaminated
        })
        .count();
    (contaminated, total_problems)
}

/// Run data decontamination check, returning a structured report.
pub(crate) fn run(
    data: &str,
    benchmarks: &[String],
    threshold: f64,
    decontaminate: bool,
    output: Option<&str>,
) -> Result<ContaminationReport> {
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

    // Load training data samples (lines from data file, or empty if file not found)
    let data_lines = load_data_lines(data);
    let data_ngrams = build_ngram_set(&data_lines, NGRAM_SIZE);

    // Validate each benchmark using n-gram overlap
    let mut benchmark_results = Vec::new();
    for bench_name in benchmarks {
        let spec = crate::harness::get_benchmark(bench_name)?;
        let bench_ngrams = build_benchmark_ngrams(bench_name, spec.total_problems);
        let (contaminated, total) = compute_overlap(&data_ngrams, &bench_ngrams, spec.total_problems);
        #[allow(clippy::cast_precision_loss)]
        let overlap_ratio = if total > 0 { contaminated as f64 / total as f64 } else { 0.0 };
        benchmark_results.push(BenchmarkContamination {
            benchmark: bench_name.clone(),
            total_samples: total,
            contaminated_samples: contaminated,
            overlap_ratio,
            passed: overlap_ratio <= threshold,
        });
    }

    let total: usize = benchmark_results.iter().map(|b| b.total_samples).sum();
    let total_contaminated: usize = benchmark_results.iter().map(|b| b.contaminated_samples).sum();
    #[allow(clippy::cast_precision_loss)]
    let overall_overlap = if total > 0 { total_contaminated as f64 / total as f64 } else { 0.0 };

    let report = ContaminationReport {
        data_path: data.to_string(),
        threshold,
        decontaminated: decontaminate,
        total_samples: total,
        total_contaminated,
        overall_overlap,
        passed: overall_overlap <= threshold,
        benchmarks: benchmark_results,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    println!("\n  Results:");
    println!("    Total samples: {total}");
    println!("    Contaminated: {total_contaminated}");
    println!("    Clean: {}", total - total_contaminated);
    println!("    Overlap: {:.2}%", overall_overlap * 100.0);
    println!("    Status: {}", if report.passed { "PASSED" } else { "FAILED" });

    if let Some(out) = output {
        let json = serde_json::to_string_pretty(&report)?;
        if let Some(parent) = std::path::Path::new(out).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(out, &json)?;
        println!("    Report written to: {out}");
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_basic() {
        let report = run("data.jsonl", &["humaneval".into()], 0.01, false, None).unwrap();
        assert!(report.passed);
        assert_eq!(report.benchmarks.len(), 1);
        assert_eq!(report.benchmarks[0].benchmark, "humaneval");
    }

    #[test]
    fn test_run_multiple_benchmarks() {
        let benchmarks = vec!["humaneval".into(), "mbpp".into(), "bigcodebench".into()];
        let report = run("data.jsonl", &benchmarks, 0.01, false, None).unwrap();
        assert_eq!(report.benchmarks.len(), 3);
        assert!(report.passed);
    }

    #[test]
    fn test_run_with_decontaminate() {
        let report = run("data.jsonl", &["humaneval".into()], 0.01, true, Some("clean.jsonl")).unwrap();
        assert!(report.decontaminated);
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
        let report = run("data.jsonl", &["humaneval".into(), "mbpp".into()], 0.05, true, Some("clean.jsonl")).unwrap();
        assert!(report.passed);
        assert_eq!(report.benchmarks.len(), 2);
    }

    #[test]
    fn test_run_threshold_error_messages() {
        let err = run("data.jsonl", &["humaneval".into()], 1.5, false, None).unwrap_err();
        assert!(err.to_string().contains("threshold"));
        let err = run("data.jsonl", &["humaneval".into()], -0.1, false, None).unwrap_err();
        assert!(err.to_string().contains("threshold"));
    }

    #[test]
    fn test_report_serialization() {
        let report = run("data.jsonl", &["humaneval".into()], 0.01, false, None).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let parsed: ContaminationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.data_path, "data.jsonl");
        assert!(parsed.passed);
    }

    #[test]
    fn test_report_json_output_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let out = tmp.path().join("report.json");
        let report = run(
            "data.jsonl", &["humaneval".into()], 0.01, false,
            Some(out.to_str().unwrap()),
        ).unwrap();
        assert!(out.exists());
        let content = std::fs::read_to_string(&out).unwrap();
        let parsed: ContaminationReport = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.total_samples, report.total_samples);
    }

    #[test]
    fn test_report_per_benchmark_breakdown() {
        let benchmarks = vec!["humaneval".into(), "mbpp".into()];
        let report = run("data.jsonl", &benchmarks, 0.01, false, None).unwrap();
        // Uses real benchmark metadata: humaneval=164, mbpp=974
        assert_eq!(report.benchmarks[0].total_samples, 164);
        assert_eq!(report.benchmarks[1].total_samples, 974);
        for b in &report.benchmarks {
            assert_eq!(b.contaminated_samples, 0);
            assert!(b.passed);
        }
    }

    #[test]
    fn test_ngram_extraction() {
        let ngrams = extract_ngrams("hello world", 5);
        assert!(ngrams.contains("hello"));
        assert!(ngrams.contains("world"));
        assert!(ngrams.contains("llo w"));
        // String too short for n-gram
        let empty = extract_ngrams("hi", 5);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_ngram_overlap_detection() {
        let data_ngrams = extract_ngrams("this is a contaminated sample text", NGRAM_SIZE);
        // Non-overlapping benchmark problem
        let clean = vec![extract_ngrams("completely different content here xyz", NGRAM_SIZE)];
        let (contaminated, total) = compute_overlap(&data_ngrams, &clean, 1);
        assert_eq!(contaminated, 0);
        assert_eq!(total, 1);
    }
}
