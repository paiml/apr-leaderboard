//! Evaluation harness for running benchmarks against .apr models.
//!
//! Supports standard coding benchmarks:
//! - HumanEval (164 problems)
//! - MBPP (974 problems)
//! - LiveCodeBench (rolling updates)
//! - BigCodeBench (1140 problems)
//! - MultiPL-E (multi-language HumanEval)
//! - DS-1000 (data science)

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::harness::{self, BenchmarkSpec};

/// Result of a single benchmark evaluation.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct EvalResult {
    pub model: String,
    pub benchmark: String,
    pub metric: String,
    pub score: f64,
    pub samples_evaluated: usize,
    pub samples_total: usize,
    pub timestamp: String,
    pub details: EvalDetails,
}

/// Detailed breakdown of evaluation results.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct EvalDetails {
    pub pass_at_1: f64,
    pub pass_at_10: Option<f64>,
    pub pass_at_100: Option<f64>,
    pub avg_tokens_generated: f64,
    pub avg_latency_ms: f64,
    pub category_scores: Vec<CategoryScore>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CategoryScore {
    pub category: String,
    pub score: f64,
    pub count: usize,
}

/// Run evaluation benchmarks against a model.
pub(crate) fn run(model_path: &str, benchmark: &str, samples: usize, output_dir: &str) -> Result<()> {
    let spec = harness::get_benchmark(benchmark)?;

    println!("Evaluating: {model_path}");
    println!("  Benchmark: {} ({})", spec.name, spec.description);
    println!(
        "  Samples: {}",
        if samples == 0 {
            format!("all ({})", spec.total_problems)
        } else {
            format!("{samples} of {}", spec.total_problems)
        }
    );

    // Load the model
    let _model_data = std::fs::read(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model {model_path}: {e}"))?;

    let n_samples = if samples == 0 {
        spec.total_problems
    } else {
        samples.min(spec.total_problems)
    };

    // Run the evaluation harness
    let result = run_benchmark(&spec, model_path, n_samples)?;

    // Write results
    std::fs::create_dir_all(output_dir)?;
    let result_path = format!(
        "{}/{}_{}.json",
        output_dir,
        benchmark,
        chrono::Utc::now().format("%Y%m%d_%H%M%S")
    );
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&result_path, &json)?;

    // Print summary
    print_summary(&result);
    println!("\n  Results saved to: {result_path}");

    Ok(())
}

/// Run a specific benchmark suite.
fn run_benchmark(spec: &BenchmarkSpec, model_path: &str, n_samples: usize) -> Result<EvalResult> {
    // Scaffold: in production this runs actual inference through each problem,
    // executes generated code in a sandbox, and computes pass@k metrics.

    let now = chrono::Utc::now();

    Ok(EvalResult {
        model: model_path.to_string(),
        benchmark: spec.name.clone(),
        metric: spec.primary_metric.clone(),
        score: 0.0, // Placeholder — real eval fills this
        samples_evaluated: n_samples,
        samples_total: spec.total_problems,
        timestamp: now.to_rfc3339(),
        details: EvalDetails {
            pass_at_1: 0.0,
            pass_at_10: if spec.compute_pass_at_10 {
                Some(0.0)
            } else {
                None
            },
            pass_at_100: None,
            avg_tokens_generated: 0.0,
            avg_latency_ms: 0.0,
            category_scores: Vec::new(),
        },
    })
}

fn print_summary(result: &EvalResult) {
    use comfy_table::{Cell, Table};

    let mut table = Table::new();
    table.set_header(vec!["Metric", "Value"]);
    table.add_row(vec![
        Cell::new("Benchmark"),
        Cell::new(&result.benchmark),
    ]);
    table.add_row(vec![
        Cell::new(&result.metric),
        Cell::new(format!("{:.2}%", result.details.pass_at_1 * 100.0)),
    ]);
    if let Some(p10) = result.details.pass_at_10 {
        table.add_row(vec![
            Cell::new("pass@10"),
            Cell::new(format!("{:.2}%", p10 * 100.0)),
        ]);
    }
    table.add_row(vec![
        Cell::new("Samples"),
        Cell::new(format!(
            "{}/{}",
            result.samples_evaluated, result.samples_total
        )),
    ]);
    table.add_row(vec![
        Cell::new("Avg Latency"),
        Cell::new(format!("{:.1}ms", result.details.avg_latency_ms)),
    ]);

    println!("\n{table}");
}

/// Show evaluation history.
pub(crate) fn show_history(model_filter: Option<&str>) -> Result<()> {
    let results_dir = "results/";
    if !std::path::Path::new(results_dir).exists() {
        bail!("No results directory found. Run `eval` first.");
    }

    let mut results: Vec<EvalResult> = Vec::new();
    for entry in std::fs::read_dir(results_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            let content = std::fs::read_to_string(&path)?;
            if let Ok(result) = serde_json::from_str::<EvalResult>(&content) {
                if let Some(filter) = model_filter {
                    if !result.model.contains(filter) {
                        continue;
                    }
                }
                results.push(result);
            }
        }
    }

    results.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    use comfy_table::{Cell, Table};
    let mut table = Table::new();
    table.set_header(vec!["Timestamp", "Model", "Benchmark", "Score"]);

    for r in &results {
        table.add_row(vec![
            Cell::new(&r.timestamp[..19]),
            Cell::new(&r.model),
            Cell::new(&r.benchmark),
            Cell::new(format!("{:.2}%", r.details.pass_at_1 * 100.0)),
        ]);
    }

    println!("{table}");
    println!("\nTotal: {} evaluation(s)", results.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_result_serialization() {
        let result = EvalResult {
            model: "test.apr".into(),
            benchmark: "humaneval".into(),
            metric: "pass@1".into(),
            score: 0.75,
            samples_evaluated: 164,
            samples_total: 164,
            timestamp: "2026-01-01T00:00:00Z".into(),
            details: EvalDetails {
                pass_at_1: 0.75,
                pass_at_10: Some(0.85),
                pass_at_100: None,
                avg_tokens_generated: 150.0,
                avg_latency_ms: 42.5,
                category_scores: vec![CategoryScore {
                    category: "arrays".into(),
                    score: 0.8,
                    count: 20,
                }],
            },
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: EvalResult = serde_json::from_str(&json).unwrap();
        assert!((parsed.details.pass_at_1 - 0.75).abs() < f64::EPSILON);
    }
}
