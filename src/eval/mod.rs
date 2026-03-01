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

/// Prompt strategy for evaluation (§8.3).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum PromptStrategy {
    Standard,
    SCoT,
    FewShot,
    Cgo,
    Reflexion,
}

impl std::fmt::Display for PromptStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::SCoT => write!(f, "scot"),
            Self::FewShot => write!(f, "few-shot"),
            Self::Cgo => write!(f, "cgo"),
            Self::Reflexion => write!(f, "reflexion"),
        }
    }
}

impl PromptStrategy {
    pub(crate) fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "standard" | "default" => Ok(Self::Standard),
            "scot" | "structured-cot" => Ok(Self::SCoT),
            "few-shot" | "fewshot" => Ok(Self::FewShot),
            "cgo" | "code-gen-opt" => Ok(Self::Cgo),
            "reflexion" | "reflect" => Ok(Self::Reflexion),
            _ => bail!("Unknown prompt strategy: {s}. Use standard, scot, few-shot, cgo, or reflexion"),
        }
    }
}

/// Reranking strategy for N-sampling (§8.2).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum RerankStrategy {
    None,
    LogProb,
    Majority,
}

impl std::fmt::Display for RerankStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::LogProb => write!(f, "logprob"),
            Self::Majority => write!(f, "majority"),
        }
    }
}

impl RerankStrategy {
    pub(crate) fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "logprob" | "log-prob" => Ok(Self::LogProb),
            "majority" | "voting" => Ok(Self::Majority),
            _ => bail!("Unknown rerank strategy: {s}. Use none, logprob, or majority"),
        }
    }
}

/// Evaluation configuration.
#[derive(Debug)]
pub(crate) struct EvalConfig {
    pub prompt_strategy: PromptStrategy,
    pub n_samples: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub rerank: RerankStrategy,
    /// Few-shot exemplars file for FewShot/SCoT strategies.
    pub exemplars: Option<String>,
    /// Custom system prompt override.
    pub system: Option<String>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            prompt_strategy: PromptStrategy::Standard,
            n_samples: 1,
            temperature: 0.0,
            top_p: 0.95,
            rerank: RerankStrategy::None,
            exemplars: None,
            system: None,
        }
    }
}

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
    pub prompt_strategy: String,
    pub n_samples: usize,
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

/// Run evaluation benchmarks against a model with default config.
#[cfg(test)]
pub(crate) fn run(model_path: &str, benchmark: &str, samples: usize, output_dir: &str) -> Result<()> {
    run_with_config(model_path, benchmark, samples, output_dir, &EvalConfig::default())
}

/// Run evaluation with full configuration (prompt strategy, n-samples).
pub(crate) fn run_with_config(
    model_path: &str,
    benchmark: &str,
    samples: usize,
    output_dir: &str,
    config: &EvalConfig,
) -> Result<()> {
    validate_config(config)?;
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
    println!("  Prompt strategy: {}", config.prompt_strategy);
    if config.n_samples > 1 {
        println!("  N-samples: {} (best-of-N selection)", config.n_samples);
    }
    if config.temperature > 0.0 {
        println!("  Temperature: {:.1}", config.temperature);
        println!("  Top-p: {:.2}", config.top_p);
    }
    if !matches!(config.rerank, RerankStrategy::None) {
        println!("  Rerank: {}", config.rerank);
    }
    if let Some(exemplars) = &config.exemplars {
        println!("  Exemplars: {exemplars}");
    }
    if let Some(system) = &config.system {
        println!("  System prompt: {system}");
    }

    // Load the model
    let _model_data = std::fs::read(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model {model_path}: {e}"))?;

    let n_samples = if samples == 0 {
        spec.total_problems
    } else {
        samples.min(spec.total_problems)
    };

    // Run the evaluation harness
    let result = run_benchmark(&spec, model_path, n_samples, config)?;

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
fn run_benchmark(
    spec: &BenchmarkSpec,
    model_path: &str,
    n_samples: usize,
    config: &EvalConfig,
) -> Result<EvalResult> {
    // Scaffold: in production this runs actual inference through each problem,
    // executes generated code in a sandbox, and computes pass@k metrics.
    // For now, we use the pass_at_k estimator with placeholder (n=0, c=0) per problem.

    let now = chrono::Utc::now();

    // Scaffold results: each problem produces (n_completions, correct_count)
    // In production, this comes from actual code execution
    let per_problem: Vec<(usize, usize)> = vec![(config.n_samples, 0); n_samples];
    let p1 = average_pass_at_k(&per_problem, 1);
    let p10 = if spec.compute_pass_at_10 && config.n_samples >= 10 {
        Some(average_pass_at_k(&per_problem, 10))
    } else if spec.compute_pass_at_10 {
        Some(0.0)
    } else {
        None
    };

    Ok(EvalResult {
        model: model_path.to_string(),
        benchmark: spec.name.clone(),
        metric: spec.primary_metric.clone(),
        score: p1,
        samples_evaluated: n_samples,
        samples_total: spec.total_problems,
        timestamp: now.to_rfc3339(),
        prompt_strategy: config.prompt_strategy.to_string(),
        n_samples: config.n_samples,
        details: EvalDetails {
            pass_at_1: p1,
            pass_at_10: p10,
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

fn validate_config(config: &EvalConfig) -> Result<()> {
    if config.temperature < 0.0 {
        bail!("temperature must be >= 0.0, got {}", config.temperature);
    }
    if !(0.0..=1.0).contains(&config.top_p) {
        bail!("top_p must be between 0.0 and 1.0, got {}", config.top_p);
    }
    Ok(())
}

/// Compute the unbiased pass@k estimator (Chen et al., 2021).
///
/// Delegates to `entrenar::eval::pass_at_k` — the sovereign stack implementation
/// of the same formula: pass@k = 1 - C(n-c, k) / C(n, k).
pub(crate) fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    entrenar::eval::pass_at_k(n, c, k)
}

/// Compute pass@k averaged across multiple problems.
///
/// Each entry in `results` is (n, c) for one problem.
pub(crate) fn average_pass_at_k(results: &[(usize, usize)], k: usize) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let sum: f64 = results.iter().map(|&(n, c)| pass_at_k(n, c, k)).sum();
    sum / results.len() as f64
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
mod tests;
