//! Benchmark harness definitions.
//!
//! Each benchmark defines its problem set, evaluation protocol,
//! and metrics (pass@k, execution accuracy, etc.).

use anyhow::{bail, Result};

/// Specification for a benchmark suite.
#[derive(Debug, Clone)]
pub(crate) struct BenchmarkSpec {
    pub name: String,
    pub description: String,
    pub total_problems: usize,
    pub primary_metric: String,
    pub compute_pass_at_10: bool,
    pub languages: Vec<String>,
    #[allow(dead_code)]
    pub source_url: String,
}

/// All supported benchmarks.
fn all_benchmarks() -> Vec<BenchmarkSpec> {
    vec![
        BenchmarkSpec {
            name: "humaneval".into(),
            description: "OpenAI HumanEval - 164 Python programming problems".into(),
            total_problems: 164,
            primary_metric: "pass@1".into(),
            compute_pass_at_10: true,
            languages: vec!["python".into()],
            source_url: "https://github.com/openai/human-eval".into(),
        },
        BenchmarkSpec {
            name: "humaneval-plus".into(),
            description: "EvalPlus HumanEval+ - 164 problems with 80x more tests".into(),
            total_problems: 164,
            primary_metric: "pass@1".into(),
            compute_pass_at_10: true,
            languages: vec!["python".into()],
            source_url: "https://github.com/evalplus/evalplus".into(),
        },
        BenchmarkSpec {
            name: "mbpp".into(),
            description: "Mostly Basic Python Programming - 974 problems".into(),
            total_problems: 974,
            primary_metric: "pass@1".into(),
            compute_pass_at_10: true,
            languages: vec!["python".into()],
            source_url: "https://github.com/google-research/google-research/tree/master/mbpp".into(),
        },
        BenchmarkSpec {
            name: "mbpp-plus".into(),
            description: "EvalPlus MBPP+ - 399 problems with rigorous tests".into(),
            total_problems: 399,
            primary_metric: "pass@1".into(),
            compute_pass_at_10: true,
            languages: vec!["python".into()],
            source_url: "https://github.com/evalplus/evalplus".into(),
        },
        BenchmarkSpec {
            name: "bigcodebench".into(),
            description: "BigCodeBench - 1140 practical programming tasks".into(),
            total_problems: 1140,
            primary_metric: "pass@1".into(),
            compute_pass_at_10: false,
            languages: vec!["python".into()],
            source_url: "https://github.com/bigcode-project/bigcodebench".into(),
        },
        BenchmarkSpec {
            name: "livecodebench".into(),
            description: "LiveCodeBench - contamination-free coding from competitions".into(),
            total_problems: 500,
            primary_metric: "pass@1".into(),
            compute_pass_at_10: false,
            languages: vec!["python".into()],
            source_url: "https://livecodebench.github.io/".into(),
        },
        BenchmarkSpec {
            name: "multipl-e".into(),
            description: "MultiPL-E - HumanEval translated to 18 languages".into(),
            total_problems: 164,
            primary_metric: "pass@1".into(),
            compute_pass_at_10: true,
            languages: vec![
                "python".into(), "javascript".into(), "typescript".into(),
                "java".into(), "cpp".into(), "rust".into(), "go".into(),
            ],
            source_url: "https://github.com/nuprl/MultiPL-E".into(),
        },
        BenchmarkSpec {
            name: "ds-1000".into(),
            description: "DS-1000 - 1000 data science problems (numpy, pandas, etc.)".into(),
            total_problems: 1000,
            primary_metric: "pass@1".into(),
            compute_pass_at_10: false,
            languages: vec!["python".into()],
            source_url: "https://ds1000-code-gen.github.io/".into(),
        },
        BenchmarkSpec {
            name: "swe-bench-lite".into(),
            description: "SWE-bench Lite - 300 real GitHub issue resolution tasks".into(),
            total_problems: 300,
            primary_metric: "resolve_rate".into(),
            compute_pass_at_10: false,
            languages: vec!["python".into()],
            source_url: "https://www.swebench.com/".into(),
        },
        BenchmarkSpec {
            name: "crux-eval".into(),
            description: "CRUXEval - code reasoning, understanding, execution".into(),
            total_problems: 800,
            primary_metric: "pass@1".into(),
            compute_pass_at_10: false,
            languages: vec!["python".into()],
            source_url: "https://crux-eval.github.io/".into(),
        },
    ]
}

/// Get a benchmark by name.
pub(crate) fn get_benchmark(name: &str) -> Result<BenchmarkSpec> {
    let benchmarks = all_benchmarks();
    for b in &benchmarks {
        if b.name == name {
            return Ok(b.clone());
        }
    }

    let available: Vec<&str> = benchmarks.iter().map(|b| b.name.as_str()).collect();
    bail!(
        "Unknown benchmark: {name}\nAvailable: {}",
        available.join(", ")
    )
}

/// List all available benchmarks.
pub(crate) fn list_benchmarks() {
    use comfy_table::{Cell, Table};

    let benchmarks = all_benchmarks();
    let mut table = Table::new();
    table.set_header(vec!["Benchmark", "Problems", "Metric", "Languages", "Description"]);

    for b in &benchmarks {
        table.add_row(vec![
            Cell::new(&b.name),
            Cell::new(b.total_problems),
            Cell::new(&b.primary_metric),
            Cell::new(b.languages.join(", ")),
            Cell::new(&b.description),
        ]);
    }

    println!("{table}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_benchmark_valid() {
        let spec = get_benchmark("humaneval").unwrap();
        assert_eq!(spec.total_problems, 164);
        assert_eq!(spec.primary_metric, "pass@1");
    }

    #[test]
    fn test_get_benchmark_invalid() {
        assert!(get_benchmark("nonexistent").is_err());
    }

    #[test]
    fn test_all_benchmarks_non_empty() {
        let benchmarks = all_benchmarks();
        assert!(benchmarks.len() >= 8);
        for b in &benchmarks {
            assert!(!b.name.is_empty());
            assert!(b.total_problems > 0);
        }
    }
}
