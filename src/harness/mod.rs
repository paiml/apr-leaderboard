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
            total_problems: 1055,
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
        let result = get_benchmark("nonexistent");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown benchmark"));
        assert!(err.contains("nonexistent"));
        assert!(err.contains("Available:"));
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

    #[test]
    fn test_all_benchmarks_exactly_10() {
        let benchmarks = all_benchmarks();
        assert_eq!(benchmarks.len(), 10);
    }

    #[test]
    fn test_each_benchmark_lookup() {
        let names = [
            "humaneval",
            "humaneval-plus",
            "mbpp",
            "mbpp-plus",
            "bigcodebench",
            "livecodebench",
            "multipl-e",
            "ds-1000",
            "swe-bench-lite",
            "crux-eval",
        ];
        for name in &names {
            let spec = get_benchmark(name).unwrap();
            assert_eq!(spec.name, *name);
        }
    }

    #[test]
    fn test_humaneval_spec() {
        let spec = get_benchmark("humaneval").unwrap();
        assert_eq!(spec.total_problems, 164);
        assert_eq!(spec.primary_metric, "pass@1");
        assert!(spec.compute_pass_at_10);
        assert_eq!(spec.languages, vec!["python"]);
    }

    #[test]
    fn test_humaneval_plus_spec() {
        let spec = get_benchmark("humaneval-plus").unwrap();
        assert_eq!(spec.total_problems, 164);
        assert!(spec.compute_pass_at_10);
    }

    #[test]
    fn test_mbpp_spec() {
        let spec = get_benchmark("mbpp").unwrap();
        assert_eq!(spec.total_problems, 974);
        assert!(spec.compute_pass_at_10);
    }

    #[test]
    fn test_mbpp_plus_spec() {
        let spec = get_benchmark("mbpp-plus").unwrap();
        assert_eq!(spec.total_problems, 399);
    }

    #[test]
    fn test_bigcodebench_spec() {
        let spec = get_benchmark("bigcodebench").unwrap();
        assert_eq!(spec.total_problems, 1140);
        assert!(!spec.compute_pass_at_10);
    }

    #[test]
    fn test_livecodebench_spec() {
        let spec = get_benchmark("livecodebench").unwrap();
        assert_eq!(spec.total_problems, 1055);
        assert!(!spec.compute_pass_at_10);
    }

    #[test]
    fn test_multipl_e_spec() {
        let spec = get_benchmark("multipl-e").unwrap();
        assert_eq!(spec.total_problems, 164);
        assert!(spec.languages.len() >= 7);
        assert!(spec.languages.contains(&"rust".to_string()));
        assert!(spec.languages.contains(&"python".to_string()));
    }

    #[test]
    fn test_ds_1000_spec() {
        let spec = get_benchmark("ds-1000").unwrap();
        assert_eq!(spec.total_problems, 1000);
    }

    #[test]
    fn test_swe_bench_lite_spec() {
        let spec = get_benchmark("swe-bench-lite").unwrap();
        assert_eq!(spec.total_problems, 300);
        assert_eq!(spec.primary_metric, "resolve_rate");
    }

    #[test]
    fn test_crux_eval_spec() {
        let spec = get_benchmark("crux-eval").unwrap();
        assert_eq!(spec.total_problems, 800);
    }

    #[test]
    fn test_benchmark_clone() {
        let spec = get_benchmark("humaneval").unwrap();
        let cloned = spec.clone();
        assert_eq!(spec.name, cloned.name);
        assert_eq!(spec.total_problems, cloned.total_problems);
    }

    #[test]
    fn test_benchmark_debug() {
        let spec = get_benchmark("humaneval").unwrap();
        let debug = format!("{spec:?}");
        assert!(debug.contains("humaneval"));
        assert!(debug.contains("164"));
    }

    #[test]
    fn test_list_benchmarks() {
        // Just verify it doesn't panic
        list_benchmarks();
    }

    #[test]
    fn test_all_benchmarks_have_source_url() {
        for b in &all_benchmarks() {
            assert!(!b.source_url.is_empty(), "{} missing source_url", b.name);
            assert!(
                b.source_url.starts_with("https://"),
                "{} source_url not https",
                b.name
            );
        }
    }

    #[test]
    fn test_all_benchmarks_have_description() {
        for b in &all_benchmarks() {
            assert!(!b.description.is_empty(), "{} missing description", b.name);
        }
    }

    #[test]
    fn test_all_benchmarks_have_languages() {
        for b in &all_benchmarks() {
            assert!(
                !b.languages.is_empty(),
                "{} missing languages",
                b.name
            );
        }
    }
}
