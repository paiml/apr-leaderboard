use super::*;
use clap::Parser;

#[test]
fn test_cli_parse_benchmarks() {
    let cli = Cli::try_parse_from(["apr-leaderboard", "benchmarks"]).unwrap();
    assert!(matches!(cli.command, Commands::Benchmarks));
}

#[test]
fn test_cli_parse_convert() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "convert", "--model-id", "Qwen/Qwen2.5-Coder-7B",
    ]).unwrap();
    match cli.command {
        Commands::Convert { model_id, output, quantization } => {
            assert_eq!(model_id, "Qwen/Qwen2.5-Coder-7B");
            assert_eq!(output, "models/");
            assert_eq!(quantization, "fp16");
        }
        _ => panic!("Expected Convert"),
    }
}

#[test]
fn test_cli_parse_eval() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "eval", "--model", "m.apr", "--benchmark", "humaneval",
        "--samples", "10", "--prompt-strategy", "scot", "--n-samples", "20",
    ]).unwrap();
    match cli.command {
        Commands::Eval { model, benchmark, samples, prompt_strategy, n_samples, .. } => {
            assert_eq!(model, "m.apr");
            assert_eq!(benchmark, "humaneval");
            assert_eq!(samples, 10);
            assert_eq!(prompt_strategy, "scot");
            assert_eq!(n_samples, 20);
        }
        _ => panic!("Expected Eval"),
    }
}

#[test]
fn test_cli_eval_defaults() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "eval", "--model", "m.apr", "--benchmark", "humaneval",
    ]).unwrap();
    match cli.command {
        Commands::Eval { samples, prompt_strategy, n_samples, temperature, top_p, rerank, .. } => {
            assert_eq!(samples, 0);
            assert_eq!(prompt_strategy, "standard");
            assert_eq!(n_samples, 1);
            assert!((temperature - 0.0).abs() < f64::EPSILON);
            assert!((top_p - 0.95).abs() < f64::EPSILON);
            assert_eq!(rerank, "none");
        }
        _ => panic!("Expected Eval"),
    }
}

#[test]
fn test_cli_eval_sampling_flags() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "eval", "--model", "m.apr", "--benchmark", "humaneval",
        "--temperature", "0.6", "--top-p", "0.9", "--n-samples", "20",
        "--rerank", "logprob",
    ]).unwrap();
    match cli.command {
        Commands::Eval { temperature, top_p, n_samples, rerank, .. } => {
            assert!((temperature - 0.6).abs() < f64::EPSILON);
            assert!((top_p - 0.9).abs() < f64::EPSILON);
            assert_eq!(n_samples, 20);
            assert_eq!(rerank, "logprob");
        }
        _ => panic!("Expected Eval"),
    }
}

#[test]
fn test_cli_parse_finetune() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "finetune", "--model", "m.apr", "--dataset", "d.jsonl",
        "--method", "qlora", "--rank", "32", "--lr", "0.001", "--epochs", "5",
        "-o", "out.apr",
    ]).unwrap();
    match cli.command {
        Commands::Finetune { model, dataset, method, rank, lr, epochs, output } => {
            assert_eq!(model, "m.apr");
            assert_eq!(dataset, "d.jsonl");
            assert_eq!(method, "qlora");
            assert_eq!(rank, 32);
            assert!((lr - 0.001).abs() < f64::EPSILON);
            assert_eq!(epochs, 5);
            assert_eq!(output, Some("out.apr".into()));
        }
        _ => panic!("Expected Finetune"),
    }
}

#[test]
fn test_cli_parse_distill() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "distill", "--teacher", "t.apr", "--student", "s.apr",
        "--strategy", "progressive", "--epochs", "10", "--data", "corpus.jsonl",
        "-o", "out.apr",
    ]).unwrap();
    match cli.command {
        Commands::Distill { teacher, student, strategy, epochs, data, output, .. } => {
            assert_eq!(teacher, "t.apr");
            assert_eq!(student, "s.apr");
            assert_eq!(strategy, "progressive");
            assert_eq!(epochs, 10);
            assert_eq!(data, Some("corpus.jsonl".into()));
            assert_eq!(output, "out.apr");
        }
        _ => panic!("Expected Distill"),
    }
}

#[test]
fn test_cli_parse_merge() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "merge", "a.apr", "b.apr", "--strategy", "ties",
        "--weights", "0.7,0.3", "--base-model", "base.apr",
        "--density", "0.2", "-o", "out.apr",
    ]).unwrap();
    match cli.command {
        Commands::Merge { models, strategy, weights, base_model, density, output, .. } => {
            assert_eq!(models, vec!["a.apr", "b.apr"]);
            assert_eq!(strategy, "ties");
            assert_eq!(weights, Some("0.7,0.3".into()));
            assert_eq!(base_model, Some("base.apr".into()));
            assert!((density.unwrap() - 0.2).abs() < f64::EPSILON);
            assert_eq!(output, "out.apr");
        }
        _ => panic!("Expected Merge"),
    }
}

#[test]
fn test_cli_parse_prune() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "prune", "--model", "m.apr", "--method", "wanda",
        "--target-ratio", "0.3", "--calibration", "calib.jsonl", "-o", "out.apr",
    ]).unwrap();
    match cli.command {
        Commands::Prune { model, method, target_ratio, calibration, analyze, output } => {
            assert_eq!(model, "m.apr");
            assert_eq!(method, "wanda");
            assert!((target_ratio - 0.3).abs() < f64::EPSILON);
            assert_eq!(calibration, Some("calib.jsonl".into()));
            assert!(!analyze);
            assert_eq!(output, "out.apr");
        }
        _ => panic!("Expected Prune"),
    }
}

#[test]
fn test_cli_parse_quantize() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "quantize", "--model", "m.apr", "--scheme", "q4k",
        "--calibration", "calib.jsonl", "-o", "out.apr",
    ]).unwrap();
    match cli.command {
        Commands::Quantize { model, scheme, calibration, plan, batch, format, output } => {
            assert_eq!(model, "m.apr");
            assert_eq!(scheme, "q4k");
            assert_eq!(calibration, Some("calib.jsonl".into()));
            assert!(!plan);
            assert_eq!(batch, None);
            assert_eq!(format, "apr");
            assert_eq!(output, "out.apr");
        }
        _ => panic!("Expected Quantize"),
    }
}

#[test]
fn test_cli_parse_compare() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "compare", "--model", "m.apr",
    ]).unwrap();
    match cli.command {
        Commands::Compare { model, .. } => assert_eq!(model, "m.apr"),
        _ => panic!("Expected Compare"),
    }
}

#[test]
fn test_cli_parse_submit() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "submit", "--results", "r.json", "--model-id", "org/model",
    ]).unwrap();
    match cli.command {
        Commands::Submit { results, model_id, leaderboard } => {
            assert_eq!(results, "r.json");
            assert_eq!(model_id, "org/model");
            assert_eq!(leaderboard, "open-llm-leaderboard");
        }
        _ => panic!("Expected Submit"),
    }
}

#[test]
fn test_cli_parse_history() {
    let cli = Cli::try_parse_from(["apr-leaderboard", "history", "--model", "qwen"]).unwrap();
    match cli.command {
        Commands::History { model } => assert_eq!(model, Some("qwen".into())),
        _ => panic!("Expected History"),
    }
}

#[test]
fn test_cli_parse_pipeline() {
    let cli = Cli::try_parse_from(["apr-leaderboard", "pipeline", "--config", "c.toml"]).unwrap();
    match cli.command {
        Commands::Pipeline { config } => assert_eq!(config, "c.toml"),
        _ => panic!("Expected Pipeline"),
    }
}

mod defaults_and_flags;
mod extended;
