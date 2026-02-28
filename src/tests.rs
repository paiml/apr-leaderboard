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
        Commands::Prune { model, method, target_ratio, calibration, output } => {
            assert_eq!(model, "m.apr");
            assert_eq!(method, "wanda");
            assert!((target_ratio - 0.3).abs() < f64::EPSILON);
            assert_eq!(calibration, Some("calib.jsonl".into()));
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
        Commands::Quantize { model, scheme, calibration, output } => {
            assert_eq!(model, "m.apr");
            assert_eq!(scheme, "q4k");
            assert_eq!(calibration, Some("calib.jsonl".into()));
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
        Commands::Compare { model } => assert_eq!(model, "m.apr"),
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

#[test]
fn test_cli_defaults_convert() {
    let cli = Cli::try_parse_from(["apr-leaderboard", "convert", "--model-id", "t/m"]).unwrap();
    match cli.command {
        Commands::Convert { output, quantization, .. } => {
            assert_eq!(output, "models/");
            assert_eq!(quantization, "fp16");
        }
        _ => panic!("Expected Convert"),
    }
}

#[test]
fn test_cli_defaults_finetune() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "finetune", "--model", "m.apr", "--dataset", "d.json",
    ]).unwrap();
    match cli.command {
        Commands::Finetune { method, rank, lr, epochs, output, .. } => {
            assert_eq!(method, "lora");
            assert_eq!(rank, 16);
            assert!((lr - 1e-4).abs() < f64::EPSILON);
            assert_eq!(epochs, 3);
            assert_eq!(output, None);
        }
        _ => panic!("Expected Finetune"),
    }
}

#[test]
fn test_cli_defaults_distill() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "distill", "--teacher", "t.apr", "--student", "s.apr", "-o", "o.apr",
    ]).unwrap();
    match cli.command {
        Commands::Distill { strategy, temperature, alpha, epochs, data, .. } => {
            assert_eq!(strategy, "progressive");
            assert!((temperature - 3.0).abs() < f64::EPSILON);
            assert!((alpha - 0.7).abs() < f64::EPSILON);
            assert_eq!(epochs, 5);
            assert_eq!(data, None);
        }
        _ => panic!("Expected Distill"),
    }
}

#[test]
fn test_cli_defaults_prune() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "prune", "--model", "m.apr", "-o", "o.apr",
    ]).unwrap();
    match cli.command {
        Commands::Prune { method, target_ratio, calibration, .. } => {
            assert_eq!(method, "wanda");
            assert!((target_ratio - 0.2).abs() < f64::EPSILON);
            assert_eq!(calibration, None);
        }
        _ => panic!("Expected Prune"),
    }
}

#[test]
fn test_cli_defaults_quantize() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "quantize", "--model", "m.apr", "-o", "o.apr",
    ]).unwrap();
    match cli.command {
        Commands::Quantize { scheme, calibration, .. } => {
            assert_eq!(scheme, "int4");
            assert_eq!(calibration, None);
        }
        _ => panic!("Expected Quantize"),
    }
}

// --- align ---
#[test]
fn test_cli_parse_align() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "align", "--model", "m.apr", "--data", "pairs.jsonl",
        "--method", "orpo", "--beta", "0.5", "--epochs", "5",
    ]).unwrap();
    match cli.command {
        Commands::Align { model, data, method, beta, epochs, ref_model, output } => {
            assert_eq!(model, "m.apr");
            assert_eq!(data, "pairs.jsonl");
            assert_eq!(method, "orpo");
            assert!((beta - 0.5).abs() < f64::EPSILON);
            assert_eq!(epochs, 5);
            assert_eq!(ref_model, None);
            assert_eq!(output, None);
        }
        _ => panic!("Expected Align"),
    }
}

#[test]
fn test_cli_defaults_align() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "align", "--model", "m.apr", "--data", "pairs.jsonl",
    ]).unwrap();
    match cli.command {
        Commands::Align { method, beta, epochs, .. } => {
            assert_eq!(method, "dpo");
            assert!((beta - 0.1).abs() < f64::EPSILON);
            assert_eq!(epochs, 3);
        }
        _ => panic!("Expected Align"),
    }
}

// --- validate ---
#[test]
fn test_cli_parse_validate() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "validate", "--data", "train.jsonl",
        "--benchmarks", "humaneval", "mbpp", "--decontaminate",
    ]).unwrap();
    match cli.command {
        Commands::Validate { data, benchmarks, decontaminate, threshold, .. } => {
            assert_eq!(data, "train.jsonl");
            assert_eq!(benchmarks, vec!["humaneval", "mbpp"]);
            assert!(decontaminate);
            assert!((threshold - 0.01).abs() < f64::EPSILON);
        }
        _ => panic!("Expected Validate"),
    }
}

// --- tune ---
#[test]
fn test_cli_parse_tune() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "tune", "--model", "m.apr", "--data", "d.jsonl",
        "--strategy", "grid", "--budget", "10",
    ]).unwrap();
    match cli.command {
        Commands::Tune { model, data, strategy, budget, max_epochs } => {
            assert_eq!(model, "m.apr");
            assert_eq!(data, "d.jsonl");
            assert_eq!(strategy, "grid");
            assert_eq!(budget, 10);
            assert_eq!(max_epochs, 3);
        }
        _ => panic!("Expected Tune"),
    }
}

// --- run ---
#[test]
fn test_cli_parse_run() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "run", "--model", "m.apr", "--prompt", "def fib(n):",
        "--speculative", "--draft-model", "draft.apr",
    ]).unwrap();
    match cli.command {
        Commands::Run { model, prompt, speculative, draft_model, .. } => {
            assert_eq!(model, "m.apr");
            assert_eq!(prompt, "def fib(n):");
            assert!(speculative);
            assert_eq!(draft_model, Some("draft.apr".into()));
        }
        _ => panic!("Expected Run"),
    }
}

// --- chat ---
#[test]
fn test_cli_parse_chat() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "chat", "--model", "m.apr", "--batch", "prompts.txt",
        "--n-samples", "5", "--temperature", "0.7", "--system", "You are a coder", "--json",
    ]).unwrap();
    match cli.command {
        Commands::Chat { model, batch, prompt, n_samples, temperature, system, json } => {
            assert_eq!(model, "m.apr");
            assert_eq!(batch, Some("prompts.txt".into()));
            assert_eq!(prompt, None);
            assert_eq!(n_samples, 5);
            assert!((temperature - 0.7).abs() < f64::EPSILON);
            assert_eq!(system, Some("You are a coder".into()));
            assert!(json);
        }
        _ => panic!("Expected Chat"),
    }
}

// --- check ---
#[test]
fn test_cli_parse_check() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "check", "--model", "m.apr",
    ]).unwrap();
    match cli.command {
        Commands::Check { model } => assert_eq!(model, "m.apr"),
        _ => panic!("Expected Check"),
    }
}

// --- compile ---
#[test]
fn test_cli_parse_compile() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "compile", "--model", "m.apr", "--release", "--lto", "-o", "binary",
    ]).unwrap();
    match cli.command {
        Commands::Compile { model, release, lto, output } => {
            assert_eq!(model, "m.apr");
            assert!(release);
            assert!(lto);
            assert_eq!(output, Some("binary".into()));
        }
        _ => panic!("Expected Compile"),
    }
}

#[test]
fn test_cli_defaults_compile() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "compile", "--model", "m.apr",
    ]).unwrap();
    match cli.command {
        Commands::Compile { release, lto, output, .. } => {
            assert!(!release);
            assert!(!lto);
            assert_eq!(output, None);
        }
        _ => panic!("Expected Compile"),
    }
}
