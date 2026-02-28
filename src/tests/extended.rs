use super::super::*;
use clap::Parser;

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
        Commands::Compile { model, release, lto, output, .. } => {
            assert_eq!(model, "m.apr");
            assert!(release);
            assert!(lto);
            assert_eq!(output, Some("binary".into()));
        }
        _ => panic!("Expected Compile"),
    }
}
