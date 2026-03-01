use super::super::*;
use clap::Parser;

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

// --- new flag tests ---

#[test]
fn test_cli_eval_json_flag() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "eval", "--model", "m.apr", "--benchmark", "humaneval", "--json",
    ]).unwrap();
    match cli.command {
        Commands::Eval { json, .. } => assert!(json),
        _ => panic!("Expected Eval"),
    }
}

#[test]
fn test_cli_eval_exemplars_and_system() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "eval", "--model", "m.apr", "--benchmark", "humaneval",
        "--exemplars", "examples.jsonl", "--system", "You are a coder",
    ]).unwrap();
    match cli.command {
        Commands::Eval { exemplars, system, .. } => {
            assert_eq!(exemplars, Some("examples.jsonl".into()));
            assert_eq!(system, Some("You are a coder".into()));
        }
        _ => panic!("Expected Eval"),
    }
}

#[test]
fn test_cli_compare_json_flag() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "compare", "--model", "m.apr", "--json",
    ]).unwrap();
    match cli.command {
        Commands::Compare { model, json } => {
            assert_eq!(model, "m.apr");
            assert!(json);
        }
        _ => panic!("Expected Compare"),
    }
}

#[test]
fn test_cli_prune_analyze_flag() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "prune", "--model", "m.apr", "-o", "out.apr", "--analyze",
    ]).unwrap();
    match cli.command {
        Commands::Prune { analyze, .. } => assert!(analyze),
        _ => panic!("Expected Prune"),
    }
}

#[test]
fn test_cli_quantize_plan_batch_format() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "quantize", "--model", "m.apr", "-o", "out.apr",
        "--plan", "--batch", "int4,int8", "--format", "gguf",
    ]).unwrap();
    match cli.command {
        Commands::Quantize { plan, batch, format, .. } => {
            assert!(plan);
            assert_eq!(batch, Some("int4,int8".into()));
            assert_eq!(format, "gguf");
        }
        _ => panic!("Expected Quantize"),
    }
}

#[test]
fn test_cli_compile_strip_and_target() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "compile", "--model", "m.apr",
        "--release", "--lto", "--strip",
        "--target", "x86_64-unknown-linux-musl", "-o", "binary",
    ]).unwrap();
    match cli.command {
        Commands::Compile { model, release, lto, strip, target, output } => {
            assert_eq!(model, "m.apr");
            assert!(release);
            assert!(lto);
            assert!(strip);
            assert_eq!(target, Some("x86_64-unknown-linux-musl".into()));
            assert_eq!(output, Some("binary".into()));
        }
        _ => panic!("Expected Compile"),
    }
}

#[test]
fn test_cli_submit_pre_submit_check() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "submit", "--results", "r.json",
        "--model-id", "org/model", "--pre-submit-check", "model.apr",
    ]).unwrap();
    match cli.command {
        Commands::Submit { results, model_id, pre_submit_check, .. } => {
            assert_eq!(results, "r.json");
            assert_eq!(model_id, "org/model");
            assert_eq!(pre_submit_check, Some("model.apr".into()));
        }
        _ => panic!("Expected Submit"),
    }
}

#[test]
fn test_cli_submit_no_pre_submit_check() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "submit", "--results", "r.json",
        "--model-id", "org/model",
    ]).unwrap();
    match cli.command {
        Commands::Submit { pre_submit_check, .. } => {
            assert!(pre_submit_check.is_none());
        }
        _ => panic!("Expected Submit"),
    }
}

#[test]
fn test_cli_submit_generate_card() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "submit", "--results", "r.json",
        "--model-id", "org/model", "--generate-card",
    ]).unwrap();
    match cli.command {
        Commands::Submit { generate_card, .. } => {
            assert!(generate_card);
        }
        _ => panic!("Expected Submit"),
    }
}

#[test]
fn test_cli_submit_no_generate_card() {
    let cli = Cli::try_parse_from([
        "apr-leaderboard", "submit", "--results", "r.json",
        "--model-id", "org/model",
    ]).unwrap();
    match cli.command {
        Commands::Submit { generate_card, .. } => {
            assert!(!generate_card);
        }
        _ => panic!("Expected Submit"),
    }
}
