use clap::{Parser, Subcommand};

mod convert;
mod eval;
mod finetune;
mod harness;
mod optimize;
mod pipeline;
mod submit;

/// Build, evaluate, and submit .apr models to Hugging Face leaderboards.
#[derive(Parser)]
#[command(name = "apr-leaderboard", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert a HuggingFace model to .apr format
    Convert {
        /// HuggingFace model ID (e.g., Qwen/Qwen2.5-Coder-7B)
        #[arg(long)]
        model_id: String,
        /// Output path for the .apr file
        #[arg(long, default_value = "models/")]
        output: String,
        /// Quantization level
        #[arg(long, default_value = "fp16")]
        quantization: String,
    },
    /// Run evaluation benchmarks against a model
    Eval {
        /// Path to the .apr model file
        #[arg(long)]
        model: String,
        /// Benchmark suite to run
        #[arg(long)]
        benchmark: String,
        /// Number of samples (0 = full benchmark)
        #[arg(long, default_value_t = 0)]
        samples: usize,
        /// Output results file
        #[arg(long, default_value = "results/")]
        output: String,
        /// Prompt strategy (standard, scot, few-shot, cgo, reflexion)
        #[arg(long, default_value = "standard")]
        prompt_strategy: String,
        /// Generate N completions per problem for best-of-N selection
        #[arg(long, default_value_t = 1)]
        n_samples: usize,
    },
    /// Fine-tune a model for improved benchmark performance
    Finetune {
        /// Path to the base .apr model
        #[arg(long)]
        model: String,
        /// Training dataset
        #[arg(long)]
        dataset: String,
        /// LoRA rank
        #[arg(long, default_value_t = 16)]
        rank: usize,
        /// Learning rate
        #[arg(long, default_value_t = 1e-4)]
        lr: f64,
        /// Number of epochs
        #[arg(long, default_value_t = 3)]
        epochs: usize,
    },
    /// Distill knowledge from a teacher model to a student model
    Distill {
        /// Path to the teacher .apr model
        #[arg(long)]
        teacher: String,
        /// Path to the student .apr model
        #[arg(long)]
        student: String,
        /// Distillation strategy (standard, progressive, ensemble)
        #[arg(long, default_value = "progressive")]
        strategy: String,
        /// Temperature for softmax
        #[arg(long, default_value_t = 3.0)]
        temperature: f64,
        /// Mixing coefficient (0=student only, 1=teacher only)
        #[arg(long, default_value_t = 0.7)]
        alpha: f64,
        /// Output model path
        #[arg(long, short)]
        output: String,
    },
    /// Merge two or more models
    Merge {
        /// Model paths to merge (at least 2)
        #[arg(required = true, num_args = 2..)]
        models: Vec<String>,
        /// Merge strategy (slerp, ties, dare, linear)
        #[arg(long, default_value = "slerp")]
        strategy: String,
        /// Output model path
        #[arg(long, short)]
        output: String,
    },
    /// Prune a model to reduce size
    Prune {
        /// Path to the .apr model
        #[arg(long)]
        model: String,
        /// Pruning method (wanda, magnitude, sparsegpt)
        #[arg(long, default_value = "wanda")]
        method: String,
        /// Target sparsity ratio (0.0 to 1.0)
        #[arg(long, default_value_t = 0.2)]
        target_ratio: f64,
        /// Output model path
        #[arg(long, short)]
        output: String,
    },
    /// Quantize a model to lower precision
    Quantize {
        /// Path to the .apr model
        #[arg(long)]
        model: String,
        /// Quantization scheme (int4, int8, q4k, q5k, q6k)
        #[arg(long, default_value = "int4")]
        scheme: String,
        /// Output model path
        #[arg(long, short)]
        output: String,
    },
    /// Compare an .apr model against HuggingFace reference
    Compare {
        /// Path to the .apr model
        #[arg(long)]
        model: String,
    },
    /// Submit results to HuggingFace leaderboard
    Submit {
        /// Path to results JSON
        #[arg(long)]
        results: String,
        /// HuggingFace model ID for the submission
        #[arg(long)]
        model_id: String,
        /// Leaderboard to target
        #[arg(long, default_value = "open-llm-leaderboard")]
        leaderboard: String,
    },
    /// List available benchmarks
    Benchmarks,
    /// Show results history
    History {
        /// Filter by model
        #[arg(long)]
        model: Option<String>,
    },
    /// Run the full pipeline: convert → eval → submit
    Pipeline {
        /// Path to pipeline config TOML
        #[arg(long)]
        config: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Convert {
            model_id,
            output,
            quantization,
        } => convert::run(&model_id, &output, &quantization),
        Commands::Eval {
            model,
            benchmark,
            samples,
            output,
            prompt_strategy,
            n_samples,
        } => {
            let strategy = eval::PromptStrategy::from_str(&prompt_strategy)?;
            let config = eval::EvalConfig {
                prompt_strategy: strategy,
                n_samples,
            };
            eval::run_with_config(&model, &benchmark, samples, &output, &config)
        }
        Commands::Finetune {
            model,
            dataset,
            rank,
            lr,
            epochs,
        } => finetune::run(&model, &dataset, rank, lr, epochs),
        Commands::Distill {
            teacher,
            student,
            strategy,
            temperature,
            alpha,
            output,
        } => optimize::distill(&teacher, &student, &strategy, temperature, alpha, &output),
        Commands::Merge {
            models,
            strategy,
            output,
        } => optimize::merge(&models, &strategy, &output),
        Commands::Prune {
            model,
            method,
            target_ratio,
            output,
        } => optimize::prune(&model, &method, target_ratio, &output),
        Commands::Quantize {
            model,
            scheme,
            output,
        } => optimize::quantize(&model, &scheme, &output),
        Commands::Compare { model } => optimize::compare(&model),
        Commands::Submit {
            results,
            model_id,
            leaderboard,
        } => submit::run(&results, &model_id, &leaderboard),
        Commands::Benchmarks => {
            harness::list_benchmarks();
            Ok(())
        }
        Commands::History { model } => eval::show_history(model.as_deref()),
        Commands::Pipeline { config } => {
            let config_content = std::fs::read_to_string(&config)?;
            let pipe: pipeline::PipelineConfig = toml::from_str(&config_content)?;
            pipeline::run_pipeline(&pipe)
        }
    }
}

#[cfg(test)]
mod tests {
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
            Commands::Eval { samples, prompt_strategy, n_samples, .. } => {
                assert_eq!(samples, 0);
                assert_eq!(prompt_strategy, "standard");
                assert_eq!(n_samples, 1);
            }
            _ => panic!("Expected Eval"),
        }
    }

    #[test]
    fn test_cli_parse_finetune() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard", "finetune", "--model", "m.apr", "--dataset", "d.jsonl",
            "--rank", "32", "--lr", "0.001", "--epochs", "5",
        ]).unwrap();
        match cli.command {
            Commands::Finetune { model, dataset, rank, lr, epochs } => {
                assert_eq!(model, "m.apr");
                assert_eq!(dataset, "d.jsonl");
                assert_eq!(rank, 32);
                assert!((lr - 0.001).abs() < f64::EPSILON);
                assert_eq!(epochs, 5);
            }
            _ => panic!("Expected Finetune"),
        }
    }

    #[test]
    fn test_cli_parse_distill() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard", "distill", "--teacher", "t.apr", "--student", "s.apr",
            "--strategy", "progressive", "-o", "out.apr",
        ]).unwrap();
        match cli.command {
            Commands::Distill { teacher, student, strategy, output, .. } => {
                assert_eq!(teacher, "t.apr");
                assert_eq!(student, "s.apr");
                assert_eq!(strategy, "progressive");
                assert_eq!(output, "out.apr");
            }
            _ => panic!("Expected Distill"),
        }
    }

    #[test]
    fn test_cli_parse_merge() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard", "merge", "a.apr", "b.apr", "--strategy", "ties", "-o", "out.apr",
        ]).unwrap();
        match cli.command {
            Commands::Merge { models, strategy, output } => {
                assert_eq!(models, vec!["a.apr", "b.apr"]);
                assert_eq!(strategy, "ties");
                assert_eq!(output, "out.apr");
            }
            _ => panic!("Expected Merge"),
        }
    }

    #[test]
    fn test_cli_parse_prune() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard", "prune", "--model", "m.apr", "--method", "wanda",
            "--target-ratio", "0.3", "-o", "out.apr",
        ]).unwrap();
        match cli.command {
            Commands::Prune { model, method, target_ratio, output } => {
                assert_eq!(model, "m.apr");
                assert_eq!(method, "wanda");
                assert!((target_ratio - 0.3).abs() < f64::EPSILON);
                assert_eq!(output, "out.apr");
            }
            _ => panic!("Expected Prune"),
        }
    }

    #[test]
    fn test_cli_parse_quantize() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard", "quantize", "--model", "m.apr", "--scheme", "q4k", "-o", "out.apr",
        ]).unwrap();
        match cli.command {
            Commands::Quantize { model, scheme, output } => {
                assert_eq!(model, "m.apr");
                assert_eq!(scheme, "q4k");
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
            Commands::Finetune { rank, lr, epochs, .. } => {
                assert_eq!(rank, 16);
                assert!((lr - 1e-4).abs() < f64::EPSILON);
                assert_eq!(epochs, 3);
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
            Commands::Distill { strategy, temperature, alpha, .. } => {
                assert_eq!(strategy, "progressive");
                assert!((temperature - 3.0).abs() < f64::EPSILON);
                assert!((alpha - 0.7).abs() < f64::EPSILON);
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
            Commands::Prune { method, target_ratio, .. } => {
                assert_eq!(method, "wanda");
                assert!((target_ratio - 0.2).abs() < f64::EPSILON);
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
            Commands::Quantize { scheme, .. } => assert_eq!(scheme, "int4"),
            _ => panic!("Expected Quantize"),
        }
    }
}
