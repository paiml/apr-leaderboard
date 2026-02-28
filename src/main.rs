use clap::{Parser, Subcommand};

mod convert;
mod eval;
mod finetune;
mod harness;
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
        } => eval::run(&model, &benchmark, samples, &output),

        Commands::Finetune {
            model,
            dataset,
            rank,
            lr,
            epochs,
        } => finetune::run(&model, &dataset, rank, lr, epochs),

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
            "apr-leaderboard",
            "convert",
            "--model-id",
            "Qwen/Qwen2.5-Coder-7B",
        ])
        .unwrap();
        match cli.command {
            Commands::Convert {
                model_id,
                output,
                quantization,
            } => {
                assert_eq!(model_id, "Qwen/Qwen2.5-Coder-7B");
                assert_eq!(output, "models/");
                assert_eq!(quantization, "fp16");
            }
            _ => panic!("Expected Convert command"),
        }
    }

    #[test]
    fn test_cli_parse_eval() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard",
            "eval",
            "--model",
            "model.apr",
            "--benchmark",
            "humaneval",
            "--samples",
            "10",
        ])
        .unwrap();
        match cli.command {
            Commands::Eval {
                model,
                benchmark,
                samples,
                output,
            } => {
                assert_eq!(model, "model.apr");
                assert_eq!(benchmark, "humaneval");
                assert_eq!(samples, 10);
                assert_eq!(output, "results/");
            }
            _ => panic!("Expected Eval command"),
        }
    }

    #[test]
    fn test_cli_parse_finetune() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard",
            "finetune",
            "--model",
            "model.apr",
            "--dataset",
            "data.jsonl",
            "--rank",
            "32",
            "--lr",
            "0.001",
            "--epochs",
            "5",
        ])
        .unwrap();
        match cli.command {
            Commands::Finetune {
                model,
                dataset,
                rank,
                lr,
                epochs,
            } => {
                assert_eq!(model, "model.apr");
                assert_eq!(dataset, "data.jsonl");
                assert_eq!(rank, 32);
                assert!((lr - 0.001).abs() < f64::EPSILON);
                assert_eq!(epochs, 5);
            }
            _ => panic!("Expected Finetune command"),
        }
    }

    #[test]
    fn test_cli_parse_submit() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard",
            "submit",
            "--results",
            "results.json",
            "--model-id",
            "org/model",
            "--leaderboard",
            "evalplus",
        ])
        .unwrap();
        match cli.command {
            Commands::Submit {
                results,
                model_id,
                leaderboard,
            } => {
                assert_eq!(results, "results.json");
                assert_eq!(model_id, "org/model");
                assert_eq!(leaderboard, "evalplus");
            }
            _ => panic!("Expected Submit command"),
        }
    }

    #[test]
    fn test_cli_parse_history() {
        let cli =
            Cli::try_parse_from(["apr-leaderboard", "history", "--model", "qwen"]).unwrap();
        match cli.command {
            Commands::History { model } => {
                assert_eq!(model, Some("qwen".into()));
            }
            _ => panic!("Expected History command"),
        }
    }

    #[test]
    fn test_cli_parse_history_no_filter() {
        let cli = Cli::try_parse_from(["apr-leaderboard", "history"]).unwrap();
        match cli.command {
            Commands::History { model } => {
                assert!(model.is_none());
            }
            _ => panic!("Expected History command"),
        }
    }

    #[test]
    fn test_cli_parse_pipeline() {
        let cli =
            Cli::try_parse_from(["apr-leaderboard", "pipeline", "--config", "config.toml"])
                .unwrap();
        match cli.command {
            Commands::Pipeline { config } => {
                assert_eq!(config, "config.toml");
            }
            _ => panic!("Expected Pipeline command"),
        }
    }

    #[test]
    fn test_cli_defaults() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard",
            "convert",
            "--model-id",
            "test/model",
        ])
        .unwrap();
        match cli.command {
            Commands::Convert {
                output,
                quantization,
                ..
            } => {
                assert_eq!(output, "models/");
                assert_eq!(quantization, "fp16");
            }
            _ => panic!("Expected Convert command"),
        }
    }

    #[test]
    fn test_cli_finetune_defaults() {
        let cli = Cli::try_parse_from([
            "apr-leaderboard",
            "finetune",
            "--model",
            "m.apr",
            "--dataset",
            "d.json",
        ])
        .unwrap();
        match cli.command {
            Commands::Finetune {
                rank, lr, epochs, ..
            } => {
                assert_eq!(rank, 16);
                assert!((lr - 1e-4).abs() < f64::EPSILON);
                assert_eq!(epochs, 3);
            }
            _ => panic!("Expected Finetune command"),
        }
    }
}
