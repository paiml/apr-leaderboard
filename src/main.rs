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
        /// Sampling temperature (0.0 = greedy, 0.1-0.8 = tuned)
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        /// Top-p (nucleus) sampling threshold
        #[arg(long, default_value_t = 0.95)]
        top_p: f64,
        /// Reranking strategy for N-sampling (none, logprob, majority)
        #[arg(long, default_value = "none")]
        rerank: String,
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
            temperature,
            top_p,
            rerank,
        } => {
            let strategy = eval::PromptStrategy::from_str(&prompt_strategy)?;
            let rerank_strategy = eval::RerankStrategy::from_str(&rerank)?;
            let config = eval::EvalConfig {
                prompt_strategy: strategy,
                n_samples,
                temperature,
                top_p,
                rerank: rerank_strategy,
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
mod tests;
