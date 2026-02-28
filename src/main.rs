use clap::{Parser, Subcommand};

mod convert;
mod eval;
mod finetune;
mod harness;
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
            let pipeline: PipelineConfig = toml::from_str(&config_content)?;
            run_pipeline(&pipeline)
        }
    }
}

#[derive(serde::Deserialize)]
struct PipelineConfig {
    model_id: String,
    output_dir: String,
    quantization: String,
    benchmarks: Vec<String>,
    submit: bool,
    leaderboard: String,
    finetune: Option<FinetuneConfig>,
}

#[derive(serde::Deserialize)]
struct FinetuneConfig {
    dataset: String,
    rank: usize,
    lr: f64,
    epochs: usize,
}

fn run_pipeline(config: &PipelineConfig) -> anyhow::Result<()> {
    println!("=== APR Leaderboard Pipeline ===\n");

    // Step 1: Convert
    println!("[1/4] Converting {} to .apr...", config.model_id);
    convert::run(&config.model_id, &config.output_dir, &config.quantization)?;

    let model_path = format!(
        "{}/{}.apr",
        config.output_dir,
        config.model_id.replace('/', "_")
    );

    // Step 2: Optional fine-tune
    if let Some(ft) = &config.finetune {
        println!("[2/4] Fine-tuning with LoRA (rank={})...", ft.rank);
        finetune::run(&model_path, &ft.dataset, ft.rank, ft.lr, ft.epochs)?;
    } else {
        println!("[2/4] Skipping fine-tune (not configured)");
    }

    // Step 3: Evaluate
    println!("[3/4] Running benchmarks...");
    for benchmark in &config.benchmarks {
        eval::run(&model_path, benchmark, 0, "results/")?;
    }

    // Step 4: Submit
    if config.submit {
        println!("[4/4] Submitting to {}...", config.leaderboard);
        submit::run(
            &format!("results/{}.json", config.model_id.replace('/', "_")),
            &config.model_id,
            &config.leaderboard,
        )?;
    } else {
        println!("[4/4] Skipping submission (submit = false)");
    }

    println!("\n=== Pipeline complete ===");
    Ok(())
}
