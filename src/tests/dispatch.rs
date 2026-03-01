//! Integration tests for the dispatch() function.
//! These exercise the actual match arms in main.rs,
//! contributing coverage for the CLI routing logic.

use super::super::*;

fn write_test_apr(path: &std::path::Path) {
    let bytes = apr_bridge::create_minimal_apr_bytes().unwrap();
    std::fs::write(path, &bytes).unwrap();
}

fn fixture_model() -> (tempfile::TempDir, String) {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("model.apr");
    write_test_apr(&model_path);
    let path_str = model_path.to_str().unwrap().to_string();
    (tmp, path_str)
}

#[test]
fn test_dispatch_benchmarks() {
    assert!(dispatch(Commands::Benchmarks).is_ok());
}

#[test]
fn test_dispatch_history() {
    assert!(dispatch(Commands::History { model: None }).is_ok());
}

#[test]
fn test_dispatch_history_with_filter() {
    assert!(dispatch(Commands::History { model: Some("qwen".into()) }).is_ok());
}

#[test]
fn test_dispatch_acceptance_list() {
    assert!(dispatch(Commands::Acceptance { category: None, verify: false }).is_ok());
}

#[test]
fn test_dispatch_acceptance_category() {
    assert!(dispatch(Commands::Acceptance {
        category: Some("technique".into()), verify: false,
    }).is_ok());
}

#[test]
fn test_dispatch_acceptance_verify() {
    assert!(dispatch(Commands::Acceptance { category: None, verify: true }).is_ok());
}

#[test]
fn test_dispatch_convert() {
    let tmp = tempfile::TempDir::new().unwrap();
    let output = tmp.path().join("out/");
    assert!(dispatch(Commands::Convert {
        model_id: "test/model".into(),
        output: output.to_str().unwrap().into(),
        quantization: "fp16".into(),
    }).is_ok());
}

#[test]
fn test_dispatch_eval() {
    let (tmp, model) = fixture_model();
    let result_dir = tmp.path().join("results/");
    assert!(dispatch(Commands::Eval {
        model, benchmark: "humaneval".into(), samples: 5,
        output: result_dir.to_str().unwrap().into(),
        prompt_strategy: "standard".into(), n_samples: 1,
        temperature: 0.0, top_p: 0.95, rerank: "none".into(),
        json: false, exemplars: None, system: None,
    }).is_ok());
}

#[test]
fn test_dispatch_eval_scot() {
    let (tmp, model) = fixture_model();
    let result_dir = tmp.path().join("results/");
    assert!(dispatch(Commands::Eval {
        model, benchmark: "mbpp".into(), samples: 0,
        output: result_dir.to_str().unwrap().into(),
        prompt_strategy: "scot".into(), n_samples: 5,
        temperature: 0.6, top_p: 0.9, rerank: "logprob".into(),
        json: true, exemplars: None, system: Some("code only".into()),
    }).is_ok());
}

#[test]
fn test_dispatch_finetune() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("model.apr");
    let output_path = tmp.path().join("out.apr");
    write_test_apr(&model_path);
    assert!(dispatch(Commands::Finetune {
        model: model_path.to_str().unwrap().into(),
        dataset: "d.jsonl".into(), method: "lora".into(),
        rank: 8, lr: 1e-4, epochs: 1,
        output: Some(output_path.to_str().unwrap().into()),
    }).is_ok());
}

#[test]
fn test_dispatch_finetune_qlora() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("model.apr");
    let output_path = tmp.path().join("out.apr");
    write_test_apr(&model_path);
    assert!(dispatch(Commands::Finetune {
        model: model_path.to_str().unwrap().into(),
        dataset: "d.jsonl".into(), method: "qlora".into(),
        rank: 4, lr: 1e-4, epochs: 1,
        output: Some(output_path.to_str().unwrap().into()),
    }).is_ok());
}

#[test]
fn test_dispatch_distill() {
    let tmp = tempfile::TempDir::new().unwrap();
    let teacher = tmp.path().join("teacher.apr");
    let student = tmp.path().join("student.apr");
    let output = tmp.path().join("distilled.apr");
    write_test_apr(&teacher);
    write_test_apr(&student);
    assert!(dispatch(Commands::Distill {
        teacher: teacher.to_str().unwrap().into(),
        student: student.to_str().unwrap().into(),
        strategy: "progressive".into(), temperature: 3.0, alpha: 0.7,
        epochs: 1, data: None,
        output: output.to_str().unwrap().into(),
    }).is_ok());
}

#[test]
fn test_dispatch_merge() {
    let tmp = tempfile::TempDir::new().unwrap();
    let a = tmp.path().join("a.apr");
    let b = tmp.path().join("b.apr");
    let output = tmp.path().join("merged.apr");
    write_test_apr(&a);
    write_test_apr(&b);
    assert!(dispatch(Commands::Merge {
        models: vec![a.to_str().unwrap().into(), b.to_str().unwrap().into()],
        strategy: "slerp".into(), weights: Some("0.7,0.3".into()),
        base_model: None, density: None, drop_rate: None,
        output: output.to_str().unwrap().into(),
    }).is_ok());
}

#[test]
fn test_dispatch_prune() {
    let (tmp, model) = fixture_model();
    let output = tmp.path().join("pruned.apr");
    assert!(dispatch(Commands::Prune {
        model, method: "magnitude".into(), target_ratio: 0.2,
        calibration: None, analyze: false,
        output: output.to_str().unwrap().into(),
    }).is_ok());
}

#[test]
fn test_dispatch_quantize() {
    let (tmp, model) = fixture_model();
    let output = tmp.path().join("quantized.apr");
    assert!(dispatch(Commands::Quantize {
        model, scheme: "int4".into(), calibration: None,
        plan: false, batch: None, format: "apr".into(),
        output: output.to_str().unwrap().into(),
    }).is_ok());
}

#[test]
fn test_dispatch_compare() {
    let (_tmp, model) = fixture_model();
    assert!(dispatch(Commands::Compare { model, json: false }).is_ok());
}

#[test]
fn test_dispatch_check() {
    let (_tmp, model) = fixture_model();
    assert!(dispatch(Commands::Check { model }).is_ok());
}

#[test]
fn test_dispatch_compile() {
    let (tmp, model) = fixture_model();
    let output = tmp.path().join("binary");
    assert!(dispatch(Commands::Compile {
        model, release: true, lto: true, strip: true,
        target: None, output: Some(output.to_str().unwrap().into()),
    }).is_ok());
}

#[test]
fn test_dispatch_align() {
    let (tmp, model) = fixture_model();
    let output = tmp.path().join("aligned.apr");
    assert!(dispatch(Commands::Align {
        model, data: "pairs.jsonl".into(), method: "dpo".into(),
        beta: 0.1, epochs: 1, ref_model: None,
        output: Some(output.to_str().unwrap().into()),
    }).is_ok());
}

#[test]
fn test_dispatch_validate() {
    assert!(dispatch(Commands::Validate {
        data: "train.jsonl".into(),
        benchmarks: vec!["humaneval".into()],
        threshold: 0.01, decontaminate: false, output: None,
    }).is_ok());
}

#[test]
fn test_dispatch_tune() {
    let (_tmp, model) = fixture_model();
    assert!(dispatch(Commands::Tune {
        model, data: "d.jsonl".into(), strategy: "tpe".into(),
        budget: 5, max_epochs: 1,
    }).is_ok());
}

#[test]
fn test_dispatch_run() {
    let (_tmp, model) = fixture_model();
    assert!(dispatch(Commands::Run {
        model, prompt: "def hello():".into(),
        speculative: false, speculation_k: 4,
        draft_model: None, json: false,
    }).is_ok());
}

#[test]
fn test_dispatch_chat() {
    let (_tmp, model) = fixture_model();
    assert!(dispatch(Commands::Chat {
        model, batch: None, prompt: Some("hello".into()),
        n_samples: 1, temperature: 0.8,
        system: None, json: false,
    }).is_ok());
}

#[test]
fn test_dispatch_submit() {
    let tmp = tempfile::TempDir::new().unwrap();
    let results_path = tmp.path().join("results.json");
    std::fs::write(&results_path, r#"{"benchmark":"humaneval","pass_at_1":0.85}"#).unwrap();
    assert!(dispatch(Commands::Submit {
        results: results_path.to_str().unwrap().into(),
        model_id: "test/model".into(),
        leaderboard: "open-llm-leaderboard".into(),
        pre_submit_check: None, generate_card: false,
    }).is_ok());
}

#[test]
fn test_dispatch_export() {
    let (tmp, model) = fixture_model();
    let output = tmp.path().join("export/");
    assert!(dispatch(Commands::Export {
        model, format: "safetensors".into(),
        output: output.to_str().unwrap().into(),
        results: None,
    }).is_ok());
}
