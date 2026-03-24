#!/usr/bin/env bash
# eval-pass-at-k.sh - Evaluate a model on code generation benchmarks
#
# Architecture:
#   Phase 1: Prepare --download benchmark, split into per-problem JSON files
#   Phase 2: Generate --batch mode (single model load via --batch-jsonl) or
#            N parallel workers each claim problems via flock.
#            Batch mode eliminates ~80s per-problem CUDA JIT overhead on gx10.
#   Phase 3: Test --sequential sandbox execution of all completions
#   Phase 4: Score --compute pass@k (Chen et al. unbiased estimator)
#
# Usage:
#   ./scripts/eval-pass-at-k.sh BENCHMARK MODEL_PATH [RESULTS_DIR] [MAX_TOKENS] [TEMPERATURE] [NUM_SAMPLES] [PROMPT_STRATEGY] [WORKERS]
#
# Environment variables:
#   APR_BATCH_MODE=auto|on|off    --batch mode control (default: auto)
#   SKIP_PARITY_GATE=1            --bypass GPU FP parity check (required for Blackwell sm_121)

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: eval-pass-at-k.sh BENCHMARK MODEL_PATH [RESULTS_DIR] [MAX_TOKENS] [TEMPERATURE] [NUM_SAMPLES] [PROMPT_STRATEGY] [WORKERS]"
    exit 1
fi

BENCHMARK="$1"
MODEL="$2"
RESULTS_DIR="${3:-results}"
MAX_TOKENS="${4:-512}"
TEMPERATURE="${5:-0.0}"
NUM_SAMPLES="${6:-1}"
PROMPT_STRATEGY="${7:-standard}"
NUM_WORKERS="${8:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_FILE="${RESULTS_DIR}/${BENCHMARK}_${TIMESTAMP}.json"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR:?}"' EXIT

# Qwen3 thinking models: thinking phase consumes 1000-6000+ tokens.
# Model produces garbage without thinking (5% vs 86% pass@1).
# Override max_tokens to give thinking room; strip_thinking_tokens()
# extracts only the post-thinking code answer.
# Qwen3 thinking: 1000-3000 tokens of reasoning before code.
# 4096 is enough for ~90% of problems (validated: 86% pass@1 at 4096).
# Higher budgets cause timeout issues with parallel workers.
EFFECTIVE_MAX_TOKENS="$MAX_TOKENS"
if [[ "$MODEL" == *qwen3* && "$MAX_TOKENS" -lt 4096 ]]; then
    EFFECTIVE_MAX_TOKENS=4096
fi

# Timeout scales with token budget: ~3 tok/s on CPU, add margin for model load
GENERATION_TIMEOUT=$(( EFFECTIVE_MAX_TOKENS / 2 + 60 ))  # ~25 min for 4096 tokens

echo "=== Pass@k Evaluation ==="
echo "Benchmark:   ${BENCHMARK}"
echo "Model:       ${MODEL}"
echo "Max tokens:  ${MAX_TOKENS} (effective: ${EFFECTIVE_MAX_TOKENS}, timeout: ${GENERATION_TIMEOUT}s)"
echo "Temperature: ${TEMPERATURE}"
echo "Samples:     ${NUM_SAMPLES}"
echo "Strategy:    ${PROMPT_STRATEGY}"
echo "Workers:     ${NUM_WORKERS}"
echo "Output:      ${RESULT_FILE}"
echo ""

# Validate prerequisites
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found at ${MODEL}"
    exit 1
fi
command -v apr >/dev/null 2>&1 || { echo "ERROR: apr CLI not found"; exit 1; }
command -v jq  >/dev/null 2>&1 || { echo "ERROR: jq not found"; exit 1; }

mkdir -p "$RESULTS_DIR"

# ── Download benchmark data ──────────────────────────────────────────────────

get_benchmark_url() {
    case "$1" in
        humaneval)     echo "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz" ;;
        mbpp)          echo "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl" ;;
        bigcodebench)  echo "https://huggingface.co/datasets/bigcode/bigcodebench/resolve/main/data/v0.1.4-00000-of-00001.parquet" ;;
        *)             echo "ERROR: Unknown benchmark: $1"; return 1 ;;
    esac
}

BENCHMARK_DATA_DIR="data/benchmarks"
mkdir -p "$BENCHMARK_DATA_DIR"
BENCHMARK_FILE="${BENCHMARK_DATA_DIR}/${BENCHMARK}.jsonl"

if [[ ! -f "$BENCHMARK_FILE" ]]; then
    URL="$(get_benchmark_url "$BENCHMARK")"
    echo "Downloading ${BENCHMARK} dataset..."
    TMPFILE="$(mktemp "${BENCHMARK_DATA_DIR}/.dl.XXXXXX")"
    if [[ "$URL" == *.gz ]]; then
        curl -sfL "$URL" | gunzip > "$TMPFILE"
    elif [[ "$URL" == *.parquet ]]; then
        # BigCodeBench: download parquet, convert to JSONL via python3
        local PARQUET_FILE="${TMPFILE}.parquet"
        curl -sfL "$URL" -o "$PARQUET_FILE"
        python3 -c "
import pandas as pd, json
df = pd.read_parquet('$PARQUET_FILE')
df.to_json('$TMPFILE', orient='records', lines=True)
print(f'  Converted {len(df)} records from parquet to JSONL')
"
        rm -f "$PARQUET_FILE"
    else
        curl -sfL "$URL" -o "$TMPFILE"
    fi
    # MBPP: filter to standard test split (task_id 11-510, 500 problems)
    # Tasks 1-10 are few-shot examples, 511-974 are training data
    if [[ "$BENCHMARK" == "mbpp" ]]; then
        jq -c 'select(.task_id >= 11 and .task_id <= 510)' "$TMPFILE" > "${TMPFILE}.filtered"
        mv "${TMPFILE}.filtered" "$TMPFILE"
        echo "  Filtered to MBPP test split (task_id 11-510)"
    fi
    mv "$TMPFILE" "$BENCHMARK_FILE"
    echo "Downloaded: ${BENCHMARK_FILE} ($(wc -l < "$BENCHMARK_FILE") problems)"
fi

TOTAL_PROBLEMS="$(wc -l < "$BENCHMARK_FILE")"
echo "Problems: ${TOTAL_PROBLEMS}"
echo ""

# ── Helper functions (sourced from eval-helpers.sh) ──────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/eval-helpers.sh"

# ── Phase 1: Prepare --split benchmark into per-problem files ────────────────

echo "Phase 1: Preparing problems..."
mkdir -p "${WORK_DIR}/problems" "${WORK_DIR}/raw" "${WORK_DIR}/completions" "${WORK_DIR}/tests"

IDX=0
while IFS= read -r line; do
    echo "$line" > "${WORK_DIR}/problems/${IDX}.json"
    IDX=$((IDX + 1))
done < "$BENCHMARK_FILE"
echo "  Prepared ${IDX} problems"

# Work queue: file listing problem indices not yet claimed
QUEUE_FILE="${WORK_DIR}/queue"
QUEUE_LOCK="${WORK_DIR}/queue.lock"
seq 0 $((TOTAL_PROBLEMS - 1)) > "$QUEUE_FILE"

# Progress tracking (atomic via flock)
PROGRESS_FILE="${WORK_DIR}/progress"
PROGRESS_LOCK="${WORK_DIR}/progress.lock"
echo "0" > "$PROGRESS_FILE"

# ── Phase 2: Generate --batch mode or parallel workers ───────────────────────

# Detect batch mode availability
# Batch mode loads model + CUDA JIT once, processes all prompts sequentially.
# Eliminates ~80s per-invocation overhead on gx10 sm_121 Blackwell GPU.
# Supports --temperature and --top-k passthrough for non-greedy sampling.
USE_BATCH=0
BATCH_MODE="${APR_BATCH_MODE:-auto}"
if [[ "$BATCH_MODE" == "on" ]]; then
    USE_BATCH=1
elif [[ "$BATCH_MODE" == "off" ]]; then
    USE_BATCH=0
elif [[ "$BATCH_MODE" == "auto" ]]; then
    # Auto-detect: use batch if --batch-jsonl available
    if apr run --help 2>&1 | grep -q -- '--batch-jsonl'; then
        USE_BATCH=1
    fi
fi


# ── Phase 2: Dispatch ────────────────────────────────────────────────────────

if (( USE_BATCH )); then
    echo "Phase 2: Generating completions (BATCH mode -- single model load, ${EFFECTIVE_MAX_TOKENS} max tokens)..."
    if ! generate_batch; then
        echo "  Batch mode failed, falling back to worker mode..." >&2
        USE_BATCH=0
    fi
    GENERATED="$TOTAL_PROBLEMS"
fi

if (( ! USE_BATCH )); then
    echo "Phase 2: Generating completions (${NUM_WORKERS} workers, ${EFFECTIVE_MAX_TOKENS} max tokens, ${GENERATION_TIMEOUT}s timeout)..."

    # Export functions and variables for subshells
    export -f generate_worker build_instruction strip_thinking_tokens strip_markdown_fences extract_python_code
    export WORK_DIR QUEUE_FILE QUEUE_LOCK PROGRESS_FILE PROGRESS_LOCK
    export MODEL BENCHMARK PROMPT_STRATEGY EFFECTIVE_MAX_TOKENS GENERATION_TIMEOUT TOTAL_PROBLEMS

    # Launch workers
    WORKER_PIDS=()
    for w in $(seq 1 "$NUM_WORKERS"); do
        generate_worker "$w" &
        WORKER_PIDS+=($!)
    done

    # Wait for all workers
    WORKER_FAILED=0
    for pid in "${WORKER_PIDS[@]}"; do
        if ! wait "$pid"; then
            WORKER_FAILED=$((WORKER_FAILED + 1))
        fi
    done

    if (( WORKER_FAILED > 0 )); then
        echo "  WARNING: ${WORKER_FAILED} workers exited with errors"
    fi

    GENERATED="$(cat "$PROGRESS_FILE")"
fi

echo "  Generation complete: ${GENERATED}/${TOTAL_PROBLEMS}"
echo ""

# ── Phase 3: Test --sequential sandbox execution ─────────────────────────────

echo "Phase 3: Testing completions..."
COMPLETED=0
PASSED=0
ERRORS=0
TASK_RESULTS_FILE="${WORK_DIR}/task_results.tsv"
: > "$TASK_RESULTS_FILE"

for problem_idx in $(seq 0 $((TOTAL_PROBLEMS - 1))); do
    local_problem_file="${WORK_DIR}/problems/${problem_idx}.json"
    local_completion_file="${WORK_DIR}/completions/${problem_idx}.py"

    TASK_ID="$(jq -r '.task_id // .name // "unknown"' < "$local_problem_file" 2>/dev/null)"
    PROMPT="$(jq -r '.instruct_prompt // .prompt // .text // .instruction // ""' < "$local_problem_file" 2>/dev/null)"

    # Check if completion exists and isn't an error marker
    if [[ ! -f "$local_completion_file" ]]; then
        ERRORS=$((ERRORS + 1))
        printf "%s\t1\t0\n" "$TASK_ID" >> "$TASK_RESULTS_FILE"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    COMPLETION_TEXT="$(cat "$local_completion_file")"
    if [[ "$COMPLETION_TEXT" == "ERROR" || "$COMPLETION_TEXT" == "SKIP" ]]; then
        ERRORS=$((ERRORS + 1))
        printf "%s\t1\t0\n" "$TASK_ID" >> "$TASK_RESULTS_FILE"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    # Assemble test file
    TEST_FILE="${WORK_DIR}/tests/${problem_idx}.py"
    ENTRY_POINT="$(jq -r '.entry_point // ""' < "$local_problem_file" 2>/dev/null)"
    TEST_SETUP="$(jq -r '.test_setup_code // ""' < "$local_problem_file" 2>/dev/null)"
    HAS_TEST="$(jq -r 'has("test")' < "$local_problem_file" 2>/dev/null)"
    HAS_TEST_LIST="$(jq -r 'has("test_list")' < "$local_problem_file" 2>/dev/null)"

    if [[ "$HAS_TEST" == "true" ]]; then
        TEST_CODE="$(jq -r '.test' < "$local_problem_file" 2>/dev/null)"
        HAS_CODE_PROMPT="$(jq -r 'has("code_prompt")' < "$local_problem_file" 2>/dev/null)"
        {
            if [[ "$HAS_CODE_PROMPT" == "true" ]]; then
                cat "$local_completion_file"
            else
                echo "$PROMPT"
                cat "$local_completion_file"
            fi
            echo ""
            echo "$TEST_CODE"
            if [[ -n "$ENTRY_POINT" && "$ENTRY_POINT" != "null" && "$HAS_CODE_PROMPT" != "true" ]]; then
                echo "check(${ENTRY_POINT})"
            fi
        } > "$TEST_FILE"
    elif [[ "$HAS_TEST_LIST" == "true" ]]; then
        {
            if [[ -n "$TEST_SETUP" && "$TEST_SETUP" != "null" ]]; then
                echo "$TEST_SETUP"
                echo ""
            fi
            cat "$local_completion_file"
            echo ""
            jq -r '.test_list[]' < "$local_problem_file" 2>/dev/null
        } > "$TEST_FILE"
    else
        {
            echo "$PROMPT"
            cat "$local_completion_file"
        } > "$TEST_FILE"
    fi

    # Execute test
    TASK_PASSED=0
    if [[ -f "$TEST_FILE" && -s "$TEST_FILE" ]]; then
        if command -v python3 >/dev/null 2>&1; then
            if timeout 10 python3 "$TEST_FILE" > /dev/null 2>&1; then
                TASK_PASSED=1
            fi
        elif command -v docker >/dev/null 2>&1; then
            if timeout 30 docker run --rm --network=none --memory=512m \
                -v "${TEST_FILE}:/test.py:ro" python:3.11-slim \
                python3 /test.py > /dev/null 2>&1; then
                TASK_PASSED=1
            fi
        fi
    fi

    printf "%s\t1\t%s\n" "$TASK_ID" "$TASK_PASSED" >> "$TASK_RESULTS_FILE"

    if (( TASK_PASSED > 0 )); then
        PASSED=$((PASSED + 1))
    fi

    COMPLETED=$((COMPLETED + 1))
    if (( COMPLETED % 10 == 0 )); then
        PCT="$(awk "BEGIN{printf \"%.1f\", ${PASSED}/${COMPLETED}*100}")"
        printf "  %s/%s passed=%s (%s%%) errors=%s\n" "$COMPLETED" "$TOTAL_PROBLEMS" "$PASSED" "$PCT" "$ERRORS"
    fi
done

# ── Phase 4: Score ───────────────────────────────────────────────────────────

if (( COMPLETED > 0 )); then
    PASS_AT_1="$(awk -F'\t' '{
        n = $2; c = $3; sum += 1
        if (n - c < 1) { p1 += 1.0; next }
        if (c == 0)    { next }
        log_ratio = log(n - c) - log(n)
        p1 += 1.0 - exp(log_ratio)
    } END { printf "%.2f", (sum > 0) ? p1/sum*100 : 0 }' "$TASK_RESULTS_FILE")"

    if (( NUM_SAMPLES >= 10 )); then
        PASS_AT_10="$(awk -F'\t' -v k=10 '{
            n = $2; c = $3; sum += 1
            if (n - c < k) { pk += 1.0; next }
            if (c == 0)    { next }
            lr = 0; for (i=0;i<k;i++) lr += log(n-c-i) - log(n-i)
            pk += 1.0 - exp(lr)
        } END { printf "%.2f", (sum > 0) ? pk/sum*100 : 0 }' "$TASK_RESULTS_FILE")"
    else
        PASS_AT_10="null"
    fi
else
    PASS_AT_1="0.0"
    PASS_AT_10="null"
fi

echo ""
echo "=== Results ==="
echo "Completed:   ${COMPLETED}/${TOTAL_PROBLEMS}"
echo "Passed:      ${PASSED}/${COMPLETED} tasks (at least 1 sample correct)"
echo "Errors:      ${ERRORS}"
echo "pass@1:      ${PASS_AT_1}%  (Chen et al. unbiased estimator)"
if [[ "$PASS_AT_10" != "null" ]]; then
    echo "pass@10:     ${PASS_AT_10}%"
fi

PASS_AT_10_JSON="${PASS_AT_10}"

# Build per-problem JSON array from task_results.tsv
PROBLEMS_JSON="["
FIRST=1
while IFS=$'\t' read -r tid nsamples npassed; do
    if (( FIRST )); then FIRST=0; else PROBLEMS_JSON+=","; fi
    PROBLEMS_JSON+="{\"task_id\":\"${tid}\",\"n\":${nsamples},\"passed\":${npassed}}"
done < "$TASK_RESULTS_FILE"
PROBLEMS_JSON+="]"

cat > "$RESULT_FILE" << EOF
{
  "benchmark": "${BENCHMARK}",
  "model": "${MODEL}",
  "timestamp": "$(date -Iseconds)",
  "config": {
    "max_tokens": ${MAX_TOKENS},
    "effective_max_tokens": ${EFFECTIVE_MAX_TOKENS},
    "temperature": ${TEMPERATURE},
    "num_samples": ${NUM_SAMPLES},
    "prompt_strategy": "${PROMPT_STRATEGY}",
    "workers": ${NUM_WORKERS},
    "timeout_seconds": ${GENERATION_TIMEOUT}
  },
  "results": {
    "total": ${TOTAL_PROBLEMS},
    "completed": ${COMPLETED},
    "passed": ${PASSED},
    "errors": ${ERRORS},
    "pass_at_1": ${PASS_AT_1},
    "pass_at_10": ${PASS_AT_10_JSON}
  },
  "problems": ${PROBLEMS_JSON}
}
EOF

echo ""
echo "Results written to: ${RESULT_FILE}"
