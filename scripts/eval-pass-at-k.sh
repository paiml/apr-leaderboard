#!/usr/bin/env bash
# eval-pass-at-k.sh - Evaluate a model on code generation benchmarks
#
# Pipeline:
#   1. Download benchmark prompts (HumanEval/MBPP/BigCodeBench)
#   2. Generate completions via `apr run --json --chat`
#   3. Strip markdown fences, combine prompt + completion + test harness
#   4. Execute in sandbox (timeout), score pass/fail
#   5. Compute pass@k and write result JSON
#
# Usage:
#   ./scripts/eval-pass-at-k.sh <benchmark> <model-path> [results-dir] [max-tokens] [temperature] [num-samples]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: eval-pass-at-k.sh <benchmark> <model-path> [results-dir] [max-tokens] [temperature] [num-samples]"
    exit 1
fi

BENCHMARK="$1"
MODEL="$2"
RESULTS_DIR="${3:-results}"
MAX_TOKENS="${4:-512}"
TEMPERATURE="${5:-0.0}"
NUM_SAMPLES="${6:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_FILE="${RESULTS_DIR}/${BENCHMARK}_${TIMESTAMP}.json"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR:?}"' EXIT

echo "=== Pass@k Evaluation ==="
echo "Benchmark:   ${BENCHMARK}"
echo "Model:       ${MODEL}"
echo "Max tokens:  ${MAX_TOKENS}"
echo "Temperature: ${TEMPERATURE}"
echo "Samples:     ${NUM_SAMPLES}"
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
        bigcodebench)  echo "https://huggingface.co/datasets/bigcode/bigcodebench/resolve/main/BigCodeBench.jsonl.gz" ;;
        *)             echo "ERROR: Unknown benchmark: $1"; return 1 ;;
    esac
}

BENCHMARK_DATA_DIR="data/benchmarks"
mkdir -p "$BENCHMARK_DATA_DIR"
BENCHMARK_FILE="${BENCHMARK_DATA_DIR}/${BENCHMARK}.jsonl"

if [[ ! -f "$BENCHMARK_FILE" ]]; then
    URL="$(get_benchmark_url "$BENCHMARK")"
    echo "Downloading ${BENCHMARK} dataset..."
    if [[ "$URL" == *.gz ]]; then
        curl -sfL "$URL" | gunzip > "$BENCHMARK_FILE"
    else
        curl -sfL "$URL" -o "$BENCHMARK_FILE"
    fi
    echo "Downloaded: ${BENCHMARK_FILE} ($(wc -l < "$BENCHMARK_FILE") problems)"
fi

TOTAL_PROBLEMS="$(wc -l < "$BENCHMARK_FILE")"
echo "Problems: ${TOTAL_PROBLEMS}"
echo ""

# ── Build instruction prompt ─────────────────────────────────────────────────

build_instruction() {
    local benchmark="$1"
    local prompt="$2"

    if [[ "$benchmark" == "humaneval" ]]; then
        # HumanEval: prompt is function signature + docstring, we need the body
        printf "Complete the following Python function. Return ONLY the function body as Python code. No markdown, no explanation.\n\n%s" "$prompt"
    elif [[ "$benchmark" == "bigcodebench" ]]; then
        # BigCodeBench: instruct_prompt is already a clear instruction
        printf "Write a Python function to solve this task. Return ONLY the complete Python function with all necessary imports. No markdown, no explanation.\n\n%s" "$prompt"
    else
        # MBPP / other: prompt is a task description
        printf "Write a Python function to solve this task. Return ONLY Python code, no explanation.\n\n%s" "$prompt"
    fi
}

# ── Main evaluation loop ─────────────────────────────────────────────────────

echo "Generating completions and evaluating..."
COMPLETED=0
PASSED=0
ERRORS=0
TOTAL_TOKENS=0
TOTAL_LATENCY=0

while IFS= read -r line; do
    TASK_ID="$(echo "$line" | jq -r '.task_id // .name // "unknown"' 2>/dev/null)"

    # Extract prompt (BigCodeBench uses instruct_prompt for instruction variant)
    PROMPT="$(echo "$line" | jq -r '.instruct_prompt // .prompt // .text // .instruction // ""' 2>/dev/null)"
    if [[ -z "$PROMPT" || "$PROMPT" == "null" ]]; then
        continue
    fi

    # Save the problem JSON for the assembler
    PROBLEM_FILE="${WORK_DIR}/problem_${COMPLETED}.json"
    echo "$line" > "$PROBLEM_FILE"

    # Build instruction for the chat model
    INSTRUCTION="$(build_instruction "$BENCHMARK" "$PROMPT")"

    # Generate completion(s) and test
    TASK_PASSED=0
    for sample_idx in $(seq 1 "$NUM_SAMPLES"); do
        RAW_FILE="${WORK_DIR}/raw_${COMPLETED}_${sample_idx}.json"
        COMPLETION_FILE="${WORK_DIR}/completion_${COMPLETED}_${sample_idx}.py"
        TEST_FILE="${WORK_DIR}/test_${COMPLETED}_${sample_idx}.py"

        # Generate via apr run --json --chat
        if ! apr run "$MODEL" \
                --prompt "$INSTRUCTION" \
                --max-tokens "$MAX_TOKENS" \
                --json --chat \
                2>/dev/null > "$RAW_FILE"; then
            ERRORS=$((ERRORS + 1))
            continue
        fi

        # Extract text from JSON response
        TEXT="$(jq -r '.text // ""' < "$RAW_FILE" 2>/dev/null)"
        TOKENS_GEN="$(jq -r '.tokens_generated // 0' < "$RAW_FILE" 2>/dev/null)"
        LATENCY="$(jq -r '.inference_time_ms // 0' < "$RAW_FILE" 2>/dev/null)"

        TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS_GEN))
        TOTAL_LATENCY="$(python3 -c "print(${TOTAL_LATENCY} + ${LATENCY})")"

        if [[ -z "$TEXT" || "$TEXT" == "null" ]]; then
            ERRORS=$((ERRORS + 1))
            continue
        fi

        # Write raw completion (markdown fences stripped by assembler)
        echo "$TEXT" > "$COMPLETION_FILE"

        # Assemble test file: prompt + completion + test harness
        if ! python3 "${SCRIPT_DIR}/assemble-test.py" \
                "$BENCHMARK" "$PROBLEM_FILE" "$COMPLETION_FILE" "$TEST_FILE" 2>/dev/null; then
            ERRORS=$((ERRORS + 1))
            continue
        fi

        # Execute in sandbox with timeout
        if [[ -f "$TEST_FILE" ]] && [[ -s "$TEST_FILE" ]]; then
            if timeout 10 python3 "$TEST_FILE" > /dev/null 2>&1; then
                TASK_PASSED=1
            fi
        fi
    done

    if (( TASK_PASSED > 0 )); then
        PASSED=$((PASSED + 1))
    fi

    COMPLETED=$((COMPLETED + 1))
    if (( COMPLETED % 10 == 0 )); then
        PCT="$(python3 -c "print(f'{${PASSED}/${COMPLETED}*100:.1f}')")"
        echo "  [${COMPLETED}/${TOTAL_PROBLEMS}] passed=${PASSED} (${PCT}%) errors=${ERRORS}"
    fi
done < "$BENCHMARK_FILE"

# ── Compute metrics ──────────────────────────────────────────────────────────

if (( COMPLETED > 0 )); then
    PASS_AT_1="$(python3 -c "print(round(${PASSED} / ${COMPLETED} * 100, 2))")"
    AVG_TOKENS="$(python3 -c "print(round(${TOTAL_TOKENS} / ${COMPLETED}, 1))")"
    AVG_LATENCY="$(python3 -c "print(round(${TOTAL_LATENCY} / ${COMPLETED}, 1))")"
else
    PASS_AT_1="0.0"
    AVG_TOKENS="0.0"
    AVG_LATENCY="0.0"
fi

echo ""
echo "=== Results ==="
echo "Completed:   ${COMPLETED}/${TOTAL_PROBLEMS}"
echo "Passed:      ${PASSED}"
echo "Errors:      ${ERRORS}"
echo "pass@1:      ${PASS_AT_1}%"
echo "Avg tokens:  ${AVG_TOKENS}"
echo "Avg latency: ${AVG_LATENCY}ms"

# Write result JSON
cat > "$RESULT_FILE" << EOF
{
  "benchmark": "${BENCHMARK}",
  "model": "${MODEL}",
  "timestamp": "$(date -Iseconds)",
  "config": {
    "max_tokens": ${MAX_TOKENS},
    "temperature": ${TEMPERATURE},
    "num_samples": ${NUM_SAMPLES}
  },
  "results": {
    "total": ${TOTAL_PROBLEMS},
    "completed": ${COMPLETED},
    "passed": ${PASSED},
    "errors": ${ERRORS},
    "pass_at_1": ${PASS_AT_1},
    "avg_tokens_generated": ${AVG_TOKENS},
    "avg_latency_ms": ${AVG_LATENCY}
  }
}
EOF

echo ""
echo "Results written to: ${RESULT_FILE}"
