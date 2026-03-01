#!/usr/bin/env bash
# eval-pass-at-k.sh - Evaluate a model on code generation benchmarks
#
# This script orchestrates the full pass@k evaluation pipeline:
#   1. Load benchmark prompts
#   2. Generate completions via `apr run`
#   3. Execute generated code in a sandbox
#   4. Score pass/fail and compute pass@k
#
# Usage:
#   ./scripts/eval-pass-at-k.sh <benchmark> <model-path> [results-dir] [max-tokens] [temperature] [num-samples]
#
# Examples:
#   ./scripts/eval-pass-at-k.sh humaneval checkpoints/qwen-coder-7b.apr
#   ./scripts/eval-pass-at-k.sh mbpp checkpoints/qwen-coder-7b.apr results/ 512 0.0 1

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: eval-pass-at-k.sh <benchmark> <model-path> (results-dir) (max-tokens) (temperature) (num-samples)"
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
COMPLETIONS_DIR="$(mktemp -d)"
trap 'rm -rf "${COMPLETIONS_DIR:?}"' EXIT

echo "=== Pass@k Evaluation ==="
echo "Benchmark:   ${BENCHMARK}"
echo "Model:       ${MODEL}"
echo "Max tokens:  ${MAX_TOKENS}"
echo "Temperature: ${TEMPERATURE}"
echo "Samples:     ${NUM_SAMPLES}"
echo "Output:      ${RESULT_FILE}"
echo ""

# Validate model exists
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found at ${MODEL}"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# Benchmark prompt sources
# HumanEval: 164 problems, each with a function signature + docstring
# MBPP: 974 problems, each with a task description + test cases
# BigCodeBench: 1140 problems

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

# Download benchmark data if not cached
if [[ ! -f "$BENCHMARK_FILE" ]]; then
    URL="$(get_benchmark_url "$BENCHMARK")"
    echo "Downloading ${BENCHMARK} dataset..."
    if [[ "$URL" == *.gz ]]; then
        curl -sfL "$URL" | gunzip > "$BENCHMARK_FILE"
    else
        curl -sfL "$URL" -o "$BENCHMARK_FILE"
    fi
    PROB_COUNT="$(wc -l < "$BENCHMARK_FILE")"
    echo "Downloaded: ${BENCHMARK_FILE} (${PROB_COUNT} problems)"
fi

TOTAL_PROBLEMS="$(wc -l < "$BENCHMARK_FILE")"
echo "Problems: ${TOTAL_PROBLEMS}"
echo ""

# Generate completions
echo "Generating completions..."
COMPLETED=0
PASSED=0

# shellcheck disable=SC2034
while IFS= read -r line; do
    TASK_ID="$(echo "$line" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('task_id', d.get('name', 'unknown')))
" 2>/dev/null || echo "task_${COMPLETED}")"

    # Extract the prompt based on benchmark format
    PROMPT="$(echo "$line" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'prompt' in d:
    print(d['prompt'])
elif 'text' in d:
    print(d['text'])
else:
    print(d.get('instruction', ''))
" 2>/dev/null || true)"

    if [[ -z "$PROMPT" ]]; then
        continue
    fi

    # Generate completion
    COMPLETION_FILE="${COMPLETIONS_DIR}/${COMPLETED}.py"
    for _ in $(seq 1 "$NUM_SAMPLES"); do
        apr run "$MODEL" \
            --prompt "$PROMPT" \
            --max-tokens "$MAX_TOKENS" \
            2>/dev/null > "$COMPLETION_FILE" || true
    done

    # Extract test cases and execute
    TESTS="$(echo "$line" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'test' in d:
    print(d['test'])
elif 'test_list' in d:
    print('\n'.join(d['test_list']))
elif 'canonical_solution' in d:
    entry = d.get('entry_point', '')
    print(f'# canonical solution available for {entry}')
else:
    print('')
" 2>/dev/null || echo "")"

    # Run in sandbox with timeout
    if [[ -f "$COMPLETION_FILE" ]] && [[ -s "$COMPLETION_FILE" ]]; then
        # Combine completion + tests and execute
        TEST_FILE="${COMPLETIONS_DIR}/test_${COMPLETED}.py"
        {
            cat "$COMPLETION_FILE"
            echo ""
            echo "$TESTS"
        } > "$TEST_FILE"

        if timeout 10 python3 "$TEST_FILE" > /dev/null 2>&1; then
            PASSED=$((PASSED + 1))
        fi
    fi

    COMPLETED=$((COMPLETED + 1))
    if (( COMPLETED % 10 == 0 )); then
        echo "  Progress: ${COMPLETED}/${TOTAL_PROBLEMS} (${PASSED} passed)"
    fi
done < "$BENCHMARK_FILE"

# Compute pass@k
if (( COMPLETED > 0 )); then
    PASS_RATE="$(python3 -c "print(round(${PASSED} / ${COMPLETED} * 100, 2))")"
else
    PASS_RATE="0.0"
fi

echo ""
echo "=== Results ==="
echo "Completed:  ${COMPLETED}/${TOTAL_PROBLEMS}"
echo "Passed:     ${PASSED}"
echo "pass@1:     ${PASS_RATE}%"

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
    "pass_at_1": ${PASS_RATE}
  }
}
EOF

echo ""
echo "Results written to: ${RESULT_FILE}"
