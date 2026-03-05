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
#   ./scripts/eval-pass-at-k.sh BENCHMARK MODEL_PATH RESULTS_DIR MAX_TOKENS TEMPERATURE NUM_SAMPLES PROMPT_STRATEGY

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: eval-pass-at-k.sh BENCHMARK MODEL_PATH RESULTS_DIR MAX_TOKENS TEMPERATURE NUM_SAMPLES PROMPT_STRATEGY"
    exit 1
fi

BENCHMARK="$1"
MODEL="$2"
RESULTS_DIR="${3:-results}"
MAX_TOKENS="${4:-512}"
TEMPERATURE="${5:-0.0}"
NUM_SAMPLES="${6:-1}"
PROMPT_STRATEGY="${7:-standard}"

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
echo "Strategy:    ${PROMPT_STRATEGY}"
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
    TMPFILE="$(mktemp "${BENCHMARK_DATA_DIR}/.dl.XXXXXX")"
    if [[ "$URL" == *.gz ]]; then
        curl -sfL "$URL" | gunzip > "$TMPFILE"
    else
        curl -sfL "$URL" -o "$TMPFILE"
    fi
    mv "$TMPFILE" "$BENCHMARK_FILE"
    echo "Downloaded: ${BENCHMARK_FILE} ($(wc -l < "$BENCHMARK_FILE") problems)"
fi

TOTAL_PROBLEMS="$(wc -l < "$BENCHMARK_FILE")"
echo "Problems: ${TOTAL_PROBLEMS}"
echo ""

# ── Markdown fence stripping ─────────────────────────────────────────────────

strip_markdown_fences() {
    # Remove ```python ... ``` or ``` ... ``` fences from model output
    sed -E '/^```(python|py)?[[:space:]]*$/d' <<< "$1"
}

# ── Chen et al. unbiased pass@k estimator (§13.1) ───────────────────────────
# pass@k = 1 - C(n-c,k) / C(n,k) where n=total samples, c=correct, k=selected
# Uses log-space to avoid overflow: log(C(a,b)) = sum(log(a-i) - log(i+1), i=0..b-1)

compute_pass_at_k() {
    local n="$1" c="$2" k="$3"
    awk -v n="$n" -v c="$c" -v k="$k" 'BEGIN {
        if (n - c < k) { print 1.0; exit }
        if (c == 0)    { print 0.0; exit }
        # log C(n-c, k) - log C(n, k)
        log_ratio = 0
        for (i = 0; i < k; i++) {
            log_ratio += log(n - c - i) - log(n - i)
        }
        printf "%.6f", 1.0 - exp(log_ratio)
    }'
}

# ── Prompt strategies (§5.6, §13) ────────────────────────────────────────────

build_instruction() {
    local benchmark="$1"
    local prompt="$2"
    local strategy="$3"

    local task_desc
    if [[ "$benchmark" == "humaneval" ]]; then
        task_desc="Complete the following Python function."
    elif [[ "$benchmark" == "bigcodebench" ]]; then
        task_desc="Write a Python function to solve this task with all necessary imports."
    else
        task_desc="Write a Python function to solve this task."
    fi

    case "$strategy" in
        standard|default)
            printf "%s Return ONLY Python code. No markdown, no explanation.\n\n%s" \
                "$task_desc" "$prompt"
            ;;
        scot|structured-cot)
            # Structured Chain-of-Thought: reason step-by-step before coding
            printf "Analyze this problem step by step:\n1. Identify the input/output types\n2. Consider edge cases\n3. Design the algorithm\n4. Write the implementation\n\n%s\n\n%s\n\nReturn ONLY the Python code after your analysis. No markdown." \
                "$task_desc" "$prompt"
            ;;
        few-shot|fewshot)
            # Few-shot: provide example before the task
            printf "Example:\ndef add(a, b):\n    return a + b\n\nNow solve:\n%s\n\n%s\n\nReturn ONLY Python code. No markdown." \
                "$task_desc" "$prompt"
            ;;
        cgo|code-gen-opt)
            # Chain of Grounded Objectives: decompose into sub-goals
            printf "Break down the implementation into clear sub-goals, then implement each one.\n\n%s\n\n%s\n\nReturn ONLY Python code. No markdown." \
                "$task_desc" "$prompt"
            ;;
        *)
            printf "%s Return ONLY Python code. No markdown, no explanation.\n\n%s" \
                "$task_desc" "$prompt"
            ;;
    esac
}

# ── Main evaluation loop ─────────────────────────────────────────────────────

echo "Generating completions and evaluating..."
COMPLETED=0
PASSED=0
ERRORS=0
TOTAL_TOKENS=0
TOTAL_LATENCY=0
# Per-task sample counts for pass@k estimator (n_i, c_i pairs)
TASK_RESULTS_FILE="${WORK_DIR}/task_results.tsv"
: > "$TASK_RESULTS_FILE"

while IFS= read -r line; do
    TASK_ID="$(jq -r '.task_id // .name // "unknown"' <<< "$line" 2>/dev/null)"

    # Extract prompt (BigCodeBench uses instruct_prompt for instruction variant)
    PROMPT="$(jq -r '.instruct_prompt // .prompt // .text // .instruction // ""' <<< "$line" 2>/dev/null)"
    if [[ -z "$PROMPT" || "$PROMPT" == "null" ]]; then
        continue
    fi

    # Save the problem JSON for the assembler
    PROBLEM_FILE="${WORK_DIR}/problem_${COMPLETED}.json"
    echo "$line" > "$PROBLEM_FILE"

    # Build instruction for the chat model
    INSTRUCTION="$(build_instruction "$BENCHMARK" "$PROMPT" "$PROMPT_STRATEGY")"

    # Generate completion(s) and test
    TASK_PASSED=0
    TASK_GENERATED=0
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
        TOTAL_LATENCY="$(awk "BEGIN{print ${TOTAL_LATENCY} + ${LATENCY}}")"

        if [[ -z "$TEXT" || "$TEXT" == "null" ]]; then
            ERRORS=$((ERRORS + 1))
            continue
        fi

        # Strip markdown fences from model output
        TEXT="$(strip_markdown_fences "$TEXT")"
        echo "$TEXT" > "$COMPLETION_FILE"

        # Assemble test file: prompt + completion + test harness
        # HumanEval: .test is a string (check function), .entry_point names the function
        # MBPP: .test_list is a JSON array of assert strings, .test_setup_code may exist
        # BigCodeBench: .test is a string
        ENTRY_POINT="$(jq -r '.entry_point // ""' < "$PROBLEM_FILE" 2>/dev/null)"
        TEST_SETUP="$(jq -r '.test_setup_code // ""' < "$PROBLEM_FILE" 2>/dev/null)"
        HAS_TEST="$(jq -r 'has("test")' < "$PROBLEM_FILE" 2>/dev/null)"
        HAS_TEST_LIST="$(jq -r 'has("test_list")' < "$PROBLEM_FILE" 2>/dev/null)"

        if [[ "$HAS_TEST" == "true" ]]; then
            # HumanEval: prompt is a Python function signature → prepend it
            # BigCodeBench instruct: prompt is natural language → don't prepend
            TEST_CODE="$(jq -r '.test' < "$PROBLEM_FILE" 2>/dev/null)"
            HAS_CODE_PROMPT="$(jq -r 'has("code_prompt")' < "$PROBLEM_FILE" 2>/dev/null)"
            {
                if [[ "$HAS_CODE_PROMPT" == "true" ]]; then
                    # BigCodeBench: completion should be standalone code
                    cat "$COMPLETION_FILE"
                else
                    # HumanEval: prompt is function signature, completion continues it
                    echo "$PROMPT"
                    cat "$COMPLETION_FILE"
                fi
                echo ""
                echo "$TEST_CODE"
                if [[ -n "$ENTRY_POINT" && "$ENTRY_POINT" != "null" && "$HAS_CODE_PROMPT" != "true" ]]; then
                    echo "check(${ENTRY_POINT})"
                fi
            } > "$TEST_FILE"
        elif [[ "$HAS_TEST_LIST" == "true" ]]; then
            # MBPP: .test_list is a JSON array of assert strings
            {
                if [[ -n "$TEST_SETUP" && "$TEST_SETUP" != "null" ]]; then
                    echo "$TEST_SETUP"
                    echo ""
                fi
                cat "$COMPLETION_FILE"
                echo ""
                jq -r '.test_list[]' < "$PROBLEM_FILE" 2>/dev/null
            } > "$TEST_FILE"
        else
            # Fallback: just the completion (no test harness)
            {
                echo "$PROMPT"
                cat "$COMPLETION_FILE"
            } > "$TEST_FILE"
        fi

        TASK_GENERATED=$((TASK_GENERATED + 1))

        # Execute test in sandbox with timeout
        # Uses Python directly if available, Docker sandbox as fallback
        if [[ -f "$TEST_FILE" && -s "$TEST_FILE" ]]; then
            if command -v python3 >/dev/null 2>&1; then
                if timeout 10 python3 "$TEST_FILE" > /dev/null 2>&1; then
                    TASK_PASSED=$((TASK_PASSED + 1))
                fi
            elif command -v docker >/dev/null 2>&1; then
                if timeout 30 docker run --rm --network=none --memory=512m \
                    -v "${TEST_FILE}:/test.py:ro" python:3.11-slim \
                    python3 /test.py > /dev/null 2>&1; then
                    TASK_PASSED=$((TASK_PASSED + 1))
                fi
            fi
        fi
    done

    # Record per-task results for pass@k estimator
    if (( TASK_GENERATED > 0 )); then
        printf "%s\t%s\t%s\n" "$TASK_ID" "$TASK_GENERATED" "$TASK_PASSED" >> "$TASK_RESULTS_FILE"
    fi

    if (( TASK_PASSED > 0 )); then
        PASSED=$((PASSED + 1))
    fi

    COMPLETED=$((COMPLETED + 1))
    if (( COMPLETED % 10 == 0 )); then
        PCT="$(awk "BEGIN{printf \"%.1f\", ${PASSED}/${COMPLETED}*100}")"
        printf "  %s/%s passed=%s (%s%%) errors=%s\n" "$COMPLETED" "$TOTAL_PROBLEMS" "$PASSED" "$PCT" "$ERRORS"
    fi
done < "$BENCHMARK_FILE"

# ── Compute metrics (Chen et al. unbiased estimator) ─────────────────────────

if (( COMPLETED > 0 )); then
    AVG_TOKENS="$(awk "BEGIN{printf \"%.1f\", ${TOTAL_TOKENS}/${COMPLETED}}")"
    AVG_LATENCY="$(awk "BEGIN{printf \"%.1f\", ${TOTAL_LATENCY}/${COMPLETED}}")"

    # Compute pass@k using the unbiased estimator (§13.1)
    # Average per-task pass@k values
    PASS_AT_1="$(awk -F'\t' '{
        n = $2; c = $3; sum += 1
        if (n - c < 1) { p1 += 1.0; next }
        if (c == 0)    { next }
        log_ratio = log(n - c) - log(n)
        p1 += 1.0 - exp(log_ratio)
    } END { printf "%.2f", (sum > 0) ? p1/sum*100 : 0 }' "$TASK_RESULTS_FILE")"

    # pass@10 (only meaningful when NUM_SAMPLES >= 10)
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
    AVG_TOKENS="0.0"
    AVG_LATENCY="0.0"
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
echo "Avg tokens:  ${AVG_TOKENS}"
echo "Avg latency: ${AVG_LATENCY}ms"

# Write result JSON
PASS_AT_10_JSON="${PASS_AT_10}"
cat > "$RESULT_FILE" << EOF
{
  "benchmark": "${BENCHMARK}",
  "model": "${MODEL}",
  "timestamp": "$(date -Iseconds)",
  "config": {
    "max_tokens": ${MAX_TOKENS},
    "temperature": ${TEMPERATURE},
    "num_samples": ${NUM_SAMPLES},
    "prompt_strategy": "${PROMPT_STRATEGY}"
  },
  "results": {
    "total": ${TOTAL_PROBLEMS},
    "completed": ${COMPLETED},
    "passed": ${PASSED},
    "errors": ${ERRORS},
    "pass_at_1": ${PASS_AT_1},
    "pass_at_10": ${PASS_AT_10_JSON},
    "avg_tokens_generated": ${AVG_TOKENS},
    "avg_latency_ms": ${AVG_LATENCY}
  }
}
EOF

echo ""
echo "Results written to: ${RESULT_FILE}"
