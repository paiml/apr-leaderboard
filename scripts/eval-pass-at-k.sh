#!/usr/bin/env bash
# eval-pass-at-k.sh - Evaluate a model on code generation benchmarks
#
# Architecture:
#   Phase 1: Prepare — download benchmark, split into per-problem JSON files
#   Phase 2: Generate — batch mode (single model load via --batch-jsonl) or
#            N parallel workers each claim problems via flock.
#            Batch mode eliminates ~80s per-problem CUDA JIT overhead on gx10.
#   Phase 3: Test — sequential sandbox execution of all completions
#   Phase 4: Score — compute pass@k (Chen et al. unbiased estimator)
#
# Usage:
#   ./scripts/eval-pass-at-k.sh BENCHMARK MODEL_PATH [RESULTS_DIR] [MAX_TOKENS] [TEMPERATURE] [NUM_SAMPLES] [PROMPT_STRATEGY] [WORKERS]
#
# Environment variables:
#   APR_BATCH_MODE=auto|on|off  — batch mode control (default: auto)
#   APR_NO_GPU=0|1              — disable GPU for batch mode (default: 0)

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

# ── Helper functions ─────────────────────────────────────────────────────────

strip_markdown_fences() {
    sed -E '/^```(python|py)?[[:space:]]*$/d' <<< "$1"
}

strip_thinking_tokens() {
    local text="$1"

    # If [151668] (</think>) present, keep only text after it
    if grep -qF '[151668]' <<< "$text"; then
        text="$(awk '/\[151668\]/{found=1; next} found{print}' <<< "$text")"
    # If </think> present (decoded form), keep only text after it
    elif grep -q '</think>' <<< "$text"; then
        text="$(awk '/<\/think>/{found=1; next} found{print}' <<< "$text")"
    # If only [151667] (no end-think), model ran out of tokens during thinking
    elif grep -qF '[151667]' <<< "$text"; then
        text="$(sed '/^\[151667\]/d' <<< "$text")"
        local extracted
        extracted="$(awk '/^```python/,/^```/ { if (!/^```/) print }' <<< "$text")"
        if [[ -n "$extracted" ]]; then
            text="$extracted"
        fi
    fi

    text="$(sed -E 's/\[1516[0-9]{2}\]//g' <<< "$text")"
    echo "$text"
}

extract_python_code() {
    awk '{
        if ($0 ~ /^Human$/ || $0 ~ /^Assistant$/ || $0 ~ /^User$/ || \
            $0 ~ /^\*\*/ || $0 ~ /^###/ || $0 ~ /^---$/) {
            exit
        }
        print
    }' <<< "$1"
}

compute_pass_at_k() {
    local n="$1" c="$2" k="$3"
    awk -v n="$n" -v c="$c" -v k="$k" 'BEGIN {
        if (n - c < k) { print 1.0; exit }
        if (c == 0)    { print 0.0; exit }
        log_ratio = 0
        for (i = 0; i < k; i++) {
            log_ratio += log(n - c - i) - log(n - i)
        }
        printf "%.6f", 1.0 - exp(log_ratio)
    }'
}

build_instruction() {
    local benchmark="$1"
    local prompt="$2"
    local strategy="$3"
    local problem_json="${4:-}"

    local task_desc
    if [[ "$benchmark" == "humaneval" ]]; then
        task_desc="Complete the following Python function."
    elif [[ "$benchmark" == "bigcodebench" ]]; then
        task_desc="Write a Python function to solve this task with all necessary imports."
    elif [[ "$benchmark" == "mbpp" && -n "$problem_json" ]]; then
        local func_name
        func_name="$(jq -r '.test_list[0] // ""' <<< "$problem_json" 2>/dev/null | grep -oP '(?<=assert )\w+' | head -1)"
        if [[ -n "$func_name" ]]; then
            task_desc="Write a Python function called \`${func_name}\` to solve this task."
        else
            task_desc="Write a Python function to solve this task."
        fi
    else
        task_desc="Write a Python function to solve this task."
    fi

    case "$strategy" in
        standard|default)
            printf "%s Return ONLY Python code. No markdown, no explanation.\n\n%s" \
                "$task_desc" "$prompt"
            ;;
        scot|structured-cot)
            printf "Analyze this problem step by step:\n1. Identify the input/output types\n2. Consider edge cases\n3. Design the algorithm\n4. Write the implementation\n\n%s\n\n%s\n\nReturn ONLY the Python code after your analysis. No markdown." \
                "$task_desc" "$prompt"
            ;;
        few-shot|fewshot)
            printf "Here are examples of completing Python functions:\n\nExample 1:\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\"\"\"\n    m = l[0]\n    for e in l:\n        if e > m:\n            m = e\n    return m\n\nExample 2:\ndef count_vowels(s: str) -> int:\n    \"\"\"Count the number of vowels in a string.\"\"\"\n    return sum(1 for c in s.lower() if c in 'aeiou')\n\nExample 3:\ndef flatten(lst):\n    \"\"\"Flatten a nested list into a single list.\"\"\"\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n\nNow complete the following:\n%s\n\n%s\n\nReturn ONLY the Python code. No markdown, no explanation." \
                "$task_desc" "$prompt"
            ;;
        cgo|code-gen-opt)
            printf "Break down the implementation into clear sub-goals, then implement each one.\n\n%s\n\n%s\n\nReturn ONLY Python code. No markdown." \
                "$task_desc" "$prompt"
            ;;
        *)
            printf "%s Return ONLY Python code. No markdown, no explanation.\n\n%s" \
                "$task_desc" "$prompt"
            ;;
    esac
}

# ── Phase 1: Prepare — split benchmark into per-problem files ────────────────

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

# ── Phase 2: Generate — batch mode or parallel workers ───────────────────────

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

# ── Phase 2a: Batch mode (single model load) ────────────────────────────────

generate_batch() {
    local batch_input="${WORK_DIR}/batch_input.jsonl"
    local batch_output="${WORK_DIR}/batch_output.jsonl"

    echo "  Building batch input..."
    local skipped=0
    for problem_idx in $(seq 0 $((TOTAL_PROBLEMS - 1))); do
        local problem_file="${WORK_DIR}/problems/${problem_idx}.json"
        local prompt instruction

        prompt="$(jq -r '.instruct_prompt // .prompt // .text // .instruction // ""' < "$problem_file" 2>/dev/null)"

        if [[ -z "$prompt" || "$prompt" == "null" ]]; then
            echo "SKIP" > "${WORK_DIR}/completions/${problem_idx}.py"
            skipped=$((skipped + 1))
            continue
        fi

        instruction="$(build_instruction "$BENCHMARK" "$prompt" "$PROMPT_STRATEGY" "$(cat "$problem_file")")"

        # task_id = problem index for correlation back to completion files
        jq -nc --arg prompt "$instruction" --arg task_id "${problem_idx}" \
            --argjson max_tokens "$EFFECTIVE_MAX_TOKENS" \
            '{prompt: $prompt, task_id: $task_id, max_tokens: $max_tokens}' >> "$batch_input"
    done

    local input_count
    input_count="$(wc -l < "$batch_input")"
    echo "  Batch input: ${input_count} prompts (${skipped} skipped)"

    # Build apr run flags
    local -a apr_flags=("$MODEL" "--batch-jsonl" "$batch_input" "--max-tokens" "$EFFECTIVE_MAX_TOKENS")
    apr_flags+=("--temperature" "$TEMPERATURE" "--top-k" "1")
    if [[ "${APR_NO_GPU:-0}" == "1" ]]; then
        apr_flags+=("--no-gpu")
    fi
    apr_flags+=("--verbose")

    # Run batch inference — stdout=JSONL results, stderr=progress
    echo "  Running batch inference (model loads once, CUDA JIT amortized)..."
    local batch_start
    batch_start="$(date +%s)"

    if ! apr run "${apr_flags[@]}" > "$batch_output"; then
        echo "  ERROR: Batch inference failed" >&2
        return 1
    fi

    local batch_end
    batch_end="$(date +%s)"
    local batch_elapsed=$(( batch_end - batch_start ))

    if [[ ! -f "$batch_output" || ! -s "$batch_output" ]]; then
        echo "  ERROR: No batch output produced" >&2
        return 1
    fi

    local output_count
    output_count="$(wc -l < "$batch_output")"
    echo "  Batch complete: ${output_count} results in ${batch_elapsed}s"

    # Parse batch output back into per-problem completion files
    local parsed=0 errors=0
    while IFS= read -r line; do
        local task_id text error
        task_id="$(jq -r '.task_id // ""' <<< "$line" 2>/dev/null)"
        error="$(jq -r '.error // null' <<< "$line" 2>/dev/null)"
        text="$(jq -r '.text // ""' <<< "$line" 2>/dev/null)"

        if [[ -z "$task_id" ]]; then continue; fi

        local completion_file="${WORK_DIR}/completions/${task_id}.py"

        if [[ "$error" != "null" && -n "$error" ]]; then
            echo "ERROR" > "$completion_file"
            echo "  problem ${task_id} FAILED: ${error}" >&2
            errors=$((errors + 1))
        elif [[ -z "$text" || "$text" == "null" ]]; then
            echo "ERROR" > "$completion_file"
            errors=$((errors + 1))
        else
            text="$(strip_thinking_tokens "$text")"
            text="$(strip_markdown_fences "$text")"
            text="$(extract_python_code "$text")"
            echo "$text" > "$completion_file"
        fi
        parsed=$((parsed + 1))
    done < "$batch_output"

    echo "  Parsed ${parsed} results (${errors} errors)"
}

# ── Phase 2b: Worker mode (per-problem model load) ──────────────────────────

generate_worker() {
    local worker_id="$1"

    while true; do
        # Claim next problem from queue (atomic via flock)
        local problem_idx
        problem_idx="$(flock "$QUEUE_LOCK" bash -c '
            line="$(head -1 "$1" 2>/dev/null)"
            if [ -n "$line" ]; then
                sed -i "1d" "$1"
                echo "$line"
            fi
        ' -- "$QUEUE_FILE")"
        problem_idx="$(echo "$problem_idx" | tr -d '[:space:]')"

        if [[ -z "$problem_idx" ]]; then
            break  # Queue empty
        fi

        local problem_file="${WORK_DIR}/problems/${problem_idx}.json"
        local raw_file="${WORK_DIR}/raw/${problem_idx}.json"
        local completion_file="${WORK_DIR}/completions/${problem_idx}.py"

        # Read problem
        local task_id prompt instruction
        task_id="$(jq -r '.task_id // .name // "unknown"' < "$problem_file" 2>/dev/null)"
        prompt="$(jq -r '.instruct_prompt // .prompt // .text // .instruction // ""' < "$problem_file" 2>/dev/null)"

        if [[ -z "$prompt" || "$prompt" == "null" ]]; then
            echo "SKIP" > "$completion_file"
            flock "$PROGRESS_LOCK" bash -c 'echo "$(( $(cat "$1") + 1 ))" > "$1"' -- "$PROGRESS_FILE"
            continue
        fi

        instruction="$(build_instruction "$BENCHMARK" "$prompt" "$PROMPT_STRATEGY" "$(cat "$problem_file")")"

        # Generate with scaled timeout
        local stderr_file="${WORK_DIR}/raw/${problem_idx}.stderr"
        if ! timeout "$GENERATION_TIMEOUT" apr run "$MODEL" \
                --prompt "$instruction" \
                --max-tokens "$EFFECTIVE_MAX_TOKENS" \
                --json --chat \
                2>"$stderr_file" > "$raw_file"; then
            echo "ERROR" > "$completion_file"
            echo "  [worker ${worker_id}] problem ${problem_idx} FAILED: $(head -1 "$stderr_file" 2>/dev/null)" >&2
            flock "$PROGRESS_LOCK" bash -c 'echo "$(( $(cat "$1") + 1 ))" > "$1"' -- "$PROGRESS_FILE"
            continue
        fi

        # Extract and clean completion
        local text
        text="$(jq -r '.text // ""' < "$raw_file" 2>/dev/null)"

        if [[ -z "$text" || "$text" == "null" ]]; then
            echo "ERROR" > "$completion_file"
        else
            text="$(strip_thinking_tokens "$text")"
            text="$(strip_markdown_fences "$text")"
            text="$(extract_python_code "$text")"
            echo "$text" > "$completion_file"
        fi

        # Update and report progress
        flock "$PROGRESS_LOCK" bash -c 'echo "$(( $(cat "$1") + 1 ))" > "$1"' -- "$PROGRESS_FILE"
        local done_count
        done_count="$(cat "$PROGRESS_FILE")"
        if (( done_count % 10 == 0 )); then
            echo "  [worker ${worker_id}] ${done_count}/${TOTAL_PROBLEMS} generated"
        fi
    done
}

# ── Phase 2: Dispatch ────────────────────────────────────────────────────────

if (( USE_BATCH )); then
    echo "Phase 2: Generating completions (BATCH mode — single model load, ${EFFECTIVE_MAX_TOKENS} max tokens)..."
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

# ── Phase 3: Test — sequential sandbox execution ─────────────────────────────

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
