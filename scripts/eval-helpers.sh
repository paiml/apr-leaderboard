#!/usr/bin/env bash
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
        task_desc="Complete the following Python function. Return ONLY the function body (no imports, no def line)."
    elif [[ "$benchmark" == "mbpp" && -n "$problem_json" ]]; then
        local func_name test_examples
        func_name="$(jq -r '.test_list[0] // ""' <<< "$problem_json" 2>/dev/null | grep -oP '(?<=assert )\w+' | head -1)"
        # Include test assertions so model knows exact function signature and I/O format
        test_examples="$(jq -r '.test_list[]? // empty' <<< "$problem_json" 2>/dev/null)"
        if [[ -n "$func_name" ]]; then
            task_desc="Write a Python function called \`${func_name}\` to solve this task."
        else
            task_desc="Write a Python function to solve this task."
        fi
        if [[ -n "$test_examples" ]]; then
            prompt="${prompt}

Examples:
${test_examples}"
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
            printf "%s Use helper functions to break the problem into smaller steps.\n\nReturn ONLY Python code. No markdown, no explanation.\n\n%s" \
                "$task_desc" "$prompt"
            ;;
        *)
            printf "%s Return ONLY Python code. No markdown, no explanation.\n\n%s" \
                "$task_desc" "$prompt"
            ;;
    esac
}

# ── Phase 2a: Batch mode (single model load) ────────────────────────────────

generate_batch() {
    local batch_input="${WORK_DIR}/batch_input.jsonl"
    local batch_output="${WORK_DIR}/batch_output.jsonl"

    echo "  Building batch input..."
    local skipped=0
    for problem_idx in $(seq 0 $((TOTAL_PROBLEMS - 1))); do
        local problem_file="${WORK_DIR}/problems/${problem_idx}.json"
        local prompt instruction

        # BigCodeBench: use code_prompt (completion mode) so model sees function name + imports.
        # Tests call task_func() directly — instruct_prompt doesn't mention the function name.
        if [[ "$BENCHMARK" == "bigcodebench" ]]; then
            prompt="$(jq -r '.code_prompt // .instruct_prompt // .prompt // ""' < "$problem_file" 2>/dev/null)"
        else
            prompt="$(jq -r '.instruct_prompt // .prompt // .text // .instruction // ""' < "$problem_file" 2>/dev/null)"
        fi

        if [[ -z "$prompt" || "$prompt" == "null" ]]; then
            echo "SKIP" > "${WORK_DIR}/completions/${problem_idx}.py"
            skipped=$((skipped + 1))
            continue
        fi

        instruction="$(build_instruction "$BENCHMARK" "$prompt" "$PROMPT_STRATEGY" "$(cat "$problem_file")")"

        # PMAT-003: N-sampling — generate NUM_SAMPLES copies per problem
        # task_id format: "{idx}" for n=1, "{idx}_s{sample}" for n>1
        local sample
        for sample in $(seq 0 $((NUM_SAMPLES - 1))); do
            local tid="${problem_idx}"
            if (( NUM_SAMPLES > 1 )); then
                tid="${problem_idx}_s${sample}"
            fi
            jq -nc --arg prompt "$instruction" --arg task_id "$tid" \
                --argjson max_tokens "$EFFECTIVE_MAX_TOKENS" \
                '{prompt: $prompt, task_id: $task_id, max_tokens: $max_tokens}' >> "$batch_input"
        done
    done

    local input_count
    input_count="$(wc -l < "$batch_input")"
    echo "  Batch input: ${input_count} prompts (${skipped} skipped)"

    # Build apr run flags (GPU mandatory -- parity gate blocks if GPU broken)
    local -a apr_flags=("$MODEL" "--batch-jsonl" "$batch_input" "--max-tokens" "$EFFECTIVE_MAX_TOKENS")
    # PMAT-003: Use top-k=40 for temperature>0 sampling (N-sampling pass@k)
    local top_k=1
    if awk "BEGIN{exit !($TEMPERATURE > 0)}" 2>/dev/null; then
        top_k=40
    fi
    apr_flags+=("--temperature" "$TEMPERATURE" "--top-k" "$top_k")
    apr_flags+=("--verbose")

    # Run batch inference --stdout=JSONL results, stderr=progress
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
        # BigCodeBench: use code_prompt (completion mode) — tests call task_func() directly
        if [[ "$BENCHMARK" == "bigcodebench" ]]; then
            prompt="$(jq -r '.code_prompt // .instruct_prompt // .prompt // ""' < "$problem_file" 2>/dev/null)"
        else
            prompt="$(jq -r '.instruct_prompt // .prompt // .text // .instruction // ""' < "$problem_file" 2>/dev/null)"
        fi

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

