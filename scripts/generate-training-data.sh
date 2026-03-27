#!/usr/bin/env bash
# generate-training-data.sh — Generate synthetic instruction-completion pairs
#
# PMAT-004: Uses a teacher model (32B) to generate high-quality code completions
# for fine-tuning a student model (7B). Follows the phi-1 textbook playbook:
# structured prompts → teacher completions → quality filter → decontaminate.
#
# Usage:
#   ./scripts/generate-training-data.sh TEACHER_MODEL OUTPUT_DIR [NUM_PROMPTS] [MAX_TOKENS]
#
# Example:
#   ./scripts/generate-training-data.sh checkpoints/qwen2.5-coder-32b-instruct-q4km.apr data/synthetic 50 512

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: generate-training-data.sh TEACHER_MODEL OUTPUT_DIR [NUM_PROMPTS] [MAX_TOKENS]"
    exit 1
fi

TEACHER="$1"
OUTPUT_DIR="$2"
NUM_PROMPTS="${3:-50}"
MAX_TOKENS="${4:-512}"

if [[ ! -f "$TEACHER" ]]; then
    echo "ERROR: Teacher model not found: $TEACHER"
    exit 1
fi

command -v apr >/dev/null 2>&1 || { echo "ERROR: apr CLI not found"; exit 1; }
command -v jq  >/dev/null 2>&1 || { echo "ERROR: jq not found"; exit 1; }

mkdir -p "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RAW_FILE="${OUTPUT_DIR}/raw_${TIMESTAMP}.jsonl"
CLEAN_FILE="${OUTPUT_DIR}/instruct_${TIMESTAMP}.jsonl"

echo "=== Synthetic Training Data Generation (PMAT-004) ==="
echo "Teacher:     $TEACHER"
echo "Output:      $OUTPUT_DIR"
echo "Prompts:     $NUM_PROMPTS"
echo "Max tokens:  $MAX_TOKENS"
echo ""

# ── Instruction prompts ──────────────────────────────────────────────────────
# Categories from phi-1 textbook playbook (albor)
CATEGORIES=(
    "string_manipulation"
    "list_operations"
    "dict_and_set"
    "math_and_number"
    "data_structures"
    "algorithms"
    "file_and_io"
    "functional"
    "oop_patterns"
    "testing"
)

PROMPTS_PER_CATEGORY=$(( NUM_PROMPTS / ${#CATEGORIES[@]} ))
if (( PROMPTS_PER_CATEGORY < 1 )); then PROMPTS_PER_CATEGORY=1; fi

# Template prompts per category
generate_prompt() {
    local cat="$1"
    local idx="$2"
    case "$cat" in
        string_manipulation)
            echo "Write a Python function that performs string manipulation. Include type hints, a docstring with examples, and handle edge cases. Function ${idx}: ";;
        list_operations)
            echo "Write a Python function that operates on lists. Include type hints, a docstring, and handle empty lists. Function ${idx}: ";;
        dict_and_set)
            echo "Write a Python function using dictionaries or sets. Include type hints and a clear docstring. Function ${idx}: ";;
        math_and_number)
            echo "Write a Python function for a mathematical operation. Include type hints, docstring, and handle edge cases like division by zero. Function ${idx}: ";;
        data_structures)
            echo "Write a Python class implementing a data structure (stack, queue, linked list, tree, or graph). Include methods with docstrings. Implementation ${idx}: ";;
        algorithms)
            echo "Write a Python function implementing a common algorithm (sorting, searching, dynamic programming). Include time complexity in docstring. Algorithm ${idx}: ";;
        file_and_io)
            echo "Write a Python function for file I/O or string parsing. Include error handling and type hints. Function ${idx}: ";;
        functional)
            echo "Write a Python function using functional programming patterns (map, filter, reduce, comprehensions, generators). Function ${idx}: ";;
        oop_patterns)
            echo "Write a Python class demonstrating an OOP design pattern (factory, observer, decorator, strategy). Pattern ${idx}: ";;
        testing)
            echo "Write a Python function with its corresponding pytest test function. Include edge case tests. Test suite ${idx}: ";;
    esac
}

# ── Phase 1: Generate completions ─────────────────────────────────────────────

echo "Phase 1: Generating ${NUM_PROMPTS} prompts across ${#CATEGORIES[@]} categories..."

BATCH_INPUT="${OUTPUT_DIR}/.batch_input.jsonl"
: > "$BATCH_INPUT"

prompt_count=0
for cat in "${CATEGORIES[@]}"; do
    for idx in $(seq 1 "$PROMPTS_PER_CATEGORY"); do
        prompt="$(generate_prompt "$cat" "$idx")"
        jq -nc --arg prompt "$prompt" --arg task_id "${cat}_${idx}" \
            --argjson max_tokens "$MAX_TOKENS" \
            '{prompt: $prompt, task_id: $task_id, max_tokens: $max_tokens}' >> "$BATCH_INPUT"
        prompt_count=$((prompt_count + 1))
    done
done

echo "  Generated ${prompt_count} prompts"

# ── Phase 2: Run teacher model ────────────────────────────────────────────────

echo "Phase 2: Running teacher model (batch mode)..."
BATCH_START="$(date +%s)"

if ! apr run "$TEACHER" --batch-jsonl "$BATCH_INPUT" --max-tokens "$MAX_TOKENS" \
    --temperature 0.8 --top-k 40 > "${OUTPUT_DIR}/.batch_output.jsonl" 2>/dev/null; then
    echo "  ERROR: Teacher inference failed"
    exit 1
fi

BATCH_END="$(date +%s)"
BATCH_ELAPSED=$(( BATCH_END - BATCH_START ))
OUTPUT_COUNT="$(wc -l < "${OUTPUT_DIR}/.batch_output.jsonl")"
echo "  Completed: ${OUTPUT_COUNT} completions in ${BATCH_ELAPSED}s"

# ── Phase 3: Format as instruction-completion pairs ───────────────────────────

echo "Phase 3: Formatting instruction-completion pairs..."
: > "$RAW_FILE"

while IFS= read -r line; do
    task_id="$(jq -r '.task_id // ""' <<< "$line" 2>/dev/null)"
    text="$(jq -r '.text // ""' <<< "$line" 2>/dev/null)"
    error="$(jq -r '.error // null' <<< "$line" 2>/dev/null)"

    if [[ "$error" != "null" && -n "$error" ]]; then continue; fi
    if [[ -z "$text" || "$text" == "null" ]]; then continue; fi

    # Extract the original prompt from the batch input
    prompt="$(jq -r --arg tid "$task_id" 'select(.task_id == $tid) | .prompt' "$BATCH_INPUT" 2>/dev/null | head -1)"

    # Format as instruction-completion pair
    jq -nc --arg instruction "$prompt" --arg output "$text" --arg category "${task_id%%_*}" \
        '{instruction: $instruction, output: $output, category: $category}' >> "$RAW_FILE"
done < "${OUTPUT_DIR}/.batch_output.jsonl"

RAW_COUNT="$(wc -l < "$RAW_FILE")"
echo "  Raw pairs: ${RAW_COUNT}"

# ── Phase 4: Quality filter (basic) ──────────────────────────────────────────

echo "Phase 4: Quality filtering..."
: > "$CLEAN_FILE"

filtered=0
while IFS= read -r line; do
    output="$(jq -r '.output' <<< "$line")"
    # Basic quality checks
    len=${#output}
    if (( len < 50 )); then continue; fi  # Too short
    if (( len > 5000 )); then continue; fi  # Too long
    if echo "$output" | grep -q '^\s*$'; then continue; fi  # Empty
    if echo "$output" | grep -qE '(Error|error|ERROR|Traceback).*:'; then continue; fi  # Error output
    echo "$line" >> "$CLEAN_FILE"
    filtered=$((filtered + 1))
done < "$RAW_FILE"

echo "  Filtered: ${filtered}/${RAW_COUNT} passed quality gate"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Results ==="
echo "Raw pairs:      ${RAW_COUNT} (${RAW_FILE})"
echo "Clean pairs:    ${filtered} (${CLEAN_FILE})"
echo "Categories:     ${#CATEGORIES[@]}"
echo "Teacher time:   ${BATCH_ELAPSED}s"
echo ""
echo "Next steps:"
echo "  1. Decontaminate: make decontaminate DATA=${CLEAN_FILE}"
echo "  2. Fine-tune:     apr finetune model.apr --method qlora --data ${CLEAN_FILE}"

# Cleanup temp files
rm -f "$BATCH_INPUT" "${OUTPUT_DIR}/.batch_output.jsonl"
