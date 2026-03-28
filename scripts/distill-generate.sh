#!/usr/bin/env bash
# Stage 1 of text-based distillation: generate teacher completions.
# Uses 32B teacher model via batch inference to generate high-quality
# coding completions from distillation prompts.
#
# This is the working path — uses apr run --batch-jsonl (proven reliable)
# instead of apr distill --stage generate (which requires apr serve,
# not yet working for .apr files per §19.6.3).
#
# Usage:
#   scripts/distill-generate.sh [CONFIG_FILE]
#   make distill-generate
#
# Environment:
#   APR_CLI     — path to apr binary (default: apr)
#   APR_NO_GPU  — set to 1 to force CPU (default: 0)

set -euo pipefail

CONFIG="${1:-configs/distill/distill-32b-7b-text.yaml}"

# Parse YAML config (bash-native, no python)
parse_yaml_value() {
    grep "^[[:space:]]*${1}:" "$CONFIG" | head -1 | sed 's/.*:[[:space:]]*//' | tr -d '"'
}

TEACHER=$(parse_yaml_value "model")
PROMPTS=$(parse_yaml_value "prompts")
OUTPUT=$(parse_yaml_value "output")
MAX_TOKENS=$(parse_yaml_value "max_tokens")
TEMPERATURE=$(parse_yaml_value "temperature")

APR="${APR_CLI:-apr}"
APR_NO_GPU="${APR_NO_GPU:-0}"

echo "=== Text-Based Distillation: Stage 1 (Generate) ==="
echo "  Teacher:     $TEACHER"
echo "  Prompts:     $PROMPTS"
echo "  Output:      $OUTPUT"
echo "  Max tokens:  $MAX_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo ""

# Validate prerequisites
if [[ ! -f "$TEACHER" ]]; then
    echo "ERROR: Teacher checkpoint not found: $TEACHER"
    echo "  Run: make import MODEL=Qwen/Qwen2.5-Coder-32B-Instruct QUANTIZE=q4km"
    exit 1
fi

if [[ ! -f "$PROMPTS" ]]; then
    echo "ERROR: Prompts file not found: $PROMPTS"
    echo "  Run: scripts/generate-distill-prompts.sh"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

# Convert prompts to batch JSONL format for apr run --batch-jsonl
BATCH_INPUT=$(mktemp /tmp/distill-batch-XXXXXX.jsonl)
trap 'rm -f "$BATCH_INPUT"' EXIT

python3 -c "
import json, sys
with open('$PROMPTS') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        # Wrap prompt in chat template for instruct model
        prompt = d['prompt']
        chat_prompt = f'<|im_start|>system\nYou are a Python programming expert. Write clean, correct, well-documented Python code.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
        batch_entry = {
            'id': f'distill-{i:04d}',
            'prompt': chat_prompt,
            'metadata': {
                'source': d.get('source', ''),
                'kind': d.get('kind', ''),
                'original_prompt': prompt
            }
        }
        print(json.dumps(batch_entry))
" > "$BATCH_INPUT"

PROMPT_COUNT=$(wc -l < "$BATCH_INPUT")
echo "Prepared $PROMPT_COUNT prompts for batch inference"

# Determine GPU flag
GPU_FLAG=""
if [[ "$APR_NO_GPU" != "1" ]]; then
    GPU_FLAG="--gpu"
fi

# Run batch inference with teacher model
echo ""
echo "Starting 32B teacher batch inference..."
echo "  This may take 1-3 hours depending on hardware."
echo ""

BATCH_OUTPUT=$(mktemp /tmp/distill-output-XXXXXX.jsonl)

"$APR" run "$TEACHER" \
    --batch-jsonl "$BATCH_INPUT" \
    --max-tokens "${MAX_TOKENS:-512}" \
    --temperature "${TEMPERATURE:-0.8}" \
    --top-k 40 \
    $GPU_FLAG \
    --verbose \
    > "$BATCH_OUTPUT" 2>&1 || {
    echo "ERROR: Batch inference failed"
    cat "$BATCH_OUTPUT"
    rm -f "$BATCH_OUTPUT"
    exit 1
}

# Convert batch output to instruction-completion JSONL format
python3 -c "
import json, sys

prompts = {}
with open('$PROMPTS') as f:
    for i, line in enumerate(f):
        prompts[f'distill-{i:04d}'] = json.loads(line)

# Parse batch output (may be mixed with log lines)
results = []
with open('$BATCH_OUTPUT') as f:
    for line in f:
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        try:
            d = json.loads(line)
            if 'id' in d and 'text' in d:
                results.append(d)
        except json.JSONDecodeError:
            continue

written = 0
min_tokens = 10
with open('$OUTPUT', 'w') as out:
    for r in results:
        rid = r.get('id', '')
        text = r.get('text', '').strip()
        tokens = r.get('num_generated', len(text.split()))

        if tokens < min_tokens:
            continue

        prompt_data = prompts.get(rid, {})
        record = {
            'instruction': prompt_data.get('prompt', ''),
            'response': text,
            'tokens': tokens,
            'metadata': {
                'source': prompt_data.get('source', 'distill-32b'),
                'kind': prompt_data.get('kind', ''),
                'teacher': 'qwen2.5-coder-32b-instruct-q4km'
            }
        }
        out.write(json.dumps(record) + '\n')
        written += 1

print(f'Wrote {written}/{len(results)} completions to $OUTPUT')
print(f'Filtered {len(results) - written} short completions (< {min_tokens} tokens)')
"

rm -f "$BATCH_OUTPUT"

# Report
FINAL_COUNT=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
echo ""
echo "=== Distillation Generation Complete ==="
echo "  Output: $OUTPUT"
echo "  Completions: $FINAL_COUNT"
echo ""
echo "Next step: fine-tune 7B student"
echo "  make distill-finetune"
