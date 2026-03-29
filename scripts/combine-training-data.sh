#!/usr/bin/env bash
# Combine training data sources for QLoRA fine-tuning.
# Merges teacher completions (PMAT-007) with instruct corpus (PMAT-004).
# Deduplicates by instruction text and shuffles.
#
# Usage: scripts/combine-training-data.sh [OUTPUT_FILE]
# Output: JSONL with {"instruction": "...", "response": "..."}

set -euo pipefail

OUTPUT="${1:-data/combined-training.jsonl}"
mkdir -p "$(dirname "$OUTPUT")"

DISTILL_DATA="data/distill/teacher-completions.jsonl"
INSTRUCT_DATA="data/instruct-corpus.jsonl"
INSTRUCT_50="data/instruct-corpus-50.jsonl"

echo "=== Combining Training Data ==="

# Collect available data sources
SOURCES=()
TOTAL=0

if [[ -f "$DISTILL_DATA" ]]; then
    COUNT=$(wc -l < "$DISTILL_DATA")
    echo "  Teacher completions: $COUNT entries"
    SOURCES+=("$DISTILL_DATA")
    TOTAL=$((TOTAL + COUNT))
fi

if [[ -f "$INSTRUCT_DATA" ]]; then
    COUNT=$(wc -l < "$INSTRUCT_DATA")
    echo "  Instruct corpus:    $COUNT entries"
    SOURCES+=("$INSTRUCT_DATA")
    TOTAL=$((TOTAL + COUNT))
elif [[ -f "$INSTRUCT_50" ]]; then
    COUNT=$(wc -l < "$INSTRUCT_50")
    echo "  Instruct corpus (small): $COUNT entries"
    SOURCES+=("$INSTRUCT_50")
    TOTAL=$((TOTAL + COUNT))
fi

if [[ ${#SOURCES[@]} -eq 0 ]]; then
    echo "ERROR: No training data found."
    echo "  Generate teacher completions: make distill-generate"
    echo "  Generate instruct corpus:     make generate-training-data"
    exit 1
fi

echo "  Total pre-dedup: $TOTAL entries"
echo ""

# Merge, normalize field names, deduplicate, shuffle
python3 -c "
import json, sys, hashlib, random

records = {}
sources_count = {}

for source_file in sys.argv[1:]:
    source_name = source_file.split('/')[-1]
    count = 0
    with open(source_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Normalize field names
            instruction = d.get('instruction', d.get('prompt', ''))
            response = d.get('response', d.get('completion', ''))

            if not instruction or not response:
                continue

            # Deduplicate by instruction hash
            key = hashlib.md5(instruction.encode()).hexdigest()
            if key not in records:
                records[key] = {
                    'instruction': instruction,
                    'response': response
                }
                count += 1
    sources_count[source_name] = count

# Shuffle
items = list(records.values())
random.seed(42)
random.shuffle(items)

with open('$OUTPUT', 'w') as out:
    for item in items:
        out.write(json.dumps(item) + '\n')

print(f'Combined: {len(items)} unique entries')
for name, count in sources_count.items():
    print(f'  {name}: {count} contributed')
" "${SOURCES[@]}"

FINAL_COUNT=$(wc -l < "$OUTPUT")
echo ""
echo "Output: $OUTPUT ($FINAL_COUNT entries)"
echo ""
echo "Next: make pipeline RECIPE=recipe-i-humaneval-qlora"
