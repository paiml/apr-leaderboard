#!/usr/bin/env bash
# Prepare calibration data for Wanda/SparseGPT pruning.
# Generates 128 random code samples from the instruct corpus.
# Per §12.0.1: calibration data is 128 random code samples.
#
# Usage: scripts/prepare-calibration-data.sh [OUTPUT] [N_SAMPLES]
# Output: JSONL with {"text": "..."} entries

set -euo pipefail

OUTPUT="${1:-data/calibration.jsonl}"
N_SAMPLES="${2:-128}"
SOURCE="data/instruct-corpus.jsonl"

mkdir -p "$(dirname "$OUTPUT")"

if [[ ! -f "$SOURCE" ]]; then
    echo "ERROR: Source corpus not found: $SOURCE"
    echo "  Run: make prep-data"
    exit 1
fi

TOTAL=$(wc -l < "$SOURCE")
echo "Preparing calibration data..."
echo "  Source: $SOURCE ($TOTAL entries)"
echo "  Samples: $N_SAMPLES"

python3 -c "
import json, random

random.seed(42)
with open('$SOURCE') as f:
    lines = f.readlines()

# Sample N random entries
indices = random.sample(range(len(lines)), min($N_SAMPLES, len(lines)))
with open('$OUTPUT', 'w') as out:
    for i in sorted(indices):
        d = json.loads(lines[i])
        # Calibration data uses raw text (response field)
        text = d.get('response', d.get('completion', ''))
        if text:
            out.write(json.dumps({'text': text}) + '\n')

written = sum(1 for _ in open('$OUTPUT'))
print(f'  Written: {written} calibration samples → $OUTPUT')
"
