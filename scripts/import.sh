#!/usr/bin/env bash
# import.sh - Download a model from HuggingFace and convert to .apr
#
# Usage:
#   ./scripts/import.sh <hf-model-id> [output-path] [--quantize scheme]
#
# Examples:
#   ./scripts/import.sh Qwen/Qwen2.5-Coder-7B-Instruct
#   ./scripts/import.sh Qwen/Qwen2.5-Coder-7B checkpoints/qwen7b.apr
#   ./scripts/import.sh Qwen/Qwen2.5-Coder-7B checkpoints/qwen7b.apr --quantize fp16

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: import.sh <hf-model-id> <output-path> <--quantize scheme>"
    exit 1
fi

MODEL_ID="$1"
NAME_LOWER="$(echo "${MODEL_ID}" | tr '/' '_' | tr 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 'abcdefghijklmnopqrstuvwxyz')"
if [[ $# -ge 2 ]] && [[ "$2" != --* ]]; then
    OUTPUT="$2"
    shift 2
else
    OUTPUT="checkpoints/${NAME_LOWER}.apr"
    shift 1
fi
QUANTIZE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quantize) QUANTIZE_FLAG="--quantize $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== APR Import ==="
echo "Source:  hf://${MODEL_ID}"
echo "Output:  ${OUTPUT}"
echo ""

# Ensure output directory exists
mkdir -p "$(dirname "${OUTPUT}")"

# Check HF Hub reachability
echo -n "Checking HF Hub... "
if curl -sf "https://huggingface.co/api/models/${MODEL_ID}" > /dev/null 2>&1; then
    echo "OK (model exists)"
else
    echo "WARNING: Could not verify model on HF Hub (may still work with cached data)"
fi

# Run import
echo ""
echo "Downloading and converting..."
# shellcheck disable=SC2086
apr import "hf://${MODEL_ID}" -o "${OUTPUT}" ${QUANTIZE_FLAG} --verbose

# Validate the output
if [[ -f "${OUTPUT}" ]]; then
    echo ""
    echo "Validating..."
    if apr check "${OUTPUT}" --json 2>/dev/null; then
        echo "Validation: PASS"
    else
        echo "Validation: check returned warnings (model may still be usable)"
    fi
    echo ""
    SIZE="$(du -h "${OUTPUT}" | cut -f1)"
    echo "Import complete: ${OUTPUT} (${SIZE})"
else
    echo "ERROR: Import failed - output file not created"
    exit 1
fi
