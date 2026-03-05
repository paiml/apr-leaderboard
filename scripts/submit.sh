#!/usr/bin/env bash
# submit.sh - Export model and publish to HuggingFace Hub
#
# Usage:
#   ./scripts/submit.sh <model.apr> <hf-repo-id> [results-dir]
#
# Examples:
#   ./scripts/submit.sh checkpoints/qwen-coder-7b.apr paiml/qwen-coder-7b-apr
#   ./scripts/submit.sh checkpoints/model.apr org/model results/

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: submit.sh <model.apr> <hf-repo-id> (results-dir)"
    exit 1
fi
MODEL="$1"
REPO_ID="$2"
RESULTS_DIR="${3:-results}"

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found at ${MODEL}"
    exit 1
fi

MODEL_NAME="$(basename "$MODEL" .apr)"
EXPORT_DIR="$(mktemp -d)"
trap 'rm -rf "${EXPORT_DIR:?}"' EXIT

echo "=== APR Submit Pipeline ==="
echo "Model:   ${MODEL}"
echo "Repo:    ${REPO_ID}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Step 1: Export to SafeTensors
echo "Step 1: Exporting to SafeTensors..."
apr export "$MODEL" \
    --format safetensors \
    --output "$EXPORT_DIR" \
    --verbose

echo ""

# Step 2: Generate model card
echo "Step 2: Generating model card..."
apr eval "$MODEL" --generate-card 2>/dev/null || true

# If apr did not generate a card, create a basic one
if [[ ! -f "${EXPORT_DIR}/README.md" ]]; then
    {
        echo "---"
        echo "license: mit"
        echo "pipeline_tag: text-generation"
        echo "library_name: aprender"
        echo "tags:"
        echo "  - apr"
        echo "  - sovereign-ai"
        echo "  - code-generation"
        echo "---"
        echo ""
        echo "# ${MODEL_NAME}"
        echo ""
        echo "Model exported from the APR sovereign AI stack."
        echo ""
        echo "## Pipeline"
        echo ""
        echo "Built with [apr-leaderboard](https://github.com/paiml/apr-leaderboard) using:"
        echo "- [aprender](https://crates.io/crates/aprender)"
        echo "- [entrenar](https://crates.io/crates/entrenar)"
        echo "- [trueno](https://crates.io/crates/trueno)"
        echo ""
        echo "## Results"
        echo ""
        if ls "${RESULTS_DIR}"/*.json > /dev/null 2>&1; then
            echo "| Benchmark | pass@1 |"
            echo "|-----------|--------|"
            for f in "${RESULTS_DIR}"/*.json; do
                bench="$(jq -r '.benchmark // ""' < "$f" 2>/dev/null || true)"
                score="$(jq -r '.results.pass_at_1 // "N/A"' < "$f" 2>/dev/null || true)"
                if [[ -n "$bench" ]]; then
                    echo "| ${bench} | ${score}% |"
                fi
            done
        else
            echo "No evaluation results available."
        fi
    } > "${EXPORT_DIR}/README.md"
fi

# Copy results into export directory
if ls "${RESULTS_DIR}"/*.json > /dev/null 2>&1; then
    mkdir -p "${EXPORT_DIR}/results"
    cp "${RESULTS_DIR}"/*.json "${EXPORT_DIR}/results/"
fi

echo ""

# Step 3: Publish to HuggingFace
echo "Step 3: Publishing to HuggingFace Hub..."
echo ""

# Dry-run first
echo "Dry-run:"
apr publish "$EXPORT_DIR" "$REPO_ID" \
    --model-name "$MODEL_NAME" \
    --pipeline-tag text-generation \
    --library-name aprender \
    --tags "apr,sovereign-ai,code-generation" \
    --dry-run

echo ""
read -rp "Proceed with upload? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    apr publish "$EXPORT_DIR" "$REPO_ID" \
        --model-name "$MODEL_NAME" \
        --pipeline-tag text-generation \
        --library-name aprender \
        --tags "apr,sovereign-ai,code-generation" \
        --verbose
    echo ""
    echo "Published: https://huggingface.co/${REPO_ID}"
else
    echo "Upload cancelled."
fi

echo ""
echo "=== Submit complete ==="
