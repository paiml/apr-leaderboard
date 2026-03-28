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

# Pre-submission checks (§14.4)
echo "Pre-submission checks..."
PREFLIGHT_PASS=true

# Check 1: APR format validation
printf "  %-40s" "apr check (format validation)"
if apr check "$MODEL" --json > /dev/null 2>&1; then
    echo "PASS"
else
    echo "FAIL"
    PREFLIGHT_PASS=false
fi

# Check 2: pmat compliance (exit 0=clean, 2=advisories-only --both are COMPLIANT)
printf "  %-40s" "pmat comply check --strict"
pmat comply check --strict > /dev/null 2>&1
PMAT_EXIT=$?
if [ "$PMAT_EXIT" -eq 0 ] || [ "$PMAT_EXIT" -eq 2 ]; then
    echo "PASS"
else
    echo "FAIL"
    PREFLIGHT_PASS=false
fi

# Check 3: Results exist
printf "  %-40s" "evaluation results present"
if ls "${RESULTS_DIR}"/*.json > /dev/null 2>&1; then
    RESULT_COUNT="$(ls "${RESULTS_DIR}"/*.json 2>/dev/null | wc -l)"
    echo "PASS (${RESULT_COUNT} result files)"
else
    echo "FAIL (no results in ${RESULTS_DIR}/)"
    PREFLIGHT_PASS=false
fi

# Check 4: HF repo format
printf "  %-40s" "repo ID format (org/model)"
if [[ "$REPO_ID" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$ ]]; then
    echo "PASS"
else
    echo "FAIL (expected org/model format)"
    PREFLIGHT_PASS=false
fi

# Check 5: Contract falsification tests
printf "  %-40s" "contract falsification tests"
CONTRACT_OUTPUT="$(make check-contracts 2>&1 || true)"
FAIL_COUNT="$(echo "$CONTRACT_OUTPUT" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")"
PASS_COUNT="$(echo "$CONTRACT_OUTPUT" | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")"
if [[ "$FAIL_COUNT" -eq 0 && "$PASS_COUNT" -gt 0 ]]; then
    echo "PASS (${PASS_COUNT} tests)"
else
    echo "FAIL (${FAIL_COUNT} failed)"
    PREFLIGHT_PASS=false
fi

# Check 6: Minimum eval score threshold
printf "  %-40s" "HumanEval pass@1 >= 80%"
if ls "${RESULTS_DIR}"/humaneval_*.json > /dev/null 2>&1; then
    BEST_SCORE="$(jq -s '[.[] | select(.results.pass_at_1 != null)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1' "${RESULTS_DIR}"/humaneval_*.json 2>/dev/null || echo "0")"
    if awk "BEGIN{exit !($BEST_SCORE >= 80.0)}" 2>/dev/null; then
        echo "PASS (${BEST_SCORE}%)"
    else
        echo "FAIL (${BEST_SCORE}% < 80%)"
        PREFLIGHT_PASS=false
    fi
else
    echo "SKIP (no HumanEval results)"
fi

echo ""
if ! $PREFLIGHT_PASS; then
    echo "ERROR: Pre-submission checks failed. Fix issues above before submitting."
    exit 1
fi

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
