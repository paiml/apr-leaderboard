#!/usr/bin/env bash
# scripts/download-benchmarks.sh — Download evaluation benchmark data
#
# Downloads HumanEval, MBPP, and BigCodeBench problem sets from their
# canonical sources into JSONL format for use by eval scripts and
# decontamination checks.
#
# Usage:
#   ./scripts/download-benchmarks.sh [OUTPUT_DIR]
#
# Refs: Spec §12, §13 (evaluation protocol), AC-016 (decontamination)

set -euo pipefail

OUTPUT_DIR="${1:-data/benchmarks}"
mkdir -p "${OUTPUT_DIR}"

HUMANEVAL_URL='https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz'
MBPP_URL='https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl'

TMP_FILES=()
cleanup() { rm -f "${TMP_FILES[@]}"; }
trap cleanup EXIT

echo "=== Benchmark Data Download ==="
echo ""

# HumanEval (164 problems)
HUMANEVAL_FILE="${OUTPUT_DIR}/humaneval.jsonl"
if [ -f "${HUMANEVAL_FILE}" ]; then
    count="$(wc -l < "${HUMANEVAL_FILE}")"
    printf "  %-20s %s problems (cached)\n" "HumanEval" "${count}"
else
    printf "  %-20s downloading...\n" "HumanEval"
    TMP="$(mktemp)"
    TMP_FILES+=("${TMP}" "${TMP}.gz")
    if curl -sfL "${HUMANEVAL_URL}" -o "${TMP}.gz"; then
        gunzip -c "${TMP}.gz" > "${HUMANEVAL_FILE}"
        count="$(wc -l < "${HUMANEVAL_FILE}")"
        printf "  %-20s %s problems\n" "HumanEval" "${count}"
    else
        printf "  %-20s FAILED (check network)\n" "HumanEval"
    fi
fi

# MBPP (974 problems, test split is 500)
MBPP_FILE="${OUTPUT_DIR}/mbpp.jsonl"
if [ -f "${MBPP_FILE}" ]; then
    count="$(wc -l < "${MBPP_FILE}")"
    printf "  %-20s %s problems (cached)\n" "MBPP" "${count}"
else
    printf "  %-20s downloading...\n" "MBPP"
    TMP="$(mktemp)"
    TMP_FILES+=("${TMP}")
    if curl -sfL "${MBPP_URL}" -o "${TMP}"; then
        mv -- "${TMP}" "${MBPP_FILE}"
        count="$(wc -l < "${MBPP_FILE}")"
        printf "  %-20s %s problems\n" "MBPP" "${count}"
    else
        printf "  %-20s FAILED (check network)\n" "MBPP"
    fi
fi

# BigCodeBench (download from HuggingFace datasets)
BCB_FILE="${OUTPUT_DIR}/bigcodebench.jsonl"
if [ -f "${BCB_FILE}" ]; then
    count="$(wc -l < "${BCB_FILE}")"
    printf "  %-20s %s problems (cached)\n" "BigCodeBench" "${count}"
else
    printf "  %-20s skipped (manual download from HF datasets)\n" "BigCodeBench"
fi

echo ""
echo "Benchmark data stored in: ${OUTPUT_DIR}"
total_files="$(find "${OUTPUT_DIR}" -name '*.jsonl' | wc -l)"
echo "Total benchmark files: ${total_files}"
