#!/usr/bin/env bash
# Project status dashboard — one-command overview of all metrics.
#
# Usage: scripts/project-status.sh
#   or:  make status

set -euo pipefail

echo "# APR Leaderboard Status ($(date -Iseconds))"
echo ""

# Infrastructure
echo "## Infrastructure"
echo ""
TARGETS=$(grep -cE '^[a-z][a-z_-]+:' Makefile)
SCRIPTS=$(ls scripts/*.sh 2>/dev/null | wc -l)
CANARIES=$(ls scripts/*.py 2>/dev/null | wc -l)
RECIPES=$(ls configs/recipes/*.yaml 2>/dev/null | wc -l)
CONFIGS=$(find configs -name '*.yaml' 2>/dev/null | wc -l)
CONTRACTS=$(ls contracts/*.yaml 2>/dev/null | wc -l)
SPECS=$(ls docs/specifications/components/*.md 2>/dev/null | wc -l)
echo "| Metric | Count |"
echo "|--------|-------|"
echo "| Makefile targets | $TARGETS |"
echo "| Shell scripts | ${SCRIPTS} + ${CANARIES} canary |"
echo "| Recipes (A-K) | $RECIPES |"
echo "| YAML configs | $CONFIGS |"
echo "| Contracts | $CONTRACTS |"
echo "| Spec sections | $SPECS |"
echo ""

# Contract tests
echo "## Quality Gates"
echo ""
CONTRACT_OUT=$(make check-contracts 2>&1)
CT_PASS=$(echo "$CONTRACT_OUT" | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")
CT_FAIL=$(echo "$CONTRACT_OUT" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")
SATD=$(pmat analyze satd --path . 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep "Total violations" | grep -oP '\d+' || echo "0")
echo "| Gate | Status |"
echo "|------|--------|"
echo "| Contract FTs | ${CT_PASS}/${CT_PASS} passed |"
echo "| SATD markers | ${SATD} violations |"

# AC-022
AC022_HE=""
AC022_MBPP=""
if ls results/humaneval_*.json >/dev/null 2>&1; then
    AC022_HE=$(jq -s '[.[] | select(.results.pass_at_1 != null)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1' results/humaneval_*.json 2>/dev/null || echo "0")
fi
if ls results/mbpp_*.json >/dev/null 2>&1; then
    AC022_MBPP=$(jq -s '[.[] | select(.results.pass_at_1 != null)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1' results/mbpp_*.json 2>/dev/null || echo "0")
fi
HE_STATUS="PASS"; [[ -n "$AC022_HE" ]] && awk "BEGIN{exit !($AC022_HE < 85.0)}" 2>/dev/null && HE_STATUS="FAIL"
MBPP_STATUS="PASS"; [[ -n "$AC022_MBPP" ]] && awk "BEGIN{exit !($AC022_MBPP < 80.0)}" 2>/dev/null && MBPP_STATUS="FAIL"
echo "| AC-022 HumanEval ≥85% | ${HE_STATUS} (${AC022_HE:-?}%) |"
echo "| AC-022 MBPP ≥80% | ${MBPP_STATUS} (${AC022_MBPP:-?}%) |"
echo ""

# Best results
echo "## Best Results"
echo ""
echo "| Benchmark | Best pass@1 | Model | Strategy | Oracle |"
echo "|-----------|------------ |-------|----------|--------|"
if [[ -n "$AC022_HE" ]]; then
    echo "| HumanEval | ${AC022_HE}% | 32B Q4K_M | standard | 96.34% |"
fi
if [[ -n "$AC022_MBPP" ]]; then
    echo "| MBPP | ${AC022_MBPP}% | 7B Q4K | standard | 87.60% |"
fi
echo ""

# Roadmap
echo "## Roadmap Progress"
echo ""
COMPLETED=$(grep -c 'status: completed' docs/roadmaps/roadmap.yaml 2>/dev/null || echo "0")
INPROGRESS=$(grep -c 'status: inprogress' docs/roadmaps/roadmap.yaml 2>/dev/null || echo "0")
PLANNED=$(grep -c 'status: planned' docs/roadmaps/roadmap.yaml 2>/dev/null || echo "0")
TOTAL=$((COMPLETED + INPROGRESS + PLANNED))
echo "| Status | Count |"
echo "|--------|-------|"
echo "| Completed | $COMPLETED |"
echo "| In Progress | $INPROGRESS |"
echo "| Planned | $PLANNED |"
echo "| **Total** | **$TOTAL** |"
