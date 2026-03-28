#!/usr/bin/env bash
# Analyze HumanEval failures across all runs.
# Identifies always-fail, sometimes-fail, and borderline problems.
#
# Usage: scripts/failure-analysis.sh [BENCHMARK]
# Output: Markdown table of problem reliability across runs

set -euo pipefail

BENCHMARK="${1:-humaneval}"
RESULTS_DIR="results"

echo "# ${BENCHMARK^} Failure Analysis"
echo ""

BENCHMARK="$BENCHMARK" RESULTS_DIR="$RESULTS_DIR" python3 << 'PYEOF'
import json, glob, sys, os

benchmark = os.environ.get("BENCHMARK", "humaneval")
results_dir = os.environ.get("RESULTS_DIR", "results")

runs = []
for f in sorted(glob.glob(f"{results_dir}/{benchmark}_*.json")):
    with open(f) as fh:
        d = json.load(fh)
    if "problems" in d and d["results"]["total"] > 0:
        config = d.get("config", {})
        strategy = config.get("prompt_strategy", "standard")
        passed = d["results"]["passed"]
        total = d["results"]["total"]
        name = os.path.basename(f).replace(".json", "")
        runs.append((name, strategy, passed, total, d))

if not runs:
    print("No result files found.")
    sys.exit(0)

print(f"**{len(runs)} runs** across {len(set(r[1] for r in runs))} strategies\n")

# Aggregate per-problem results
all_results = {}
for name, strategy, passed, total, d in runs:
    for p in d["problems"]:
        tid = p["task_id"]
        if tid not in all_results:
            all_results[tid] = {"passes": 0, "total": 0, "strategies": {}}
        all_results[tid]["total"] += 1
        if p["passed"]:
            all_results[tid]["passes"] += 1
        if strategy not in all_results[tid]["strategies"]:
            all_results[tid]["strategies"][strategy] = {"passes": 0, "total": 0}
        all_results[tid]["strategies"][strategy]["total"] += 1
        if p["passed"]:
            all_results[tid]["strategies"][strategy]["passes"] += 1

# Categorize
always_fail = []
always_pass = []
inconsistent = []

for tid, data in sorted(all_results.items()):
    rate = data["passes"] / data["total"]
    if rate == 0:
        always_fail.append((tid, data))
    elif rate == 1:
        always_pass.append((tid, data))
    else:
        inconsistent.append((tid, data, rate))

inconsistent.sort(key=lambda x: x[2])

print(f"| Category | Count | Note |")
print(f"|----------|-------|------|")
print(f"| Always pass | {len(always_pass)} | Reliable across all strategies |")
print(f"| Inconsistent | {len(inconsistent)} | Pass in some runs, fail in others |")
print(f"| Always fail | {len(always_fail)} | Never solved by any strategy |")
print()

if always_fail:
    print("## Always-Fail Problems (model limitation)")
    print()
    print("| Problem | Entry Point | Runs |")
    print("|---------|-------------|------|")
    # Load benchmark for entry points
    bench_file = f"data/benchmarks/{benchmark}.jsonl"
    entry_points = {}
    if os.path.exists(bench_file):
        with open(bench_file) as f:
            for line in f:
                p = json.loads(line)
                entry_points[p["task_id"]] = p.get("entry_point", "?")
    for tid, data in always_fail:
        ep = entry_points.get(tid, "?")
        print(f"| {tid} | `{ep}` | 0/{data['total']} |")
    print()

if inconsistent:
    print("## Inconsistent Problems (borderline)")
    print()
    print("| Problem | Entry Point | Pass Rate | Best Strategy |")
    print("|---------|-------------|-----------|---------------|")
    bench_file = f"data/benchmarks/{benchmark}.jsonl"
    entry_points = {}
    if os.path.exists(bench_file):
        with open(bench_file) as f:
            for line in f:
                p = json.loads(line)
                entry_points[p["task_id"]] = p.get("entry_point", "?")
    for tid, data, rate in inconsistent:
        ep = entry_points.get(tid, "?")
        best = "—"
        best_rate = 0
        for strat, sdata in data["strategies"].items():
            sr = sdata["passes"] / sdata["total"] if sdata["total"] > 0 else 0
            if sr > best_rate:
                best_rate = sr
                best = strat
        print(f"| {tid} | `{ep}` | {data['passes']}/{data['total']} ({rate:.0%}) | {best} ({best_rate:.0%}) |")
    print()

# Summary
print("## Distillation Priority")
print()
print(f"**High priority (always fail, {len(always_fail)} problems):** These need teacher knowledge transfer.")
print(f"**Medium priority (borderline, {len([x for x in inconsistent if x[2] < 0.5])} problems with <50% pass rate):** These need better prompting or slight model improvement.")
print(f"**Low priority (borderline, {len([x for x in inconsistent if x[2] >= 0.5])} problems with >=50% pass rate):** N-sampling can recover these.")
PYEOF
