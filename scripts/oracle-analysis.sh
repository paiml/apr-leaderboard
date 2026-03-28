#!/usr/bin/env bash
# Oracle analysis: compute upper bound if best strategy were selected per problem.
# Shows the theoretical maximum from strategy ensembling.
#
# Usage: scripts/oracle-analysis.sh [BENCHMARK]
# Output: Oracle pass@1 and strategy breakdown

set -euo pipefail

BENCHMARK="${1:-humaneval}"
RESULTS_DIR="${RESULTS_DIR:-results}"

echo "# ${BENCHMARK^} Oracle Analysis"
echo ""

python3 << 'PYEOF'
import json, glob, sys, os

benchmark = sys.argv[1] if len(sys.argv) > 1 else "humaneval"
results_dir = os.environ.get("RESULTS_DIR", "results")

runs = []
for f in sorted(glob.glob(f"{results_dir}/{benchmark}_*.json")):
    with open(f) as fh:
        d = json.load(fh)
    if "problems" in d and d["results"]["total"] > 0:
        config = d.get("config", {})
        strategy = config.get("prompt_strategy", "standard")
        model = os.path.basename(d.get("model", "unknown"))
        runs.append((os.path.basename(f), strategy, model, d))

if not runs:
    print("No result files found.")
    sys.exit(0)

# Per-problem: did ANY run solve it?
all_results = {}
total_problems = 0
for name, strategy, model, d in runs:
    total_problems = d["results"]["total"]
    for p in d["problems"]:
        tid = p["task_id"]
        if tid not in all_results:
            all_results[tid] = {}
        key = f"{strategy}"
        if key not in all_results[tid]:
            all_results[tid][key] = 0
        all_results[tid][key] = max(all_results[tid][key], p["passed"])

# Oracle: solved if ANY strategy solved it
oracle_solved = sum(1 for tid, strats in all_results.items()
                    if any(v == 1 for v in strats.values()))
oracle_pct = 100 * oracle_solved / total_problems

# Per-strategy best
strategies = set()
for strats in all_results.values():
    strategies.update(strats.keys())

print(f"**Oracle (best per problem): {oracle_solved}/{total_problems} = {oracle_pct:.2f}%**\n")
print("| Strategy | Solved | pass@1 | Unique Wins |")
print("|----------|--------|--------|-------------|")

for strat in sorted(strategies):
    solved = sum(1 for tid, strats in all_results.items()
                 if strats.get(strat, 0) == 1)
    pct = 100 * solved / total_problems
    # Unique wins: problems ONLY this strategy solves
    unique = sum(1 for tid, strats in all_results.items()
                 if strats.get(strat, 0) == 1 and
                 all(v == 0 for k, v in strats.items() if k != strat))
    print(f"| {strat} | {solved}/{total_problems} | {pct:.2f}% | {unique} |")

print()

# Problems no strategy solves
never_solved = [tid for tid, strats in sorted(all_results.items())
                if all(v == 0 for v in strats.values())]
print(f"**Never solved ({len(never_solved)} problems):** {', '.join(never_solved)}")
print()
print(f"**Gap to oracle: {oracle_pct - max(100*sum(1 for tid,strats in all_results.items() if strats.get(s,0)==1)/total_problems for s in strategies):.2f}pp** — recoverable via strategy routing or N-sampling.")
PYEOF
