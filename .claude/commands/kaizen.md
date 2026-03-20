Continuously implement and improve the spec at docs/specifications/ using pmat work items. Push changes frequently (every 1-3 logical edits).

## Process

1. **Check eval status** on gx10 — report progress of any running evals, copy and commit new results if available.

2. **Sweep spec sections** for:
   - Stale data (old numbers, "in progress" notes for completed work, outdated counts)
   - Internal inconsistencies (cross-reference mismatches, conflicting numbers between sections)
   - Missing findings (dogfooding results not reflected in relevant sections)
   - Claims contradicted by measured data

3. **Fix every issue** using five-whys analysis:
   - Identify the symptom
   - Trace to root cause (5 levels deep)
   - Fix the root cause, not just the symptom
   - Document the five-whys in commit messages or spec text where non-obvious

4. **Push frequently** — commit after every 1-3 logical edits with descriptive messages referencing (Refs PMAT-037).

5. **Update README.md** and configs when spec changes affect them.

6. **Report** what was fixed, what evals are running, and what remains.

## Scope

- All 24 spec sections in docs/specifications/components/
- README.md, CLAUDE.md, configs/eval/coding-benchmarks.yaml
- Result files in results/
- Scripts in scripts/ (if eval issues found)

## Do NOT

- Run new evals without asking (they take hours)
- Publish to HuggingFace
- Create branches (work on main only)
