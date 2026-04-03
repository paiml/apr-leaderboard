# Quality Gates (pmat comply)

Every pipeline step and every commit MUST pass the `pmat comply` quality gates. This is the enforcement mechanism for the claims in this spec.

## 17.1 Specification Compliance

This spec itself is validated by `pmat comply`:

```bash
# Score this specification (must achieve ≥95/100)
pmat spec score docs/specifications/leaderboard-spec.md --verbose

# Extract falsifiable claims and generate review checklist
pmat comply review docs/specifications/leaderboard-spec.md --format markdown

# Full compliance audit with signed evidence
pmat comply audit -o audit.json
```

## 17.2 Mandatory Pre-Commit Checks

```bash
# Full compliance check (blocks commit on failure)
pmat comply check --strict --format json

# Key checks enforced:
#   CB-200  TDG Grade Gate — no function below grade A
#   CB-303  Equation-Driven Development — contract bindings present
#   CB-125  Coverage quality — ≥95% with no exclusion gaming
#   CB-304  Dead code — 0% tolerance
#   CB-120  OIP Tarantula — no NaN, no unwrap in production paths
```

## 17.3 Pipeline Quality Gates

Each recipe step has a `pmat comply` gate:

| Pipeline Step | pmat Gate | Blocks On |
|---|---|---|
| Import | `apr check model.apr` + `pmat comply check` | Format validation failure, contract binding gaps |
| Distill | `pv proof-status` for attention/softmax contracts | Unverified kernel obligations |
| Finetune | `pmat comply check --strict` + coverage ≥95% | TDG regression, coverage drop |
| Merge | `pv audit` for merge strategy contracts | Unbound merge kernel |
| Prune | `apr eval` before/after + `pmat comply baseline` | Quality regression beyond threshold |
| Quantize | `pv proof-status` for Q4K/Q6K contracts | Kani proof failure |
| Eval | `pmat comply review` extracts claims → validates | Untested falsifiable claims |
| Submit | `pmat comply audit` signed evidence | Incomplete audit trail |

## 17.4 Cross-Crate Consistency

The sovereign stack (aprender, entrenar, trueno) MUST maintain cross-crate consistency:

```bash
# Detect API divergence and copy-paste duplication across stack
pmat comply cross-crate \
    --crates ../aprender ../entrenar ../trueno . \
    --similarity-threshold 0.80 \
    --strict

# Verify no contract drift between crates
pv diff ../provable-contracts/contracts/old/ ../provable-contracts/contracts/
```

## 17.5 Documentation Publishing

This specification is published as an [mdBook](https://rust-lang.github.io/mdBook/) via GitHub Actions. On every push to `main` that modifies `docs/` or `book.toml`, the workflow builds and deploys to GitHub Pages at:

> **https://paiml.github.io/apr-leaderboard/**

The mdBook source lives in `docs/src/` with chapters split from the canonical spec at `docs/specifications/leaderboard-spec.md`. The build output (`docs/book/`) is gitignored.

```bash
# Local preview
mdbook serve    # http://localhost:3000

# Build only
mdbook build    # outputs to docs/book/
```

## 17.6 Contract Falsification Gate

`make check-contracts` runs all provable contract falsification tests as a single gate. This is the primary automated quality check for the project.

```bash
make check-contracts    # runs all falsification tests + contract structure validation
```

**Test categories (64/65 passing, 2026-04-03):**

| Category | Count | What it checks |
|---|---|---|
| pass@k estimator | 5 | Chen et al. boundary conditions, monotonicity |
| throughput bounds | 2 | tok/s >= 1.0, TTFT < 500ms |
| benchmark data | 3 | HumanEval/MBPP/BigCodeBench problem counts |
| decontamination | 1 | Zero HE/MBPP prompt overlap |
| eval results | 3 | Best pass@1, run count, latest score |
| distillation | 2 | Teacher > student, category coverage |
| MBPP eval | 1 | Best MBPP pass@1 >= 70% |
| AC-022 gate | 1 | HE >= 85% AND MBPP >= 80% (compound) |
| quantization | 3 | Q4K size, apr check, golden ordering |
| distillation data | 3 | Teacher completions count + JSONL validity |
| oracle analysis | 2 | Oracle upper bound, never-solved count |
| pipeline | 3 | Script count, config count, Make target count |
| compile | 1 | apr compile subcommand available |
| data catalog | 2 | Contract bindings, dataset documentation |
| leaderboard coverage | 2 | Eval run count, benchmark coverage |
| HF parity | 1 | HumanEval gap < 5pp vs HF reference |
| contract coverage | 1 | >= 25 contract YAMLs |
| contract structure | 29 | All YAMLs have metadata/equations/proof_obligations/falsification_tests |

**Single known failure:** FT-GATE-001 (AC-022 compound gate) — MBPP at 76.2% vs 80% target. Closing via PMAT-008 (DPO) + PMAT-007 (distillation).

**pv proof-status:** Validates contract YAML schema via provable-contracts tooling. 28/28 contracts parsed, 98 proof obligations, 10 Kani harnesses. See §16.5.
