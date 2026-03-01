#!/usr/bin/env bash
# pipeline.sh - Run a multi-stage recipe from a TOML config
#
# Reads a recipe TOML config and runs each stage in order via apr CLI.
# Stages: import -> distill -> finetune -> align -> merge -> prune -> quantize -> eval -> submit
#
# Usage:
#   ./scripts/pipeline.sh <recipe.toml>
#   ./scripts/pipeline.sh --plan <recipe.toml>    # dry-run: validate only
#
# Examples:
#   ./scripts/pipeline.sh configs/recipes/recipe-a-quick-lora.toml
#   ./scripts/pipeline.sh --plan configs/recipes/recipe-c-full-pipeline.toml

set -euo pipefail

PLAN_MODE=false
if [[ "${1:-}" == "--plan" ]]; then
    PLAN_MODE=true
    shift
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: pipeline.sh (--plan) <recipe.toml>"
    exit 1
fi
CONFIG="$1"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

MODE_LABEL="EXECUTE"
if $PLAN_MODE; then
    MODE_LABEL="PLAN (dry-run)"
fi

echo "=== APR Pipeline ==="
echo "Config: ${CONFIG}"
echo "Mode:   ${MODE_LABEL}"
echo ""

# Parse TOML config using Python
read_toml() {
    local default="${2:-}"
    python3 -c "
import tomllib, sys, json
with open('${CONFIG}', 'rb') as f:
    config = tomllib.load(f)
path = '${1}'.split('.')
val = config
for p in path:
    if isinstance(val, dict) and p in val:
        val = val[p]
    else:
        val = '${default}'
        break
if isinstance(val, (list, dict)):
    print(json.dumps(val))
else:
    print(val)
" 2>/dev/null
}

has_section() {
    python3 -c "
import tomllib
with open('${CONFIG}', 'rb') as f:
    config = tomllib.load(f)
print('yes' if '${1}' in config else 'no')
" 2>/dev/null
}

# Read top-level config
MODEL_ID="$(read_toml model_id)"
OUTPUT_DIR="$(read_toml output_dir checkpoints)"
LEADERBOARD="$(read_toml leaderboard bigcode)"
SUBMIT="$(read_toml submit false)"
BENCHMARKS="$(read_toml benchmarks)"

if [[ -z "$MODEL_ID" ]]; then
    echo "ERROR: model_id not found in config"
    exit 1
fi

MODEL_NAME="$(echo "${MODEL_ID}" | tr '/' '_' | tr 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 'abcdefghijklmnopqrstuvwxyz')"
CHECKPOINT="${OUTPUT_DIR}/${MODEL_NAME}.apr"
CURRENT="${CHECKPOINT}"

echo "Model:       ${MODEL_ID}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Leaderboard: ${LEADERBOARD}"
echo ""

# Determine pipeline stages
STAGES=()
STAGES+=("import")

[[ "$(has_section distill)" == "yes" ]] && STAGES+=("distill")
[[ "$(has_section finetune)" == "yes" ]] && STAGES+=("finetune")
[[ "$(has_section align)" == "yes" ]] && STAGES+=("align")
[[ "$(has_section merge)" == "yes" ]] && STAGES+=("merge")
[[ "$(has_section prune)" == "yes" ]] && STAGES+=("prune")
[[ "$(has_section quantize)" == "yes" ]] && STAGES+=("quantize")

STAGES+=("eval")

if [[ "$SUBMIT" == "true" ]] || [[ "$SUBMIT" == "True" ]]; then
    STAGES+=("submit")
fi
[[ "$(has_section compile)" == "yes" ]] && STAGES+=("compile")

echo "Pipeline stages: ${STAGES[*]}"
echo ""

if $PLAN_MODE; then
    echo "=== Plan Mode - Commands that would run ==="
    echo ""
    for stage in "${STAGES[@]}"; do
        case "$stage" in
            import)
                echo "[import] apr import hf://${MODEL_ID} -o ${CHECKPOINT}"
                ;;
            distill)
                TEACHER="$(read_toml distill.teacher)"
                STRATEGY="$(read_toml distill.strategy standard)"
                TEMP="$(read_toml distill.temperature 3.0)"
                ALPHA="$(read_toml distill.alpha 0.7)"
                echo "[distill] apr distill ${TEACHER} --student ${CURRENT} --output ${OUTPUT_DIR}/${MODEL_NAME}-distilled.apr --strategy ${STRATEGY} --temperature ${TEMP} --alpha ${ALPHA}"
                ;;
            finetune)
                DATASET="$(read_toml finetune.dataset)"
                METHOD="$(read_toml finetune.method lora)"
                RANK="$(read_toml finetune.rank 16)"
                LR="$(read_toml finetune.lr 0.0002)"
                EPOCHS="$(read_toml finetune.epochs 3)"
                echo "[finetune] apr finetune ${CURRENT} --method ${METHOD} --rank ${RANK} --learning-rate ${LR} --epochs ${EPOCHS} --data ${DATASET} --output ${OUTPUT_DIR}/${MODEL_NAME}-finetuned.apr"
                ;;
            align)
                DATA="$(read_toml align.data)"
                METHOD="$(read_toml align.method dpo)"
                echo "[align] apr finetune ${CURRENT} --method ${METHOD} --data ${DATA} --output ${OUTPUT_DIR}/${MODEL_NAME}-aligned.apr"
                ;;
            merge)
                MODELS="$(read_toml merge.models)"
                STRATEGY="$(read_toml merge.strategy slerp)"
                echo "[merge] apr merge ${CURRENT} ${MODELS} --strategy ${STRATEGY} --output ${OUTPUT_DIR}/${MODEL_NAME}-merged.apr"
                ;;
            prune)
                METHOD="$(read_toml prune.method magnitude)"
                RATIO="$(read_toml prune.target_ratio 0.5)"
                echo "[prune] apr prune ${CURRENT} --method ${METHOD} --target-ratio ${RATIO} --output ${OUTPUT_DIR}/${MODEL_NAME}-pruned.apr"
                ;;
            quantize)
                SCHEME="$(read_toml quantize.scheme int4)"
                echo "[quantize] apr quantize ${CURRENT} --scheme ${SCHEME} --output ${OUTPUT_DIR}/${MODEL_NAME}-${SCHEME}.apr"
                ;;
            eval)
                echo "[eval] ./scripts/eval-pass-at-k.sh <benchmark> ${CURRENT}"
                echo "       Benchmarks: ${BENCHMARKS}"
                ;;
            submit)
                echo "[submit] ./scripts/submit.sh ${CURRENT} <hf-repo>"
                ;;
            compile)
                echo "[compile] apr compile ${CURRENT} --output ${OUTPUT_DIR}/${MODEL_NAME} --release --strip --lto"
                ;;
        esac
    done
    echo ""
    echo "=== Plan complete (no changes made) ==="
    exit 0
fi

# Execute pipeline stages
mkdir -p "$OUTPUT_DIR" results

for stage in "${STAGES[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "Stage: ${stage}"
    echo "----------------------------------------------"

    case "$stage" in
        import)
            apr import "hf://${MODEL_ID}" -o "${CHECKPOINT}" --verbose
            CURRENT="${CHECKPOINT}"
            ;;

        distill)
            TEACHER="$(read_toml distill.teacher)"
            STRATEGY="$(read_toml distill.strategy standard)"
            TEMP="$(read_toml distill.temperature 3.0)"
            ALPHA="$(read_toml distill.alpha 0.7)"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-distilled.apr"
            apr distill "$TEACHER" \
                --student "$CURRENT" \
                --output "$NEXT" \
                --strategy "$STRATEGY" \
                --temperature "$TEMP" \
                --alpha "$ALPHA" \
                --verbose
            CURRENT="$NEXT"
            ;;

        finetune)
            DATASET="$(read_toml finetune.dataset)"
            METHOD="$(read_toml finetune.method lora)"
            RANK="$(read_toml finetune.rank 16)"
            LR="$(read_toml finetune.lr 0.0002)"
            EPOCHS="$(read_toml finetune.epochs 3)"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-finetuned.apr"
            apr finetune "$CURRENT" \
                --method "$METHOD" \
                --rank "$RANK" \
                --learning-rate "$LR" \
                --epochs "$EPOCHS" \
                --data "$DATASET" \
                --output "$NEXT" \
                --verbose
            CURRENT="$NEXT"
            ;;

        align)
            DATA="$(read_toml align.data)"
            METHOD="$(read_toml align.method dpo)"
            EPOCHS="$(read_toml align.epochs 3)"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-aligned.apr"
            # DPO alignment is handled via finetune with method=dpo
            apr finetune "$CURRENT" \
                --method "$METHOD" \
                --data "$DATA" \
                --epochs "$EPOCHS" \
                --output "$NEXT" \
                --verbose
            CURRENT="$NEXT"
            ;;

        merge)
            MODELS_JSON="$(read_toml merge.models)"
            STRATEGY="$(read_toml merge.strategy slerp)"
            BASE_MODEL="$(read_toml merge.base_model "")"
            DENSITY="$(read_toml merge.density 0.2)"
            # Parse JSON array of model paths
            MERGE_FILES="$(echo "$MODELS_JSON" | python3 -c "import sys,json; print(' '.join(json.load(sys.stdin)))" 2>/dev/null || echo "$MODELS_JSON")"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-merged.apr"
            EXTRA_FLAGS=""
            if [[ -n "$BASE_MODEL" ]]; then
                EXTRA_FLAGS="--base-model $BASE_MODEL"
            fi
            # shellcheck disable=SC2086
            apr merge "$CURRENT" $MERGE_FILES \
                --strategy "$STRATEGY" \
                --density "$DENSITY" \
                --output "$NEXT" \
                $EXTRA_FLAGS \
                --verbose
            CURRENT="$NEXT"
            ;;

        prune)
            METHOD="$(read_toml prune.method magnitude)"
            RATIO="$(read_toml prune.target_ratio 0.5)"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-pruned.apr"
            apr prune "$CURRENT" \
                --method "$METHOD" \
                --target-ratio "$RATIO" \
                --output "$NEXT" \
                --verbose
            CURRENT="$NEXT"
            ;;

        quantize)
            SCHEME="$(read_toml quantize.scheme int4)"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-${SCHEME}.apr"
            apr quantize "$CURRENT" \
                --scheme "$SCHEME" \
                --output "$NEXT" \
                --verbose
            CURRENT="$NEXT"
            ;;

        eval)
            # Run each benchmark
            BENCH_LIST="$(echo "$BENCHMARKS" | python3 -c "
import sys, json
for b in json.load(sys.stdin):
    print(b)
" 2>/dev/null || echo "$BENCHMARKS")"
            while IFS= read -r bench; do
                bench="$(echo "$bench" | tr -d '[:space:]')"
                [[ -z "$bench" ]] && continue
                echo ""
                echo "  Evaluating: ${bench}"
                case "$bench" in
                    humaneval|mbpp|bigcodebench)
                        ./scripts/eval-pass-at-k.sh "$bench" "$CURRENT" results/ 512 0.0 1 || echo "  WARNING: ${bench} evaluation failed"
                        ;;
                    *)
                        echo "  SKIP: ${bench} (not yet supported in eval script)"
                        ;;
                esac
            done <<< "$BENCH_LIST"
            ;;

        submit)
            echo "Submit stage - requires HF_REPO environment variable"
            if [[ -n "${HF_REPO:-}" ]]; then
                ./scripts/submit.sh "$CURRENT" "$HF_REPO" results/
            else
                echo "  SKIP: Set HF_REPO to enable submission"
            fi
            ;;

        compile)
            RELEASE="$(read_toml compile.release true)"
            LTO="$(read_toml compile.lto true)"
            OUTPUT_NAME="$(read_toml compile.output "${MODEL_NAME}")"
            COMPILE_FLAGS=""
            if [[ "$RELEASE" == "true" ]] || [[ "$RELEASE" == "True" ]]; then
                COMPILE_FLAGS="--release"
            fi
            if [[ "$LTO" == "true" ]] || [[ "$LTO" == "True" ]]; then
                COMPILE_FLAGS="${COMPILE_FLAGS} --lto"
            fi
            # shellcheck disable=SC2086
            apr compile "$CURRENT" \
                --output "${OUTPUT_DIR}/${OUTPUT_NAME}" \
                --strip \
                $COMPILE_FLAGS \
                --verbose
            ;;
    esac

    echo "  Stage ${stage}: DONE (current model: ${CURRENT})"
done

echo ""
echo "=== Pipeline complete ==="
echo "Final model: ${CURRENT}"
