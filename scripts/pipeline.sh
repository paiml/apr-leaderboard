#!/usr/bin/env bash
# pipeline.sh - Run a multi-stage recipe from a YAML config
#
# Reads a recipe YAML config and runs each stage in order via apr CLI.
# Stages: import -> distill -> finetune -> align -> merge -> prune -> quantize -> eval -> submit
#
# Zero Python. Config parsing uses bash-native YAML extraction.
#
# Usage:
#   ./scripts/pipeline.sh <recipe.yaml>
#   ./scripts/pipeline.sh --plan <recipe.yaml>    # dry-run: validate only
#
# Examples:
#   ./scripts/pipeline.sh configs/recipes/recipe-a-quick-lora.yaml
#   ./scripts/pipeline.sh --plan configs/recipes/recipe-c-full-pipeline.yaml

set -euo pipefail

PLAN_MODE=false
if [[ "${1:-}" == "--plan" ]]; then
    PLAN_MODE=true
    shift
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: pipeline.sh (--plan) <recipe.yaml>"
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

# ── Bash-native YAML reader ──────────────────────────────────────────────────
# Handles flat and one-level-nested YAML keys. No python3 dependency.
# Supports: scalars, quoted strings, simple lists (- item).
# Does NOT handle multi-line values, anchors, or complex nesting.

read_yaml() {
    local key="$1"
    local default="${2:-}"
    local result=""

    if [[ "$key" == *.* ]]; then
        # Nested key: "finetune.method" → find [finetune:] section, then [method:]
        local section="${key%%.*}"
        local field="${key#*.}"
        local in_section=false

        while IFS= read -r line; do
            # Skip comments and blank lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// /}" ]] && continue

            # Check if we entered the target section
            if [[ "$line" =~ ^${section}:[[:space:]]*$ ]]; then
                in_section=true
                continue
            fi

            # Check if we left the section (non-indented line that's a new key)
            if $in_section && [[ "$line" =~ ^[a-zA-Z_] ]]; then
                in_section=false
                continue
            fi

            # Read field within section
            if $in_section && [[ "$line" =~ ^[[:space:]]+${field}:[[:space:]]*(.*) ]]; then
                result="${BASH_REMATCH[1]}"
                # Strip quotes
                result="${result#\"}"
                result="${result%\"}"
                result="${result#\'}"
                result="${result%\'}"
                # Strip inline comments
                result="${result%%[[:space:]]#*}"
                break
            fi
        done < "$CONFIG"
    else
        # Top-level key
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// /}" ]] && continue

            if [[ "$line" =~ ^${key}:[[:space:]]*(.*) ]]; then
                result="${BASH_REMATCH[1]}"
                result="${result#\"}"
                result="${result%\"}"
                result="${result#\'}"
                result="${result%\'}"
                result="${result%%[[:space:]]#*}"
                break
            fi
        done < "$CONFIG"
    fi

    if [[ -z "$result" ]]; then
        echo "$default"
    else
        echo "$result"
    fi
}

# Read a YAML list (returns newline-separated items)
read_yaml_list() {
    local key="$1"
    local in_section=false
    local is_nested=false
    local section=""
    local field=""

    if [[ "$key" == *.* ]]; then
        is_nested=true
        section="${key%%.*}"
        field="${key#*.}"
    fi

    local in_list=false
    local in_target_section=false

    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue

        if $is_nested; then
            # Find section first
            if [[ "$line" =~ ^${section}:[[:space:]]*$ ]]; then
                in_target_section=true
                continue
            fi
            if $in_target_section && [[ "$line" =~ ^[a-zA-Z_] ]]; then
                in_target_section=false
                in_list=false
                continue
            fi
            if $in_target_section && [[ "$line" =~ ^[[:space:]]+${field}:[[:space:]]*$ ]]; then
                in_list=true
                continue
            fi
        else
            if [[ "$line" =~ ^${key}:[[:space:]]*$ ]]; then
                in_list=true
                continue
            fi
            # Left the list (new top-level key)
            if $in_list && [[ "$line" =~ ^[a-zA-Z_] ]]; then
                break
            fi
        fi

        if $in_list; then
            if [[ "$line" =~ ^[[:space:]]+-[[:space:]]*(.*) ]]; then
                local item="${BASH_REMATCH[1]}"
                item="${item#\"}"
                item="${item%\"}"
                item="${item#\'}"
                item="${item%\'}"
                echo "$item"
            elif [[ "$line" =~ ^[a-zA-Z_] ]]; then
                break
            fi
        fi
    done < "$CONFIG"
}

# Check if a section exists
has_section() {
    grep -qE "^${1}:" "$CONFIG" 2>/dev/null && echo "yes" || echo "no"
}

# ── Read config ──────────────────────────────────────────────────────────────

MODEL_ID="$(read_yaml model.id)"
OUTPUT_DIR="$(read_yaml model.output_dir checkpoints)"
LEADERBOARD="$(read_yaml model.leaderboard bigcode)"
SUBMIT="$(read_yaml model.submit false)"

if [[ -z "$MODEL_ID" ]]; then
    echo "ERROR: model.id not found in config"
    exit 1
fi

MODEL_NAME="$(echo "${MODEL_ID}" | tr '/' '_' | tr 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 'abcdefghijklmnopqrstuvwxyz')"
CHECKPOINT="${OUTPUT_DIR}/${MODEL_NAME}.apr"
CURRENT="${CHECKPOINT}"

echo "Model:       ${MODEL_ID}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Leaderboard: ${LEADERBOARD}"
echo ""

# ── Determine pipeline stages ────────────────────────────────────────────────

STAGES=()
EXPLICIT_STAGES=()
while IFS= read -r stage; do
    [[ -n "$stage" ]] && EXPLICIT_STAGES+=("$stage")
done < <(read_yaml_list stages)

if [[ ${#EXPLICIT_STAGES[@]} -gt 0 ]]; then
    STAGES=("${EXPLICIT_STAGES[@]}")
else
    # Infer stages from config sections
    STAGES+=("import")
    [[ "$(has_section distill)" == "yes" ]] && STAGES+=("distill")
    [[ "$(has_section finetune)" == "yes" ]] && STAGES+=("finetune")
    [[ "$(has_section align)" == "yes" ]] && STAGES+=("align")
    [[ "$(has_section merge)" == "yes" ]] && STAGES+=("merge")
    [[ "$(has_section prune)" == "yes" ]] && STAGES+=("prune")
    [[ "$(has_section quantize)" == "yes" ]] && STAGES+=("quantize")
    STAGES+=("eval")
    if [[ "$SUBMIT" == "true" ]]; then
        STAGES+=("submit")
    fi
    [[ "$(has_section compile)" == "yes" ]] && STAGES+=("compile")
fi

echo "Pipeline stages: ${STAGES[*]}"
echo ""

# ── Validate ordering (§10 golden ordering: distill→finetune→merge→prune→quantize)
saw_ft=false prev_stage=""
for s in "${STAGES[@]}"; do
    [[ "$s" == "finetune" ]] && saw_ft=true
    [[ "$s" == "merge" && "$saw_ft" == "false" ]] && echo "WARNING: Merge without finetune: merging untrained variants is suboptimal."
    [[ "$s" == "finetune" && "$prev_stage" == "prune" ]] && echo "WARNING: Finetune after prune is an anti-pattern."
    [[ "$s" == "distill" && "$prev_stage" == "finetune" ]] && echo "WARNING: Distill after finetune overwrites fine-tuned specialization."
    prev_stage="$s"
done

# ── Read benchmarks ──────────────────────────────────────────────────────────

BENCH_LIST=()
while IFS= read -r bench; do
    [[ -n "$bench" ]] && BENCH_LIST+=("$bench")
done < <(read_yaml_list benchmarks)

# ── Plan mode ────────────────────────────────────────────────────────────────

if $PLAN_MODE; then
    echo "=== Plan Mode - Commands that would run ==="
    echo ""
    for stage in "${STAGES[@]}"; do
        case "$stage" in
            import)
                echo "[import] apr import hf://${MODEL_ID} -o ${CHECKPOINT}"
                ;;
            distill)
                TEACHER="$(read_yaml distill.teacher)"
                STRATEGY="$(read_yaml distill.strategy standard)"
                TEMP="$(read_yaml distill.temperature 3.0)"
                ALPHA="$(read_yaml distill.alpha 0.7)"
                echo "[distill] apr distill ${TEACHER} --student ${CURRENT} --output ${OUTPUT_DIR}/${MODEL_NAME}-distilled.apr --strategy ${STRATEGY} --temperature ${TEMP} --alpha ${ALPHA}"
                ;;
            finetune)
                DATASET="$(read_yaml finetune.dataset)"
                METHOD="$(read_yaml finetune.method lora)"
                RANK="$(read_yaml finetune.rank 16)"
                LR="$(read_yaml finetune.lr 0.0002)"
                EPOCHS="$(read_yaml finetune.epochs 3)"
                QUANTIZE_NF4="$(read_yaml finetune.quantize_nf4 false)"
                MAX_SEQ_LEN="$(read_yaml finetune.max_seq_len)"
                VRAM="$(read_yaml finetune.vram)"
                TASK="$(read_yaml finetune.task)"
                MODEL_SIZE="$(read_yaml finetune.model_size)"
                WAIT_GPU="$(read_yaml finetune.wait_gpu 0)"
                EXTRA_FLAGS=""
                [[ "$QUANTIZE_NF4" == "true" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --quantize-nf4"
                [[ -n "$MAX_SEQ_LEN" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --max-seq-len ${MAX_SEQ_LEN}"
                [[ -n "$VRAM" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --vram ${VRAM}"
                [[ -n "$TASK" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --task ${TASK}"
                [[ -n "$MODEL_SIZE" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --model-size ${MODEL_SIZE}"
                [[ "$WAIT_GPU" != "0" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --wait-gpu ${WAIT_GPU}"
                echo "[finetune] apr finetune ${CURRENT} --method ${METHOD} --rank ${RANK} --learning-rate ${LR} --epochs ${EPOCHS} --data ${DATASET} --output ${OUTPUT_DIR}/${MODEL_NAME}-finetuned.apr${EXTRA_FLAGS}"
                ;;
            align)
                DATA="$(read_yaml align.data)"
                METHOD="$(read_yaml align.method dpo)"
                echo "[align] apr finetune ${CURRENT} --method ${METHOD} --data ${DATA} --output ${OUTPUT_DIR}/${MODEL_NAME}-aligned.apr"
                ;;
            merge)
                STRATEGY="$(read_yaml merge.strategy slerp)"
                echo "[merge] apr merge ${CURRENT} --strategy ${STRATEGY} --output ${OUTPUT_DIR}/${MODEL_NAME}-merged.apr"
                ;;
            prune)
                METHOD="$(read_yaml prune.method magnitude)"
                RATIO="$(read_yaml prune.target_ratio 0.5)"
                echo "[prune] apr prune ${CURRENT} --method ${METHOD} --target-ratio ${RATIO} --output ${OUTPUT_DIR}/${MODEL_NAME}-pruned.apr"
                ;;
            quantize)
                SCHEME="$(read_yaml quantize.scheme int4)"
                echo "[quantize] apr quantize ${CURRENT} --scheme ${SCHEME} --output ${OUTPUT_DIR}/${MODEL_NAME}-${SCHEME}.apr"
                ;;
            prep-data) echo "[prep-data] apr data audit data/instruct-corpus.jsonl" ;;
            eval)      echo "[eval] ./scripts/eval-pass-at-k.sh <benchmark> ${CURRENT} (strategy=$(read_yaml eval.prompt_strategy standard))" ;;
            submit)    echo "[submit] ./scripts/submit.sh ${CURRENT} <hf-repo>" ;;
            compile)  PLAN_FLAGS="--release --strip"; [[ "$(read_yaml compile.lto true)" == "true" ]] && PLAN_FLAGS+=" --lto"; echo "[compile] apr compile ${CURRENT} ${PLAN_FLAGS}" ;;
        esac
    done
    echo ""
    echo "=== Plan complete ==="
    exit 0
fi

# ── Execute pipeline stages ──────────────────────────────────────────────────

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
            TEACHER="$(read_yaml distill.teacher)"
            STRATEGY="$(read_yaml distill.strategy standard)"
            TEMP="$(read_yaml distill.temperature 3.0)"
            ALPHA="$(read_yaml distill.alpha 0.7)"
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
            DATASET="$(read_yaml finetune.dataset)"
            METHOD="$(read_yaml finetune.method lora)"
            RANK="$(read_yaml finetune.rank 16)"
            LR="$(read_yaml finetune.lr 0.0002)"
            EPOCHS="$(read_yaml finetune.epochs 3)"
            QUANTIZE_NF4="$(read_yaml finetune.quantize_nf4 false)"
            MAX_SEQ_LEN="$(read_yaml finetune.max_seq_len)"
            VRAM="$(read_yaml finetune.vram)"
            TASK="$(read_yaml finetune.task)"
            MODEL_SIZE="$(read_yaml finetune.model_size)"
            WAIT_GPU="$(read_yaml finetune.wait_gpu 0)"
            EXTRA_FLAGS=""
            [[ "$QUANTIZE_NF4" == "true" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --quantize-nf4"
            [[ -n "$MAX_SEQ_LEN" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --max-seq-len ${MAX_SEQ_LEN}"
            [[ -n "$VRAM" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --vram ${VRAM}"
            [[ -n "$TASK" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --task ${TASK}"
            [[ -n "$MODEL_SIZE" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --model-size ${MODEL_SIZE}"
            [[ "$WAIT_GPU" != "0" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --wait-gpu ${WAIT_GPU}"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-finetuned.apr"
            # shellcheck disable=SC2086
            apr finetune "$CURRENT" \
                --method "$METHOD" \
                --rank "$RANK" \
                --learning-rate "$LR" \
                --epochs "$EPOCHS" \
                --data "$DATASET" \
                --output "$NEXT" \
                $EXTRA_FLAGS \
                --verbose
            CURRENT="$NEXT"
            ;;

        align)
            DATA="$(read_yaml align.data)"
            METHOD="$(read_yaml align.method dpo)"
            EPOCHS="$(read_yaml align.epochs 3)"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-aligned.apr"
            apr finetune "$CURRENT" \
                --method "$METHOD" \
                --data "$DATA" \
                --epochs "$EPOCHS" \
                --output "$NEXT" \
                --verbose
            CURRENT="$NEXT"
            ;;

        merge)
            STRATEGY="$(read_yaml merge.strategy slerp)"
            BASE_MODEL="$(read_yaml merge.base_model)"
            DENSITY="$(read_yaml merge.density 0.2)"
            # Read merge model list
            MERGE_FILES=""
            while IFS= read -r mf; do
                [[ -n "$mf" ]] && MERGE_FILES="${MERGE_FILES} ${mf}"
            done < <(read_yaml_list merge.models)
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
            METHOD="$(read_yaml prune.method magnitude)"
            RATIO="$(read_yaml prune.target_ratio 0.5)"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-pruned.apr"
            apr prune "$CURRENT" \
                --method "$METHOD" \
                --target-ratio "$RATIO" \
                --output "$NEXT" \
                --verbose
            CURRENT="$NEXT"
            ;;

        quantize)
            SCHEME="$(read_yaml quantize.scheme int4)"
            NEXT="${OUTPUT_DIR}/${MODEL_NAME}-${SCHEME}.apr"
            apr quantize "$CURRENT" \
                --scheme "$SCHEME" \
                --output "$NEXT" \
                --verbose
            CURRENT="$NEXT"
            ;;

        prep-data)
            apr data audit data/instruct-corpus.jsonl --verbose
            ;;

        eval)
            EVAL_MAX_TOKENS="$(read_yaml eval.max_tokens 512)"
            EVAL_TEMPERATURE="$(read_yaml eval.temperature 0.0)"
            EVAL_SAMPLES="$(read_yaml eval.num_samples 1)"
            EVAL_STRATEGY="$(read_yaml eval.prompt_strategy standard)"
            for bench in "${BENCH_LIST[@]}"; do
                echo ""
                echo "  Evaluating: ${bench} (strategy=${EVAL_STRATEGY})"
                case "$bench" in
                    humaneval|mbpp|bigcodebench)
                        ./scripts/eval-pass-at-k.sh "$bench" "$CURRENT" results/ \
                            "$EVAL_MAX_TOKENS" "$EVAL_TEMPERATURE" "$EVAL_SAMPLES" \
                            "$EVAL_STRATEGY" \
                            || echo "  WARNING: ${bench} evaluation failed"
                        ;;
                    *)
                        echo "  SKIP: ${bench} (not yet supported in eval script)"
                        ;;
                esac
            done
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
            RELEASE="$(read_yaml compile.release true)"
            LTO="$(read_yaml compile.lto true)"
            OUTPUT_NAME="$(read_yaml compile.output "${MODEL_NAME}")"
            COMPILE_FLAGS=""
            [[ "$RELEASE" == "true" ]] && COMPILE_FLAGS="--release"
            [[ "$LTO" == "true" ]] && COMPILE_FLAGS="${COMPILE_FLAGS} --lto"
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
