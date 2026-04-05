# Optimization targets (finetune, align, merge, prune, quantize, distill, compile)
# Included from top-level Makefile

# -- Optimization ----------------------------------------------------------------

finetune-instruct:
	@test -f "data/instruct-corpus.jsonl" || { echo "ERROR: Run 'make prep-data' first"; exit 1; }
	$(APR) finetune --task instruct \
		--data data/instruct-corpus.jsonl \
		--model-size $(MODEL_SIZE) \
		--rank $(RANK) \
		--epochs $(EPOCHS) \
		--learning-rate $(LR) \
		--output "$(OUTPUT_DIR)/$(NAME)-instruct.apr" \
		--verbose

finetune:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) finetune "$(CHECKPOINT)" \
		--method $(METHOD) \
		--rank $(RANK) \
		--learning-rate $(LR) \
		--epochs $(EPOCHS) \
		--data "$(DATA)" \
		--output "$(OUTPUT_DIR)/$(NAME)-finetuned.apr" \
		--verbose

align:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@test -f "$(PREFS_DATA)" || { echo "ERROR: Preference data not found at $(PREFS_DATA)"; exit 1; }
	$(APR) finetune "$(CHECKPOINT)" \
		--method $(ALIGN_METHOD) \
		--data "$(PREFS_DATA)" \
		--output "$(OUTPUT_DIR)/$(NAME)-aligned.apr" \
		--verbose

dpo-pipeline:
	@bash scripts/run-dpo-pipeline.sh "$(or $(PREFS_DATA),data/preference-pairs.jsonl)" "$(CHECKPOINT)"

merge:
	@test -n "$(MODELS)" || { echo "ERROR: MODELS required (space-separated .apr paths)"; exit 1; }
	$(APR) merge $(MODELS) \
		--strategy $(STRATEGY) \
		--output "$(OUTPUT_DIR)/merged.apr" \
		--verbose

prune:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) prune "$(CHECKPOINT)" \
		--method $(PRUNE_METHOD) \
		--target-ratio $(SPARSITY) \
		--output "$(OUTPUT_DIR)/$(NAME)-pruned.apr" \
		--verbose

quantize:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) quantize "$(CHECKPOINT)" \
		--scheme $(SCHEME) \
		--output "$(OUTPUT_DIR)/$(NAME)-$(SCHEME).apr" \
		--verbose

distill:
	@test -f "$(TEACHER)" || { echo "ERROR: TEACHER model not found at $(TEACHER)"; exit 1; }
	@test -f "$(STUDENT)" || { echo "ERROR: STUDENT model not found at $(STUDENT)"; exit 1; }
	$(APR) distill "$(TEACHER)" \
		--student "$(STUDENT)" \
		--output "$(OUTPUT_DIR)/distilled.apr" \
		--strategy $(DIST_STRATEGY) \
		--temperature $(DIST_TEMP) \
		--alpha $(DIST_ALPHA) \
		--verbose

# -- Text-Based Distillation (PMAT-007) ----------------------------------------

DISTILL_CONFIG ?= configs/distill/distill-32b-7b-text.yaml
DISTILL_PROMPTS ?= data/distill/distill-prompts.jsonl
DISTILL_OUTPUT ?= data/distill/teacher-completions.jsonl
DISTILL_STUDENT ?= checkpoints/qwen2.5-coder-7b-instruct-q4k.apr
DISTILL_MODEL ?= checkpoints/qwen2.5-coder-7b-distilled-q4k.apr

validate-teacher:
	@test -f "$(TEACHER)" || { echo "ERROR: TEACHER model not found: $(TEACHER)"; exit 1; }
	./scripts/validate-teacher.sh "$(TEACHER)" humaneval 0.60

distill-generate: $(DISTILL_PROMPTS)
	@echo "=== Stage 1: Generate teacher completions ==="
	./scripts/distill-generate.sh $(DISTILL_CONFIG)

distill-finetune: $(DISTILL_OUTPUT)
	@echo "=== Stage 2: Fine-tune 7B student on teacher completions ==="
	@test -f "$(DISTILL_STUDENT)" || { echo "ERROR: Student not found: $(DISTILL_STUDENT)"; exit 1; }
	@test -f "$(DISTILL_OUTPUT)" || { echo "ERROR: Teacher completions not found. Run: make distill-generate"; exit 1; }
	$(APR) finetune "$(DISTILL_STUDENT)" \
		--method qlora \
		--rank 32 \
		--data "$(DISTILL_OUTPUT)" \
		--output "$(DISTILL_MODEL)" \
		--epochs 3 \
		--learning-rate 0.0002 \
		--verbose

distill-eval: $(DISTILL_MODEL)
	@echo "=== Stage 3: Evaluate distilled model ==="
	@test -f "$(DISTILL_MODEL)" || { echo "ERROR: Distilled model not found. Run: make distill-finetune"; exit 1; }
	@mkdir -p $(RESULTS_DIR)
	./scripts/eval-pass-at-k.sh humaneval "$(DISTILL_MODEL)" "$(RESULTS_DIR)"

$(DISTILL_PROMPTS):
	./scripts/generate-distill-prompts.sh

compile:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) compile "$(CHECKPOINT)" \
		--output "$(OUTPUT_DIR)/$(NAME)" \
		--release --strip --lto \
		--verbose
