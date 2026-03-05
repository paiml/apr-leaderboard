# apr-leaderboard: Dev convenience targets over `apr` CLI + batuta playbook
#
# The pipeline DAG is defined in configs/pipeline/leaderboard-playbook.yaml.
# This Makefile provides quick dev shortcuts for common operations.
#
# Usage:
#   make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
#   make eval-humaneval CHECKPOINT=checkpoints/qwen-coder-7b.apr
#   make pipeline RECIPE=recipe-a-quick-lora
#   make verify

.SUFFIXES:
.DELETE_ON_ERROR:

.PHONY: import import-plan \
        prep-data prep-data-audit \
        finetune finetune-instruct align merge prune quantize distill compile \
        eval-humaneval eval-mbpp eval-bigcodebench eval-all eval-perplexity \
        export publish model-card \
        pipeline pipeline-plan \
        check inspect verify dogfood validate clean \
        prove-wgpu \
        docs docs-serve book

SHELL := /bin/bash
APR   := apr

# -- Defaults --
MODEL         ?=
CHECKPOINT    ?= checkpoints/$(shell echo "$(MODEL)" | tr '/' '_' | tr 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 'abcdefghijklmnopqrstuvwxyz').apr
NAME          ?= $(shell basename "$(CHECKPOINT)" .apr)
OUTPUT_DIR    ?= checkpoints
RESULTS_DIR   ?= results
QUANTIZE      ?= fp16
STRATEGY      ?= slerp
PRUNE_METHOD  ?= wanda
SPARSITY      ?= 0.5
SCHEME        ?= int4
RECIPE        ?=
HF_REPO       ?=
MAX_TOKENS    ?= 512
TEMPERATURE   ?= 0.0
NUM_SAMPLES   ?= 1

# -- Import ---------------------------------------------------------------------

import:
	@test -n "$(MODEL)" || { echo "ERROR: MODEL required (e.g., make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct)"; exit 1; }
	@mkdir -p $(OUTPUT_DIR)
	$(APR) import "hf://$(MODEL)" -o "$(CHECKPOINT)" --quantize $(QUANTIZE) --verbose

import-plan:
	@test -n "$(MODEL)" || { echo "ERROR: MODEL required"; exit 1; }
	@echo "=== Import Plan ==="
	@echo "Source:  hf://$(MODEL)"
	@echo "Output:  $(CHECKPOINT)"
	@echo "Command: $(APR) import hf://$(MODEL) -o $(CHECKPOINT)"
	@echo ""
	@echo "Checking HF Hub reachability..."
	@curl -sf "https://huggingface.co/api/models/$(MODEL)" > /dev/null && echo "OK: Model exists on HF Hub" || echo "WARN: Could not reach HF Hub"

# -- Data Preparation -----------------------------------------------------------

prep-data:
	@echo "=== Preparing instruction corpus from ground truth corpora ==="
	@mkdir -p data
	$(APR) data audit data/instruct-corpus.jsonl --verbose 2>/dev/null || echo "  No existing corpus — run alimentar to generate"
	@test -f data/instruct-corpus.jsonl && wc -l data/instruct-corpus.jsonl | awk '{print "  " $$1 " instruction pairs"}' || true

prep-data-audit:
	$(APR) data audit data/instruct-corpus.jsonl --verbose

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

compile:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) compile "$(CHECKPOINT)" \
		--output "$(OUTPUT_DIR)/$(NAME)" \
		--release --strip --lto \
		--verbose

# -- Evaluation ------------------------------------------------------------------

eval-humaneval:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@mkdir -p $(RESULTS_DIR)
	./scripts/eval-pass-at-k.sh humaneval "$(CHECKPOINT)" "$(RESULTS_DIR)" \
		$(MAX_TOKENS) $(TEMPERATURE) $(NUM_SAMPLES)

eval-mbpp:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@mkdir -p $(RESULTS_DIR)
	./scripts/eval-pass-at-k.sh mbpp "$(CHECKPOINT)" "$(RESULTS_DIR)" \
		$(MAX_TOKENS) $(TEMPERATURE) $(NUM_SAMPLES)

eval-bigcodebench:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@mkdir -p $(RESULTS_DIR)
	./scripts/eval-pass-at-k.sh bigcodebench "$(CHECKPOINT)" "$(RESULTS_DIR)" \
		$(MAX_TOKENS) $(TEMPERATURE) $(NUM_SAMPLES)

eval-all:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@mkdir -p $(RESULTS_DIR)
	@for bench in humaneval mbpp bigcodebench; do \
		echo "=== Evaluating $$bench ==="; \
		./scripts/eval-pass-at-k.sh $$bench "$(CHECKPOINT)" "$(RESULTS_DIR)" \
			$(MAX_TOKENS) $(TEMPERATURE) $(NUM_SAMPLES); \
	done

eval-perplexity:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) eval "$(CHECKPOINT)" --dataset wikitext-2 --json

# -- Submission ------------------------------------------------------------------

export:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) export "$(CHECKPOINT)" \
		--format $(EXPORT_FORMAT) \
		--output "$(OUTPUT_DIR)/export/" \
		--verbose

publish:
	@test -n "$(HF_REPO)" || { echo "ERROR: HF_REPO required (e.g., make publish HF_REPO=org/model)"; exit 1; }
	./scripts/submit.sh "$(CHECKPOINT)" "$(HF_REPO)" "$(RESULTS_DIR)"

model-card:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) eval "$(CHECKPOINT)" --generate-card --json

# -- Pipeline (YAML configs) -----------------------------------------------------

pipeline:
	@test -n "$(RECIPE)" || { echo "ERROR: RECIPE required (e.g., make pipeline RECIPE=recipe-a-quick-lora)"; exit 1; }
	@test -f "configs/recipes/$(RECIPE).yaml" || { echo "ERROR: Recipe not found: configs/recipes/$(RECIPE).yaml"; exit 1; }
	./scripts/pipeline.sh "configs/recipes/$(RECIPE).yaml"

pipeline-plan:
	@test -n "$(RECIPE)" || { echo "ERROR: RECIPE required"; exit 1; }
	@test -f "configs/recipes/$(RECIPE).yaml" || { echo "ERROR: Recipe not found: configs/recipes/$(RECIPE).yaml"; exit 1; }
	./scripts/pipeline.sh --plan "configs/recipes/$(RECIPE).yaml"

# -- Inspection ------------------------------------------------------------------

check:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) check "$(CHECKPOINT)" --json

inspect:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) inspect "$(CHECKPOINT)"

# -- Verification ----------------------------------------------------------------

validate:
	@echo "=== Config Validation (sovereign stack — zero Python) ==="
	@echo ""
	@echo "YAML configs (bashrs lint):"
	@for f in configs/models/*.yaml configs/recipes/*.yaml configs/eval/*.yaml configs/pipeline/*.yaml data_catalog.yaml; do \
		if [ -f "$$f" ]; then \
			printf "  %-50s" "$$f"; \
			bashrs config lint "$$f" > /dev/null 2>&1 && echo "valid" || echo "INVALID"; \
		fi; \
	done
	@echo ""
	@echo "Shell scripts (bashrs lint):"
	@for f in scripts/*.sh; do \
		if [ -f "$$f" ]; then \
			printf "  %-50s" "$$f"; \
			bashrs lint "$$f" > /dev/null 2>&1 && echo "valid" || echo "INVALID"; \
		fi; \
	done
	@echo ""
	@echo "Makefile (bashrs make lint):"
	@printf "  %-50s" "Makefile"
	@bashrs make lint Makefile > /dev/null 2>&1 && echo "valid" || echo "INVALID"

verify:
	@echo "=== APR Leaderboard Pipeline Verification ==="
	@echo ""
	@echo -n "apr CLI:     " && $(APR) --version
	@echo -n "Location:    " && which $(APR)
	@echo ""
	@echo "Subcommand smoke test:"
	@for cmd in import run serve finetune merge prune quantize distill eval \
	            export publish check compile bench chat inspect; do \
		printf "  %-12s" "$$cmd"; \
		$(APR) $$cmd --help > /dev/null 2>&1 && echo "OK" || echo "FAIL"; \
	done
	@echo ""
	@echo "YAML configs:"
	@ls configs/models/*.yaml 2>/dev/null | while read f; do printf "  %s\n" "$$f"; done
	@ls configs/recipes/*.yaml 2>/dev/null | while read f; do printf "  %s\n" "$$f"; done
	@ls configs/eval/*.yaml 2>/dev/null | while read f; do printf "  %s\n" "$$f"; done
	@ls configs/pipeline/*.yaml 2>/dev/null | while read f; do printf "  %s\n" "$$f"; done
	@echo ""
	@echo "Scripts:"
	@ls scripts/*.sh 2>/dev/null | while read f; do printf "  %s (executable: %s)\n" "$$f" "$$(test -x $$f && echo yes || echo no)"; done
	@echo ""
	@echo "=== Verification complete ==="

dogfood:
	@echo "=== Dogfood: End-to-end smoke test ==="
	@echo ""
	@echo "Step 1: Verify apr CLI..."
	@$(APR) --version
	@echo ""
	@echo "Step 2: Check subcommands..."
	@$(APR) import --help > /dev/null 2>&1 && echo "  import: OK"
	@$(APR) run --help > /dev/null 2>&1 && echo "  run:    OK"
	@$(APR) eval --help > /dev/null 2>&1 && echo "  eval:   OK"
	@$(APR) finetune --help > /dev/null 2>&1 && echo "  finetune: OK"
	@$(APR) merge --help > /dev/null 2>&1 && echo "  merge:  OK"
	@$(APR) prune --help > /dev/null 2>&1 && echo "  prune:  OK"
	@$(APR) quantize --help > /dev/null 2>&1 && echo "  quantize: OK"
	@$(APR) distill --help > /dev/null 2>&1 && echo "  distill: OK"
	@$(APR) export --help > /dev/null 2>&1 && echo "  export: OK"
	@$(APR) publish --help > /dev/null 2>&1 && echo "  publish: OK"
	@echo ""
	@echo "Step 3: Validate YAML configs (bashrs)..."
	@for f in configs/models/*.yaml configs/recipes/*.yaml configs/eval/*.yaml configs/pipeline/*.yaml data_catalog.yaml; do \
		if [ -f "$$f" ]; then \
			printf "  %-50s" "$$f"; \
			bashrs config lint "$$f" > /dev/null 2>&1 && echo "valid" || echo "INVALID"; \
		fi; \
	done
	@echo ""
	@echo "Step 4: Lint shell scripts (bashrs)..."
	@for f in scripts/*.sh; do \
		printf "  %-50s" "$$f"; \
		bashrs lint "$$f" > /dev/null 2>&1 && echo "valid" || echo "INVALID"; \
	done
	@echo ""
	@echo "Step 5: Build docs/book (mdbook)..."
	@if command -v mdbook > /dev/null 2>&1; then \
		cd docs && mdbook build > /dev/null 2>&1 && echo "  mdbook build: OK" || echo "  mdbook build: FAIL"; \
	else \
		echo "  mdbook: not installed (skipped)"; \
	fi
	@echo ""
	@echo "=== Dogfood complete (zero Python) ==="

# -- wgpu Proof ------------------------------------------------------------------

prove-wgpu:
	@echo "=== wgpu Training Proof ==="
	@echo "Proving GPU training works via wgpu (no CUDA toolkit)."
	@echo "This can run in parallel with other work."
	@echo ""
	./scripts/prove-wgpu.sh

clean:
	rm -rf checkpoints/*.apr
	rm -rf $(RESULTS_DIR)/*.json

# -- Documentation ---------------------------------------------------------------

docs:
	cd docs && mdbook build

docs-serve:
	cd docs && mdbook serve

book:
	cd docs && mdbook build

# -- Distill variables (with defaults) --
DIST_STRATEGY ?= standard
DIST_TEMP     ?= 3.0
DIST_ALPHA    ?= 0.7
EXPORT_FORMAT ?= safetensors
METHOD        ?= lora
RANK          ?= 16
LR            ?= 0.0002
EPOCHS        ?= 3
DATA          ?= data/instruct-corpus.jsonl
MODEL_SIZE    ?= 7B
TEACHER       ?=
STUDENT       ?=
ALIGN_METHOD  ?= dpo
PREFS_DATA    ?= data/preferences.jsonl

# -- Usage -----------------------------------------------------------------------
#
# Import a model from HuggingFace:
#   make import MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
#
# Evaluate on HumanEval:
#   make eval-humaneval CHECKPOINT=checkpoints/qwen-coder-7b.apr
#
# Run a recipe pipeline:
#   make pipeline RECIPE=recipe-a-quick-lora
#
# Validate all configs:
#   make validate
#
# Verify the pipeline:
#   make verify
#   make dogfood
#
# Full pipeline DAG (batuta):
#   batuta playbook run configs/pipeline/leaderboard-playbook.yaml
