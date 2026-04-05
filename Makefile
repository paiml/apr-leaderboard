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
        prep-data prep-data-audit data-split data-balance decontaminate data-quality benchmark-download generate-training-data generate-preference-pairs combine-training-data prepare-calibration-data \
        finetune finetune-instruct align merge prune quantize distill distill-generate distill-finetune distill-eval compile \
        eval-humaneval eval-mbpp eval-bigcodebench eval-all eval-perplexity eval-sweep compare-results results-history leaderboard \
        export publish model-card \
        pipeline pipeline-plan \
        check inspect qa compare-hf bench verify dogfood validate clean failure-analysis validate-teacher validate-ac022 status \
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
PROMPT_STRATEGY ?= standard
CORPUS_DIR    ?=
CORPUS_NAME   ?= corpus
MIN_LINES     ?= 3
MAX_LINES     ?= 200

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
BALANCE_STRATEGY ?= oversample

# -- Include split-out target groups --
include make/data.mk
include make/optimization.mk
include make/eval.mk
include make/contracts.mk

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

qa:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) qa "$(CHECKPOINT)" --verbose

compare-hf:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@test -n "$(MODEL)" || { echo "ERROR: MODEL required (e.g., make compare-hf MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct)"; exit 1; }
	$(APR) compare-hf --hf "$(MODEL)" --json "$(CHECKPOINT)"

bench:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) bench "$(CHECKPOINT)" --json

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
	            export publish check compile bench chat inspect data qa compare-hf; do \
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
	@echo "Step 2: Check subcommands (19 expected)..."
	@passed=0; failed=0; \
	for cmd in import run serve finetune merge prune quantize distill eval \
	           export publish check compile bench chat inspect data qa compare-hf; do \
		if $(APR) $$cmd --help > /dev/null 2>&1; then \
			printf "  %-12s OK\n" "$$cmd"; \
			passed=$$((passed + 1)); \
		else \
			printf "  %-12s FAIL\n" "$$cmd"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo "  $$passed/$$((passed + failed)) subcommands verified"
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
	@echo "Step 6: Contract falsification tests..."
	@$(MAKE) --no-print-directory check-contracts
	@echo ""
	@echo "=== Dogfood complete (zero Python) ==="

# -- wgpu Proof ------------------------------------------------------------------

prove-wgpu:
	@echo "=== wgpu Training Proof ==="
	@echo "Proving GPU training works via wgpu (no CUDA toolkit)."
	@echo "This can run in parallel with other work."
	@echo ""
	./scripts/prove-wgpu.sh

proof-status:
	@pv proof-status

status:
	@./scripts/project-status.sh

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
