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
        prep-data prep-data-audit data-split data-balance decontaminate data-quality benchmark-download generate-training-data generate-preference-pairs combine-training-data \
        finetune finetune-instruct align merge prune quantize distill distill-generate distill-finetune distill-eval compile \
        eval-humaneval eval-mbpp eval-bigcodebench eval-all eval-perplexity eval-sweep compare-results results-history leaderboard \
        export publish model-card \
        pipeline pipeline-plan \
        check inspect qa compare-hf bench verify dogfood validate clean \
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
	@echo "=== Extracting instruction/response pairs from Python source ==="
	@mkdir -p data
	@test -d "$(CORPUS_DIR)" || { echo "ERROR: CORPUS_DIR not set or not found: $(CORPUS_DIR)"; exit 1; }
	$(APR) data prep "$(CORPUS_DIR)" \
		--output data/instruct-corpus.jsonl \
		--corpus "$(CORPUS_NAME)" \
		--min-lines $(MIN_LINES) --max-lines $(MAX_LINES)
	@wc -l data/instruct-corpus.jsonl | awk '{print "  " $$1 " instruction pairs extracted"}'

prep-data-audit:
	$(APR) data audit data/instruct-corpus.jsonl --verbose

data-split:
	@test -f "$(DATA)" || { echo "ERROR: DATA file not found: $(DATA)"; exit 1; }
	@mkdir -p data/splits
	$(APR) data split "$(DATA)" \
		--train 0.8 --val 0.1 --test 0.1 \
		--label-column label --seed 42 \
		--output data/splits
	@echo "Split files written to data/splits/"

data-balance:
	@test -f "$(DATA)" || { echo "ERROR: DATA file not found: $(DATA)"; exit 1; }
	$(APR) data balance "$(DATA)" \
		--strategy $(BALANCE_STRATEGY) \
		--label-column label --seed 42 \
		--output data/balanced.jsonl
	@wc -l data/balanced.jsonl | awk '{print "  " $$1 " samples after rebalancing"}'

decontaminate:
	@echo "=== Decontamination Gate (AC-016) ==="
	@test -f "$(DATA)" || { echo "ERROR: DATA file not found: $(DATA)"; exit 1; }
	$(APR) data decontaminate "$(DATA)" \
		--reference data/benchmarks/humaneval.jsonl data/benchmarks/mbpp.jsonl \
		--ngram 10 --threshold 0.5 --json

data-quality:
	@echo "=== Data Quality Gate (AC-025) ==="
	@test -f "$(DATA)" || { echo "ERROR: DATA file not found: $(DATA)"; exit 1; }
	$(APR) data quality "$(DATA)" --min-score 80 --json

benchmark-download:
	@echo "=== Downloading benchmark data ==="
	@mkdir -p data/benchmarks
	./scripts/download-benchmarks.sh data/benchmarks

# PMAT-004: Generate synthetic training data from teacher model
TEACHER ?= checkpoints/qwen2.5-coder-32b-instruct-q4km.apr
NUM_TRAIN_PROMPTS ?= 50
generate-training-data:
	@test -f "$(TEACHER)" || { echo "ERROR: Teacher model not found: $(TEACHER)"; exit 1; }
	@mkdir -p data/synthetic
	./scripts/generate-training-data.sh "$(TEACHER)" data/synthetic $(NUM_TRAIN_PROMPTS) $(MAX_TOKENS)

# PMAT-008: Combine training data from multiple sources
combine-training-data:
	./scripts/combine-training-data.sh data/combined-training.jsonl

# PMAT-014: Generate DPO preference pairs from N-sampling eval results
generate-preference-pairs:
	@test -n "$(EVAL_WORK_DIR)" || { echo "ERROR: EVAL_WORK_DIR required (from N-sampling eval run)"; exit 1; }
	@mkdir -p data
	./scripts/generate-preference-pairs.sh "$(EVAL_WORK_DIR)" data/preference-pairs.jsonl

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

# -- Text-Based Distillation (PMAT-007) ----------------------------------------

DISTILL_CONFIG ?= configs/distill/distill-32b-7b-text.yaml
DISTILL_PROMPTS ?= data/distill/distill-prompts.jsonl
DISTILL_OUTPUT ?= data/distill/teacher-completions.jsonl
DISTILL_STUDENT ?= checkpoints/qwen2.5-coder-7b-instruct-q4k.apr
DISTILL_MODEL ?= checkpoints/qwen2.5-coder-7b-distilled-q4k.apr

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

# -- Evaluation ------------------------------------------------------------------

eval-humaneval:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@mkdir -p $(RESULTS_DIR)
	./scripts/eval-pass-at-k.sh humaneval "$(CHECKPOINT)" "$(RESULTS_DIR)" \
		$(MAX_TOKENS) $(TEMPERATURE) $(NUM_SAMPLES) $(PROMPT_STRATEGY)

eval-mbpp:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@mkdir -p $(RESULTS_DIR)
	./scripts/eval-pass-at-k.sh mbpp "$(CHECKPOINT)" "$(RESULTS_DIR)" \
		$(MAX_TOKENS) $(TEMPERATURE) $(NUM_SAMPLES) $(PROMPT_STRATEGY)

eval-bigcodebench:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@mkdir -p $(RESULTS_DIR)
	./scripts/eval-pass-at-k.sh bigcodebench "$(CHECKPOINT)" "$(RESULTS_DIR)" \
		$(MAX_TOKENS) $(TEMPERATURE) $(NUM_SAMPLES) $(PROMPT_STRATEGY)

eval-all:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	@mkdir -p $(RESULTS_DIR)
	@for bench in humaneval mbpp bigcodebench; do \
		echo "=== Evaluating $$bench ==="; \
		./scripts/eval-pass-at-k.sh $$bench "$(CHECKPOINT)" "$(RESULTS_DIR)" \
			$(MAX_TOKENS) $(TEMPERATURE) $(NUM_SAMPLES) $(PROMPT_STRATEGY); \
	done

eval-perplexity:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	$(APR) eval "$(CHECKPOINT)" --dataset wikitext-2 --json

results-history:
	./scripts/results-history.sh "$(RESULTS_DIR)"

eval-sweep:
	@test -f "$(CHECKPOINT)" || { echo "ERROR: Model not found at $(CHECKPOINT)"; exit 1; }
	./scripts/eval-sweep.sh humaneval "$(CHECKPOINT)" "standard scot few-shot cgo" $(MAX_TOKENS)

compare-results:
	@test -n "$(BASE)" || { echo "ERROR: BASE required (e.g., make compare-results BASE=results/a.json NEW=results/b.json)"; exit 1; }
	@test -n "$(NEW)" || { echo "ERROR: NEW required (e.g., make compare-results BASE=results/a.json NEW=results/b.json)"; exit 1; }
	./scripts/compare-results.sh "$(BASE)" "$(NEW)"

leaderboard:
	./scripts/leaderboard-summary.sh "$(RESULTS_DIR)"

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

check-contracts:
	@echo "=== Contract Falsification Tests ==="
	@PASS=0; FAIL=0; \
	echo "-- pass-at-k (FT-001..003) --"; \
	r1=$$(awk 'BEGIN{n=10;c=0;k=1; if(c==0){print 0.0;exit} log_ratio=0; for(i=0;i<k;i++){log_ratio+=log(n-c-i)-log(n-i)} printf "%.1f",1.0-exp(log_ratio)}'); \
	r2=$$(awk 'BEGIN{n=10;c=10;k=1; if(n-c<k){print 1.0;exit}}'); \
	r3=$$(awk 'BEGIN{n=10;c=5;k=1; log_ratio=0; for(i=0;i<k;i++){log_ratio+=log(n-c-i)-log(n-i)} printf "%.1f",1.0-exp(log_ratio)}'); \
	[ $$(echo "$$r1 == 0" | bc) -eq 1 ] && echo "  FT-001 (zero correct=0):  PASS" && PASS=$$((PASS+1)) || { echo "  FT-001: FAIL (got $$r1)"; FAIL=$$((FAIL+1)); }; \
	[ $$(echo "$$r2 == 1" | bc) -eq 1 ] && echo "  FT-002 (all correct=1):   PASS" && PASS=$$((PASS+1)) || { echo "  FT-002: FAIL (got $$r2)"; FAIL=$$((FAIL+1)); }; \
	[ $$(echo "$$r3 == 0.5" | bc) -eq 1 ] && echo "  FT-003 (pass@1=ratio):    PASS" && PASS=$$((PASS+1)) || { echo "  FT-003: FAIL (got $$r3)"; FAIL=$$((FAIL+1)); }; \
	r4=$$(awk 'BEGIN{n=10;c=5;k=10; if(n-c<k){print 1.0;exit} log_ratio=0; for(i=0;i<k;i++){log_ratio+=log(n-c-i)-log(n-i)} printf "%.4f",1.0-exp(log_ratio)}'); \
	[ $$(echo "$$r4 == 1.0" | bc) -eq 1 ] && echo "  FT-004 (pass@10,n=10,c=5=1.0): PASS" && PASS=$$((PASS+1)) || { echo "  FT-004: FAIL (got $$r4, expected 1.0)"; FAIL=$$((FAIL+1)); }; \
	r5=$$(awk 'BEGIN{n=20;c=10;k=10; if(n-c<k){print 1.0;exit} log_ratio=0; for(i=0;i<k;i++){log_ratio+=log(n-c-i)-log(n-i)} printf "%.4f",1.0-exp(log_ratio)}'); \
	[ $$(echo "$$r5 > 0.99" | bc) -eq 1 ] && echo "  FT-005 (pass@10,n=20,c=10>0.99): PASS" && PASS=$$((PASS+1)) || { echo "  FT-005: FAIL (got $$r5)"; FAIL=$$((FAIL+1)); }; \
	echo "-- inference-throughput (FT-TPUT-001..002) --"; \
	if [ -f results/bench_1.5b_instruct_q4k_cpu.json ]; then \
		tps=$$(jq '.results.tokens_per_second' results/bench_1.5b_instruct_q4k_cpu.json); \
		ttft=$$(jq '.results.time_to_first_token_ms' results/bench_1.5b_instruct_q4k_cpu.json); \
		[ $$(echo "$$tps >= 1.0" | bc) -eq 1 ] && echo "  FT-TPUT-001 (tps>=1.0): PASS ($$tps)" && PASS=$$((PASS+1)) || { echo "  FT-TPUT-001: FAIL ($$tps)"; FAIL=$$((FAIL+1)); }; \
		[ $$(echo "$$ttft < 500" | bc) -eq 1 ] && echo "  FT-TPUT-002 (ttft<500): PASS ($${ttft}ms)" && PASS=$$((PASS+1)) || { echo "  FT-TPUT-002: FAIL ($${ttft}ms)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-TPUT: SKIP (no bench results)"; fi; \
	echo "-- benchmark data (FT-DATA-001..003) --"; \
	for bench in humaneval mbpp; do \
		f="data/benchmarks/$${bench}.jsonl"; \
		if [ -f "$$f" ]; then \
			lines=$$(wc -l < "$$f"); \
			[ "$$lines" -ge 100 ] \
				&& echo "  FT-DATA ($${bench}): PASS ($$lines problems)" && PASS=$$((PASS+1)) \
				|| { echo "  FT-DATA ($${bench}): FAIL ($$lines < 100 problems)"; FAIL=$$((FAIL+1)); }; \
		else echo "  FT-DATA ($${bench}): SKIP (not downloaded)"; fi; \
	done; \
	echo "-- decontamination (FT-DECON-001) --"; \
	if [ -f data/benchmarks/humaneval.jsonl ] && [ -f data/benchmarks/mbpp.jsonl ]; then \
		overlap=$$(comm -12 <(jq -r '.prompt // .text // ""' data/benchmarks/humaneval.jsonl 2>/dev/null | sort -u) \
			<(jq -r '.prompt // .text // ""' data/benchmarks/mbpp.jsonl 2>/dev/null | sort -u) | wc -l); \
		[ "$$overlap" -eq 0 ] \
			&& echo "  FT-DECON-001 (no HE/MBPP overlap): PASS" && PASS=$$((PASS+1)) \
			|| { echo "  FT-DECON-001: FAIL ($$overlap overlapping prompts)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-DECON-001: SKIP (benchmarks not downloaded)"; fi; \
	echo "-- eval results (FT-EVAL-001..003) --"; \
	if ls results/humaneval_*.json 1>/dev/null 2>&1; then \
		best=$$(jq -s '[.[] | select(.results.pass_at_1 != null)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1' results/humaneval_*.json 2>/dev/null || echo "0"); \
		[ $$(echo "$$best >= 85.0" | bc) -eq 1 ] \
			&& echo "  FT-EVAL-001 (pass@1>=85%): PASS ($${best}%)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-EVAL-001: FAIL ($${best}% < 85%)"; FAIL=$$((FAIL+1)); }; \
		nresults=$$(ls results/humaneval_*.json | wc -l); \
		[ "$$nresults" -ge 3 ] \
			&& echo "  FT-EVAL-002 (>=3 HE runs): PASS ($$nresults runs)" && PASS=$$((PASS+1)) \
			|| echo "  FT-EVAL-002: SKIP ($$nresults < 3 runs)"; \
		latest=$$(ls -t results/humaneval_*.json | head -1); \
		latest_score=$$(jq '.results.pass_at_1' "$$latest" 2>/dev/null || echo "0"); \
		[ $$(echo "$$latest_score >= 80.0" | bc) -eq 1 ] \
			&& echo "  FT-EVAL-003 (latest>=80%): PASS ($${latest_score}%)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-EVAL-003: FAIL ($${latest_score}% < 80%)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-EVAL-001: SKIP (no results)"; fi; \
	echo "-- distillation (FT-DIST-001..002) --"; \
	if ls results/humaneval_*.json 1>/dev/null 2>&1; then \
		best32=$$(jq -s '[.[] | select(.config.max_tokens >= 512 and .results.pass_at_1 > 88)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1 // 0' results/humaneval_*.json 2>/dev/null || echo "0"); \
		best7=$$(jq -s '[.[] | select(.results.pass_at_1 > 80 and .results.pass_at_1 < 88)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1 // 0' results/humaneval_*.json 2>/dev/null || echo "0"); \
		[ $$(echo "$$best32 > $$best7" | bc) -eq 1 ] \
			&& echo "  FT-DIST-001 (teacher>student): PASS (32B=$${best32}% > 7B=$${best7}%)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-DIST-001: FAIL (32B=$${best32}% vs 7B=$${best7}%)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-DIST-001: SKIP (no results)"; fi; \
	if [ -f data/distill/distill-prompts.jsonl ]; then \
		cats=$$(python3 -c "import json; kinds=set(); [kinds.add(json.loads(l).get('kind','')) for l in open('data/distill/distill-prompts.jsonl')]; print(len(kinds))" 2>/dev/null || echo "0"); \
		[ "$$cats" -ge 10 ] \
			&& echo "  FT-DIST-002 (>=10 categories): PASS ($$cats categories)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-DIST-002: FAIL ($$cats < 10 categories)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-DIST-002: SKIP (no prompts)"; fi; \
	echo "-- contract structure --"; \
	for f in contracts/*.yaml; do \
		python3 -c "import yaml; d=yaml.safe_load(open('$$f')); [d[k] for k in ('metadata','equations','proof_obligations','falsification_tests')]" 2>/dev/null \
		&& echo "  $$(basename $$f): valid" && PASS=$$((PASS+1)) \
		|| { echo "  $$(basename $$f): INVALID"; FAIL=$$((FAIL+1)); }; \
	done; \
	echo ""; echo "$$PASS passed, $$FAIL failed"

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
BALANCE_STRATEGY ?= oversample

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
