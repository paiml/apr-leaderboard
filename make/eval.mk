# Evaluation targets
# Included from top-level Makefile

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

failure-analysis:
	@RESULTS_DIR=$(RESULTS_DIR) ./scripts/failure-analysis.sh $(BENCHMARK)

validate-ac022:
	./scripts/validate-ac022.sh $(RESULTS_DIR)
