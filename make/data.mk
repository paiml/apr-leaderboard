# Data preparation targets
# Included from top-level Makefile

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

# §12.0.1: Calibration data for Wanda/SparseGPT pruning
prepare-calibration-data:
	./scripts/prepare-calibration-data.sh data/calibration.jsonl 128

# PMAT-008: Combine training data from multiple sources
combine-training-data:
	./scripts/combine-training-data.sh data/combined-training.jsonl

# PMAT-014: Generate DPO preference pairs from N-sampling eval results
generate-preference-pairs:
	@test -n "$(EVAL_WORK_DIR)" || { echo "ERROR: EVAL_WORK_DIR required (from N-sampling eval run)"; exit 1; }
	@mkdir -p data
	./scripts/generate-preference-pairs.sh "$(EVAL_WORK_DIR)" data/preference-pairs.jsonl
