.PHONY: build test lint fmt check clean coverage docs dogfood

# --- Quality gates ---

build:
	cargo build --release

test:
	cargo test --all-features

lint:
	cargo clippy --all-targets -- -D warnings

fmt:
	cargo fmt --all -- --check

check: fmt lint test

coverage:
	cargo llvm-cov --all-features --lcov --output-path lcov.info
	cargo llvm-cov --all-features --text

clean:
	cargo clean

# --- Documentation ---

docs:
	cd docs && mdbook build

docs-serve:
	cd docs && mdbook serve

# --- Evaluation targets ---

eval-humaneval:
	cargo run --release -- eval --model $(MODEL) --benchmark humaneval --output results/

eval-mbpp:
	cargo run --release -- eval --model $(MODEL) --benchmark mbpp --output results/

eval-bigcodebench:
	cargo run --release -- eval --model $(MODEL) --benchmark bigcodebench --output results/

eval-all:
	cargo run --release -- pipeline --config $(CONFIG)

# --- Convenience ---

benchmarks:
	cargo run --release -- benchmarks

history:
	cargo run --release -- history

dogfood:
	cargo run --release -- benchmarks
	cargo run --release -- history
	cargo run --release -- convert --model-id test/model
	cargo run --release -- distill --teacher t.apr --student s.apr -o o.apr
	cargo run --release -- merge a.apr b.apr -o o.apr
	cargo run --release -- prune --model m.apr -o o.apr
	cargo run --release -- quantize --model m.apr -o o.apr
	cargo run --release -- compare --model m.apr
	@echo "=== All CLI subcommands work ==="

# --- Example usage ---
# make check                                    # fmt + lint + test
# make coverage                                 # test coverage report
# make eval-humaneval MODEL=models/qwen.apr     # run HumanEval
# make eval-all CONFIG=configs/qwen-coder-7b.toml
# make dogfood                                  # exercise all CLI subcommands
