.PHONY: build test lint fmt check clean eval submit

build:
	cargo build --release

test:
	cargo test --all-features

lint:
	cargo clippy --all-targets -- -D warnings

fmt:
	cargo fmt --all -- --check

check: fmt lint test

clean:
	cargo clean

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

# --- Example usage ---
# make eval-humaneval MODEL=models/Qwen_Qwen2.5-Coder-7B.apr
# make eval-all CONFIG=configs/qwen-coder-7b.toml
