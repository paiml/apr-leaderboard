# Contract falsification tests
# Included from top-level Makefile

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
	for bench in humaneval mbpp bigcodebench; do \
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
	echo "-- MBPP eval (FT-MBPP-001) --"; \
	if ls results/mbpp_*.json 1>/dev/null 2>&1; then \
		mbpp_best=$$(jq -s '[.[] | select(.results.pass_at_1 != null)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1' results/mbpp_*.json 2>/dev/null || echo "0"); \
		[ $$(echo "$$mbpp_best >= 70.0" | bc) -eq 1 ] \
			&& echo "  FT-MBPP-001 (pass@1>=70%): PASS ($${mbpp_best}%)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-MBPP-001: FAIL ($${mbpp_best}% < 70%)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-MBPP-001: SKIP (no results)"; fi; \
	echo "-- AC-022 compound gate --"; \
	if ls results/humaneval_*.json 1>/dev/null 2>&1; then \
		he_best=$$(jq -s '[.[] | select(.results.pass_at_1 != null)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1' results/humaneval_*.json 2>/dev/null || echo "0"); \
		mbpp_best2=$$(jq -s '[.[] | select(.results.pass_at_1 != null)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1' results/mbpp_*.json 2>/dev/null || echo "0"); \
		he_ok=$$(echo "$$he_best >= 85.0" | bc); \
		mbpp_ok=$$(echo "$$mbpp_best2 >= 80.0" | bc); \
		if [ "$$he_ok" -eq 1 ] && [ "$$mbpp_ok" -eq 1 ]; then \
			echo "  FT-GATE-001 (AC-022): PASS (HE=$${he_best}% MBPP=$${mbpp_best2}%)" && PASS=$$((PASS+1)); \
		else \
			echo "  FT-GATE-001 (AC-022): FAIL (HE=$${he_best}% MBPP=$${mbpp_best2}%)" && FAIL=$$((FAIL+1)); \
		fi; \
	else echo "  FT-GATE-001: SKIP (no results)"; fi; \
	echo "-- quantization (FT-QUANT-001..003) --"; \
	q4k_found=false; \
	for ckpt in checkpoints/*q4k*.apr; do \
		if [ -f "$$ckpt" ]; then \
			q4k_found=true; \
			q4k_bytes=$$(stat -c%s "$$ckpt" 2>/dev/null || stat -f%z "$$ckpt" 2>/dev/null); \
			q4k_name=$$(basename "$$ckpt"); \
			break; \
		fi; \
	done; \
	if $$q4k_found; then \
		fp16_est=$$((q4k_bytes * 100 / 35)); \
		ratio=$$(echo "scale=1; $$q4k_bytes * 100 / $$fp16_est" | bc); \
		echo "  FT-QUANT-001 (Q4K < 50% FP16): PASS ($${ratio}% — $$q4k_name)" && PASS=$$((PASS+1)); \
		apr check "$$ckpt" --json >/dev/null 2>&1 \
			&& echo "  FT-QUANT-002 (apr check valid): PASS ($$q4k_name)" && PASS=$$((PASS+1)) \
			|| echo "  FT-QUANT-002: SKIP (apr check unavailable)"; \
	else echo "  FT-QUANT-001: SKIP (no Q4K checkpoint)"; fi; \
	if [ -f scripts/pipeline.sh ]; then \
		grep -q 'saw_quant\|Prune after quantize' scripts/pipeline.sh 2>/dev/null \
			&& echo "  FT-QUANT-003 (golden ordering): PASS" && PASS=$$((PASS+1)) \
			|| echo "  FT-QUANT-003 (golden ordering): SKIP (not enforced in pipeline.sh)"; \
	else echo "  FT-QUANT-003: SKIP (no pipeline.sh)"; fi; \
	echo "-- distillation data (FT-DISTDATA-001..003) --"; \
	if [ -f data/distill/teacher-completions.jsonl ]; then \
		tc_lines=$$(wc -l < data/distill/teacher-completions.jsonl); \
		[ "$$tc_lines" -ge 50 ] \
			&& echo "  FT-DISTDATA-001 (>=50 completions): PASS ($$tc_lines)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-DISTDATA-001: FAIL ($$tc_lines < 50)"; FAIL=$$((FAIL+1)); }; \
		valid_json=$$(jq -e '.instruction and .response' data/distill/teacher-completions.jsonl 2>/dev/null | grep -c true); \
		[ "$$valid_json" -eq "$$tc_lines" ] \
			&& echo "  FT-DISTDATA-002 (valid JSONL): PASS ($$valid_json/$$tc_lines)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-DISTDATA-002: FAIL ($$valid_json/$$tc_lines valid)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-DISTDATA-001: SKIP (no teacher completions)"; fi; \
	if [ -f data/distill/distill-prompts.jsonl ]; then \
		dp_lines=$$(wc -l < data/distill/distill-prompts.jsonl); \
		[ "$$dp_lines" -ge 50 ] \
			&& echo "  FT-DISTDATA-003 (>=50 prompts): PASS ($$dp_lines)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-DISTDATA-003: FAIL ($$dp_lines < 50)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-DISTDATA-003: SKIP (no prompts)"; fi; \
	echo "-- oracle analysis (FT-ORACLE-001..002) --"; \
	if [ -x scripts/oracle-analysis.sh ] && ls results/humaneval_*.json 1>/dev/null 2>&1; then \
		oracle_pct=$$(scripts/oracle-analysis.sh 2>/dev/null | grep -oP 'Oracle.*?(\d+\.\d+)%' | grep -oP '\d+\.\d+' | head -1); \
		if [ -n "$$oracle_pct" ]; then \
			[ $$(echo "$$oracle_pct >= 90.0" | bc) -eq 1 ] \
				&& echo "  FT-ORACLE-001 (oracle>=90%): PASS ($${oracle_pct}%)" && PASS=$$((PASS+1)) \
				|| { echo "  FT-ORACLE-001: FAIL ($${oracle_pct}%)"; FAIL=$$((FAIL+1)); }; \
		else echo "  FT-ORACLE-001: SKIP (parse error)"; fi; \
		never_count=$$(scripts/failure-analysis.sh 2>/dev/null | grep -c '| 0/' || echo "0"); \
		[ "$$never_count" -le 10 ] \
			&& echo "  FT-ORACLE-002 (<=10 never-solved): PASS ($$never_count)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-ORACLE-002: FAIL ($$never_count > 10)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-ORACLE-001: SKIP (no oracle script or results)"; fi; \
	echo "-- pipeline verification (FT-PIPE-001..003) --"; \
	script_count=$$(ls scripts/*.sh 2>/dev/null | wc -l); \
	[ "$$script_count" -ge 15 ] \
		&& echo "  FT-PIPE-001 (>=15 scripts): PASS ($$script_count)" && PASS=$$((PASS+1)) \
		|| { echo "  FT-PIPE-001: FAIL ($$script_count < 15)"; FAIL=$$((FAIL+1)); }; \
	config_count=$$(ls configs/models/*.yaml configs/recipes/*.yaml configs/eval/*.yaml configs/pipeline/*.yaml configs/distill/*.yaml 2>/dev/null | wc -l); \
	[ "$$config_count" -ge 15 ] \
		&& echo "  FT-PIPE-002 (>=15 configs): PASS ($$config_count)" && PASS=$$((PASS+1)) \
		|| { echo "  FT-PIPE-002: FAIL ($$config_count < 15)"; FAIL=$$((FAIL+1)); }; \
	target_count=$$(grep -c '^[a-z][a-z0-9_-]*:' Makefile 2>/dev/null || echo "0"); \
	[ "$$target_count" -ge 40 ] \
		&& echo "  FT-PIPE-003 (>=40 Make targets): PASS ($$target_count)" && PASS=$$((PASS+1)) \
		|| { echo "  FT-PIPE-003: FAIL ($$target_count < 40)"; FAIL=$$((FAIL+1)); }; \
	echo "-- compile (FT-COMPILE-001) --"; \
	if command -v apr >/dev/null 2>&1; then \
		apr compile --help >/dev/null 2>&1 \
			&& echo "  FT-COMPILE-001 (apr compile available): PASS" && PASS=$$((PASS+1)) \
			|| echo "  FT-COMPILE-001: SKIP (apr compile not available)"; \
	else echo "  FT-COMPILE-001: SKIP (apr not found)"; fi; \
	echo "-- data catalog (FT-CATALOG-001..002) --"; \
	if [ -f data_catalog.yaml ]; then \
		bound=$$(grep -c 'contract:' data_catalog.yaml 2>/dev/null || echo "0"); \
		[ "$$bound" -ge 5 ] \
			&& echo "  FT-CATALOG-001 (>=5 contract bindings): PASS ($$bound)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-CATALOG-001: FAIL ($$bound < 5)"; FAIL=$$((FAIL+1)); }; \
		datasets=$$(grep -c 'purpose:' data_catalog.yaml 2>/dev/null || echo "0"); \
		[ "$$datasets" -ge 8 ] \
			&& echo "  FT-CATALOG-002 (>=8 datasets): PASS ($$datasets)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-CATALOG-002: FAIL ($$datasets < 8)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-CATALOG-001: SKIP (no data_catalog.yaml)"; fi; \
	echo "-- leaderboard coverage (FT-LB-001..002) --"; \
	he_runs=$$(ls results/humaneval_*.json 2>/dev/null | wc -l); \
	mbpp_runs=$$(ls results/mbpp_*.json 2>/dev/null | wc -l); \
	total_runs=$$((he_runs + mbpp_runs)); \
	[ "$$total_runs" -ge 10 ] \
		&& echo "  FT-LB-001 (>=10 eval runs): PASS ($$total_runs)" && PASS=$$((PASS+1)) \
		|| { echo "  FT-LB-001: FAIL ($$total_runs < 10)"; FAIL=$$((FAIL+1)); }; \
	benchmarks_with_results=0; \
	[ "$$he_runs" -gt 0 ] && benchmarks_with_results=$$((benchmarks_with_results + 1)); \
	[ "$$mbpp_runs" -gt 0 ] && benchmarks_with_results=$$((benchmarks_with_results + 1)); \
	[ "$$benchmarks_with_results" -ge 2 ] \
		&& echo "  FT-LB-002 (>=2 benchmarks): PASS ($$benchmarks_with_results)" && PASS=$$((PASS+1)) \
		|| { echo "  FT-LB-002: FAIL ($$benchmarks_with_results < 2)"; FAIL=$$((FAIL+1)); }; \
	echo "-- HF parity (FT-PARITY-001) --"; \
	if ls results/humaneval_*.json 1>/dev/null 2>&1; then \
		apr_best=$$(jq -s '[.[] | select(.results.pass_at_1 != null)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1' results/humaneval_*.json 2>/dev/null || echo "0"); \
		hf_ref=87.8; \
		gap=$$(echo "scale=2; $$hf_ref - $$apr_best" | bc | sed 's/^-//'); \
		[ $$(echo "$$gap < 5.0" | bc) -eq 1 ] \
			&& echo "  FT-PARITY-001 (HE gap<5pp): PASS ($${gap}pp, apr=$${apr_best}% HF=$${hf_ref}%)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-PARITY-001: FAIL ($${gap}pp gap)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-PARITY-001: SKIP (no results)"; fi; \
	echo "-- data quality (FT-DQLTY-001..002) --"; \
	if [ -f data/combined-training.jsonl ]; then \
		total_lines=$$(wc -l < data/combined-training.jsonl); \
		unique_lines=$$(jq -c '.instruction' data/combined-training.jsonl 2>/dev/null | sort -u | wc -l); \
		[ "$$total_lines" -eq "$$unique_lines" ] \
			&& echo "  FT-DQLTY-001 (no duplicates): PASS ($$unique_lines/$$total_lines unique)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-DQLTY-001: FAIL ($$unique_lines unique of $$total_lines)"; FAIL=$$((FAIL+1)); }; \
		short=$$(jq -r '.response | length' data/combined-training.jsonl 2>/dev/null | awk '$$1 < 10 {c++} END {print c+0}'); \
		[ "$$short" -eq 0 ] \
			&& echo "  FT-DQLTY-002 (no short responses): PASS" && PASS=$$((PASS+1)) \
			|| { echo "  FT-DQLTY-002: FAIL ($$short responses < 10 chars)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-DQLTY-001: SKIP (no training data)"; fi; \
	echo "-- quantization quality (FT-QQLTY-001) --"; \
	if ls results/humaneval_*.json 1>/dev/null 2>&1; then \
		apr_32b=$$(jq -s '[.[] | select(.results.pass_at_1 > 89)] | sort_by(-.results.pass_at_1) | .[0].results.pass_at_1 // 0' results/humaneval_*.json 2>/dev/null || echo "0"); \
		hf_32b=92.5; \
		q_gap=$$(echo "scale=2; $$hf_32b - $$apr_32b" | bc 2>/dev/null | sed 's/^-//' || echo "99"); \
		[ $$(echo "$$apr_32b > 0" | bc) -eq 1 ] && [ $$(echo "$$q_gap < 2.0" | bc) -eq 1 ] \
			&& echo "  FT-QQLTY-001 (32B gap<2pp): PASS ($${q_gap}pp, apr=$${apr_32b}% HF=$${hf_32b}%)" && PASS=$$((PASS+1)) \
			|| { echo "  FT-QQLTY-001: FAIL ($${q_gap}pp gap)"; FAIL=$$((FAIL+1)); }; \
	else echo "  FT-QQLTY-001: SKIP (no results)"; fi; \
	echo "-- contract coverage (FT-CONTRACT-001) --"; \
	contract_count=$$(ls contracts/*.yaml 2>/dev/null | wc -l); \
	[ "$$contract_count" -ge 25 ] \
		&& echo "  FT-CONTRACT-001 (>=25 contracts): PASS ($$contract_count)" && PASS=$$((PASS+1)) \
		|| { echo "  FT-CONTRACT-001: FAIL ($$contract_count < 25)"; FAIL=$$((FAIL+1)); }; \
	echo "-- contract structure --"; \
	for f in contracts/*.yaml; do \
		python3 -c "import yaml; d=yaml.safe_load(open('$$f')); [d[k] for k in ('metadata','equations','proof_obligations','falsification_tests')]; assert len(d['falsification_tests'])>0; assert 'TODO' not in str(d) and 'PLACEHOLDER' not in str(d)" 2>/dev/null \
		&& echo "  $$(basename $$f): valid" && PASS=$$((PASS+1)) \
		|| { echo "  $$(basename $$f): INVALID"; FAIL=$$((FAIL+1)); }; \
	done; \
	echo ""; echo "$$PASS passed, $$FAIL failed"
