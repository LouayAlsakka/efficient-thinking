#!/usr/bin/env bash
# Cross-Family Frontier Ladder (ET-II extraction robustness) — see docs/et2-cross-family-frontier-spec.md
# Replicates the ET-II frontier protocol on the Llama-3.x family. MATH n=500 first (where ET-II's
# significant result lives), GSM8K second; small models first (scoreable early); 70B last (may stall).
# True-greedy pass@1 decoded separately (flip-decider rule); temperature caches for sc@N/oracle@N.
# Idle-time work: launch ONLY on a free machine — do not co-run with another training (Metal contention).
#
#   bash scripts/run_cross_family.sh            # 1B,3B,8B  (default; 70B skipped)
#   WITH_70B=1 bash scripts/run_cross_family.sh # append the 70B cells last
set -uo pipefail
cd "$(dirname "$0")/.."
V=./.venv/bin/python
MATH=reasoning/data/math500.jsonl
GSM=reasoning/data/gsm8k_test.jsonl
N=500; MAXTOK=1024
mkdir -p reasoning/cache reasoning/results

# tag -> mlx-community model id (Llama 3.2 for 1B/3B, Llama 3.1 for 8B/70B)
declare -a LADDER=("1B:mlx-community/Llama-3.2-1B-Instruct-4bit"
                   "3B:mlx-community/Llama-3.2-3B-Instruct-4bit"
                   "8B:mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
[ "${WITH_70B:-0}" = "1" ] && LADDER+=("70B:mlx-community/Meta-Llama-3.1-70B-Instruct-4bit")

gen () { # $1 tag  $2 model  $3 data  $4 mathflag  $5 nmax  $6 temp  $7 out
  echo "  [gen] $1 $(basename $3) nmax=$5 temp=$6 -> $7"; date
  $V reasoning/reason_cache.py generate --model "$2" $4 --data "$3" --problems $N \
     --nmax "$5" --temp "$6" --max-tokens $MAXTOK --out "$7" 2>&1 | grep -vE 'it/s|Fetching' | tail -3
}

echo "=== CROSS-FAMILY LADDER START ==="; date
# ---- MATH n=500 first: greedy pass@1 (temp 0, 1 sample), then 16-sample cache ----
for spec in "${LADDER[@]}"; do t=${spec%%:*}; m=${spec#*:}
  gen "$t" "$m" "$MATH" "--math" 1  0.0 "reasoning/cache/llama_math_${t}_greedy.jsonl"
  gen "$t" "$m" "$MATH" "--math" 16 0.8 "reasoning/cache/llama_math_${t}.jsonl"
done
# ---- GSM8K n=500 second: greedy pass@1, then 32-sample cache ----
for spec in "${LADDER[@]}"; do t=${spec%%:*}; m=${spec#*:}
  gen "$t" "$m" "$GSM" ""       1  0.0 "reasoning/cache/llama_gsm8k_${t}_greedy.jsonl"
  gen "$t" "$m" "$GSM" ""       32 0.8 "reasoning/cache/llama_gsm8k_${t}.jsonl"
done
echo "=== CROSS-FAMILY LADDER DONE ==="; date
