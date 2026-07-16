#!/bin/bash
# Queued overnight: after the current MATH generation frees the machine, boost the flip-decider power.
#  (1) extend MATH 7B/14B caches to the full 500 (the max from MATH-500; true n~1000 needs the full MATH test set)
#  (2) mode-correctness: TRUE greedy (temp 0) single-shot comparators for the 'no-search' arm, replacing the
#      temp-0.8 sample[0] proxy the first McNemar used.
# McNemar is re-scored next session once these land (reasoning/mcnemar_flips.py, + a --greedy variant).
cd "$(dirname "$0")/.." || exit 1
LOG=reasoning/results/flip_decider.log; V=./.venv/bin/python
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
say "QUEUED flip-decider armed — waiting for MATH generation to free the machine"
while pgrep -f "reason_cache.py generate --math" >/dev/null 2>&1; do sleep 300; done
say "machine free — starting"
for t in 7B 14B; do
  say "extend MATH $t -> 500"
  caffeinate -i $V reasoning/reason_cache.py generate --math \
    --model mlx-community/Qwen2.5-${t}-Instruct-4bit --problems 500 --nmax 16 --max-tokens 1536 \
    --out reasoning/cache/math_${t}.jsonl >> "$LOG" 2>&1
done
for spec in "gsm8k:14B" "gsm8k:32B" "math:14B"; do
  bench=${spec%%:*}; t=${spec#*:}; mf=""; [ "$bench" = "math" ] && mf="--math"
  say "greedy comparator $bench $t (temp 0, nmax 1)"
  caffeinate -i $V reasoning/reason_cache.py generate $mf \
    --model mlx-community/Qwen2.5-${t}-Instruct-4bit --temp 0 --nmax 1 --problems 500 \
    --out reasoning/cache/${bench}_${t}_greedy.jsonl >> "$LOG" 2>&1
done
say "DONE — MATH-500 caches + true-greedy comparators ready for McNemar re-score"
