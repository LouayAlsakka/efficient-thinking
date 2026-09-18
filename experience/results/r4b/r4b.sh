#!/usr/bin/env bash
# R4 second model (理 11300): a DIFFERENT FAMILY at comparable size. Llama-3.1-8B-Instruct-4bit.
# Floor 5% on 300, unchanged. If it floors, R4 = (c) and NO THIRD MODEL is run.
#
# THE GUARD r4.sh LACKED, now a rule: a slice that does not produce its full 75 episodes STOPS THE
# CHAIN. The first R4 run reported 262 episodes and 262 is not 300 -- a crash inside parse_action
# killed the run and the chain carried on as though the arm were complete. A short slice is not a
# smaller measurement, it is an unfinished one, and a rate computed over it is over an unknown
# population.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
M="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1
mkdir -p "$SP/r4b_base"
say "R4b START model=$M expect=${EXPECT}/slice x4 = 300"
for s in 1 2 3 4; do
  "$PY" experience/et8_agent.py --tasks "$SP/v3s/s$s" --out "$SP/r4b_base/r4b8_s$s" \
     --model "$M" --budget 12 --inspect-first >> "$SP/r4b_base/s$s.log" 2>&1
  rc=$?
  n=$(wc -l < "$SP/r4b_base/r4b8_s$s.episodes.jsonl" 2>/dev/null | tr -d ' ')
  n=${n:-0}
  say "slice $s rc=$rc episodes=$n"
  if [ "$n" -ne "$EXPECT" ]; then
    say "STOP: slice $s produced $n of $EXPECT episodes. The chain refuses to continue on a short"
    say "      slice. Fix the cause, re-run this slice, and do NOT quote a rate over the partial set."
    tail -25 "$SP/r4b_base/s$s.log"
    exit 1
  fi
done
say "R4b BASE COMPLETE: 300/300. The floor check decides whether any probe is fit."
