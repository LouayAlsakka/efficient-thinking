#!/usr/bin/env bash
# R6 — the strongest SIMPLE baselines at G's own compute (理 11398). Readings pre-written in
# experience/et8_r6_baselines.py, before this ran.
#
#   (a) self-rerank    the frozen model's own mean log-prob over the SAME enumerated candidates
#   (b) majority vote  5 sampled hypotheses at that decision
#   (c) base + search  = R5-E's budget-16 arm, ALREADY MEASURED (22.3%) — not re-run
#
# Both arms go through et8_agent's OWN head_state plumbing with an identity linear layer, so they
# take exactly the decision G takes. Nothing in the measured agent is edited.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1
mkdir -p "$SP/r6"
for arm in rerank vote; do
  for s in 1 2 3 4; do
    "$PY" "$SP/r6_run.py" --arm "$arm" --tasks "$SP/v3s/s$s" --out "$SP/r6/${arm}_s$s" \
       >> "$SP/r6/${arm}_s$s.log" 2>&1
    n=$(wc -l < "$SP/r6/${arm}_s$s.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
    say "$arm slice $s episodes=$n"
    [ "$n" -eq "$EXPECT" ] || { say "STOP: $arm slice $s produced $n of $EXPECT."; tail -20 "$SP/r6/${arm}_s$s.log"; exit 1; }
  done
  say "$arm: 300/300"
done
say "R6 COMPLETE -> paired stats vs base@12 and vs G@12"
for arm in rerank vote; do
  "$PY" experience/paired_stats.py \
    --base "$SP"/v3b/v3b_s{1,2,3,4}.episodes.jsonl \
    --g "$SP"/r6/${arm}_s{1,2,3,4}.episodes.jsonl \
    --name "R6 $arm vs base@12" --expect 300 --out "$SP/r6_${arm}_vs_base.json"
  "$PY" experience/paired_stats.py \
    --base "$SP"/r6/${arm}_s{1,2,3,4}.episodes.jsonl \
    --g "$SP"/v3g300/v3g300_s{1,2,3,4}.episodes.jsonl \
    --name "R6 G@12 vs $arm (positive = G better)" --expect 300 --out "$SP/r6_g_vs_${arm}.json"
done
