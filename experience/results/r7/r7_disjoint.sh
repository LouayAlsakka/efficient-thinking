#!/usr/bin/env bash
# R7 DISJOINT leg — seed 89 (seed 47 FAILED P3 at 61.3% and its run was stopped), ZERO signature overlap with seed 21 (enforced by --exclude, verified).
# The head was fitted on seed 21, so ALL 300 here are valid eval. BASE first: a G number without its
# paired base is a number about nothing.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
HEAD="$SP/r7states/head_layer18.npz"; EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1; [ -f "$HEAD" ] || { say "STOP: no head"; exit 1; }
mkdir -p "$SP/r7d_base" "$SP/r7d_g"
run(){ local arm="$1" dir="$2" pre="$3"; shift 3
  for s in 1 2 3 4; do
    "$PY" experience/et8_sql_agent.py --tasks "$SP/sql89s/s$s" --out "$dir/${pre}_s$s" --budget 12 \
       "$@" >> "$dir/s$s.log" 2>&1
    local n; n=$(wc -l < "$dir/${pre}_s$s.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
    say "$arm slice $s episodes=$n"
    [ "$n" -eq "$EXPECT" ] || { say "STOP: $arm slice $s produced $n of $EXPECT"; tail -15 "$dir/s$s.log"; return 1; }
  done; }
say "R7 DISJOINT: seed 89, BASE arm first"
run BASE "$SP/r7d_base" db || exit 1
say "BASE 300/300 -> G arm, head layer 18"
run G "$SP/r7d_g" dg --head "$HEAD" --head-layer 18 || exit 1
say "R7 DISJOINT COMPLETE 300/300 both arms"
"$PY" experience/paired_stats.py \
  --base "$SP"/r7d_base/db_s{1,2,3,4}.episodes.jsonl \
  --g "$SP"/r7d_g/dg_s{1,2,3,4}.episodes.jsonl \
  --name "R7 SQL DISJOINT (seed 47, zero signature overlap): G vs base" --expect 300 \
  --out "$SP/r7_disjoint_stats.json"
