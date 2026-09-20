#!/usr/bin/env bash
# R7 G arm — head layer 18, HELD-OUT slice only (tasks 226-300 = slice 4). The fit trained on
# tasks < 226; evaluating on all four slices would put 225 TRAINING tasks in the eval set.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
HEAD="$SP/r7states/head_layer18.npz"; EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1
[ -f "$HEAD" ] || { say "STOP: no head at $HEAD"; exit 1; }
mkdir -p "$SP/r7g"
"$PY" experience/et8_sql_agent.py --tasks "$SP/sqls/s4" --out "$SP/r7g/g_s4" --budget 12 \
   --head "$HEAD" --head-layer 18 >> "$SP/r7g/s4.log" 2>&1
n=$(wc -l < "$SP/r7g/g_s4.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
say "G episodes=$n"
[ "$n" -eq "$EXPECT" ] || { say "STOP: $n of $EXPECT"; tail -20 "$SP/r7g/s4.log"; exit 1; }
say "R7 G COMPLETE -> paired vs the base on the SAME 75"
"$PY" experience/paired_stats.py --base "$SP"/r7base/b_s4.episodes.jsonl \
  --g "$SP"/r7g/g_s4.episodes.jsonl --name "R7 SQL: G vs base on the held-out 75" --expect 75 \
  --out "$SP/r7_g_stats.json"
