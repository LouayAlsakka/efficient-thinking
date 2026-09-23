#!/usr/bin/env bash
# R5-C (理, amended 11397): the frontier as a CURVE, not one point.
#
# BASE budgets 8, 12, 14, 16, 18, 24  (~5k -> ~15k tokens/episode)   12 and 16 already measured
# HEAD budgets 6, 8, 10, 12                                          12 measured; 6/8/10 from R5-Q
#
# SLICE-1 REUSE, and why it is legitimate: the ladder already ran budgets 14 and 18 on slice 1 with
# THIS EXACT AGENT (same model, same tasks, same budget, temp 0). R2d's calibration measured
# uncached re-runs as 40/40 byte-identical on this hardware. Budgets 8 and 24 have no slice 1 and
# run all four.
#
# EACH BUDGET IS A DIFFERENT AGENT. The budget is in the prompt. No point on either curve is a
# truncation or extension of another, and the table must never be read that way.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
M="Qwen/Qwen2.5-7B-Instruct"; EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1
mkdir -p "$SP/r5c"
for b in 14 18 8 24; do          # 14/18 first: they are 3 slices each, so the curve fills sooner
  for s in 1 2 3 4; do
    out="$SP/r5c/b${b}_s${s}"
    if [ "$s" = "1" ] && [ -f "$SP/r5e/b${b}_s1.episodes.jsonl" ]; then
      cp "$SP/r5e/b${b}_s1.episodes.jsonl" "${out}.episodes.jsonl"
      cp "$SP/r5e/b${b}_s1.steps.jsonl"    "${out}.steps.jsonl"
      say "b$b slice 1 REUSED from the ladder (identical agent)"
    else
      "$PY" experience/et8_agent.py --tasks "$SP/v3s/s$s" --out "$out" \
         --model "$M" --budget "$b" --inspect-first >> "$SP/r5c/b${b}_s${s}.log" 2>&1
    fi
    n=$(wc -l < "${out}.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
    [ "$n" -eq "$EXPECT" ] || { say "STOP: base b$b slice $s produced $n of $EXPECT."; exit 1; }
  done
  say "BASE budget $b: 300/300"
done
say "R5-C BASE CURVE COMPLETE -> building the table"
"$PY" "$SP/r5c_table.py" "$SP"
