#!/usr/bin/env bash
# R5-E phase 2: the SELECTED budget on the full 300, paired against G's 300.
# The budget comes from r5e_ladder.json, which applied a rule fixed before the ladder ran.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
M="Qwen/Qwen2.5-7B-Instruct"; EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1
B=$("$PY" -c "import json;print(json.load(open('$SP/r5e_ladder.json'))['SELECTED_BUDGET'])" 2>/dev/null)
[ -n "$B" ] || { say "STOP: no SELECTED_BUDGET in r5e_ladder.json"; exit 1; }
say "R5-E 300: base, NO head, budget $B (selected by the pre-fixed nearest-to-9170 rule)"
mkdir -p "$SP/r5e300"
for s in 1 2 3 4; do
  # slice 1 was already run at every ladder budget -- reuse it rather than re-run the same agent
  if [ "$s" = "1" ] && [ -f "$SP/r5e/b${B}_s1.episodes.jsonl" ]; then
    cp "$SP/r5e/b${B}_s1.episodes.jsonl" "$SP/r5e300/r5e_s1.episodes.jsonl"
    cp "$SP/r5e/b${B}_s1.steps.jsonl"    "$SP/r5e300/r5e_s1.steps.jsonl"
    say "slice 1 REUSED from the ladder (identical agent: budget $B, same tasks, temp 0)"
  else
    "$PY" experience/et8_agent.py --tasks "$SP/v3s/s$s" --out "$SP/r5e300/r5e_s$s" \
       --model "$M" --budget "$B" --inspect-first >> "$SP/r5e300/s$s.log" 2>&1
  fi
  n=$(wc -l < "$SP/r5e300/r5e_s$s.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
  say "slice $s episodes=$n"
  [ "$n" -eq "$EXPECT" ] || { say "STOP: slice $s produced $n of $EXPECT."; exit 1; }
done
say "R5-E 300/300 COMPLETE -> paired stats vs G's 300"
"$PY" experience/paired_stats.py \
  --base "$SP"/r5e300/r5e_s1.episodes.jsonl "$SP"/r5e300/r5e_s2.episodes.jsonl \
         "$SP"/r5e300/r5e_s3.episodes.jsonl "$SP"/r5e300/r5e_s4.episodes.jsonl \
  --g "$SP"/v3g300/v3g300_s1.episodes.jsonl "$SP"/v3g300/v3g300_s2.episodes.jsonl \
      "$SP"/v3g300/v3g300_s3.episodes.jsonl "$SP"/v3g300/v3g300_s4.episodes.jsonl \
  --name "R5-E: base at budget $B (compute-matched) vs G at budget 12" --expect 300 \
  --out "$SP/r5e_300_stats.json"
