#!/usr/bin/env bash
# R7 base: no head, budget 12, 4 slices of 75. A slice under 75 STOPS the chain (the rule R4 earned).
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
EXPECT=75; say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1; mkdir -p "$SP/r7base"
for s in 1 2 3 4; do
  "$PY" experience/et8_sql_agent.py --tasks "$SP/sqls/s$s" --out "$SP/r7base/b_s$s" --budget 12 \
     >> "$SP/r7base/s$s.log" 2>&1
  n=$(wc -l < "$SP/r7base/b_s$s.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
  say "slice $s episodes=$n"
  [ "$n" -eq "$EXPECT" ] || { say "STOP: slice $s produced $n of $EXPECT."; tail -20 "$SP/r7base/s$s.log"; exit 1; }
done
say "R7 BASE 300/300 COMPLETE"
"$PY" - "$SP" <<'PY'
import json,sys,glob,collections
SP=sys.argv[1]; e=[]
for f in sorted(glob.glob(SP+"/r7base/b_s*.episodes.jsonl")): e+=[json.loads(l) for l in open(f)]
n=len(e); g=sum(x["green"] for x in e)
ch=sum(x["chance_pct"] for x in e)/n
print("  R7 BASE: %d/%d = %.1f%%  actions %.2f  tokens/ep %.1f  mean chance %.1f%%"%(
  g,n,100*g/n,sum(x["actions"] for x in e)/n,
  sum(x.get("tokens_in",0)+x.get("tokens_out",0) for x in e)/n,ch))
print("  CEILING CHECK (pre-registered): base must be BELOW ~80%% or R7 cannot answer -> %s"%
      ("PASSES, there is headroom" if 100*g/n<80 else "FAILS, no headroom"))
PY
