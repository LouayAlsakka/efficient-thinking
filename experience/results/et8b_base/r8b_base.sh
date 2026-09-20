#!/usr/bin/env bash
# ET-8b G4: the base LOOP's own base — no head, 300 episodes, budget 12 across up to 3 decisions.
# The 8a single-decision numbers are NEVER reused as this base (gate G4's own words).
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
EXPECT=75; say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1; mkdir -p "$SP/r8b_base"
for s in 1 2 3 4; do
  "$PY" experience/et8b_loop.py --tasks "$SP/v3s/s$s" --out "$SP/r8b_base/lb_s$s" --budget 12 \
     >> "$SP/r8b_base/s$s.log" 2>&1
  n=$(wc -l < "$SP/r8b_base/lb_s$s.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
  say "slice $s episodes=$n"
  [ "$n" -eq "$EXPECT" ] || { say "STOP: slice $s produced $n of $EXPECT"; tail -20 "$SP/r8b_base/s$s.log"; exit 1; }
done
say "8b BASE LOOP 300/300 COMPLETE -> G1, G2 gates"
"$PY" - "$SP" <<'PY'
import json,sys,glob,collections
SP=sys.argv[1]
eps=[];steps=[]
for f in sorted(glob.glob(SP+"/r8b_base/lb_s*.episodes.jsonl")): eps+=[json.loads(l) for l in open(f)]
for f in sorted(glob.glob(SP+"/r8b_base/lb_s*.steps.jsonl")): steps+=[json.loads(l) for l in open(f)]
n=len(eps); g=sum(x["green"] for x in eps)
print("  BASE LOOP: %d/%d = %.1f%%  decisions %.2f  actions %.2f  tokens/ep %.1f"%(
 g,n,100*g/n,sum(x["decisions"] for x in eps)/n,sum(x["actions"] for x in eps)/n,
 sum(x.get("tokens_in",0)+x.get("tokens_out",0) for x in eps)/n))
print("  G1 task gate: distinct task_ids %d / %d episodes = %.3f  (>= 0.9) -> %s"%(
 len({e["task_id"] for e in eps}),n,len({e["task_id"] for e in eps})/n,
 "PASSES" if len({e["task_id"] for e in eps})/n>=0.9 else "FAILS"))
print("  G2 prompt gate, PER DECISION (distinct decision states / episodes reaching it):")
for d in (1,2,3):
    hy=[s for s in steps if s.get("action")=="hypothesize" and s.get("decision")==d and s.get("decision_state")]
    if not hy: print("    decision %d: none reached"%d); continue
    epi=len({s["task_id"] for s in hy}); dis=len({s["decision_state"] for s in hy})
    print("    decision %d: %d distinct / %d episodes = %.3f  -> %s"%(
      d,dis,epi,dis/epi,"PASSES" if dis/epi>=0.9 else "FAILS — not scored"))
PY
