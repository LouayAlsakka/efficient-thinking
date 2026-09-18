#!/usr/bin/env bash
# R4 G arm — the Llama head WIRED, on the HELD-OUT quarter only.
#
# SLICE 4 ONLY, AND THIS IS NOT AN ECONOMY. cmd_fit trains on tasks < cut (~226); slice 4 is tasks
# 226-300, which the head has never seen. Running all four slices would put ~225 TRAINING tasks in
# the eval set -- leakage, and no bar can be read through it. n = 75.
#
# The paired base is already measured: r4b8_s4, the same model on the same 75 with no head.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
M="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
HEAD="$SP/r4b_states.npz/head_layer18.npz"
EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1

[ -f "$HEAD" ] || { say "STOP: no head at $HEAD. The fit did not produce one; nothing to wire."; exit 1; }
mkdir -p "$SP/r4b_g"
say "G WIRED, head layer 18, held-out slice 4 (tasks 226-300)"
"$PY" experience/et8_agent.py --tasks "$SP/v3s/s4" --out "$SP/r4b_g/r4bg_s4" \
   --model "$M" --budget 12 --inspect-first \
   --head "$HEAD" --head-layer 18 >> "$SP/r4b_g/s4.log" 2>&1
rc=$?; n=$(wc -l < "$SP/r4b_g/r4bg_s4.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
say "G rc=$rc episodes=$n"
if [ "$n" -ne "$EXPECT" ]; then
  say "STOP: $n of $EXPECT episodes. No rate is quoted over a partial slice."
  tail -25 "$SP/r4b_g/s4.log"; exit 1
fi
say "R4 G COMPLETE — paired against r4b8_s4 (same model, same 75, no head)"
"$PY" - "$SP" <<'PY'
import json,sys
SP=sys.argv[1]
def g(p): return [json.loads(l) for l in open(p)]
b={e["task_id"]:e for e in g(SP+"/r4b_base/r4b8_s4.episodes.jsonl")}
h={e["task_id"]:e for e in g(SP+"/r4b_g/r4bg_s4.episodes.jsonl")}
k=sorted(set(b)&set(h))
bg=sum(b[t]["green"] for t in k); hg=sum(h[t]["green"] for t in k)
print("  paired on %d tasks:  base %d/%d = %.1f%%   G %d/%d = %.1f%%   delta %+.1f pts"
      %(len(k),bg,len(k),100*bg/len(k),hg,len(k),100*hg/len(k),100*(hg-bg)/len(k)))
print("  actions: base %.2f  G %.2f"%(sum(b[t]["actions"] for t in k)/len(k),
                                      sum(h[t]["actions"] for t in k)/len(k)))
PY
