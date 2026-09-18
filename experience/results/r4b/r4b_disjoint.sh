#!/usr/bin/env bash
# R4 disjoint leg — r4_prereg.json: "G on the held-out quarter AND THEN ON A DISJOINT SET."
#
# WHY ALL 300 HERE AND ONLY 75 ON v3. The head was fitted on v3 tasks < 226. It has seen NOTHING of
# v3rep -- a different seed, disjoint problems -- so every one of v3rep's 300 is valid eval. That
# makes this the STRONGER leg, not a bigger version of the same one: slice 4 tests held-out TASKS
# from the fitting distribution; v3rep tests a distribution the head never touched.
#
# ⚠️ READ THIS BEFORE BELIEVING THE BANNER. et8_agent prints "G head loaded: ... trained on tasks
# < 226" on every run. v3rep REUSES THE SAME task_id STRINGS (task_0001..task_0300) for entirely
# different programs -- verified: same id, different seed, different program text. So on THIS set
# that 226 is a v3 index and means nothing, and a later reader must not conclude that 225 of these
# 300 were training tasks. The disjointness here is BY CONSTRUCTION (a different seed generates
# different programs), not by task index. train_task_max is printed, never enforced -- checked.
#
# BOTH ARMS RUN HERE. The base for this model on v3rep does not exist yet, and a G number without
# its paired base is a number about nothing.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
M="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
HEAD="$SP/r4b_states.npz/head_layer18.npz"
EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1
[ -f "$HEAD" ] || { say "STOP: no head at $HEAD"; exit 1; }
mkdir -p "$SP/r4b_rep_base" "$SP/r4b_rep_g"

run(){ # arm dir prefix extra...
  local arm="$1" dir="$2" pre="$3"; shift 3
  for s in 1 2 3 4; do
    "$PY" experience/et8_agent.py --tasks "$SP/v3rs/s$s" --out "$dir/${pre}_s$s" \
       --model "$M" --budget 12 --inspect-first "$@" >> "$dir/s$s.log" 2>&1
    local rc=$? n; n=$(wc -l < "$dir/${pre}_s$s.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
    say "$arm slice $s rc=$rc episodes=$n"
    if [ "$n" -ne "$EXPECT" ]; then
      say "STOP: $arm slice $s produced $n of $EXPECT. The chain refuses to continue on a short slice."
      tail -25 "$dir/s$s.log"; return 1
    fi
  done
}

say "R4 DISJOINT: v3rep (seed 47), BASE arm first -- a G number without its paired base is a number about nothing"
run BASE "$SP/r4b_rep_base" r4brb || exit 1
say "BASE 300/300 -> G arm, head layer 18"
run G "$SP/r4b_rep_g" r4brg --head "$HEAD" --head-layer 18 || exit 1
say "R4 DISJOINT COMPLETE 300/300 both arms"
"$PY" - "$SP" <<'PY'
import json,sys
SP=sys.argv[1]
def load(d,p):
    o={}
    for s in (1,2,3,4):
        for l in open("%s/%s_s%d.episodes.jsonl"%(d,p,s)): 
            e=json.loads(l); o[e["task_id"]]=e
    return o
b=load(SP+"/r4b_rep_base","r4brb"); h=load(SP+"/r4b_rep_g","r4brg")
k=sorted(set(b)&set(h)); n=len(k)
bg=sum(b[t]["green"] for t in k); hg=sum(h[t]["green"] for t in k)
print("  paired on %d tasks:  base %d = %.1f%%   G %d = %.1f%%   delta %+.1f pts"
      %(n,bg,100*bg/n,hg,100*hg/n,100*(hg-bg)/n))
print("  actions: base %.2f  G %.2f"%(sum(b[t]["actions"] for t in k)/n,
                                      sum(h[t]["actions"] for t in k)/n))
PY
