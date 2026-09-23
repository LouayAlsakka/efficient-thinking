#!/usr/bin/env bash
# R5-E ladder (理: give the BASE more compute and ask whether it buys G's success.
#
# WHY THIS EXPERIMENT EXISTS. R2's "+34%" compared TWO POINTS AT DIFFERENT COMPUTE, so it decides
# nothing about the frontier. The reviewer's criterion: an experience prior must improve the
# QUALITY-COST FRONTIER. So spend G's total budget on the base instead and see what it buys.
#
#   base budget 12   18.7%   6,854 model tokens/episode
#   G    budget 12   31.7%   6,706 model tokens + the controller's candidate passes  ~= 9,170 all-in
#   target                   9,170 tokens/episode for the base, bought with a LARGER BUDGET
#
# THE SELECTION RULE IS FIXED BEFORE THE LADDER RUNS (理): measure mean tokens/episode on SLICE 1
# for budgets 14, 16, 18; pick the budget NEAREST 9,170 by that measurement; run THAT budget on the
# full 300. Nearest, not "nearest from below" and not "whichever reads best" -- so the choice cannot
# be made by its answer.
#
# EACH BUDGET IS A DIFFERENT AGENT. The budget is in the prompt. A budget-18 arm is not a budget-12
# arm truncated; it is told it has 18 actions. Labelled as such everywhere, never read as a
# truncation of anything.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
M="Qwen/Qwen2.5-7B-Instruct"
EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1
mkdir -p "$SP/r5e"
say "R5-E LADDER: base, NO head, budgets 14/16/18 on slice 1. Target 9,170 tok/ep."
for b in 14 16 18; do
  "$PY" experience/et8_agent.py --tasks "$SP/v3s/s1" --out "$SP/r5e/b${b}_s1" \
     --model "$M" --budget "$b" --inspect-first >> "$SP/r5e/b${b}_s1.log" 2>&1
  rc=$?; n=$(wc -l < "$SP/r5e/b${b}_s1.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
  say "budget $b rc=$rc episodes=$n"
  if [ "$n" -ne "$EXPECT" ]; then
    say "STOP: budget $b produced $n of $EXPECT. No rate over a partial slice."
    tail -20 "$SP/r5e/b${b}_s1.log"; exit 1
  fi
done
say "LADDER COMPLETE -- the selection rule now runs on the measurements"
"$PY" - "$SP" <<'PY'
import json,sys
SP=sys.argv[1]; TARGET=9170.0
rows=[]
for b in (14,16,18):
    e=[json.loads(l) for l in open("%s/r5e/b%d_s1.episodes.jsonl"%(SP,b))]
    n=len(e); tok=sum(x.get("tokens_in",0)+x.get("tokens_out",0) for x in e)/n
    rows.append({"budget":b,"n":n,"tokens_per_episode":round(tok,1),
                 "distance_to_target":round(abs(tok-TARGET),1),
                 "success_pct_slice1":round(100*sum(x["green"] for x in e)/n,1),
                 "actions":round(sum(x["actions"] for x in e)/n,2)})
pick=min(rows,key=lambda r:r["distance_to_target"])
out={"document":"R5-E ladder — slice 1, the budget selection (理",
 "target_tokens_per_episode":TARGET,
 "rule_fixed_before_the_run":"pick the budget NEAREST the target by measured mean tokens/episode on "
   "slice 1. Nearest in absolute distance -- not nearest-from-below, not whichever reads best.",
 "ladder":rows,"SELECTED_BUDGET":pick["budget"],
 "note":"slice-1 success is printed for completeness and is NOT the result: n=75, and the arm that "
        "matters is the SELECTED budget on the full 300, paired against G's 300.",
 "each_budget_is_a_different_agent":"the budget is in the prompt; a budget-18 arm is told it has 18 "
        "actions. It is not a truncated budget-12 arm."}
json.dump(out,open(SP+"/r5e_ladder.json","w"),indent=1)
print(json.dumps(out,indent=1))
PY
