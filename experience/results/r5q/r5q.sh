#!/usr/bin/env bash
# R5-Q (理 11364): G at SMALLER budgets. The smallest budget whose success >= 18.7% on the 300,
# and its tokens/episode against the base's 6,854.
#
# SAME HEAD AS THE MEASURED G-300, checked rather than assumed: v3repstates/head_layer18.npz --
# the seed-47 head transferred to seed 21. A different head would not be comparable with 31.7%.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
M="Qwen/Qwen2.5-7B-Instruct"; HEAD="$SP/v3repstates/head_layer18.npz"; EXPECT=75
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1
[ -f "$HEAD" ] || { say "STOP: head missing at $HEAD"; exit 1; }
mkdir -p "$SP/r5q"
for b in 10 8 6; do
  for s in 1 2 3 4; do
    "$PY" experience/et8_agent.py --tasks "$SP/v3s/s$s" --out "$SP/r5q/g${b}_s$s" \
       --model "$M" --budget "$b" --inspect-first --head "$HEAD" --head-layer 18 \
       >> "$SP/r5q/g${b}_s$s.log" 2>&1
    n=$(wc -l < "$SP/r5q/g${b}_s$s.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
    [ "$n" -eq "$EXPECT" ] || { say "STOP: G budget $b slice $s produced $n of $EXPECT."; exit 1; }
  done
  say "G budget $b: 300/300"
done
say "R5-Q COMPLETE"
"$PY" - "$SP" <<'PY'
import json,sys
SP=sys.argv[1]; BAR=18.7; BASE_TOK=6854.4
rows=[]
for b in (10,8,6):
    e=[]
    for s in (1,2,3,4): e+=[json.loads(l) for l in open("%s/r5q/g%d_s%d.episodes.jsonl"%(SP,b,s))]
    n=len(e); g=sum(x["green"] for x in e)
    rows.append({"budget":b,"n":n,"success_pct":round(100*g/n,1),
                 "tokens_per_episode":round(sum(x.get("tokens_in",0)+x.get("tokens_out",0) for x in e)/n,1),
                 "actions":round(sum(x["actions"] for x in e)/n,2),
                 "clears_base_18.7":100*g/n>=BAR})
ok=[r for r in rows if r["clears_base_18.7"]]
pick=min(ok,key=lambda r:r["budget"]) if ok else None
out={"document":"R5-Q — G at smaller budgets (理 11364)",
 "bar":"success >= %.1f%% (the base's own 300 at budget 12)"%BAR,
 "base_tokens_per_episode":BASE_TOK,"ladder":rows,
 "SMALLEST_BUDGET_CLEARING_THE_BAR":(pick["budget"] if pick else None),
 "its_tokens_per_episode":(pick["tokens_per_episode"] if pick else None),
 "reading":("G reaches the base's success at %s model tokens/episode vs the base's %.1f"
            %(pick["tokens_per_episode"],BASE_TOK) if pick else
            "NO budget in {10,8,6} clears the base's 18.7%% -- G cannot buy the base's success more cheaply on this ladder."),
 "⚠️ model tokens only":"these are MODEL tokens. The controller's candidate passes are NOT included "
   "here and must be added before any frontier claim -- that addition is what made R2 read +34%."}
json.dump(out,open(SP+"/r5q.json","w"),indent=1,ensure_ascii=False)
print(json.dumps(out,indent=1,ensure_ascii=False))
PY
