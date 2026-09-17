#!/usr/bin/env python3
"""G step 2, measured before it is spent: HOW MANY DISTINCT CANDIDATES DOES THE MODEL OFFER?

    python3 et8_candidate_diversity.py --tasks <slice dir> --runs experience/traj/base_v1_7b \
        --temps 0.8 1.2 --k 8 --out <dir>/candidate_diversity.json

WHY THIS RUNS BEFORE G-TRIVIAL-PRIME AND BEFORE THE HEAD. Both of those SELECT among candidates
the model proposes. A two-task smoke showed k=8 draws at temp 0.8 collapsing to ONE region --
`k_distinct` 1 and 2, `regions_offered` ['producer'] both times. If that holds at n=80, then
selection has nothing to select from and no head, however good, can move the agent: the candidate
set IS the action space, and a one-element action space is not steerable.

That is a result about the MODEL, not a defect of the harness, and it is cheap to establish (one
decision point per task, no episodes, no patching). Spending 80 episodes of G-trivial-prime first
and discovering it afterwards would be the expensive order.

WHAT IT REPORTS, per temperature:
  regions_per_task    distinct REGIONS across k draws -- the quantity selection needs
  actions_per_task    distinct (action, region, bug_class) triples -- a looser diversity
  symptom_offered     share of tasks where ANY draw names the symptom region. This is the CEILING
                      on G-trivial-prime: where the symptom region is never proposed, the control
                      falls back to the greedy action and IS the baseline, by construction.
  bug_offered         share where any draw names the TRUE bug region -- the ceiling on the head.

TEMPERATURE IS A CONFOUND AND IT IS REPORTED, NOT TUNED AWAY. Raising temperature buys diversity
and costs validity; `invalid` counts the draws that stopped parsing as an action. A temperature
that offers four regions by producing three malformed objects has not made the agent steerable.
"""
from __future__ import annotations
import argparse, collections, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True, help="slice dirs (symlinks are fine)")
    ap.add_argument("--task-source", required=True,
                    help="the REAL task directory. The slice dirs hold symlinks, and "
                         "first_decision_states() loads every task id it finds in the step log "
                         "from ONE directory -- pointing it at a slice makes it miss the other 60.")
    ap.add_argument("--runs", nargs="+", required=True, help="baseline runs, for the decision prompt")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--temps", nargs="+", type=float, default=[0.8, 1.2])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import et8_agent as A
    import et8_inject as I
    from mlx_lm import load

    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    want = {os.path.basename(f)[:-5] for d in a.tasks for f in glob.glob(os.path.join(d, "task_*.json"))}
    dec = [d for d in I.first_decision_states(steps, a.task_source) if d["task_id"] in want]
    seen, uniq = set(), []
    for d in dec:                      # the two runs share task ids; one decision per TASK
        if d["task_id"] not in seen:
            seen.add(d["task_id"]); uniq.append(d)
    print(f"  tasks: {len(uniq)} of {len(want)}", file=sys.stderr)

    model, tok = load(a.model)
    out = {"model": a.model, "k": a.k, "n_tasks": len(uniq), "temps": {}}
    for temp in a.temps:
        regs, acts, sym, bug, invalid, rows = [], [], 0, 0, 0, []
        for i, d in enumerate(uniq):
            task = json.load(open(os.path.join(a.task_source, d["task_id"] + ".json")))
            R, T = set(), set()
            for j in range(a.k):
                text, _, _ = A.generate(model, tok, d["messages"], temp=0.0 if j == 0 else temp)
                p = A.parse_action(text)
                if p.get("action") not in ("hypothesize", "inspect", "patch"):
                    invalid += 1; continue
                if p.get("region"):
                    R.add(p["region"])
                T.add((p.get("action"), p.get("region"), p.get("bug_class")))
            regs.append(len(R)); acts.append(len(T))
            sym += task["symptom_region"] in R
            bug += task["bug_region"] in R
            rows.append({"task_id": d["task_id"], "regions": sorted(R), "n_actions": len(T),
                         "symptom_region": task["symptom_region"], "bug_region": task["bug_region"]})
            if (i + 1) % 20 == 0:
                print(f"  temp {temp}  {i+1}/{len(uniq)}", file=sys.stderr)
        n = len(uniq)
        out["temps"][str(temp)] = {
            "mean_distinct_regions": round(sum(regs) / n, 2),
            "region_count_histogram": dict(sorted(collections.Counter(regs).items())),
            "mean_distinct_actions": round(sum(acts) / n, 2),
            "symptom_region_offered_pct": round(100 * sym / n, 1),
            "bug_region_offered_pct": round(100 * bug / n, 1),
            "invalid_draws": invalid, "draws": n * a.k, "rows": rows}
        p = out["temps"][str(temp)]
        print(f"temp {temp}: regions/task {p['mean_distinct_regions']}  hist {p['region_count_histogram']}  "
              f"symptom offered {p['symptom_region_offered_pct']}%  bug offered {p['bug_region_offered_pct']}%  "
              f"invalid {invalid}/{n*a.k}", file=sys.stderr)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
