#!/usr/bin/env python3
"""
et8_verify.py — Efficient Thinking VIII VERIFY gate v0 (proposal §3.3): does a candidate lesson generalize?

A lesson is ADMITTED only if, on HELD-OUT tasks in its scope, giving the frozen model that ONE lesson (as text memory,
mechanism A — the cheapest carrier, used here purely as the test instrument) improves decisions relative to no memory, AND
does not improve decisions on a SHUFFLED-FAMILY control (tasks outside its scope with the same symptom region), which would
mean the lesson is a generic prompt effect rather than experience about this condition.

Decision metric (per episode): actions-to-green if green, else budget+1; secondary: distinct regions touched, wasted steps.
Admit if:  mean_actions(in-scope, with lesson) <= mean_actions(in-scope, without) - MIN_GAIN  on n >= MIN_N episodes,
       and mean_actions(control, with) >= mean_actions(control, without) - CONTROL_TOL  (no leak).
Register the REJECTION RATE (P0): the fraction of candidate lessons this gate refuses.

Usage:
  python experience/et8_verify.py --lessons experience/lessons/v0/lessons.json --tasks experience/tasks/v1 \
      --model Qwen/Qwen2.5-3B-Instruct --n 20 --out experience/lessons/v0/verified.json
Cost: per lesson, 2 conditions x 2 arms x n episodes. Run on llm1 (Sautee); a Studio can do 1-2 lessons for a smoke.
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys, uuid, time
sys.path.insert(0, os.path.dirname(__file__))
import et8_agent as A
import et8_distill as D

MIN_GAIN, CONTROL_TOL, MIN_N = 1.0, 0.5, 8   # actions; conservative for v0, tuned on P0 after the first run

def lesson_text(L: dict) -> str:
    return D.render_text([L])

def pick_tasks(task_dir: str, family: str | None, symptom_region: str, n: int, rng: random.Random, exclude_family: str | None = None):
    files = sorted(glob.glob(os.path.join(task_dir, "task_*.json")))
    pool = []
    for f in files:
        t = json.load(open(f))
        if t["symptom_region"] != symptom_region: continue
        if family and t["family"] != family: continue
        if exclude_family and t["family"] == exclude_family: continue
        pool.append(t)
    rng.shuffle(pool)
    return pool[:n]

def run_arm(model, tok, tasks, memory, budget, run_id, log, model_id):
    out = []
    for t in tasks:
        s = A.run_episode(model, tok, t, budget, memory, run_id, log, model_id)
        out.append(s)
    return out

def cost(summ) -> float:
    return sum((s["actions"] if s["green"] else s["actions"] + 1) for s in summ) / max(1, len(summ))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lessons", required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="verify only the first N lessons (smoke)")
    a = ap.parse_args()
    lessons = json.load(open(a.lessons))
    if a.limit: lessons = lessons[: a.limit]
    rng = random.Random(a.seed)
    model, tok = A.load_model(a.model)
    run_id = "verify-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    log = open(a.out.replace(".json", "") + ".steps.jsonl", "a")
    results = []
    for L in lessons:
        fam, sym = L["condition"]["family"], L["condition"]["symptom_region"]
        if not L["action_prior"] and not L["action_avoid"]:
            results.append({**L, "status": "rejected", "reason": "empty lesson"}); continue
        in_scope = pick_tasks(a.tasks, fam, sym, a.n, rng)
        control = pick_tasks(a.tasks, None, sym, a.n, rng, exclude_family=fam)
        if len(in_scope) < MIN_N:
            results.append({**L, "status": "unevaluable", "reason": f"only {len(in_scope)} held-out tasks in scope"}); continue
        mem = lesson_text(L)
        base_in = run_arm(model, tok, in_scope, None, a.budget, run_id, log, a.model)
        with_in = run_arm(model, tok, in_scope, mem, a.budget, run_id, log, a.model)
        base_ctl = run_arm(model, tok, control, None, a.budget, run_id, log, a.model) if control else []
        with_ctl = run_arm(model, tok, control, mem, a.budget, run_id, log, a.model) if control else []
        gain = cost(base_in) - cost(with_in)
        leak = (cost(base_ctl) - cost(with_ctl)) if control else 0.0
        admitted = gain >= MIN_GAIN and leak <= CONTROL_TOL
        results.append({**L, "status": "admitted" if admitted else "rejected",
                        "confound_check": {"in_scope": {"n": len(in_scope), "cost_base": round(cost(base_in), 2), "cost_with": round(cost(with_in), 2),
                                                        "green_base": sum(s["green"] for s in base_in), "green_with": sum(s["green"] for s in with_in)},
                                           "control": {"n": len(control), "cost_base": round(cost(base_ctl), 2) if control else None,
                                                       "cost_with": round(cost(with_ctl), 2) if control else None},
                                           "gain": round(gain, 2), "leak": round(leak, 2), "rule": f"gain>={MIN_GAIN} and leak<={CONTROL_TOL}"},
                        "verified_by": {"run_id": run_id, "model": a.model, "seed": a.seed}})
        print(f"{fam}|{sym}: gain={gain:+.2f} leak={leak:+.2f} -> {'ADMIT' if admitted else 'reject'}", file=sys.stderr)
    n_eval = sum(1 for r in results if r["status"] in ("admitted", "rejected"))
    summary = {"lessons": len(results), "admitted": sum(1 for r in results if r["status"] == "admitted"),
               "rejected": sum(1 for r in results if r["status"] == "rejected"),
               "unevaluable": sum(1 for r in results if r["status"] == "unevaluable"),
               "rejection_rate_P0": round(sum(1 for r in results if r["status"] == "rejected") / n_eval, 3) if n_eval else None,
               "rule": {"MIN_GAIN": MIN_GAIN, "CONTROL_TOL": CONTROL_TOL, "MIN_N": MIN_N}, "run_id": run_id}
    json.dump({"summary": summary, "lessons": results}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
