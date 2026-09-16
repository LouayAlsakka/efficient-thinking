#!/usr/bin/env python3
"""Mechanism G, step 1: is the productive/wasted probe reading the DECISION or the TASK?

    python3 et8_probe_taskcond.py --runs experience/traj/base_v1_7b --tasks experience/tasks/v1 \
        --model Qwen/Qwen2.5-7B-Instruct --layers 18 27 --out experience/results/g_probe_taskcond.json

理's gate (README:84): fit the probe WITHIN task. If d' >= 2 within task, the vector carries a
decision signal and G proceeds. If not, the vector reads DIFFICULTY — it separates hard tasks from
easy ones, which is useless for choosing between candidate actions inside one task — and G stops.

WHY THE POOLED d' OF 3.1-4.9 CANNOT ANSWER THIS. Pooled across tasks, a state that merely encodes
"this task is hard" separates tau+ from tau- almost perfectly, because hard tasks generate most of
the wasted steps. The pooled number is real and it is not evidence of a decision signal. The
within-task number is the one that discriminates the two readings, and it is the only one 理 will
read before step 2.

TWO ESTIMATORS, both reported, because they fail differently:
  demeaned   subtract each task's own mean state, then fit one vector over everything. Uses every
             task, including those with only one label present — where it contributes nothing but
             noise-free zero.
  within     fit and score ONLY on tasks that carry BOTH labels. Fewer tasks, but every number in
             it is a real within-task contrast.
A gap between them is itself informative: if `demeaned` passes and `within` fails, the signal is
coming from tasks that never showed both outcomes.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def d_prime(p, n):
    if len(p) < 2 or len(n) < 2:
        return None
    return float((p.mean() - n.mean()) / (np.sqrt(0.5 * (p.var() + n.var())) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layers", nargs="+", type=int, default=[18, 27])
    ap.add_argument("--limit", type=int, default=0, help="cap decisions (smoke)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import et8_inject as I
    from mlx_lm import load
    import mlx.core as mx

    steps = []
    for r in a.runs:
        p = r + ".steps.jsonl"
        steps += [json.loads(l) for l in open(p)]
    decisions = I.first_decision_states(steps, a.tasks)
    if a.limit:
        decisions = decisions[: a.limit]
    print(f"  decisions: {len(decisions)}", file=sys.stderr)

    model, tok = load(a.model)
    by_task = {}
    for i, d in enumerate(decisions):
        hs = I.hidden_at_layers(model, tok, d["messages"], a.layers)
        rec = by_task.setdefault(d["task_id"], {"pos": {l: [] for l in a.layers},
                                                "neg": {l: [] for l in a.layers}})
        for l, h in hs.items():
            rec["pos" if d["label"] == "tau+" else "neg"][l].append(np.array(h.astype(mx.float32)))
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(decisions)}", file=sys.stderr)

    out = {"spec": "理 README:84 — probe fitted WITHIN task; gate d' >= 2 else STOP",
           "model": a.model, "runs": a.runs, "n_decisions": len(decisions),
           "n_tasks": len(by_task),
           "n_tasks_with_both_labels": sum(1 for v in by_task.values()
                                           if v["pos"][a.layers[0]] and v["neg"][a.layers[0]]),
           "layers": {}}
    for l in a.layers:
        # --- pooled, for comparison with the existing 3.1-4.9 figures
        P = np.stack([x for v in by_task.values() for x in v["pos"][l]])
        N = np.stack([x for v in by_task.values() for x in v["neg"][l]])
        v_pool = P.mean(0) - N.mean(0); v_pool /= (np.linalg.norm(v_pool) + 1e-8)
        pooled = d_prime(P @ v_pool, N @ v_pool)

        # --- demeaned: subtract each task's own mean state
        dp, dn = [], []
        for rec in by_task.values():
            allx = rec["pos"][l] + rec["neg"][l]
            if not allx:
                continue
            mu = np.stack(allx).mean(0)
            dp += [x - mu for x in rec["pos"][l]]
            dn += [x - mu for x in rec["neg"][l]]
        demeaned = None
        if len(dp) >= 2 and len(dn) >= 2:
            DP, DN = np.stack(dp), np.stack(dn)
            v = DP.mean(0) - DN.mean(0); v /= (np.linalg.norm(v) + 1e-8)
            demeaned = d_prime(DP @ v, DN @ v)

        # --- within: only tasks carrying BOTH labels
        wp, wn = [], []
        for rec in by_task.values():
            if rec["pos"][l] and rec["neg"][l]:
                mu = np.stack(rec["pos"][l] + rec["neg"][l]).mean(0)
                wp += [x - mu for x in rec["pos"][l]]
                wn += [x - mu for x in rec["neg"][l]]
        within = None
        if len(wp) >= 2 and len(wn) >= 2:
            WP, WN = np.stack(wp), np.stack(wn)
            v = WP.mean(0) - WN.mean(0); v /= (np.linalg.norm(v) + 1e-8)
            within = d_prime(WP @ v, WN @ v)

        out["layers"][str(l)] = {"pooled_d_prime": pooled, "demeaned_d_prime": demeaned,
                                 "within_task_d_prime": within,
                                 "n_pos": int(len(P)), "n_neg": int(len(N)),
                                 "n_within_pos": len(wp), "n_within_neg": len(wn)}
        print(f"  layer {l}: pooled d'={pooled:.3f}  demeaned={demeaned if demeaned is None else round(demeaned,3)}"
              f"  WITHIN-TASK={within if within is None else round(within,3)}"
              f"  (within n: {len(wp)}+/{len(wn)}-)")

    best = max((out["layers"][k].get("within_task_d_prime") or -9) for k in out["layers"])
    out["gate"] = {"threshold": 2.0, "best_within_task_d_prime": best,
                   "verdict": "PROCEED to G step 2" if best >= 2.0 else
                              "STOP — the vector reads difficulty, not the decision"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  GATE: best within-task d' = {best:.3f} -> {out['gate']['verdict']}")


if __name__ == "__main__":
    main()
