#!/usr/bin/env python3
"""Mechanism G, steps 2-4: can a READ-ONLY head on the frozen state pick the bug region?

    # step 2+3 -- enumerate candidates, read the teacher-forced state, cache it
    python3 et8_head_region.py states --runs experience/traj/base_v1_7b experience/traj/base_v1_7b_s2 \
        --tasks experience/tasks/v1 --layers 18 27 --out experience/results/g_head

    # step 4 -- fit the head, held out BY SLICE, and print the only bar that matters
    python3 et8_head_region.py fit --states experience/results/g_head --out experience/results/g_head_fit.json

WHAT THE CANDIDATE SET TURNED OUT TO BE. Step 2 was specified as "candidate enumeration at the one
decision point, k <= 8". Enumerating it gives k = 4 for ALL 2,000 tasks and the SAME four names:
producer / transform / aggregate / consumer. So the head is not a ranker over a task-specific list,
it is a fixed 4-way classifier, and that is a WEAKER instrument than the one the plan imagined --
a fixed vocabulary lets a head score well by learning the marginal instead of reading the state.
The marginal is printed beside every number for exactly that reason.

THE BAR, WRITTEN BEFORE THE FIT. The head does not compete with chance (25%) or with the majority
class (producer, 36.3%). It competes with THE AGENT, which already picks the right region on the
first hypothesis 88.05% of the time (3,522/4,000 over the two 2,000-task runs). A head below 88.05%
makes the agent worse and G is dead on the spot. This is the number G-trivial's -36.2 points was
the first evidence about: a prior that overrides a well-calibrated chooser pays for every override.

AND THE CEILING, ALSO WRITTEN FIRST. Even a PERFECT head buys at most the 478 missed episodes, and
the G-trivial control just measured that group: where symptom == bug, sending the agent to the
right region moved success by -4.4 points (51.1 -> 46.7, n=45). So perfect localisation is not
worth the whole gap. The head's accuracy and the episodes it would save are two different
quantities and this file reports only the first.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REGIONS = ["producer", "transform", "aggregate", "consumer"]


def cmd_states(a):
    import et8_inject as I
    from mlx_lm import load
    import mlx.core as mx

    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = I.first_decision_states(steps, a.tasks)
    if a.limit:
        dec = dec[: a.limit]
    print(f"  decisions: {len(dec)}", file=sys.stderr)

    model, tok = load(a.model)
    os.makedirs(a.out, exist_ok=True)
    X = {l: [] for l in a.layers}
    meta = []
    t0 = time.time()
    for i, d in enumerate(dec):
        task = json.load(open(os.path.join(a.tasks, d["task_id"] + ".json")))
        hs = I.hidden_at_layers(model, tok, d["messages"], a.layers)
        for l in a.layers:
            X[l].append(np.array(hs[l].astype(mx.float32), copy=False))
        meta.append({"task_id": d["task_id"], "step": d["step"],
                     "bug_region": task["bug_region"], "symptom_region": task["symptom_region"],
                     "agent_pick": d["region"], "agent_hit": d["label"] == "tau+",
                     "k": len(task["regions"])})
        if (i + 1) % 200 == 0:
            r = (time.time() - t0) / (i + 1)
            print(f"  {i+1}/{len(dec)}  {r:.2f}s/state  eta {r*(len(dec)-i-1)/60:.0f}m", file=sys.stderr)
    for l in a.layers:
        np.save(os.path.join(a.out, f"X_layer{l}.npy"), np.stack(X[l]))
    with open(os.path.join(a.out, "meta.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"  wrote {len(meta)} states to {a.out}", file=sys.stderr)


def softmax_fit(Xtr, ytr, C=4, iters=300, lr=0.5, l2=1e-3):
    """Plain multinomial logistic by full-batch GD. No sklearn on this box; and a linear head is
    the whole point -- anything stronger stops being a read of the state and starts being a model."""
    n, d = Xtr.shape
    W = np.zeros((d, C), dtype=np.float64); b = np.zeros(C)
    Y = np.zeros((n, C)); Y[np.arange(n), ytr] = 1.0
    for _ in range(iters):
        Z = Xtr @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (Xtr.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def cmd_fit(a):
    meta = [json.loads(l) for l in open(os.path.join(a.states, "meta.jsonl"))]
    y = np.array([REGIONS.index(m["bug_region"]) for m in meta])
    tid = np.array([int(m["task_id"].split("_")[1]) for m in meta])
    agent_hit = np.array([m["agent_hit"] for m in meta])
    sym_eq = np.array([m["symptom_region"] == m["bug_region"] for m in meta])
    layers = sorted(int(f.split("layer")[1].split(".")[0])
                    for f in os.listdir(a.states) if f.startswith("X_layer"))
    out = {"n": len(meta), "k": 4, "regions": REGIONS,
           "marginal_majority_pct": round(100 * np.bincount(y).max() / len(y), 1),
           "agent_first_hyp_pct": round(100 * agent_hit.mean(), 2), "layers": {}}
    for l in layers:
        X = np.load(os.path.join(a.states, f"X_layer{l}.npy")).astype(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        per_slice = []
        for s in range(4):
            lo, hi = 20 * s + 1, 20 * s + 20
            te = (tid >= lo) & (tid <= hi)
            tr = ~((tid >= 1) & (tid <= 80))      # train on 81..2000 ONLY: no eval task, either slice
            W, b = softmax_fit(X[tr], y[tr])
            pred = (X[te] @ W + b).argmax(1)
            per_slice.append({"slice": s + 1, "n": int(te.sum()),
                              "head_pct": round(100 * float((pred == y[te]).mean()), 1),
                              "agent_pct": round(100 * float(agent_hit[te].mean()), 1)})
        W, b = softmax_fit(X[~((tid >= 1) & (tid <= 80))], y[~((tid >= 1) & (tid <= 80))])
        ev = (tid >= 1) & (tid <= 80)
        pred = (X[ev] @ W + b).argmax(1)
        out["layers"][str(l)] = {
            "per_slice": per_slice,
            "pooled_head_pct": round(100 * float((pred == y[ev]).mean()), 1),
            "pooled_agent_pct": round(100 * float(agent_hit[ev].mean()), 1),
            "head_on_agent_misses_pct": round(100 * float((pred == y[ev])[~agent_hit[ev]].mean()), 1)
                if (~agent_hit[ev]).any() else None,
            "n_agent_misses": int((~agent_hit[ev]).sum()),
            "head_pct_symptom_eq_bug": round(100 * float((pred == y[ev])[sym_eq[ev]].mean()), 1),
            "head_pct_symptom_ne_bug": round(100 * float((pred == y[ev])[~sym_eq[ev]].mean()), 1),
            "agent_pct_symptom_ne_bug": round(100 * float(agent_hit[ev][~sym_eq[ev]].mean()), 1),
        }
        p = out["layers"][str(l)]
        print(f"layer {l}: head {p['pooled_head_pct']}%  agent {p['pooled_agent_pct']}%  "
              f"majority {out['marginal_majority_pct']}%  (on agent misses: {p['head_on_agent_misses_pct']}%)",
              file=sys.stderr)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("states"); s.set_defaults(fn=cmd_states)
    s.add_argument("--runs", nargs="+", required=True); s.add_argument("--tasks", required=True)
    s.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    s.add_argument("--layers", nargs="+", type=int, default=[18, 27])
    s.add_argument("--limit", type=int, default=0); s.add_argument("--out", required=True)
    f = sub.add_parser("fit"); f.set_defaults(fn=cmd_fit)
    f.add_argument("--states", required=True); f.add_argument("--out", required=True)
    a = ap.parse_args(); a.fn(a)
