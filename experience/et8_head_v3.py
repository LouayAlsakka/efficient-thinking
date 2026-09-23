#!/usr/bin/env python3
"""G on v3: a read-only head that SCORES each candidate region at the post-inspect decision.

    python3 et8_head_v3.py states --runs <base slices> --tasks experience/tasks/v3 \
        --layers 18 27 --out <dir>
    python3 et8_head_v3.py fit --states <dir> --out <fit.json>

WHY THIS IS NOT et8_head_region.py. That file is a fixed FOUR-WAY classifier over the region names
producer/transform/aggregate/consumer, because v1 and v2 had exactly those four in every task. v3
tasks carry 4-6 regions drawn from eight names, so a K-way classifier over a fixed vocabulary does
not apply. This scores each candidate SEPARATELY -- one state per (decision, candidate region),
built by teacher-forcing the opening fields of a hypothesis naming that region -- and picks the
argmax. Variable region counts and names fall out for free, and the head never sees a name it must
have been trained on.

WHERE IT READS. The FIRST HYPOTHESIS AFTER AN INSPECT, under the inspect-first loop (理 10941). At
that point the prompt carries the symptom AND the inspected region's source, and the prompt gate
reads 1.000 on v3 (75 distinct prompts from 75 episodes) against 0.005-0.088 on every earlier
decision point in this programme. This is the first place a head can be fit at all.

THE CONTROLS ARE NOT OPTIONAL AND THEY RUN IN THE SAME COMMAND:
  permutation   shuffle the TRAINING labels, refit, score. Must collapse to chance.
  PCA-8         refit in 8 dimensions. A signal that survives is a key being copied, not a readout.
  n=100         refit on 100 examples. Same reasoning.
A 100% held-out result that PASSES its permutation control is what v1 produced, and it was the
label sitting in the prompt. Those three together are what caught it; all three are printed.

THE BAR (理): problems solved and cost. NOT localisation -- the arm that localised better and
solved half as many would have passed a localisation bar, twice, on two different task sets.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def post_inspect_decisions(steps, tasks_dir):
    """The first hypothesis that follows at least one inspect, with the prompt the agent saw."""
    import et8_inject as I
    import collections
    by = collections.defaultdict(list)
    for s in steps:
        by[(s["run_id"], s["task_id"])].append(s)
    out = []
    for (rid, tid), st in by.items():
        st.sort(key=lambda s: s["step"])
        task = json.load(open(os.path.join(tasks_dir, tid + ".json")))
        hist, insp = [], {}
        import et8_agent as A
        for s in st:
            if s["action"] == "hypothesize":
                if not insp:
                    break                       # no observation yet: not the decision G acts at
                out.append({"task_id": tid, "run_id": rid, "step": s["step"],
                            "bug_region": task["bug_region"], "regions": task["regions"],
                            "agent_region": s["region"], "agent_hit": bool(s["region_hit"]),
                            "messages": I.decision_prompt(task, hist, insp, 12, s["step"])})
                break
            if s["action"] == "inspect" and s["region"]:
                insp[s["region"]] = A.region_source(task["program"], s["region"]) or ""
                hist.append("inspect %s" % s["region"])
            elif s.get("region"):
                hist.append("%s %s" % (s["action"], s["region"]))
    return out


def cmd_states(a):
    import et8_inject as I
    from mlx_lm import load
    import mlx.core as mx

    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = post_inspect_decisions(steps, a.tasks)
    if a.limit:
        dec = dec[: a.limit]
    print("  decisions: %d" % len(dec), file=sys.stderr)

    model, tok = load(a.model)
    os.makedirs(a.out, exist_ok=True)
    X = {l: [] for l in a.layers}
    meta = []
    t0 = time.time()
    for i, d in enumerate(dec):
        for r in d["regions"]:
            pre = '{"action": "hypothesize", "region": "%s", "bug_class": "' % r
            msgs = list(d["messages"])
            msgs[-1] = dict(msgs[-1])
            hs = hidden_at(model, tok, msgs, a.layers, prefix=pre)
            for l in a.layers:
                X[l].append(np.array(hs[l].astype(mx.float32), copy=False))
            meta.append({"task_id": d["task_id"], "candidate": r,
                         "label": int(r == d["bug_region"]),
                         "agent_region": d["agent_region"], "agent_hit": d["agent_hit"],
                         "n_regions": len(d["regions"])})
        if (i + 1) % 25 == 0:
            rate = (time.time() - t0) / (i + 1)
            print("  %d/%d  %.2fs/decision  eta %.0fm"
                  % (i + 1, len(dec), rate, rate * (len(dec) - i - 1) / 60), file=sys.stderr)
    for l in a.layers:
        np.save(os.path.join(a.out, "X_layer%d.npy" % l), np.stack(X[l]))
    with open(os.path.join(a.out, "meta.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print("  wrote %d candidate states" % len(meta), file=sys.stderr)


def hidden_at(model, tok, messages, layers, prefix=""):
    import mlx.core as mx
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
    ids = mx.array([tok.encode(prompt)])
    inner = model.model
    h = inner.embed_tokens(ids)
    mask = None
    try:
        from mlx_lm.models.base import create_attention_mask
        mask = create_attention_mask(h, None)
    except Exception:
        pass
    got = {}
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, None)
        if i in layers:
            got[i] = h[0, -1, :]
    return got


def logistic_fit(X, y, l2=1e-3, **_):
    """Binary scorer: is THIS candidate the bug region? Linear, on purpose.

    FIT BY L-BFGS (scipy), NOT BY HAND-TUNED GRADIENT DESCENT (理, and then a defect of mine
    that ruling exposed). The first version ran a fixed 400 iterations with the gradient averaged
    over the pool, so a larger pool took smaller effective steps. Told to fix it, I switched to a
    loss-change criterion -- which hit a 40,000-iteration cap without converging -- and then raised
    the learning rate to 2.0, at which point the SAME pool scored 58.7% at lr 0.5 and 24.0% at
    lr 2.0.

    A probe accuracy that moves 35 points with the optimizer's step size is not a measurement of the
    state. L-BFGS removes the free parameter entirely: it converges to the regularised optimum or it
    reports that it did not, and `last_iters` / `hit_cap` carry that into every table.
    """
    from scipy.optimize import minimize
    n, d = X.shape
    def f(th):
        w, b = th[:d], th[d]
        z = X @ w + b
        # log(1+exp(z)) computed stably
        ll = np.logaddexp(0.0, z) - y * z
        loss = ll.mean() + 0.5 * l2 * float(w @ w)
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y) / n
        return loss, np.concatenate([X.T @ g + l2 * w, [g.sum()]])
    res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    logistic_fit.last_iters = int(res.nit)
    logistic_fit.hit_cap = not bool(res.success)
    return res.x[:d], float(res.x[d])


def pick_accuracy(scores, meta, idx):
    """argmax WITHIN each decision, not per candidate: the head's job is to choose a region."""
    import collections
    by = collections.defaultdict(list)
    for j in idx:
        by[meta[j]["task_id"]].append(j)
    hit = 0
    for tid, js in by.items():
        best = max(js, key=lambda j: scores[j])
        hit += meta[best]["label"]
    return hit / max(1, len(by)), len(by)


def cmd_fit(a):
    meta = [json.loads(l) for l in open(os.path.join(a.states, "meta.jsonl"))]
    y = np.array([m["label"] for m in meta], dtype=float)
    tid = np.array([int(m["task_id"].split("_")[1]) for m in meta])
    layers = sorted(int(f.split("layer")[1].split(".")[0])
                    for f in os.listdir(a.states) if f.startswith("X_layer"))
    tasks = sorted(set(tid))
    cut = tasks[int(0.75 * len(tasks))]
    tr = tid < cut
    te = ~tr
    out = {"candidate_states": len(meta), "decisions": len(set(tid)),
           "train_tasks": int((tid[tr]).max() if tr.any() else 0),
           "held_out_tasks": len(set(tid[te])),
           "agent_pick_accuracy_on_heldout": None, "layers": {}}
    import collections
    seen = {}
    for m, t in zip(meta, tid):
        if t >= cut:
            seen[m["task_id"]] = m["agent_hit"]
    if seen:
        out["agent_pick_accuracy_on_heldout"] = round(100 * sum(seen.values()) / len(seen), 1)

    for l in layers:
        X = np.load(os.path.join(a.states, "X_layer%d.npy" % l)).astype(np.float64)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        w, b = logistic_fit(Z[tr], y[tr])
        s = Z @ w + b
        acc, nd = pick_accuracy(s, meta, np.where(te)[0])
        # CONTROL 1: permuted training labels
        rng = np.random.default_rng(7); perm = []
        for _ in range(5):
            ys = y.copy(); ys[tr] = rng.permutation(y[tr])
            w2, b2 = logistic_fit(Z[tr], ys[tr])
            perm.append(pick_accuracy(Z @ w2 + b2, meta, np.where(te)[0])[0])
        # CONTROL 2: PCA to 8 dims
        Xc = Z[tr] - Z[tr].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt[:8].T
        w3, b3 = logistic_fit((Z @ P)[tr], y[tr])
        pca8 = pick_accuracy((Z @ P) @ w3 + b3, meta, np.where(te)[0])[0]
        # CONTROL 3: 100 training candidate-states, FIVE DRAWS.
        # One draw was not a control. Two different 100-subsamples of the SAME pool scored 33.3% and
        # 46.7% held out -- a 13-point spread from the subsample alone -- so a single draw could sit
        # either side of the full head and I read one of those as an anomaly and reported it as
        # possible under-convergence. Five draws, all printed, mean compared.
        idx = np.where(tr)[0]
        n100s = []
        for _ in range(5):
            sub = rng.choice(idx, size=min(100, len(idx)), replace=False)
            w4, b4 = logistic_fit(Z[sub], y[sub])
            n100s.append(pick_accuracy(Z @ w4 + b4, meta, np.where(te)[0])[0])
        n100 = float(np.mean(n100s))
        np.savez(os.path.join(a.states, "head_layer%d.npz" % l), w=w, b=np.array([b]),
                 mu=mu, sd=sd, train_task_max=np.array([cut]))
        out["layers"][str(l)] = {
            "weights": "head_layer%d.npz in the states dir (w, b, mu, sd, train_task_max)" % l,
            "head_pick_pct": round(100 * acc, 1), "held_out_decisions": nd,
            "CONTROL_permuted_labels_pct": [round(100 * p, 1) for p in perm],
            "CONTROL_pca8_pct": round(100 * pca8, 1),
            "CONTROL_n100_pct": round(100 * n100, 1),
            "CONTROL_n100_draws_pct": [round(100 * x, 1) for x in n100s],
            "fit_iterations": int(logistic_fit.last_iters),
            "fit_hit_iteration_cap": bool(logistic_fit.hit_cap),
            "chance_pct": round(100 * float(np.mean([1.0 / m["n_regions"] for m in meta])), 1)}
        p = out["layers"][str(l)]
        print("layer %d: head %.1f%%  agent %s%%  chance %.1f%%  | permuted %s  pca8 %.1f%%  n100 %.1f%%"
              % (l, p["head_pick_pct"], out["agent_pick_accuracy_on_heldout"], p["chance_pct"],
                 p["CONTROL_permuted_labels_pct"], p["CONTROL_pca8_pct"], p["CONTROL_n100_pct"]),
              file=sys.stderr)
    out["how_to_read_the_controls"] = (
        "permuted must collapse to chance; if PCA-8 or n=100 match the full head, the signal needs "
        "neither dimensions nor examples and is a key being copied, not a readout. v1 scored 100% "
        "held out, PASSED its permutation control, and survived both of those -- which is how the "
        "label-in-the-prompt was found. All three are printed, always.")
    out["the_bar"] = ("problems solved and cost on a paired run. Head accuracy is NOT the bar: the "
                      "arm that localised better and solved half as many would have passed a "
                      "localisation bar twice, on two different task sets (理).")
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
