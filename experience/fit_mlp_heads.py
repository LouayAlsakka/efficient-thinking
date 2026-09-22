#!/usr/bin/env python
"""ET-8b §10 — the NONLINEAR head: a 2-layer MLP (hidden 256, ReLU) on the same frozen states and
verifier labels as the linear gen0 head. Pre-registered in docs/et8b-loop-gates.md §10.

Same three decisions, same train/eval split, same three probe controls. Saved in the SAME npz format
the loop already reads, plus the MLP weights — so et8b_loop.py needs no edit and the comparison is
head-vs-head with everything else identical.

⚠️ THE COMPUTE CHARGE IS NOT OPTIONAL. §10 says "charged its own inference". A 3584->256->1 MLP is
~918k multiply-adds per candidate against the linear head's ~3.6k — 256x the controller arithmetic.
That is recorded here and printed beside any win, because a nonlinear head that wins while costing
more is not the same finding as one that wins for free.
"""
import argparse, collections, json, os, sys
import numpy as np


def mlp_fit(Xtr, ytr, hidden=256, epochs=300, lr=0.05, l2=1e-3, seed=0):
    """Plain numpy MLP — no new dependency, and the arithmetic is visible rather than delegated."""
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    W1 = rng.normal(0, (2.0 / d) ** 0.5, (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, (2.0 / hidden) ** 0.5, hidden); b2 = 0.0
    n = len(ytr)
    for ep in range(epochs):
        H = Xtr @ W1 + b1
        A = np.maximum(H, 0)
        z = A @ W2 + b2
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = (p - ytr) / n
        gW2 = A.T @ g + l2 * W2; gb2 = g.sum()
        dA = np.outer(g, W2) * (H > 0)
        gW1 = Xtr.T @ dA + l2 * W1; gb1 = dA.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return W1, b1, W2, b2


def mlp_score(X, W1, b1, W2, b2):
    return np.maximum(X @ W1 + b1, 0) @ W2 + b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--cut", type=int, default=226)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out-heads", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    meta = [json.loads(l) for l in open(os.path.join(a.states, "meta.jsonl"))]
    X = np.load(os.path.join(a.states, "X_layer%d.npy" % a.layer)).astype(np.float64)
    y = np.array([m["label"] for m in meta], float)
    tid = np.array([int(m["task_id"].split("_")[1]) for m in meta])
    dec = np.array([m["decision"] for m in meta])
    tr, te = tid < a.cut, tid >= a.cut
    print("  states %s · train %d · held out %d" % (X.shape, tr.sum(), te.sum()))

    def pick(s, idx):
        g = collections.defaultdict(list)
        for i in idx:
            g[(meta[i]["task_id"], meta[i]["decision"])].append((s[i], meta[i]["label"]))
        return 100.0 * sum(1 for v in g.values() if max(v)[1] == 1) / max(1, len(g)), len(g)

    os.makedirs(a.out_heads, exist_ok=True)
    rng = np.random.default_rng(7)
    res = {}
    for d in (1, 2, 3):
        mtr, mte = tr & (dec == d), te & (dec == d)
        mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-6
        Z = (X - mu) / sd
        W1, b1, W2, b2 = mlp_fit(Z[mtr], y[mtr], hidden=a.hidden)
        s = mlp_score(Z, W1, b1, W2, b2)
        acc, n = pick(s, np.where(mte)[0])
        tracc, _ = pick(s, np.where(mtr)[0])       # §10: print training accuracy if MLP < linear
        perm = []
        for _ in range(5):
            ys = y.copy(); ys[mtr] = rng.permutation(y[mtr])
            w_ = mlp_fit(Z[mtr], ys[mtr], hidden=a.hidden, seed=1)
            perm.append(round(pick(mlp_score(Z, *w_), np.where(mte)[0])[0], 1))
        Xc = Z[mtr] - Z[mtr].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False); P = Vt[:8].T
        w8 = mlp_fit((Z @ P)[mtr], y[mtr], hidden=a.hidden, seed=2)
        pca8 = round(pick(mlp_score(Z @ P, *w8), np.where(mte)[0])[0], 1)
        idx = np.where(mtr)[0]
        sub = rng.choice(idx, size=min(500, len(idx)), replace=False)
        ws = mlp_fit(Z[sub], y[sub], hidden=a.hidden, seed=3)
        sacc = round(pick(mlp_score(Z, *ws), np.where(mte)[0])[0], 1)
        np.savez(os.path.join(a.out_heads, "head_d%d_layer%d.npz" % (d, a.layer)),
                 w=np.zeros(X.shape[1]), b=np.array([0.0]), mu=mu, sd=sd,
                 train_task_max=np.array([a.cut]),
                 train_ranges=np.array([[int(tid[mtr].min()), int(tid[mtr].max())]], dtype=int),
                 mlp_W1=W1, mlp_b1=b1, mlp_W2=W2, mlp_b2=np.array([b2]))
        chance = 100 * np.mean([1.0 / meta[i]["n_regions"] for i in np.where(mte)[0]])
        res["decision_%d" % d] = {"held_out_pick_pct": round(acc, 1), "n": n,
                                  "TRAIN_pick_pct": round(tracc, 1), "chance_pct": round(chance, 1),
                                  "CONTROL_permuted": perm, "CONTROL_pca8": pca8,
                                  "CONTROL_row_subsample": sacc}
        print("  decision %d: held out %.1f%% (train %.1f%%, chance %.1f) perm %s pca8 %.1f sub %.1f"
              % (d, acc, tracc, chance, perm, pca8, sacc))
    dmodel = X.shape[1]
    json.dump({"document": "ET-8b §10 — the NONLINEAR (MLP) gen0 head",
               "prereg": "docs/et8b-loop-gates.md §10",
               "architecture": "2-layer MLP, hidden %d, ReLU, on the SAME frozen states and verifier "
                               "labels as the linear gen0 head" % a.hidden,
               "controller_arithmetic_per_candidate": {
                   "linear": dmodel, "mlp": dmodel * a.hidden + a.hidden,
                   "ratio": round((dmodel * a.hidden + a.hidden) / dmodel, 1),
                   "note": "§10 charges the head its own inference. A win that costs 256x the "
                           "controller arithmetic is not the same finding as a win for free."},
               "heads": res, "signed": "Sautee (sha-ta)"},
              open(a.json, "w"), indent=1, ensure_ascii=False)
    print("\n  controller arithmetic per candidate: linear %d · MLP %d (%.0fx)"
          % (dmodel, dmodel * a.hidden + a.hidden, (dmodel * a.hidden + a.hidden) / dmodel))
    print("  wrote %s" % a.json)


if __name__ == "__main__":
    main()
