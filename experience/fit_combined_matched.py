#!/usr/bin/env python
"""ET-8b §11 — two arms that separate ORIGIN of training data from AMOUNT of it (理).

ARM 1  COMBINED: one head on the UNION of gen0's and gen1's training trajectories, applied once.
       If combined ≈ gen1, VIII-b's sentence becomes "a steered agent generates better training
       data" rather than "the loop accumulates once" — 理's registered reading, and the interesting
       one, because it moves the claim from the loop to the data.

ARM 2  MATCHED ROWS: gen1 re-fitted at gen0's row count and gen0 at gen1's. gen1 was fitted on 2,824
       rows against gen0's 4,052 — FEWER — so the published rise is conservative but confounded:
       origin and amount vary together. This holds amount fixed.

Both reuse states already on disk. No new episodes are generated to FIT; the loop runs after.
"""
import argparse, collections, json, os, sys
import numpy as np


def load(d, layer):
    meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    X = np.load(os.path.join(d, "X_layer%d.npy" % layer)).astype(np.float64)
    assert len(meta) == X.shape[0], "%s: meta %d vs X %d" % (d, len(meta), X.shape[0])
    return meta, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen0-states", required=True)
    ap.add_argument("--gen1-states", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--cut", type=int, default=226)
    ap.add_argument("--mode", required=True, choices=["combined", "gen1_at_gen0_rows", "gen0_at_gen1_rows"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-heads", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--repo", default=os.path.expanduser("~/github/efficient-thinking/experience"))
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import et8_head_v3 as H

    m0, X0 = load(a.gen0_states, a.layer)
    m1, X1 = load(a.gen1_states, a.layer)
    print("  gen0 states %d · gen1 states %d" % (len(m0), len(m1)))

    if a.mode == "combined":
        meta, X = m0 + m1, np.concatenate([X0, X1], 0)
        note = "UNION of gen0's and gen1's training trajectories (%d + %d rows)" % (len(m0), len(m1))
    elif a.mode == "gen1_at_gen0_rows":
        # gen1 has FEWER rows; upsampling would invent data, so this arm is only meaningful the
        # other way. Refuse rather than silently bootstrap.
        if len(m1) < len(m0):
            sys.exit("STOP: gen1 has %d rows against gen0's %d. Matching UP would mean resampling "
                     "gen1's own rows and calling it more data. Run gen0_at_gen1_rows instead — "
                     "that is the direction the data supports." % (len(m1), len(m0)))
    if a.mode == "gen0_at_gen1_rows":
        # gen0 DOWN-sampled to gen1's count, BY TASK so a task's candidates stay together
        rng = np.random.default_rng(a.seed)
        tid0 = np.array([int(m["task_id"].split("_")[1]) for m in m0])
        # DOWN-SAMPLE THE TRAINING TASKS ONLY. The first version sampled over ALL tasks, which
        # removed held-out ones too and shrank the evaluation set from 75 decisions to 52 — the two
        # arms would then have been compared on DIFFERENT eval sets, which is not a control, it is
        # two different measurements. The held-out range is kept whole.
        n1_train = sum(1 for m in m1 if int(m["task_id"].split("_")[1]) < a.cut)
        train_tasks = sorted({t for t in tid0.tolist() if t < a.cut})
        rng.shuffle(train_tasks)
        keep, n = set(), 0
        for t in train_tasks:
            c = int((tid0 == t).sum())
            if n + c > n1_train:
                continue
            keep.add(t); n += c
        sel = np.array([(t in keep) or (t >= a.cut) for t in tid0])
        meta = [m for m, s in zip(m0, sel) if s]
        X = X0[sel]
        note = ("gen0's TRAINING states down-sampled to gen1's training row count: %d training rows "
                "over %d tasks (gen1 has %d). Sampled BY TASK — dropping half a task's candidates "
                "would leave decisions whose correct answer was never in the training set. The "
                "HELD-OUT range is kept whole so both arms are scored on the same evaluation set."
                % (n, len(keep), n1_train))
    print("  %s" % note)

    y = np.array([m["label"] for m in meta], float)
    tid = np.array([int(m["task_id"].split("_")[1]) for m in meta])
    dec = np.array([m["decision"] for m in meta])
    tr, te = tid < a.cut, tid >= a.cut

    def pick(s, idx):
        g = collections.defaultdict(list)
        for i in idx:
            g[(meta[i]["task_id"], meta[i]["decision"])].append((s[i], meta[i]["label"]))
        return 100.0 * sum(1 for v in g.values() if max(v)[1] == 1) / max(1, len(g)), len(g)

    os.makedirs(a.out_heads, exist_ok=True)
    rng = np.random.default_rng(7)
    res = {}
    CV = te.sum() == 0
    for d in (1, 2, 3):
        mtr = tr & (dec == d)
        mte = te & (dec == d)
        if mtr.sum() < 50:
            sys.exit("STOP: decision %d has %d training rows" % (d, mtr.sum()))
        mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-6
        Z = (X - mu) / sd
        w, b = H.logistic_fit(Z[mtr], y[mtr])
        if CV:
            tasks_d = sorted(set(tid[mtr].tolist()))
            order = np.random.default_rng(13).permutation(len(tasks_d))
            folds = [set(tasks_d[i] for i in order[k::3]) for k in range(3)]
            hits = tot = 0
            for f in folds:
                inf = np.array([t in f for t in tid])
                ftr, fte = mtr & ~inf, mtr & inf
                if ftr.sum() < 50 or fte.sum() < 20:
                    continue
                mu2, sd2 = X[ftr].mean(0), X[ftr].std(0) + 1e-6
                Z2 = (X - mu2) / sd2
                w2, b2 = H.logistic_fit(Z2[ftr], y[ftr])
                fa, fn = pick(Z2 @ w2 + b2, np.where(fte)[0])
                hits += fa * fn / 100.0; tot += fn
            acc, n = (100.0 * hits / tot if tot else 0.0), int(tot)
        else:
            acc, n = pick(Z @ w + b, np.where(mte)[0])
        perm = []
        for _ in range(5):
            ys = y.copy(); ys[mtr] = rng.permutation(y[mtr])
            w3, b3 = H.logistic_fit(Z[mtr], ys[mtr])
            perm.append(round(pick(Z @ w3 + b3, np.where(mte if not CV else mtr)[0])[0], 1))
        np.savez(os.path.join(a.out_heads, "head_d%d_layer%d.npz" % (d, a.layer)),
                 w=w, b=np.array([b]), mu=mu, sd=sd, train_task_max=np.array([a.cut]),
                 train_ranges=np.array([[int(tid[mtr].min()), int(tid[mtr].max())]], dtype=int))
        res["decision_%d" % d] = {("cv_pick_pct" if CV else "held_out_pick_pct"): round(acc, 1),
                                  "n": n, "train_rows": int(mtr.sum()), "CONTROL_permuted": perm}
        print("  decision %d: %.1f%% (n=%d, train rows %d) perm %s" % (d, acc, n, mtr.sum(), perm))
    json.dump({"document": "ET-8b §11 — %s" % a.mode, "prereg": "docs/et8b-loop-gates.md §11",
               "construction": note, "evaluation": "3-fold CV over tasks" if CV else "held out",
               "heads": res, "signed": "Sautee (sha-ta)"},
              open(a.json, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % a.json)


if __name__ == "__main__":
    main()
