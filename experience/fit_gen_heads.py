#!/usr/bin/env python
"""ET-8b: fit the three PER-DECISION heads for one generation.

THE MERGED FITTER. This was briefly two files. fit_gen_heads.py could cross-fit (--train-range /
--eval-range) but wrote `0.0% (n=0, chance nan)` when the eval mask came out empty; the _cv fork
refused that and reported K-fold CV instead, but could not cross-fit. They were STRICTLY
COMPLEMENTARY, which is the worst shape a fork can have: the next cross-fitted generation with an
empty eval mask would have needed both and had neither. The fork was deliberate at the time --
fit_gen_heads.py was being re-invoked per fold by a running chain, and editing a script a running
loop re-reads kills the later arms only -- and this is the reconciliation.

WHY THE EMPTY-MASK REFUSAL EXISTS. After cross-fitting, the source runs cover tasks 1-225 only, so a head fitted on
"task_id < 226" has an EMPTY held-out mask. The parent file does not crash on that — I ran it, and
it writes `0.0% (n=0, chance nan)` with EVERY CONTROL AT 0.0. That artifact is worse than a crash
twice over: the accuracy looks like catastrophic failure, and the controls look like they PASSED
(a permutation control reading 0.0 is indistinguishable from one that collapsed to chance), when in
fact nothing was evaluated at all.

So here an empty eval mask is either a K-fold CV over the training tasks — accuracy AND all three
controls computed inside the CV loop, so none of them can report a number it did not measure — or a
hard refusal. It never writes a zero.

THE ARCHITECTURE IS NOT RE-CHOSEN HERE. gen0 applied the spec's rule (docs/et8b-loop-gates.md §3)
before any loop ran and selected per-decision; every later generation inherits it. Probe accuracy is
reported descriptively and nothing is selected on it. And per 理 11068 the bar is problems solved on
a paired run — probe accuracy is a sanity check, not the result.
"""
import argparse, collections, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--cut", type=int, default=226)
    ap.add_argument("--train-range", action="append", default=None, metavar="LO:HI",
                    help="repeatable, inclusive. Fit on these task ids INSTEAD of task_id < cut. "
                         "Cross-fitting: fold f is run by a head that never saw fold f.")
    ap.add_argument("--eval-range", action="append", default=None, metavar="LO:HI",
                    help="repeatable, inclusive. Report on these instead of task_id >= cut.")
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="when the eval mask is empty, report K-fold CV over the TRAINING TASKS. "
                         "Descriptive only; the deployed head is still fitted on all of them.")
    ap.add_argument("--out-heads", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--repo", default=os.path.expanduser("~/github/efficient-thinking/experience"))
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import et8_head_v3 as H

    meta = [json.loads(l) for l in open(os.path.join(a.states, "meta.jsonl"))]
    X = np.load(os.path.join(a.states, "X_layer%d.npy" % a.layer)).astype(np.float64)
    if len(meta) != X.shape[0]:
        sys.exit("STOP: meta %d rows but X %d rows" % (len(meta), X.shape[0]))
    y = np.array([m["label"] for m in meta], float)
    tid = np.array([int(m["task_id"].split("_")[1]) for m in meta])
    dec = np.array([m["decision"] for m in meta])
    def mask_of(ranges, fallback):
        if not ranges:
            return fallback, None
        m = np.zeros(len(tid), bool); spans = []
        for r in ranges:
            lo, hi = (int(x) for x in r.split(":")); spans.append([lo, hi])
            m |= (tid >= lo) & (tid <= hi)
        return m, spans
    tr, train_spans = mask_of(a.train_range, tid < a.cut)
    te, eval_spans = mask_of(a.eval_range, tid >= a.cut)
    if (tr & te).any():
        sys.exit("STOP: train and eval ranges overlap on %d rows" % int((tr & te).sum()))
    if train_spans:
        print("  cross-fit: training on task ids %s" % train_spans)
    if eval_spans:
        print("  evaluating on task ids %s" % eval_spans)
    print("  states %s  meta %d  train rows %d  held-out rows %d"
          % (X.shape, len(meta), tr.sum(), te.sum()))
    print("  per decision: %s" % dict(sorted(collections.Counter(dec.tolist()).items())))
    CV = te.sum() == 0
    if CV and not a.cv_folds:
        sys.exit("STOP: the held-out mask is EMPTY (no rows at or above task %d). That would write "
                 "`0.0%% (n=0, chance nan)` with every control at 0.0 and call it a measurement. "
                 "Pass --cv-folds K to report CV over the training tasks instead." % a.cut)

    def pick(s, idx):
        g = collections.defaultdict(list)
        for i in idx:
            g[(meta[i]["task_id"], meta[i]["decision"])].append((s[i], meta[i]["label"]))
        return 100.0 * sum(1 for v in g.values() if max(v)[1] == 1) / max(1, len(g)), len(g)

    def evaluate(ftr, fte, rng):
        """One fit + its three controls on one (train, eval) split. Returns (acc, n, perm, pca8, sub)."""
        mu, sd = X[ftr].mean(0), X[ftr].std(0) + 1e-6
        Z = (X - mu) / sd
        idx_te = np.where(fte)[0]
        w, b = H.logistic_fit(Z[ftr], y[ftr])
        acc, n = pick(Z @ w + b, idx_te)
        perm = []
        for _ in range(5):
            ys = y.copy(); ys[ftr] = rng.permutation(y[ftr])
            w2, b2 = H.logistic_fit(Z[ftr], ys[ftr])
            perm.append(pick(Z @ w2 + b2, idx_te)[0])
        Xc = Z[ftr] - Z[ftr].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt[:8].T
        w3, b3 = H.logistic_fit((Z @ P)[ftr], y[ftr])
        pca8 = pick((Z @ P) @ w3 + b3, idx_te)[0]
        ii = np.where(ftr)[0]
        sub_i = rng.choice(ii, size=min(500, len(ii)), replace=False)
        w4, b4 = H.logistic_fit(Z[sub_i], y[sub_i])
        sub = pick(Z @ w4 + b4, idx_te)[0]
        return acc, n, perm, pca8, sub

    os.makedirs(a.out_heads, exist_ok=True)
    rng = np.random.default_rng(7)
    res = {}
    for d in (1, 2, 3):
        mtr = tr & (dec == d)
        if mtr.sum() < 50:
            sys.exit("STOP: decision %d has %d training rows, under the 50 floor" % (d, mtr.sum()))
        if CV:
            # folds over TASKS, never over rows: two candidates of one decision split across folds
            # would put the answer in the training set
            tasks = sorted(set(tid[mtr].tolist()))
            order = np.random.default_rng(13).permutation(len(tasks))
            folds = [set(tasks[i] for i in order[k::a.cv_folds]) for k in range(a.cv_folds)]
            accs, ns, perms, pcas, subs = [], [], [], [], []
            for fk in folds:
                inf = np.array([t in fk for t in tid])
                ftr, fte = mtr & ~inf, mtr & inf
                if ftr.sum() < 50 or fte.sum() < 20:
                    continue
                A_, n_, p_, q_, s_ = evaluate(ftr, fte, rng)
                accs.append(A_ * n_); ns.append(n_); perms.append(p_); pcas.append(q_ * n_)
                subs.append(s_ * n_)
            if not ns:
                sys.exit("STOP: decision %d — CV produced no evaluable fold" % d)
            tot = float(sum(ns))
            acc, n = sum(accs) / tot, int(tot)
            perm = [round(sum(pf[k] * ns[j] for j, pf in enumerate(perms)) / tot, 1) for k in range(5)]
            pca8, sub = sum(pcas) / tot, sum(subs) / tot
            eval_label = "%d-fold CV over the TRAINING tasks (descriptive; the deployed head is fitted on all of them)" % a.cv_folds
            chance_src = np.where(mtr)[0]
        else:
            mte = te & (dec == d)
            if mte.sum() < 20:
                sys.exit("STOP: decision %d has only %d held-out rows" % (d, mte.sum()))
            acc, n, perm, pca8, sub = evaluate(mtr, mte, rng)
            perm = [round(p, 1) for p in perm]
            eval_label = "held out"
            chance_src = np.where(mte)[0]

        # the DEPLOYED head is always fitted on the full training set
        mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-6
        w, b = H.logistic_fit(((X - mu) / sd)[mtr], y[mtr])
        out = os.path.join(a.out_heads, "head_d%d_layer%d.npz" % (d, a.layer))
        spans = np.array(train_spans if train_spans
                         else [[int(tid[mtr].min()), int(tid[mtr].max())]], dtype=int)
        np.savez(out, w=w, b=np.array([b]), mu=mu, sd=sd, train_task_max=np.array([a.cut]),
                 train_ranges=spans)
        chance = 100 * np.mean([1.0 / meta[i]["n_regions"] for i in chance_src])
        key = "cv_pick_pct" if CV else "held_out_pick_pct"
        res["decision_%d" % d] = {
            key: round(acc, 1), "n": n, "chance_pct": round(chance, 1), "evaluation": eval_label,
            "train_rows": int(mtr.sum()),
            "CONTROL_permuted": perm, "CONTROL_pca8": round(pca8, 1),
            "CONTROL_row_subsample": round(sub, 1), "weights": os.path.basename(out)}
        print("  decision %d [%s]: %.1f%% (n=%d, chance %.1f) perm %s pca8 %.1f sub %.1f -> %s"
              % (d, "CV" if CV else "held out", acc, n, chance, perm, pca8, sub,
                 os.path.basename(out)))

    json.dump({
        "document": "ET-8b %s heads — per-decision, the architecture FIXED at gen0" % a.gen,
        "evaluation_mode": ("%d-fold CV over training tasks" % a.cv_folds) if CV else "held out",
        "fit_on": ("task ids %s (CROSS-FIT)" % train_spans) if train_spans else "task_id < %d" % a.cut,
        "⚠️_why_CV_here": (
            "the cross-fitted source runs cover tasks 1-225 only, so a head fitted on task_id < 226 "
            "has an empty held-out mask. The parent fitter writes 0.0%% with every control at 0.0 in "
            "that case, which reads as a failed head with PASSING controls when nothing was measured "
            "at all. Accuracy and all three controls are computed inside the CV loop here."
        ) if CV else "held-out rows exist; no CV needed",
        "architecture_not_rechosen": (
            "gen0 applied docs/et8b-loop-gates.md §3 before any loop ran and selected per-decision; "
            "this generation inherits it. Nothing here is selected on the evaluation."),
        "the_bar": "problems solved on a paired run (理 11068). Probe accuracy is a sanity check.",
        "source_states": os.path.abspath(a.states), "layer": a.layer, "heads": res,
        "how_to_read_the_controls": (
            "permuted must collapse to chance. If PCA-8 or the row subsample matches the full head, "
            "the signal needs neither dimensions nor examples and is a key being copied."),
        "signed": "Sautee (sha-ta)"},
        open(a.json, "w"), indent=1, ensure_ascii=False)
    print("\n  wrote %s" % a.json)


if __name__ == "__main__":
    main()
