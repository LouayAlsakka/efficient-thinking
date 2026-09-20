#!/usr/bin/env python
"""ET-8b: fit the three PER-DECISION heads for one generation.

THE ARCHITECTURE IS NOT RE-CHOSEN HERE. gen0 applied the spec's rule (docs/et8b-loop-gates.md §3,
shared vs per-decision on held-out probe accuracy) BEFORE any loop ran and selected per-decision.
That choice is fixed for the campaign. This script fits per-decision heads and REPORTS held-out
accuracy descriptively -- it never selects on it. Selecting on the set the loop is then evaluated
on is the contamination that cost gen0 its headline (+35.0 -> +17.3).

Fit is on task_id < --cut. Everything at or above --cut is held out and MUST be the loop's
evaluation set. Each head's npz carries train_task_max so the boundary travels with the weights.
"""
import argparse, collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, help="dir with meta.jsonl + X_layer<L>.npy")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--cut", type=int, default=226, help="fit on task_id < cut; >= cut is held out")
    ap.add_argument("--out-heads", required=True, help="dir to write head_d{1,2,3}_layer<L>.npz")
    ap.add_argument("--json", required=True, help="artifact path")
    ap.add_argument("--gen", required=True, help="generation label, e.g. gen1")
    ap.add_argument("--train-range", action="append", default=None, metavar="LO:HI",
                    help="repeatable, inclusive. Fit on these task ids INSTEAD of task_id < cut. "
                         "Used for cross-fitting: fold f is run by a head that never saw fold f.")
    ap.add_argument("--eval-range", action="append", default=None, metavar="LO:HI",
                    help="repeatable, inclusive. Report held-out accuracy on these instead of >= cut.")
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
    print("  eval on task ids %s" % (eval_spans or ">= %d" % a.cut))
    print("  states %s  meta %d  train rows %d  held-out rows %d"
          % (X.shape, len(meta), tr.sum(), te.sum()))
    print("  per decision: %s" % dict(sorted(collections.Counter(dec.tolist()).items())))

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
        if mtr.sum() < 50:
            sys.exit("STOP: decision %d has %d training rows, under the 50 floor" % (d, mtr.sum()))
        mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-6
        Z = (X - mu) / sd
        w, b = H.logistic_fit(Z[mtr], y[mtr])
        acc, n = pick(Z @ w + b, np.where(mte)[0])
        # CONTROL 1 -- permuted labels x5. Must collapse to chance.
        perm = []
        for _ in range(5):
            ys = y.copy(); ys[mtr] = rng.permutation(y[mtr])
            w2, b2 = H.logistic_fit(Z[mtr], ys[mtr])
            perm.append(round(pick(Z @ w2 + b2, np.where(mte)[0])[0], 1))
        # CONTROL 2 -- PCA-8. If 8 dims suffice, it is a key being copied, not a readout.
        Xc = Z[mtr] - Z[mtr].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt[:8].T
        w3, b3 = H.logistic_fit((Z @ P)[mtr], y[mtr])
        pca8 = round(pick((Z @ P) @ w3 + b3, np.where(mte)[0])[0], 1)
        # CONTROL 3 -- n=100 decisions' worth of rows. If examples are not needed, same.
        idx = np.where(mtr)[0]
        sub = rng.choice(idx, size=min(500, len(idx)), replace=False)
        w4, b4 = H.logistic_fit(Z[sub], y[sub])
        n100 = round(pick(Z @ w4 + b4, np.where(mte)[0])[0], 1)

        out = os.path.join(a.out_heads, "head_d%d_layer%d.npz" % (d, a.layer))
        spans = np.array(train_spans if train_spans else [[0, a.cut - 1]], dtype=int)
        np.savez(out, w=w, b=np.array([b]), mu=mu, sd=sd, train_task_max=np.array([a.cut]),
                 train_ranges=spans)
        chance = 100 * np.mean([1.0 / meta[i]["n_regions"] for i in np.where(mte)[0]])
        res["decision_%d" % d] = {
            "held_out_pick_pct": round(acc, 1), "n": n, "chance_pct": round(chance, 1),
            "train_decisions": int(len(set((meta[i]["task_id"], meta[i]["decision"])
                                           for i in np.where(mtr)[0]))),
            "CONTROL_permuted": perm, "CONTROL_pca8": pca8, "CONTROL_n500rows": n100,
            "weights": os.path.basename(out)}
        print("  decision %d: %.1f%% (n=%d, chance %.1f) perm %s pca8 %.1f nsub %.1f -> %s"
              % (d, acc, n, chance, perm, pca8, n100, os.path.basename(out)))

    json.dump({
        "document": "ET-8b %s heads — per-decision, the architecture FIXED at gen0" % a.gen,
        "architecture_not_rechosen": (
            "gen0 applied docs/et8b-loop-gates.md §3 (shared vs per-decision, on held-out probe "
            "accuracy, before any loop run) and selected per-decision. This generation inherits "
            "that choice. Nothing here is selected on the held-out set."),
        "fit_on": ("task ids %s (CROSS-FIT)" % train_spans) if train_spans
                  else "task_id < %d" % a.cut,
        "held_out": "task_id >= %d — THE LOOP'S EVALUATION SET, and the loop must run only this"
                    % a.cut,
        "⚠️_why_the_split_is_stated_twice": (
            "gen0's chain fitted on task_id < 226 and ran the loop on all 300, so 225 of 300 "
            "evaluation tasks sat in the head's own training set. It printed +35.0 and the "
            "result is +17.3. The boundary is in the npz (train_task_max) so it travels."),
        "source_states": os.path.abspath(a.states),
        "layer": a.layer,
        "heads": res,
        "how_to_read_the_controls": (
            "permuted must collapse to chance. If PCA-8 or the row subsample matches the full "
            "head, the signal needs neither dimensions nor examples and is a key being copied."),
        "signed": "Sautee (sha-ta)"},
        open(a.json, "w"), indent=1, ensure_ascii=False)
    print("\n  wrote %s" % a.json)


if __name__ == "__main__":
    main()
