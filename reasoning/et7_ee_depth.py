#!/usr/bin/env python
"""ET-VII E-E — the probe across RELATIVE DEPTH. Pre-registered: docs/et7-ee-prereg.md §3c.

§9's scaling limb probed every judge at ABSOLUTE layer 18. `hidden_at` indexes inner.layers[i], and
the models are not the same depth: 18 is 64% of the way through 1.5B and 7B (28 blocks), 37.5%
through 14B (48), 28% through 32B (64). So "A* is flat across scale" was measured by a probe that
walks steadily EARLIER in the network as the model grows. This reads a fixed grid of RELATIVE depths
instead, so the scaling claim can be read at matched depth.

ONE forward pass per cell yields EVERY layer in the grid -- hidden_at already takes a list -- so the
sweep costs what a single-layer pass costs. No generation: the judge's picks are read from the
existing meta.json, unchanged.

SELF-CHECK: the grid always contains layer 18, and for a judge with published fold-wise numbers the
layer-18 column must reproduce them. If a re-extraction of the same prompt at the same layer does not
return the published fold, the sweep is measuring something else and says so before any curve.
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "experience"))
from et7_ee_forced import BAL, SYS_F1, USER, TAIL_F1, K, folds_over_problems   # one fold rule, shared

GRID = (0.25, 0.375, 0.50, 0.64, 0.75)


def layers_for(n_blocks, grid=GRID, always=(18,)):
    """The grid in absolute indices, plus 18 so the published setting is visible in every curve."""
    ls = {min(n_blocks - 1, max(0, int(round(r * n_blocks)))) for r in grid}
    ls |= {l for l in always if l < n_blocks}
    return sorted(ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(HERE, "et7_ee_cells.json"))
    ap.add_argument("--states", required=True, help="existing states dir -- meta.json is the labels")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--cache", default="", help="dir to write X_layer<N>.npy so a fit failure is cheap")
    ap.add_argument("--foldwise", default="", help="published fold-wise json to check layer 18 against")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import et8_head_v3 as H
    from mlx_lm import load
    import mlx.core as mx

    cells = json.load(open(a.cells))["cells"]
    meta = json.load(open(os.path.join(a.states, "meta.json")))
    idx = np.array([i for i, m in enumerate(meta) if tuple(m["pair"]) in BAL])
    if a.limit:
        idx = idx[:a.limit]
    model, tok = load(a.judge)
    n_blocks = len(model.model.layers)
    LAYERS = layers_for(n_blocks)
    print("  %s · %d blocks · layers %s (relative %s) · %d primary cells"
          % (a.judge, n_blocks, LAYERS, [round(l / n_blocks, 3) for l in LAYERS], len(idx)),
          file=sys.stderr)

    Xs = {l: [] for l in LAYERS}
    cached = a.cache and all(os.path.exists(os.path.join(a.cache, "X_layer%d.npy" % l)) for l in LAYERS)
    if cached:
        Xs = {l: np.load(os.path.join(a.cache, "X_layer%d.npy" % l)) for l in LAYERS}
        print("  reusing cached states from %s" % a.cache, file=sys.stderr)
    else:
        for n, i in enumerate(idx):
            c, m = cells[i], meta[i]
            flip = (m["correct"] != c["correct_side"])      # the order the states were taken under
            left, right = (c["answer_B"], c["answer_A"]) if flip else (c["answer_A"], c["answer_B"])
            msgs = [{"role": "system", "content": SYS_F1},
                    {"role": "user", "content": USER % (c["problem"], left, right, TAIL_F1)}]
            hs = H.hidden_at(model, tok, msgs, LAYERS)      # ONE pass, every layer in the grid
            for l in LAYERS:
                Xs[l].append(np.array(hs[l].astype(mx.float32), copy=False))
            if (n + 1) % 25 == 0:
                print("    %d/%d" % (n + 1, len(idx)), file=sys.stderr)
        Xs = {l: np.stack(v).astype(np.float64) for l, v in Xs.items()}
        if a.cache:
            os.makedirs(a.cache, exist_ok=True)
            for l, v in Xs.items():
                np.save(os.path.join(a.cache, "X_layer%d.npy" % l), v)
            print("  cached -> %s" % a.cache, file=sys.stderr)

    sub = [meta[i] for i in idx]
    y_cor = np.array([1.0 if m["correct"] == "A" else 0.0 for m in sub])
    y_str = np.array([1.0 if m["strong_side"] == "A" else 0.0 for m in sub])
    probsn = np.array([m["problem"] for m in sub])
    FOLDS = folds_over_problems(set(probsn.tolist()))
    rngnp = np.random.default_rng(11)

    def fit(X, mtr, mte, y, P=None, subi=None):
        mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-6
        Z = (X - mu) / sd
        if P is not None:
            Z = Z @ P
        ti = np.where(mtr)[0] if subi is None else subi
        w, b = H.logistic_fit(Z[ti], y[ti])
        return ((Z[mte] @ w + b) > 0).astype(float) == y[mte]

    per_layer = {}
    for l in LAYERS:
        X = Xs[l]
        rows = []
        for k, f in enumerate(FOLDS):
            mte = np.array([p in f for p in probsn]); mtr = ~mte
            acc = float(fit(X, mtr, mte, y_cor).mean())
            perm = []
            for _ in range(5):
                ys = y_cor.copy(); ti = np.where(mtr)[0]
                ys[ti] = rngnp.permutation(ys[ti])
                perm.append(round(float(fit(X, mtr, mte, ys).mean()), 4))
            Zt = (X[mtr] - X[mtr].mean(0)) / (X[mtr].std(0) + 1e-6)
            _, _, Vt = np.linalg.svd(Zt - Zt.mean(0), full_matrices=False)
            ti = np.where(mtr)[0]
            sn = min(30, len(ti))
            rows.append({"fold": k + 1, "cells": int(mte.sum()), "A_star": round(acc, 4),
                         "CONTROL_permuted": perm,
                         "CONTROL_pca8": round(float(fit(X, mtr, mte, y_cor, P=Vt[:8].T).mean()), 4),
                         "CONTROL_subsample": round(float(fit(X, mtr, mte, y_cor,
                                                    subi=rngnp.choice(ti, size=sn, replace=False)).mean()), 4),
                         "CONTROL_subsample_n": sn, "CONTROL_subsample_train_n": int(len(ti)),
                         "CONTROL_policy_identity": round(float(fit(X, mtr, mte, y_str).mean()), 4)})
        v = np.array([r["A_star"] for r in rows])
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        per_layer[str(l)] = {
            "layer": l, "relative_depth": round(l / n_blocks, 4), "per_fold": rows,
            "A_star_fold_mean": round(float(v.mean()), 4), "fold_se": round(se, 4),
            "CI95_t_df4": [round(float(v.mean() - 2.776 * se), 4), round(float(v.mean() + 2.776 * se), 4)]}
        print("    layer %2d (%.0f%%)  A* %.3f  CI %s" % (l, 100 * l / n_blocks, v.mean(),
              per_layer[str(l)]["CI95_t_df4"]), file=sys.stderr)

    check = {"checked": False}
    if a.foldwise and os.path.exists(a.foldwise) and "18" in per_layer:
        pub = [round(f["A_star"], 4) for f in json.load(open(a.foldwise))["folds"]]
        got = [r["A_star"] for r in per_layer["18"]["per_fold"]]
        check = {"checked": True, "published_layer18": pub, "recomputed_layer18": got,
                 "max_abs_diff": round(max(abs(x - y) for x, y in zip(got, pub)), 4),
                 "reproduces": all(abs(x - y) < 0.0005 for x, y in zip(got, pub)),
                 "why": ("a re-extraction of the same prompt at the same layer must return the "
                         "published fold, or this sweep is measuring something else")}
        print("  layer-18 reproduction: %s" % check["reproduces"], file=sys.stderr)

    best = max(per_layer.values(), key=lambda d: d["A_star_fold_mean"])
    json.dump({"document": "ET-VII E-E — probe across relative depth",
               "prereg": "docs/et7-ee-prereg.md §3c",
               "prereg_sha": os.environ.get("PREREG_SHA", ""),
               "why": ("§9 read every judge at absolute layer 18, which is 64% of the way through "
                       "1.5B and 7B and 37.5% through 14B; the flat-A* curve is confounded with "
                       "reading earlier as the model grows"),
               "judge": a.judge, "n_blocks": n_blocks, "grid_relative": list(GRID),
               "layers": LAYERS, "n_cells": int(len(idx)), "folds": K,
               "prompt": "the ORIGINAL three-way prompt the published states were taken under",
               "layer18_reproduction": check,
               "per_layer": per_layer,
               "best_layer": {"layer": best["layer"], "relative_depth": best["relative_depth"],
                              "A_star_fold_mean": best["A_star_fold_mean"]},
               "how_to_read": ("the scaling claim is read at MATCHED RELATIVE DEPTH across judges; "
                               "best-layer-per-judge is reported and is NOT the headline -- a probe "
                               "accuracy that moves with an instrument setting is not a measurement, "
                               "which is why the grid was fixed before any of it was seen"),
               "signed": "Sautee (sha-ta)"},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
