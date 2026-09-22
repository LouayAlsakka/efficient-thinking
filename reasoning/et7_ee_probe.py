#!/usr/bin/env python
"""ET-VII E-E stage 2 — A(W), A*(W) and Δ. Pre-registered: docs/et7-ee-prereg.md (+ §3a amendment).

A(W)   the judge's own pick accuracy on decisive pairs (TIE counts wrong: a refusal to express what
       it may still know is the paper's subject).
A*(W)  a LOWER BOUND on what the state knows — a linear probe on the judge's hidden states at the
       decision point, predicting which side is correct.
Δ      A* − A.

HELD OUT BY PROBLEM, never by pair (§3 — two pairs from one problem share its text).
STRATA (§3a): PRIMARY = the two balanced policy pairs, where the stronger policy is correct only
49.0%/43.9% of the time so policy identity buys nothing. SECONDARY = the eight confounded pairs,
reported separately and never as the headline.
FOUR controls (§3 + §3a): permutation x5, PCA-8, row subsample, and the POLICY-IDENTITY probe.
"""
import argparse, collections, hashlib, json, os, random, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "experience"))

SYS = ("You are a strict, impartial grader. Two assistants answered the same problem. Decide whose "
       "FINAL answer is correct. Reply with EXACTLY one token: A, B, or TIE.")
BAL = {("Qwen2.5-1.5B-Instruct-4bit", "Qwen2.5-3B-Instruct-4bit"),
       ("Qwen2.5-14B-Instruct-4bit", "Qwen2.5-7B-Instruct-4bit")}


def pair_key(c):
    return tuple(sorted((c["model_A"], c["model_B"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(HERE, "et7_ee_cells.json"))
    ap.add_argument("--judge", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import et8_head_v3 as H
    from mlx_lm import load, generate as gen
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    cells = json.load(open(a.cells))["cells"]
    if a.limit:
        cells = cells[:a.limit]
    rng = random.Random(0)
    model, tok = load(a.judge)
    sampler = make_sampler(temp=0.0)
    print("  judge %s · layer %d · %d cells" % (a.judge, a.layer, len(cells)), file=sys.stderr)

    X, meta = [], []
    for i, c in enumerate(cells):
        # ORDER RANDOMISED and the judge BLINDED — the existing harness's own presentation
        flip = rng.random() < 0.5
        left, right = (c["answer_B"], c["answer_A"]) if flip else (c["answer_A"], c["answer_B"])
        correct = ("B" if c["correct_side"] == "A" else "A") if flip else c["correct_side"]
        user = ("[Problem]\n%s\n\n[Assistant A]\n%s\n\n[Assistant B]\n%s\n\nWhich is correct? A, B, or TIE."
                % (c["problem"], left, right))
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
        t = (gen(model, tok, prompt=pr, max_tokens=4, sampler=sampler, verbose=False) or "").strip().upper()
        pick = "A" if t.startswith("A") else ("B" if t.startswith("B") else "TIE")
        hs = H.hidden_at(model, tok, msgs, [a.layer])
        X.append(np.array(hs[a.layer].astype(mx.float32), copy=False))
        stronger = max((c["model_A"], c["model_B"]), key=lambda m: float(re.search(r"(\d+\.?\d*)B", m).group(1)))
        strong_side = "A" if c["model_A"] == stronger else "B"
        if flip:
            strong_side = "B" if strong_side == "A" else "A"
        meta.append({"problem": c["problem_index"], "pair": pair_key(c), "correct": correct,
                     "judge_pick": pick, "strong_side": strong_side,
                     "balanced": pair_key(c) in BAL})
        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, len(cells)), file=sys.stderr)
    X = np.stack(X).astype(np.float64)

    probs = sorted({m["problem"] for m in meta})
    rng2 = random.Random(7); rng2.shuffle(probs)
    cut = int(0.75 * len(probs)); train_p = set(probs[:cut])
    tr = np.array([m["problem"] in train_p for m in meta])
    y_cor = np.array([1.0 if m["correct"] == "A" else 0.0 for m in meta])
    y_str = np.array([1.0 if m["strong_side"] == "A" else 0.0 for m in meta])

    def fit_eval(mask_tr, mask_te, y, rngnp):
        mu, sd = X[mask_tr].mean(0), X[mask_tr].std(0) + 1e-6
        Z = (X - mu) / sd
        w, b = H.logistic_fit(Z[mask_tr], y[mask_tr])
        acc = float((((Z[mask_te] @ w + b) > 0).astype(float) == y[mask_te]).mean())
        perm = []
        for _ in range(5):
            ys = y.copy(); ys[mask_tr] = rngnp.permutation(y[mask_tr])
            w2, b2 = H.logistic_fit(Z[mask_tr], ys[mask_tr])
            perm.append(float((((Z[mask_te] @ w2 + b2) > 0).astype(float) == y[mask_te]).mean()))
        Xc = Z[mask_tr] - Z[mask_tr].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False); P = Vt[:8].T
        w3, b3 = H.logistic_fit((Z @ P)[mask_tr], y[mask_tr])
        pca8 = float(((((Z @ P)[mask_te] @ w3 + b3) > 0).astype(float) == y[mask_te]).mean())
        idx = np.where(mask_tr)[0]
        sub = rngnp.choice(idx, size=min(200, len(idx)), replace=False)
        w4, b4 = H.logistic_fit(Z[sub], y[sub])
        sacc = float((((Z[mask_te] @ w4 + b4) > 0).astype(float) == y[mask_te]).mean())
        return acc, perm, pca8, sacc

    out = {"document": "ET-VII E-E — A, A* and Δ", "prereg": "docs/et7-ee-prereg.md",
           "judge": a.judge, "layer": a.layer, "cells": len(cells), "strata": {}}
    rngnp = np.random.default_rng(11)
    for tag, sel in (("PRIMARY_balanced_pairs", np.array([m["balanced"] for m in meta])),
                     ("SECONDARY_confounded_pairs", np.array([not m["balanced"] for m in meta]))):
        mtr, mte = tr & sel, (~tr) & sel
        if mtr.sum() < 40 or mte.sum() < 20:
            out["strata"][tag] = {"status": "too few cells", "train": int(mtr.sum()), "test": int(mte.sum())}
            print("    %-28s too few (train %d, test %d)" % (tag, mtr.sum(), mte.sum()), file=sys.stderr)
            continue
        A = float(np.mean([m["judge_pick"] == m["correct"] for m, s in zip(meta, sel) if s and not tr[meta.index(m)]])) \
            if False else float(np.mean([meta[i]["judge_pick"] == meta[i]["correct"] for i in np.where(mte)[0]]))
        Astar, perm, pca8, sacc = fit_eval(mtr, mte, y_cor, rngnp)
        pid, _, _, _ = fit_eval(mtr, mte, y_str, rngnp)
        base = float(max(np.mean(y_cor[mte]), 1 - np.mean(y_cor[mte])))
        out["strata"][tag] = {"n_test": int(mte.sum()), "n_train": int(mtr.sum()),
                              "A_judge_pick_accuracy": round(A, 3),
                              "A_star_probe_accuracy": round(Astar, 3),
                              "delta": round(Astar - A, 3),
                              "majority_class_baseline": round(base, 3),
                              "CONTROL_permuted": [round(x, 3) for x in perm],
                              "CONTROL_pca8": round(pca8, 3),
                              "CONTROL_row_subsample": round(sacc, 3),
                              "CONTROL_policy_identity_probe": round(pid, 3)}
        print("    %-28s n=%3d  A %.3f  A* %.3f  Δ %+.3f  base %.3f | perm %s pca8 %.3f sub %.3f POLICY %.3f"
              % (tag, mte.sum(), A, Astar, Astar - A, base,
                 [round(x, 2) for x in perm], pca8, sacc, pid), file=sys.stderr)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
