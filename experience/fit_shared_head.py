#!/usr/bin/env python
"""ET-8b §13b: fit ONE SHARED head from a state set that has no per-decision structure.

WHY THIS FILE EXISTS, AND WHY IT IS NOT A FLAG ON fit_gen_heads.py.
§13b asks whether a generation's gain survives when the EXPERIENCE COMES FROM ANOTHER TASK FAMILY
at the same amount. The SQL side of that comparison (R7) was a SINGLE-DECISION experiment: its
states carry no `decision` field at all, so the three per-decision heads fit_gen_heads.py builds
cannot be built from them (`KeyError: 'decision'`). 理 12436 ruled the form: BOTH sides of the
§13b pair are fitted here as one shared head over all rows -- gen1' on the 1,437 SQL rows, and
gen1-matched on 1,437 gen1 rows subsampled at a fixed seed -- so the pair holds AMOUNT fixed and
FORM fixed and varies ORIGIN alone. `et8b_loop --head` consumes exactly this artifact; the flag
predates §13b and is not being introduced for it.

WHAT THIS IS NOT. It is NOT the published gen1 head. The published gen1 is three per-decision heads
on 2,824 rows. `gen1 - gen1-matched` would therefore vary three-heads-vs-one AND 2824-vs-1437 at
once; 理 dropped that row rather than report a confounded number, and the form difference is printed
in every artifact this writes so that a form effect is never later quoted as an origin effect.

fit_gen_heads.py is untouched: a running chain re-reads it, and the two fitters answer different
questions. The evaluation machinery below is deliberately the SAME code shape as its `evaluate()` --
same three controls, same folds-over-TASKS rule (never over rows: two candidates of one task split
across folds would put the answer in the training set).

THE EMPTY-MASK REFUSAL IS INHERITED AND IS THE POINT. An empty held-out mask is a K-fold CV over
the training tasks -- accuracy AND all three controls computed inside the CV loop -- or a hard
refusal. It never writes a zero that reads as a failed head with passing controls.

Probe accuracy here is descriptive. The bar is problems solved on a paired run (理 11068).
"""
import argparse, collections, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--cut", type=int, default=226)
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="when the eval mask is empty, report K-fold CV over the TRAINING TASKS. "
                         "Descriptive only; the deployed head is still fitted on all of them.")
    ap.add_argument("--out-head", required=True, help="the .npz et8b_loop --head reads")
    ap.add_argument("--json", required=True)
    ap.add_argument("--name", required=True, help="arm name for the artifact, e.g. gen1prime")
    ap.add_argument("--form-note", default="", help="one line naming this arm's form vs published gen1")
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
    has_dec = all("decision" in m for m in meta)

    tr, te = tid < a.cut, tid >= a.cut
    print("  states %s  meta %d  train rows %d  held-out rows %d"
          % (X.shape, len(meta), tr.sum(), te.sum()))
    print("  per-decision field present: %s%s" % (has_dec,
          ("  %s" % dict(sorted(collections.Counter(m["decision"] for m in meta).items())))
          if has_dec else "  (single-decision source — this is why the head is shared)"))
    CV = te.sum() == 0
    if CV and not a.cv_folds:
        sys.exit("STOP: the held-out mask is EMPTY (no rows at or above task %d). That would write "
                 "`0.0%% (n=0, chance nan)` with every control at 0.0 and call it a measurement. "
                 "Pass --cv-folds K to report CV over the training tasks instead." % a.cut)
    if tr.sum() < 50:
        sys.exit("STOP: %d training rows, under the 50 floor" % tr.sum())

    # ONE pick per (task, decision) where a decision exists, else per task: a shared head still
    # decides once per decision point, and pooling them into one task-level pick would score a
    # three-decision episode as if it were one.
    def gkey(i):
        return (meta[i]["task_id"], meta[i].get("decision", 0))

    def pick(s, idx):
        g = collections.defaultdict(list)
        for i in idx:
            g[gkey(i)].append((s[i], meta[i]["label"]))
        return 100.0 * sum(1 for v in g.values() if max(v)[1] == 1) / max(1, len(g)), len(g)

    def evaluate(ftr, fte, rng):
        """One fit + its three controls on one (train, eval) split."""
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
        sub_n = min(500, len(ii))
        sub_i = rng.choice(ii, size=sub_n, replace=False)
        w4, b4 = H.logistic_fit(Z[sub_i], y[sub_i])
        sub = pick(Z @ w4 + b4, idx_te)[0]
        return acc, n, perm, pca8, sub, sub_n

    rng = np.random.default_rng(7)
    if CV:
        tasks = sorted(set(tid[tr].tolist()))
        order = np.random.default_rng(13).permutation(len(tasks))
        folds = [set(tasks[i] for i in order[k::a.cv_folds]) for k in range(a.cv_folds)]
        accs, ns, perms, pcas, subs, subns = [], [], [], [], [], []
        for fk in folds:
            inf = np.array([t in fk for t in tid])
            ftr, fte = tr & ~inf, tr & inf
            if ftr.sum() < 50 or fte.sum() < 20:
                continue
            A_, n_, p_, q_, s_, sn_ = evaluate(ftr, fte, rng)
            accs.append(A_ * n_); ns.append(n_); perms.append(p_)
            pcas.append(q_ * n_); subs.append(s_ * n_); subns.append(sn_)
        if not ns:
            sys.exit("STOP: CV produced no evaluable fold")
        tot = float(sum(ns))
        acc, n = sum(accs) / tot, int(tot)
        perm = [round(sum(pf[k] * ns[j] for j, pf in enumerate(perms)) / tot, 1) for k in range(5)]
        pca8, sub = sum(pcas) / tot, sum(subs) / tot
        eval_label = ("%d-fold CV over the TRAINING tasks (descriptive; the deployed head is "
                      "fitted on all of them)" % a.cv_folds)
        chance_src = np.where(tr)[0]
        sub_realised = "%s of %d train rows per fold" % (sorted(set(subns)), int(tr.sum()))
    else:
        if te.sum() < 20:
            sys.exit("STOP: only %d held-out rows" % te.sum())
        acc, n, perm, pca8, sub, sn = evaluate(tr, te, rng)
        perm = [round(p, 1) for p in perm]
        eval_label = "held out"
        chance_src = np.where(te)[0]
        sub_realised = "%d of %d train rows" % (sn, int(tr.sum()))

    # the DEPLOYED head is always fitted on the full training set
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    w, b = H.logistic_fit(((X - mu) / sd)[tr], y[tr])
    os.makedirs(os.path.dirname(os.path.abspath(a.out_head)), exist_ok=True)
    np.savez(a.out_head, w=w, b=np.array([b]), mu=mu, sd=sd,
             train_task_max=np.array([a.cut]),
             train_ranges=np.array([[int(tid[tr].min()), int(tid[tr].max())]], dtype=int))
    chance = 100 * np.mean([1.0 / meta[i]["n_regions"] for i in chance_src])
    key = "cv_pick_pct" if CV else "held_out_pick_pct"
    print("  SHARED head [%s]: %.1f%% (n=%d, chance %.1f) perm %s pca8 %.1f sub %.1f -> %s"
          % ("CV" if CV else "held out", acc, n, chance, perm, pca8, sub,
             os.path.basename(a.out_head)))
    if pca8 >= acc or sub >= acc:
        print("  ⚠️ a control matched or beat the full head — read the controls note before using this")

    json.dump({
        "document": "ET-8b §13b %s — ONE SHARED head (理 12436, option (i))" % a.name,
        "arm": a.name,
        "head_form": "shared: one head for every decision point",
        "⚠️_form_vs_published_gen1": (a.form_note or
            "the published gen1 is THREE per-decision heads on 2,824 rows. This arm is one shared "
            "head. A difference between this arm and published gen1 is a FORM difference as much as "
            "anything else and must never be quoted as an origin effect. The §13b pair is "
            "gen1' vs gen1-matched, both fitted here, both one shared head, both 1,437 rows."),
        "why_shared": (
            "R7's SQL states carry no `decision` field — that experiment was single-decision — so "
            "three per-decision heads cannot be fitted from them. 理 12436 ruled both sides of the "
            "pair to this form rather than spend 3 h of GPU regenerating SQL loop states."),
        "evaluation_mode": ("%d-fold CV over training tasks" % a.cv_folds) if CV else "held out",
        "fit_on": "task_id < %d" % a.cut,
        "rows": int(X.shape[0]), "train_rows": int(tr.sum()),
        "source_has_decision_field": has_dec,
        "layer": a.layer, "source_states": os.path.abspath(a.states),
        "head": {
            key: round(acc, 1), "n": n, "chance_pct": round(chance, 1), "evaluation": eval_label,
            "CONTROL_permuted": perm, "CONTROL_pca8": round(pca8, 1),
            "CONTROL_row_subsample": round(sub, 1),
            "CONTROL_row_subsample_realised_n": sub_realised,
            "weights": os.path.basename(a.out_head)},
        "how_to_read_the_controls": (
            "permuted must collapse to chance. If PCA-8 or the row subsample matches the full head, "
            "the signal needs neither dimensions nor examples and is a key being copied."),
        "the_bar": "problems solved on a paired run (理 11068). Probe accuracy is a sanity check.",
        "signed": "Sautee (sha-ta)"},
        open(a.json, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % a.json)


if __name__ == "__main__":
    main()
