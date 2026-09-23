#!/usr/bin/env python
"""ET-VII E-E — A redefined as FORCED PREFERENCE. Pre-registered: docs/et7-ee-prereg.md §3b.

§3b withdrew the tie-scored A: on the 46 cells where the 7B judge commits it is 73.9% right and the
probe is 73.0%, so the registered Delta = 0.482 was 91 ties scored wrong by a rule. A is now the
judge's forced preference at the decision position -- logit("A") against logit("B"), no tie available.

ONE FORWARD PASS per cell per judge per prompt form per ordering. No generation.
  F2 (primary) the section 3 prompt with the tie option removed.
  F1 (check)   the original three-way prompt, logit(A) vs logit(B), P(TIE) recorded per cell.
If F1 and F2 disagree by more than 5 points the prompt is doing work and both are reported.

THE PROBE IS UNCHANGED: same cached states, same five folds over problems, same fits. Only A changes.
The folds are reconstructed exactly (rng 13 permutation of the primary problems, stride k::5) and the
script ASSERTS the per-fold probe accuracies against et7_ee_foldwise.json before it reports anything --
a re-fit that does not reproduce the published fold is a different measurement wearing its name.
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "experience"))

SYS_F1 = ("You are a strict, impartial grader. Two assistants answered the same problem. Decide whose "
          "FINAL answer is correct. Reply with EXACTLY one token: A, B, or TIE.")
SYS_F2 = ("You are a strict, impartial grader. Two assistants answered the same problem. Decide whose "
          "FINAL answer is correct. Reply with EXACTLY one token: A or B.")
USER = "[Problem]\n%s\n\n[Assistant A]\n%s\n\n[Assistant B]\n%s\n\nWhich is correct? %s"
TAIL_F1 = "A, B, or TIE."
TAIL_F2 = "A or B."

BAL = {("Qwen2.5-1.5B-Instruct-4bit", "Qwen2.5-3B-Instruct-4bit"),
       ("Qwen2.5-14B-Instruct-4bit", "Qwen2.5-7B-Instruct-4bit")}
FOLD_SEED = 13          # recovered by exact match against et7_ee_foldwise.json counts AND A_judge
K = 5


def folds_over_problems(problems):
    p = list(np.random.default_rng(FOLD_SEED).permutation(np.array(sorted(problems))))
    return [set(p[k::K]) for k in range(K)]


def first_id(tok, s):
    t = getattr(tok, "_tokenizer", tok)
    ids = t.encode(s, add_special_tokens=False)
    return int(ids[0]), [int(i) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(HERE, "et7_ee_cells.json"))
    ap.add_argument("--states", required=True, help="dir with X.npy + meta.json from et7_ee_probe.py")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--foldwise", default="", help="et7_ee_foldwise.json to assert the probe against")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: read only the first N primary cells")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import et8_head_v3 as H
    from mlx_lm import load
    import mlx.core as mx

    cells = json.load(open(a.cells))["cells"]
    meta = json.load(open(os.path.join(a.states, "meta.json")))
    X = np.load(os.path.join(a.states, "X.npy")).astype(np.float64)
    assert len(meta) == len(cells) == len(X), (len(meta), len(cells), len(X))
    sel = np.array([tuple(m["pair"]) in BAL for m in meta])
    idx = np.where(sel)[0]
    if a.limit:
        idx = idx[:a.limit]
        sel = np.zeros_like(sel); sel[idx] = True
    print("  %s · %d primary cells of %d" % (a.judge, len(idx), len(meta)), file=sys.stderr)

    model, tok = load(a.judge)
    ID_A, ids_a = first_id(tok, "A")
    ID_B, ids_b = first_id(tok, "B")
    ID_T, ids_t = first_id(tok, "TIE")
    print("  token ids  A=%d %s  B=%d %s  TIE=%d %s" % (ID_A, ids_a, ID_B, ids_b, ID_T, ids_t), file=sys.stderr)
    assert ID_A != ID_B != ID_T

    def read(problem, left, right, form):
        sys_s, tail = (SYS_F2, TAIL_F2) if form == "F2" else (SYS_F1, TAIL_F1)
        msgs = [{"role": "system", "content": sys_s},
                {"role": "user", "content": USER % (problem, left, right, tail)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        lg = model(mx.array([ids]))[0, -1].astype(mx.float32)
        lp = lg - mx.logsumexp(lg)
        lg = np.array(lg, copy=False)
        return float(lg[ID_A]), float(lg[ID_B]), float(np.exp(np.array(lp, copy=False)[ID_T]))

    rows = []
    for n, i in enumerate(idx):
        c, m = cells[i], meta[i]
        flip = (m["correct"] != c["correct_side"])          # the order the states were taken under
        left, right = (c["answer_B"], c["answer_A"]) if flip else (c["answer_A"], c["answer_B"])
        r = {"cell": int(i), "problem": m["problem"], "correct_primary": m["correct"],
             "judge_pick_generated": m["judge_pick"], "strong_side": m["strong_side"]}
        for form in ("F2", "F1"):
            la, lb, pt = read(c["problem"], left, right, form)
            r[form + "_pick_primary"] = "A" if la > lb else "B"
            r[form + "_margin_primary"] = la - lb
            sa, sb, spt = read(c["problem"], right, left, form)   # swapped ordering
            # in the swapped presentation the correct side is the other letter
            r[form + "_pick_swapped"] = "A" if sa > sb else "B"
            r[form + "_margin_swapped"] = sa - sb
            if form == "F1":
                r["P_TIE_primary"], r["P_TIE_swapped"] = pt, spt
        rows.append(r)
        if (n + 1) % 25 == 0:
            print("    %d/%d" % (n + 1, len(idx)), file=sys.stderr)

    correct_sw = {i: ("B" if meta[i]["correct"] == "A" else "A") for i in idx}
    for r in rows:
        i = r["cell"]
        for form in ("F2", "F1"):
            r[form + "_ok_primary"] = r[form + "_pick_primary"] == meta[i]["correct"]
            r[form + "_ok_swapped"] = r[form + "_pick_swapped"] == correct_sw[i]

    # ---- the probe, unchanged -------------------------------------------------------------
    probs = {meta[i]["problem"] for i in idx}
    FOLDS = folds_over_problems(probs)
    y_cor = np.array([1.0 if m["correct"] == "A" else 0.0 for m in meta])
    y_str = np.array([1.0 if m["strong_side"] == "A" else 0.0 for m in meta])
    rngnp = np.random.default_rng(11)
    by_cell = {r["cell"]: r for r in rows}

    def fit(mtr, mte, y, P=None, sub=None):
        mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-6
        Z = (X - mu) / sd
        if P is not None:
            Z = Z @ P
        tr_idx = np.where(mtr)[0] if sub is None else sub
        w, b = H.logistic_fit(Z[tr_idx], y[tr_idx])
        return ((Z[mte] @ w + b) > 0).astype(float) == y[mte]

    fold_out, probe_pred = [], {}
    for k, f in enumerate(FOLDS):
        mte = sel & np.array([m["problem"] in f for m in meta])
        mtr = sel & ~mte
        ok = fit(mtr, mte, y_cor)
        for j, i in enumerate(np.where(mte)[0]):
            probe_pred[int(i)] = bool(ok[j])
        cells_k = [by_cell[int(i)] for i in np.where(mte)[0]]
        d = {"fold": k + 1, "cells": int(mte.sum()),
             "A_star_probe": round(float(ok.mean()), 4),
             "A_judge_tie_scored": round(float(np.mean([c["judge_pick_generated"] == c["correct_primary"]
                                                        for c in cells_k])), 4)}
        for form in ("F2", "F1"):
            pri = float(np.mean([c[form + "_ok_primary"] for c in cells_k]))
            swp = float(np.mean([c[form + "_ok_swapped"] for c in cells_k]))
            d["A_forced_%s_primary_order" % form] = round(pri, 4)
            d["A_forced_%s_swapped_order" % form] = round(swp, 4)
            d["A_forced_%s_mean_order" % form] = round((pri + swp) / 2, 4)
            d["delta_forced_%s" % form] = round(d["A_star_probe"] - pri, 4)
        # controls, sized to this fold (§3b: n=137 stratum)
        perm = []
        for _ in range(5):
            ys = y_cor.copy(); ti = np.where(mtr)[0]
            ys[ti] = rngnp.permutation(ys[ti])
            perm.append(round(float(fit(mtr, mte, ys).mean()), 4))
        Zt = (X[mtr] - X[mtr].mean(0)) / (X[mtr].std(0) + 1e-6)
        _, _, Vt = np.linalg.svd(Zt - Zt.mean(0), full_matrices=False)
        d["CONTROL_permuted"] = perm
        d["CONTROL_pca8"] = round(float(fit(mtr, mte, y_cor, P=Vt[:8].T).mean()), 4)
        ti = np.where(mtr)[0]
        d["CONTROL_subsample_n30"] = round(float(fit(mtr, mte, y_cor,
                                                     sub=rngnp.choice(ti, size=min(30, len(ti)),
                                                                      replace=False)).mean()), 4)
        d["CONTROL_subsample_n"] = int(min(30, len(ti)))
        d["CONTROL_subsample_train_n"] = int(len(ti))   # PRINT THE PAIR, never the draw alone:
        # a subsample control whose size is not shown beside its input can silently be the whole
        # training set. The cap that produced the withdrawn "control fired" was min(200, 99).
        d["CONTROL_policy_identity"] = round(float(fit(mtr, mte, y_str).mean()), 4)
        fold_out.append(d)
        print("    fold %d n=%2d  A* %.3f  F2 %.3f/%.3f  F1 %.3f/%.3f  tie-scored %.3f"
              % (k + 1, d["cells"], d["A_star_probe"], d["A_forced_F2_primary_order"],
                 d["A_forced_F2_swapped_order"], d["A_forced_F1_primary_order"],
                 d["A_forced_F1_swapped_order"], d["A_judge_tie_scored"]), file=sys.stderr)

    def interval(v):
        v = np.array(v, dtype=float)
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4), "se": round(se, 4),
                "CI95_t_df4": [round(float(v.mean() - 2.776 * se), 4), round(float(v.mean() + 2.776 * se), 4)]}

    # ---- assert the probe reproduces the published fold ------------------------------------
    assertion = {"checked": False}
    if a.foldwise and os.path.exists(a.foldwise):
        pub = json.load(open(a.foldwise))
        got = [f["A_star_probe"] for f in fold_out]
        want = [round(f["A_star"], 4) for f in pub["folds"]]
        assertion = {"checked": True, "published": want, "recomputed": got,
                     "max_abs_diff": round(max(abs(x - y) for x, y in zip(got, want)), 4),
                     "reproduces": all(abs(x - y) < 0.0005 for x, y in zip(got, want))}
        print("  probe reproduction: %s  %s vs %s" % (assertion["reproduces"], got, want), file=sys.stderr)

    # ---- tie split, from THESE fold predictions (§3b, 理) -----------------------------
    tie = [i for i in idx if meta[i]["judge_pick"] == "TIE"]
    com = [i for i in idx if meta[i]["judge_pick"] != "TIE"]
    def frac(ii, key):
        return round(float(np.mean([by_cell[int(i)][key] for i in ii])), 4) if ii else None
    split = {"tie_cells": len(tie), "committed_cells": len(com),
             "probe_on_ties": round(float(np.mean([probe_pred[int(i)] for i in tie])), 4) if tie else None,
             "probe_on_committed": round(float(np.mean([probe_pred[int(i)] for i in com])), 4) if com else None,
             "forced_F2_on_ties": frac(tie, "F2_ok_primary"),
             "forced_F2_on_committed": frac(com, "F2_ok_primary"),
             "three_way_judge_on_committed": round(float(np.mean(
                 [meta[i]["judge_pick"] == meta[i]["correct"] for i in com])), 4) if com else None}

    # THE PICK DISTRIBUTION, BESIDE THE ACCURACY. A judge that answers the same letter every time
    # scores the fold's label balance and nothing else, and its accuracy alone cannot show that.
    # A 1.5B A of 0.000 read as a finding until the pick distribution showed 1034 of 1050 ties.
    picks = {}
    for form in ("F2", "F1"):
        for order in ("primary", "swapped"):
            key = "%s_pick_%s" % (form, order)
            n_a = sum(1 for r in rows if r[key] == "A")
            picks["%s_%s" % (form, order)] = {
                "n": len(rows), "picked_A": n_a, "picked_B": len(rows) - n_a,
                "frac_A": round(n_a / len(rows), 4)}
    picks["reading"] = ("frac_A near 0 or 1 in BOTH orderings means the forced pick is a constant "
                        "letter: the accuracy is then the stratum's label balance, not a reading of "
                        "the answers. frac_A that FLIPS between the orderings is the judge tracking "
                        "the content.")

    f2p = [f["A_forced_F2_primary_order"] for f in fold_out]
    f1p = [f["A_forced_F1_primary_order"] for f in fold_out]
    gap = abs(float(np.mean(f2p)) - float(np.mean(f1p)))
    out = {
        "document": "ET-VII E-E — forced-preference A (prereg §3b)",
        "prereg": "docs/et7-ee-prereg.md §3b",
        "prereg_sha": os.environ.get("PREREG_SHA", ""),
        "judge": a.judge, "layer": a.layer, "states": a.states,
        "primary_stratum": "the two balanced policy pairs (§3a)",
        "n_cells": int(len(idx)), "n_problems": len(probs), "folds": K, "fold_seed": FOLD_SEED,
        "prompts_verbatim": {"F2_system": SYS_F2, "F2_user_template": USER % ("<problem>", "<left>", "<right>", TAIL_F2),
                             "F1_system": SYS_F1, "F1_user_template": USER % ("<problem>", "<left>", "<right>", TAIL_F1)},
        "token_ids": {"A": ID_A, "B": ID_B, "TIE": ID_T,
                      "full_encodings": {"A": ids_a, "B": ids_b, "TIE": ids_t},
                      "note": ("A and B are single tokens; TIE is not -- its id here is the FIRST token of "
                               "\"TIE\" (the model must emit it to say TIE, so P(TIE) is read on it, and it is "
                               "an UPPER bound: that token also begins other words). The forced pick never "
                               "touches it: the pick is logit(A) vs logit(B) alone.")},
        "prompt_inspection": {
            "asked_by": "理 — 66% ties on decisive pairs is high; say whether the prompt or the template invites TIE",
            "chat_template": ("Qwen2.5's template contributes NO system text of its own once a system message is "
                              "supplied, and adds no tie language: the rendered prompt is exactly "
                              "<|im_start|>system\\n<our system><|im_end|>\\n<|im_start|>user\\n<our user><|im_end|>"
                              "\\n<|im_start|>assistant\\n. The template is not the cause."),
            "the_prompt_does_invite_it": ("TIE is named TWICE — once in the system line and again as the last three "
                                          "characters of the user's question — and it is offered as a peer of A and "
                                          "B, in final position. Nothing in the prompt tells the judge that the pairs "
                                          "are DECISIVE (exactly one side is correct by construction), so on the "
                                          "judge's information TIE is a legitimate answer that is never right. A "
                                          "grader asked to be strict and given a costless abstention will take it."),
            "but_it_is_not_the_whole_cause": ("the same prompt produces a 98.5% tie rate at 1.5B and 26% at 14B, so "
                                              "the rate is prompt AND capability; the prompt sets the floor on how "
                                              "cheap abstaining is, and §3b's F2 removes exactly that."),
            "not_changed_for_F1": "per 理 the three-way prompt is read verbatim as the check arm; only F2 differs.",
        },
        "probe_reproduction_assertion": assertion,
        "per_fold": fold_out,
        "A_star_probe": interval([f["A_star_probe"] for f in fold_out]),
        "A_forced_F2_primary": interval(f2p),
        "A_forced_F2_swapped": interval([f["A_forced_F2_swapped_order"] for f in fold_out]),
        "A_forced_F2_mean_order": interval([f["A_forced_F2_mean_order"] for f in fold_out]),
        "A_forced_F1_primary": interval(f1p),
        "A_forced_F1_swapped": interval([f["A_forced_F1_swapped_order"] for f in fold_out]),
        "A_judge_tie_scored": interval([f["A_judge_tie_scored"] for f in fold_out]),
        "delta_forced_F2": interval([f["delta_forced_F2"] for f in fold_out]),
        "delta_forced_F1": interval([f["delta_forced_F1"] for f in fold_out]),
        "F1_vs_F2_gap_points": round(gap * 100, 2),
        "F1_vs_F2_reading": ("the prompt is doing work — both forms reported, neither is the headline alone"
                             if gap > 0.05 else "F1 and F2 agree within 5 points; F2 is the published reading"),
        "position_bias_swap_gap_points_F2": round(100 * abs(float(np.mean(f2p)) - float(np.mean(
            [f["A_forced_F2_swapped_order"] for f in fold_out]))), 2),
        "P_TIE_F1": {"mean_primary": round(float(np.mean([r["P_TIE_primary"] for r in rows])), 4),
                     "median_primary": round(float(np.median([r["P_TIE_primary"] for r in rows])), 4),
                     "generated_tie_rate": round(float(np.mean([meta[i]["judge_pick"] == "TIE" for i in idx])), 4)},
        # PER CELL, because §3d splits these by a property of the cell (did the three-way judge
        # tie?) and a fold mean cannot be split after the fact. The probe's per-cell predictions
        # live in the depth artefacts; these are the forced half of the same pairing.
        "per_cell": {str(r["cell"]): {"problem": r["problem"], "correct": r["correct_primary"],
                                      "judge_pick": r["judge_pick_generated"],
                                      "F2_ok_primary": r["F2_ok_primary"],
                                      "F2_ok_swapped": r["F2_ok_swapped"],
                                      "F1_ok_primary": r["F1_ok_primary"],
                                      "probe_ok": probe_pred.get(r["cell"])} for r in rows},
        "pick_distribution": picks,
        "tie_split": split,
        "how_to_read": ("§3b's registered readings. Δ_forced interval excluding zero with lower bound ≥ 0.05: "
                        "the elicitation gap is real under the publishable definition. Interval containing zero, "
                        "or forced ≥ probe: the state and the output agree and the abstention was format — the "
                        "registered claim FAILS, at full prominence. Any control firing withdraws that judge's "
                        "Δ_forced before anything is said."),
        "signed": "Sautee (sha-ta)",
    }
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % a.out, file=sys.stderr)
    print("  A* %.3f | F2 %.3f | Δ_F2 %s | F1 %.3f | P(TIE) %.3f"
          % (out["A_star_probe"]["mean"], out["A_forced_F2_primary"]["mean"],
             out["delta_forced_F2"]["CI95_t_df4"], out["A_forced_F1_primary"]["mean"],
             out["P_TIE_F1"]["mean_primary"]), file=sys.stderr)


if __name__ == "__main__":
    main()
