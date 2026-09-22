#!/usr/bin/env python
"""ET-VII E-E — the 32B arm, in ONE run and ONE artefact (理 12125).

Everything the other judges needed three passes to produce, from one walk over the 137 primary cells:
  · the TIE-SCORED A      — the three-way prompt, generated, 4 tokens (§2's original definition)
  · the FORCED A          — §3b: logit(A) vs logit(B), F2 and F1, both orderings, P(TIE) per cell
  · the DEPTH GRID        — §3c: layers [16, 18, 24, 32, 41, 48] of 64, one forward pass for all
  · the CROSSING          — §3d: probe vs forced on the cells the three-way judge TIED, at matched
                            relative depth 0.64 (layer 41), paired per fold
  · the pick distribution and every control's `sub n / train n`

THE PRESENTATION MUST MATCH THE OTHER JUDGES OR THE CELLS ARE NOT THE SAME CELLS. The left/right
order was drawn by `random.Random(0)` walking all 1,050 cells in file order; that walk is replayed
here and ASSERTED against an existing judge's meta.json before a single token is generated. A judge
reading a different presentation would produce a curve that looks comparable and is not.

States and picks are written to --states BEFORE any fit: the first version of the probe threw away
1,050 judge calls when the fit died on a missing scipy.
"""
import argparse, json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "experience"))
from et7_ee_forced import BAL, SYS_F1, SYS_F2, USER, TAIL_F1, TAIL_F2, K, folds_over_problems
from et7_ee_depth import layers_for

MATCHED_REL = 0.64
MIN_TIE_CELLS = 30          # §3d: judges under this are reported, not scored


def flips_for(cells):
    """Replay the presentation draw exactly as et7_ee_probe.py made it."""
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in cells]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(HERE, "et7_ee_cells.json"))
    ap.add_argument("--judge", default="mlx-community/Qwen2.5-32B-Instruct-4bit")
    ap.add_argument("--states", required=True, help="cache dir, written BEFORE the fit")
    ap.add_argument("--verify-against", default="",
                    help="an existing judge's states dir; its meta.json pins the presentation order")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import et8_head_v3 as H
    from mlx_lm import load, generate as gen
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    cells = json.load(open(a.cells))["cells"]
    flips = flips_for(cells)
    prim = [i for i, c in enumerate(cells) if tuple(sorted((c["model_A"], c["model_B"]))) in BAL]

    if a.verify_against:
        ref = json.load(open(os.path.join(a.verify_against, "meta.json")))
        assert len(ref) == len(cells), (len(ref), len(cells))
        ref_flips = [ref[i]["correct"] != cells[i]["correct_side"] for i in range(len(cells))]
        bad = [i for i in range(len(cells)) if ref_flips[i] != flips[i]]
        assert not bad, "presentation order does not match %s at %d cells" % (a.verify_against, len(bad))
        print("  presentation verified against %s — %d/%d flips identical"
              % (a.verify_against, len(cells), len(cells)), file=sys.stderr)

    if a.limit:
        prim = prim[:a.limit]

    meta_path = os.path.join(a.states, "meta.json")
    have = os.path.exists(meta_path)
    model = tok = None
    if not have:
        model, tok = load(a.judge)
        n_blocks = len(model.model.layers)
    else:
        n_blocks = json.load(open(meta_path))[0]["n_blocks"]
    LAYERS = layers_for(n_blocks)
    matched = min(LAYERS, key=lambda l: abs(l / n_blocks - MATCHED_REL))
    print("  %s · %d blocks · layers %s · matched %.2f -> layer %d · %d primary cells"
          % (a.judge, n_blocks, LAYERS, MATCHED_REL, matched, len(prim)), file=sys.stderr)

    if have:
        meta = json.load(open(meta_path))
        Xs = {l: np.load(os.path.join(a.states, "X_layer%d.npy" % l)) for l in LAYERS}
        print("  reusing %d cached cells from %s" % (len(meta), a.states), file=sys.stderr)
    else:
        sampler = make_sampler(temp=0.0)
        ID_A = tok._tokenizer.encode("A", add_special_tokens=False)[0]
        ID_B = tok._tokenizer.encode("B", add_special_tokens=False)[0]
        ID_T = tok._tokenizer.encode("TIE", add_special_tokens=False)[0]
        print("  token ids A=%d B=%d TIE(first)=%d" % (ID_A, ID_B, ID_T), file=sys.stderr)

        def msgs_for(problem, left, right, form):
            s, t = (SYS_F2, TAIL_F2) if form == "F2" else (SYS_F1, TAIL_F1)
            return [{"role": "system", "content": s},
                    {"role": "user", "content": USER % (problem, left, right, t)}]

        def logits_at(ms):
            ids = tok.apply_chat_template(ms, add_generation_prompt=True)
            lg = model(mx.array([ids]))[0, -1].astype(mx.float32)
            lp = np.array(lg - mx.logsumexp(lg), copy=False)
            lg = np.array(lg, copy=False)
            return float(lg[ID_A]), float(lg[ID_B]), float(np.exp(lp[ID_T]))

        meta, Xs = [], {l: [] for l in LAYERS}
        for n, i in enumerate(prim):
            c = cells[i]
            left, right = (c["answer_B"], c["answer_A"]) if flips[i] else (c["answer_A"], c["answer_B"])
            correct = ("B" if c["correct_side"] == "A" else "A") if flips[i] else c["correct_side"]
            m1 = msgs_for(c["problem"], left, right, "F1")
            txt = (gen(model, tok, prompt=tok.apply_chat_template(m1, add_generation_prompt=True),
                       max_tokens=4, sampler=sampler, verbose=False) or "").strip().upper()
            pick = "A" if txt.startswith("A") else ("B" if txt.startswith("B") else "TIE")
            hs = H.hidden_at(model, tok, m1, LAYERS)
            for l in LAYERS:
                Xs[l].append(np.array(hs[l].astype(mx.float32), copy=False))
            rec = {"cell": i, "problem": c["problem_index"], "correct": correct,
                   "judge_pick": pick, "judge_raw": txt[:8], "n_blocks": n_blocks,
                   "flip": bool(flips[i])}
            stronger = max((c["model_A"], c["model_B"]),
                           key=lambda mm: float(__import__("re").search(r"(\d+\.?\d*)B", mm).group(1)))
            ss = "A" if c["model_A"] == stronger else "B"
            rec["strong_side"] = ("B" if ss == "A" else "A") if flips[i] else ss
            for form in ("F2", "F1"):
                la, lb, pt = logits_at(msgs_for(c["problem"], left, right, form))
                sa, sb, spt = logits_at(msgs_for(c["problem"], right, left, form))
                rec[form + "_pick_primary"] = "A" if la > lb else "B"
                rec[form + "_pick_swapped"] = "A" if sa > sb else "B"
                rec[form + "_margin_primary"] = la - lb
                if form == "F1":
                    rec["P_TIE_primary"], rec["P_TIE_swapped"] = pt, spt
            meta.append(rec)
            if (n + 1) % 10 == 0:
                print("    %d/%d" % (n + 1, len(prim)), file=sys.stderr)
        os.makedirs(a.states, exist_ok=True)
        for l in LAYERS:
            np.save(os.path.join(a.states, "X_layer%d.npy" % l), np.stack(Xs[l]).astype(np.float64))
        json.dump(meta, open(meta_path, "w"))
        Xs = {l: np.stack(v).astype(np.float64) for l, v in Xs.items()}
        print("  cached %d cells -> %s" % (len(meta), a.states), file=sys.stderr)

    # ---------------- analysis ----------------
    for m in meta:
        cs = "B" if m["correct"] == "A" else "A"
        for form in ("F2", "F1"):
            m[form + "_ok_primary"] = m[form + "_pick_primary"] == m["correct"]
            m[form + "_ok_swapped"] = m[form + "_pick_swapped"] == cs
    y_cor = np.array([1.0 if m["correct"] == "A" else 0.0 for m in meta])
    y_str = np.array([1.0 if m["strong_side"] == "A" else 0.0 for m in meta])
    probsn = np.array([m["problem"] for m in meta])
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

    def interval(v):
        v = np.array(v, dtype=float)
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        return {"mean": round(float(v.mean()), 4), "se": round(se, 4),
                "CI95_t_df4": [round(float(v.mean() - 2.776 * se), 4),
                               round(float(v.mean() + 2.776 * se), 4)]}

    per_layer, pred_at = {}, {}
    for l in LAYERS:
        X, rows, cellpred = Xs[l], [], {}
        for k, f in enumerate(FOLDS):
            mte = np.array([p in f for p in probsn]); mtr = ~mte
            ok = fit(X, mtr, mte, y_cor)
            for j, ci in enumerate(np.where(mte)[0]):
                cellpred[int(ci)] = bool(ok[j])
            perm = []
            for _ in range(5):
                ys = y_cor.copy(); ti = np.where(mtr)[0]
                ys[ti] = rngnp.permutation(ys[ti])
                perm.append(round(float(fit(X, mtr, mte, ys).mean()), 4))
            Zt = (X[mtr] - X[mtr].mean(0)) / (X[mtr].std(0) + 1e-6)
            _, _, Vt = np.linalg.svd(Zt - Zt.mean(0), full_matrices=False)
            ti = np.where(mtr)[0]; sn = min(30, len(ti))
            rows.append({"fold": k + 1, "cells": int(mte.sum()), "A_star": round(float(ok.mean()), 4),
                         "CONTROL_permuted": perm,
                         "CONTROL_pca8": round(float(fit(X, mtr, mte, y_cor, P=Vt[:8].T).mean()), 4),
                         "CONTROL_subsample": round(float(fit(X, mtr, mte, y_cor,
                                             subi=rngnp.choice(ti, size=sn, replace=False)).mean()), 4),
                         "CONTROL_subsample_n": int(sn), "CONTROL_subsample_train_n": int(len(ti)),
                         "CONTROL_policy_identity": round(float(fit(X, mtr, mte, y_str).mean()), 4)})
        pred_at[l] = cellpred
        # KEY BY THE ABSOLUTE CELL INDEX, as the other judges' depth artefacts do. Internally this
        # loop indexes `meta` (0..136); the other scripts key by the cell's position in
        # et7_ee_cells.json. Two conventions in one series is how a later join silently scores the
        # wrong stratum — et7_ee_crossing.py refuses on a key mismatch, but a refusal is a
        # consolation prize, not a design.
        per_layer[str(l)] = {"layer": l, "relative_depth": round(l / meta[0]["n_blocks"], 4),
                             "per_fold": rows,
                             "per_cell_correct": {str(meta[a_]["cell"]): b_
                                                  for a_, b_ in sorted(cellpred.items())},
                             "per_cell_key": "absolute index into et7_ee_cells.json",
                             **interval([r["A_star"] for r in rows])}
        print("    layer %2d (%.0f%%)  A* %.3f" % (l, 100 * l / meta[0]["n_blocks"],
                                                   per_layer[str(l)]["mean"]), file=sys.stderr)

    fold_rows = []
    for k, f in enumerate(FOLDS):
        ii = [j for j, p in enumerate(probsn) if p in f]
        d = {"fold": k + 1, "cells": len(ii),
             "A_judge_tie_scored": round(float(np.mean([meta[j]["judge_pick"] == meta[j]["correct"]
                                                        for j in ii])), 4)}
        for form in ("F2", "F1"):
            pri = float(np.mean([meta[j][form + "_ok_primary"] for j in ii]))
            swp = float(np.mean([meta[j][form + "_ok_swapped"] for j in ii]))
            d.update({"A_forced_%s_primary" % form: round(pri, 4),
                      "A_forced_%s_swapped" % form: round(swp, 4),
                      "A_forced_%s_mean_order" % form: round((pri + swp) / 2, 4)})
        for l in (matched, 18):
            d["A_star_layer%d" % l] = per_layer[str(l)]["per_fold"][k]["A_star"]
            d["delta_forced_F2_layer%d" % l] = round(d["A_star_layer%d" % l] - d["A_forced_F2_primary"], 4)
        fold_rows.append(d)

    # §3d — the crossing, on the cells the THREE-WAY judge tied, at matched depth
    tie_idx = [j for j in range(len(meta)) if meta[j]["judge_pick"] == "TIE"]
    com_idx = [j for j in range(len(meta)) if meta[j]["judge_pick"] != "TIE"]
    d3 = {"matched_layer": matched, "tie_cells": len(tie_idx), "committed_cells": len(com_idx),
          "min_to_score": MIN_TIE_CELLS}
    if len(tie_idx) < MIN_TIE_CELLS:
        d3["status"] = ("REPORTED, NOT SCORED — %d tie cells is under §3d's floor of %d"
                        % (len(tie_idx), MIN_TIE_CELLS))
    else:
        d3["status"] = "scored"
        pf = []
        for k, f in enumerate(FOLDS):
            ii = [j for j in tie_idx if probsn[j] in f]
            if not ii:
                continue
            pr = float(np.mean([pred_at[matched][j] for j in ii]))
            fo = float(np.mean([meta[j]["F2_ok_primary"] for j in ii]))
            pf.append({"fold": k + 1, "tie_cells": len(ii), "probe": round(pr, 4),
                       "forced": round(fo, 4), "paired_diff": round(pr - fo, 4)})
        d3["per_fold"] = pf
        d3["paired_probe_minus_forced"] = interval([x["paired_diff"] for x in pf])
    for name, ii in (("on_tied_cells", tie_idx), ("on_committed_cells", com_idx)):
        if ii:
            d3[name] = {"n": len(ii),
                        "probe_layer%d" % matched: round(float(np.mean([pred_at[matched][j] for j in ii])), 4),
                        "forced_F2": round(float(np.mean([meta[j]["F2_ok_primary"] for j in ii])), 4),
                        "three_way_judge": round(float(np.mean([meta[j]["judge_pick"] == meta[j]["correct"]
                                                                for j in ii])), 4)}

    picks = {}
    for form in ("F2", "F1"):
        for o in ("primary", "swapped"):
            n_a = sum(1 for m in meta if m["%s_pick_%s" % (form, o)] == "A")
            picks["%s_%s" % (form, o)] = {"n": len(meta), "picked_A": n_a,
                                          "frac_A": round(n_a / len(meta), 4)}

    out = {"document": "ET-VII E-E — the 32B arm, one run (理 12125)",
           "prereg": "docs/et7-ee-prereg.md §3b + §3c + §3d",
           "prereg_sha": os.environ.get("PREREG_SHA", ""),
           "judge": a.judge, "n_blocks": meta[0]["n_blocks"], "n_cells": len(meta),
           "layers": LAYERS, "matched_relative_depth": MATCHED_REL, "matched_layer": matched,
           "presentation_verified_against": a.verify_against,
           "prompts_verbatim": {"F1_system": SYS_F1, "F2_system": SYS_F2,
                                "user_template_F1": USER % ("<problem>", "<left>", "<right>", TAIL_F1),
                                "user_template_F2": USER % ("<problem>", "<left>", "<right>", TAIL_F2)},
           "per_layer": per_layer, "per_fold": fold_rows,
           "A_judge_tie_scored": interval([r["A_judge_tie_scored"] for r in fold_rows]),
           "A_forced_F2_primary": interval([r["A_forced_F2_primary"] for r in fold_rows]),
           "A_forced_F2_swapped": interval([r["A_forced_F2_swapped"] for r in fold_rows]),
           "A_forced_F2_mean_order": interval([r["A_forced_F2_mean_order"] for r in fold_rows]),
           "A_forced_F1_primary": interval([r["A_forced_F1_primary"] for r in fold_rows]),
           "A_star_matched_depth": interval([r["A_star_layer%d" % matched] for r in fold_rows]),
           "A_star_layer18": interval([r["A_star_layer18"] for r in fold_rows]),
           "delta_forced_F2_matched_depth": interval([r["delta_forced_F2_layer%d" % matched] for r in fold_rows]),
           "delta_forced_F2_layer18": interval([r["delta_forced_F2_layer18"] for r in fold_rows]),
           "P_TIE_F1": {"mean": round(float(np.mean([m["P_TIE_primary"] for m in meta])), 4),
                        "median": round(float(np.median([m["P_TIE_primary"] for m in meta])), 4),
                        "generated_tie_rate": round(float(np.mean([m["judge_pick"] == "TIE" for m in meta])), 4)},
           "pick_distribution": picks,
           "crossing_3d": d3,
           "signed": "Sautee (sha-ta)"}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % a.out, file=sys.stderr)
    print("  tie-scored %.3f | forced F2 %.3f | A* matched(L%d) %.3f | Δ %s | tie rate %.3f"
          % (out["A_judge_tie_scored"]["mean"], out["A_forced_F2_primary"]["mean"], matched,
             out["A_star_matched_depth"]["mean"],
             out["delta_forced_F2_matched_depth"]["CI95_t_df4"],
             out["P_TIE_F1"]["generated_tie_rate"]), file=sys.stderr)


if __name__ == "__main__":
    main()
