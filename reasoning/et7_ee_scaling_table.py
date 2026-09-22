#!/usr/bin/env python
"""ET-VII E-E — the scaling table, read at MATCHED relative depth (§3c).

The four judges' numbers live in three different file shapes: the forced arms in
`et7_ee_forced_<j>.json`, the probe curves in `et7_ee_depth_<j>.json`, and 32B in its own
single-run `et7_ee_32B.json`. Joining them by hand is how a Δ gets read against a probe fitted at
the wrong depth — which is exactly what happened to 14B, whose Δ was reported against layer 18
(37.5% of its stack) until this was written.

TWO ROWS, TWO DIFFERENT KINDS OF COMPARABILITY, and the file says which is which:
  A_forced   read at the OUTPUT. No layer. Comparable across sizes as it stands.
  A_star     read at a LAYER. Comparable only at matched relative depth, per §3c.
  Δ_forced   inherits A_star's, so it is recomputed here PAIRED PER FOLD at the matched layer.

§3c's reading is applied, not chosen: a judge is "inside" if its matched-depth A* falls in the 7B
fold interval that §3c names as the yardstick.
"""
import argparse, json, os, sys
import numpy as np

MATCHED_REL = 0.64


def interval(v, tcrit=2.776):
    v = np.array(v, dtype=float)
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return {"mean": round(float(v.mean()), 4), "se": round(se, 4),
            "CI95_t_df4": [round(float(v.mean() - tcrit * se), 4),
                           round(float(v.mean() + tcrit * se), 4)]}


def matched_layer(per_layer):
    return min(per_layer.values(), key=lambda L: abs(L["relative_depth"] - MATCHED_REL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "experience", "results"))
    ap.add_argument("--judges", nargs="+", default=["1.5B", "7B", "14B", "32B"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for j in a.judges:
        p32 = os.path.join(a.dir, "et7_ee_32B.json")
        if j == "32B" and os.path.exists(p32):
            d = json.load(open(p32))
            L = matched_layer(d["per_layer"])
            f2 = [r["A_forced_F2_primary"] for r in d["per_fold"]]
            astar = [r["A_star_layer%d" % L["layer"]] for r in d["per_fold"]]
            row = {"judge": j, "model": d["judge"], "n_blocks": d["n_blocks"],
                   "matched_layer": L["layer"], "matched_relative": L["relative_depth"],
                   "A_star_matched": interval(astar),
                   "A_star_layer18": interval([r["A_star_layer18"] for r in d["per_fold"]]),
                   "A_forced_F2": interval(f2),
                   "A_tie_scored": interval([r["A_judge_tie_scored"] for r in d["per_fold"]]),
                   "tie_rate": d["P_TIE_F1"]["generated_tie_rate"],
                   "delta_forced_matched": interval(np.array(astar) - np.array(f2)),
                   "source": "et7_ee_32B.json (one run)"}
        else:
            dep = json.load(open(os.path.join(a.dir, "et7_ee_depth_%s.json" % j)))
            fo = json.load(open(os.path.join(a.dir, "et7_ee_forced_%s.json" % j)))
            L = matched_layer(dep["per_layer"])
            astar = [r["A_star"] for r in L["per_fold"]]
            a18 = [r["A_star"] for r in dep["per_layer"]["18"]["per_fold"]]
            f2 = [r["A_forced_F2_primary_order"] for r in fo["per_fold"]]
            # CROSS-CHECK: the depth sweep's layer-18 column and the forced run's own probe are
            # produced by different scripts from different extractions. They must agree exactly.
            own = [r["A_star_probe"] for r in fo["per_fold"]]
            agree = all(abs(x - y) < 5e-4 for x, y in zip(a18, own))
            row = {"judge": j, "model": fo["judge"], "n_blocks": dep["n_blocks"],
                   "matched_layer": L["layer"], "matched_relative": L["relative_depth"],
                   "A_star_matched": interval(astar), "A_star_layer18": interval(a18),
                   "A_forced_F2": interval(f2),
                   "A_tie_scored": interval([r["A_judge_tie_scored"] for r in fo["per_fold"]]),
                   "tie_rate": fo["P_TIE_F1"]["generated_tie_rate"],
                   "delta_forced_matched": interval(np.array(astar) - np.array(f2)),
                   "delta_forced_layer18_as_first_reported": fo["delta_forced_F2"]["CI95_t_df4"],
                   "layer18_column_agrees_with_forced_run": agree,
                   "source": "et7_ee_depth_%s.json + et7_ee_forced_%s.json" % (j, j)}
        rows.append(row)

    yard = [r for r in rows if r["judge"] == "7B"]
    yardstick = yard[0]["A_star_matched"]["CI95_t_df4"] if yard else None
    for r in rows:
        if yardstick:
            m = r["A_star_matched"]["mean"]
            r["inside_7B_fold_interval"] = bool(yardstick[0] <= m <= yardstick[1])
        ci = r["delta_forced_matched"]["CI95_t_df4"]
        r["delta_reading_3b"] = ("gap real under the publishable definition" if ci[0] >= 0.05 else
                                 "excludes zero but under the 0.05 floor" if ci[0] > 0 else
                                 "FAILS — the state and the output agree; the abstention was format")

    out = {"document": "ET-VII E-E — scaling at matched relative depth 0.64",
           "prereg": "docs/et7-ee-prereg.md §3b + §3c",
           "yardstick": {"judge": "7B", "A_star_matched_CI": yardstick,
                         "why": "§3c names the 7B fold interval at its headline setting as the test"},
           "comparability": {
               "A_forced": "read at the OUTPUT, no layer — comparable across sizes as it stands",
               "A_star": "read at a LAYER — comparable only at matched relative depth",
               "delta_forced": "inherits A_star's, so it is recomputed paired per fold at the matched layer"},
           "per_judge": rows,
           "flat_verdict": ("§9's 'flat' STANDS, restated at matched depth — every judge's matched-depth "
                            "A* lies inside the 7B fold interval"
                            if all(r.get("inside_7B_fold_interval") for r in rows) else
                            "§9's 'flat' is WITHDRAWN as written — at least one judge falls outside "
                            "the 7B fold interval at matched depth"),
           "signed": "Sautee (sha-ta)"}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  judge  blk  L(matched)  A*matched   A*@18    A_forced  tie-sc  tie%   Δ_forced matched")
    for r in rows:
        print("  %-5s  %3d  %2d (%.0f%%)   %.3f      %.3f    %.3f    %.3f   %.0f%%   %+.4f %s%s"
              % (r["judge"], r["n_blocks"], r["matched_layer"], 100 * r["matched_relative"],
                 r["A_star_matched"]["mean"], r["A_star_layer18"]["mean"], r["A_forced_F2"]["mean"],
                 r["A_tie_scored"]["mean"], 100 * r["tie_rate"],
                 r["delta_forced_matched"]["mean"], r["delta_forced_matched"]["CI95_t_df4"],
                 "" if r.get("inside_7B_fold_interval") else "  <-- OUTSIDE"))
    print("\n  %s\n  wrote %s" % (out["flat_verdict"], a.out))


if __name__ == "__main__":
    main()
