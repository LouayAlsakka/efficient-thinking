#!/usr/bin/env python
"""ET-VII E-E — assemble the per-judge forced artefacts into the registered file.

理 asked for `experience/results/et7_ee_forced.json`. The run writes one file per judge so an
arm can be reported the moment it lands; this joins them, applies §3b's readings mechanically rather
than by choosing them, and builds the cross-judge table §3b's third reading needs.

ONE DISTINCTION THIS FILE EXISTS TO KEEP STRAIGHT. The §3c depth confound applies to A*, not to A.
A* is read at absolute layer 18, which is 64% of depth at 1.5B and 7B and 37.5% at 14B, so the
A* row of the scaling table is NOT comparable across sizes until the sweep lands. A_forced is read
at the OUTPUT -- logit(A) against logit(B) at the decision position -- and has no layer, so the
forced curve across sizes is clean now. Δ_forced inherits the confound from its A* term. Every row
is labelled with which it is.
"""
import argparse, json, os, sys

READ_FAIL = ("the state and the output agree; the abstention was format. The registered claim "
             "FAILS under the definition that matters (§3b)")
READ_REAL = ("the elicitation gap is real under the publishable definition — the registered claim, "
             "restated under §3b")
READ_THIN = ("Δ_forced excludes zero but its lower bound is under 0.05; §3b's first reading is not "
             "met and the gap is not claimed at this n")


def readings(d):
    """§3b's table, applied. Controls first: a control firing withdraws Δ before anything is said."""
    fired, notes = [], []
    lo_probe = d["A_star_probe"]["CI95_t_df4"][0]
    probe = d["A_star_probe"]["mean"]
    folds = d["per_fold"]
    sub = sum(f["CONTROL_subsample_n30"] for f in folds) / len(folds)
    if abs(sub - probe) <= 0.02:
        fired.append("subsample n=30 matched the probe within 0.02 (%.3f vs %.3f)" % (sub, probe))
    pca = sum(f["CONTROL_pca8"] for f in folds) / len(folds)
    notes.append("PCA-8 %.3f %s the probe's fold-interval lower bound %.3f — %s"
                 % (pca, "below" if pca < lo_probe else "within", lo_probe,
                    "eight dimensions do not suffice" if pca < lo_probe else
                    "the probe's excess over eight dimensions is not established at this n"))
    perm = [x for f in folds for x in f["CONTROL_permuted"]]
    pm = sum(perm) / len(perm)
    if pm >= lo_probe:
        fired.append("permutation pooled mean %.3f did not collapse below the probe's interval" % pm)
    notes.append("permutation pooled %.3f, range %.3f–%.3f" % (pm, min(perm), max(perm)))

    # POLICY IDENTITY IS NOT A WITHDRAWAL TRIGGER ON THIS STRATUM, and an earlier version of this
    # function made it one. §3a: "If it scores near the correctness probe ON THE CONFOUNDED PAIRS,
    # the two are not separable there and THAT STRATUM's Δ is withdrawn. Its score ON THE BALANCED
    # PAIRS is the check that the balancing worked." Everything here is the balanced stratum, so the
    # reading is descriptive. It has to be: the balancing makes policy identity UNINFORMATIVE ABOUT
    # CORRECTNESS (the stronger policy is right 49.0% / 43.9% of the time), not undetectable in the
    # states — a model plainly encodes which answer came from the bigger policy. A correctness probe
    # cannot ride on a feature that predicts correctness below chance; if it did, it would be dragged
    # toward 0.439, not lifted. Reporting a high number here as a fired control would have withdrawn
    # a Δ the pre-registration does not license withdrawing.
    pol = sum(f["CONTROL_policy_identity"] for f in folds) / len(folds)
    notes.append("policy-identity %.3f on the BALANCED stratum — descriptive per §3a, not a trigger; "
                 "the stronger policy is correct 49.0%%/43.9%% here, so this feature cannot lift a "
                 "correctness probe%s" % (pol, " (and it reads close to the correctness probe, so say "
                 "so out loud rather than leaving it in the file)" if abs(pol - probe) < 0.05 else ""))

    ci = d["delta_forced_F2"]["CI95_t_df4"]
    forced = d["A_forced_F2_primary"]["mean"]
    if fired:
        verdict = "WITHDRAWN before reading: " + "; ".join(fired)
    elif ci[0] > 0 and ci[0] >= 0.05:
        verdict = READ_REAL
    elif ci[0] > 0:
        verdict = READ_THIN
    else:
        verdict = READ_FAIL
    return {"verdict": verdict, "controls_fired": fired, "control_notes": notes,
            "forced_ge_probe": forced >= probe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="per-judge et7_ee_forced_*.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {}
    for p in a.inputs:
        d = json.load(open(p))
        tag = os.path.basename(p).replace("et7_ee_forced_", "").replace(".json", "")
        arms[tag] = d
    order = sorted(arms, key=lambda t: float(t.replace("B", "")))

    table = []
    for t in order:
        d = arms[t]
        r = readings(d)
        table.append({
            "judge_tag": t, "judge": d["judge"], "n_cells": d["n_cells"],
            "A_forced_F2": d["A_forced_F2_primary"]["mean"],
            "A_forced_F2_CI": d["A_forced_F2_primary"]["CI95_t_df4"],
            "A_forced_F1": d["A_forced_F1_primary"]["mean"],
            "A_star_probe_LAYER18": d["A_star_probe"]["mean"],
            "A_star_probe_CI": d["A_star_probe"]["CI95_t_df4"],
            "A_tie_scored": d["A_judge_tie_scored"]["mean"],
            "delta_forced_F2": d["delta_forced_F2"]["mean"],
            "delta_forced_F2_CI": d["delta_forced_F2"]["CI95_t_df4"],
            "generated_tie_rate": d["P_TIE_F1"]["generated_tie_rate"],
            "P_TIE_median": d["P_TIE_F1"]["median_primary"],
            "F1_vs_F2_gap_points": d["F1_vs_F2_gap_points"],
            "swap_gap_points": d["position_bias_swap_gap_points_F2"],
            "reading": r["verdict"], "controls_fired": r["controls_fired"],
            "control_notes": r["control_notes"],
            "probe_reproduces_published_fold": d["probe_reproduction_assertion"].get("reproduces"),
        })

    any_real = [r for r in table if r["reading"] == READ_REAL]
    scaling = ("no judge shows a gap under the publishable definition" if not any_real else
               "gap present at: " + ", ".join(r["judge_tag"] for r in any_real))
    out = {
        "document": "ET-VII E-E — forced-preference A, all judges (prereg §3b)",
        "prereg": "docs/et7-ee-prereg.md §3b",
        "prereg_sha": {t: arms[t].get("prereg_sha", "") for t in order},
        "primary_stratum": "the two balanced policy pairs (§3a)",
        "WHAT_IS_COMPARABLE_ACROSS_SIZES": (
            "A_forced is read at the OUTPUT and has no layer, so its row IS comparable across "
            "judges now. A_star_probe is read at ABSOLUTE layer 18 — 64% of depth at 1.5B and 7B, "
            "37.5% at 14B — so its row is NOT comparable until the §3c relative-depth sweep lands, "
            "and delta_forced inherits that from its A* term. Read the A_forced row for the "
            "scaling question; read the A* row only within a judge."),
        "scaling_under_forced_A": scaling,
        "per_judge": table,
        "signed": "Sautee (sha-ta)",
    }
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  %-6s %-7s %-7s %-7s %-8s  %s" % ("judge", "A_forc", "A*(18)", "tie-sc", "Δ_forced", "reading"))
    for r in table:
        print("  %-6s %.3f   %.3f   %.3f   %+.3f %-16s %s"
              % (r["judge_tag"], r["A_forced_F2"], r["A_star_probe_LAYER18"], r["A_tie_scored"],
                 r["delta_forced_F2"], str(r["delta_forced_F2_CI"]), r["reading"][:60]))
    print("\n  scaling under forced A: %s\n  wrote %s" % (scaling, a.out))


if __name__ == "__main__":
    main()
