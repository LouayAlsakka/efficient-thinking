#!/usr/bin/env python
"""VIII-b §13b/§13c — the artefact, assembled from the scored games and the head fits.

WHY A SEPARATE FILE FROM THE SCORER. The scorer answers "what do the arms say"; this answers "what
may a reader conclude from them", and those are different jobs with different failure modes. The
scorer must not know about confounds; this must not recompute a number.

EVERY CLAIM CARRIES ITS OWN DISQUALIFIER, because the alternative is a reader supplying one:

  the g1' - g1-matched pair is CONFOUNDED BY THE MATCHING UNIT and is not read as origin. It is
  reported because the confound worked AGAINST g1-matched and g1-matched still won, which makes
  the domain-shift conclusion conservative rather than fragile.

  the probe accuracies are DESCRIPTIVE. The bar is problems solved on a paired run. §13c game 1
  is the fourth time in this programme that a better-fitting head did not solve more problems.

  a two-draw spread is a RANGE, not an sd. Rows built on two games say so.
"""
import argparse, json, os, sys


def load(p):
    return json.load(open(p, encoding="utf8")) if p and os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-score", required=True)
    ap.add_argument("--c-score", default="")
    ap.add_argument("--heads-dec", default="")
    ap.add_argument("--heads-row", default="")
    ap.add_argument("--redundancy", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b, c = load(a.b_score), load(a.c_score)
    if not b:
        sys.exit("STOP: no §13b score file")

    out = {
        "document": "VIII-b §13b and §13c — independence of experience, as measured",
        "the_bar": "problems solved on a paired run. Probe accuracy is a diagnostic and decides nothing.",
        "§14": b.get("§14"),
        "§13b": {
            "games": b.get("green_counts"), "within_arm": b.get("within_arm"),
            "pairings": b.get("pairings"),
            "⚠️_the_g1prime_pair_is_confounded": (
                "g1' is an INTACT collection that happens to be 1,437 rows; g1-matched was cut to a "
                "row budget, which splits decision groups and leaves 46% of them with no positive "
                "label. Not read as origin. Reported because the confound worked AGAINST "
                "g1-matched and g1-matched still won — the domain-shift conclusion is therefore "
                "conservative."),
            "⚠️_game_1_is_asymmetric_in_time": b.get("⚠️_game_1_is_asymmetric_in_time"),
        },
        "§13c": ({"games": c.get("green_counts"), "within_arm": c.get("within_arm"),
                  "pairings": c.get("pairings"), "plan_note": c.get("plan_note")} if c else
                 "not yet scored"),
        "head_fits": {
            "decision_matched": load(a.heads_dec), "row_matched_SUPERSEDED": load(a.heads_row),
            "why_both_are_here": (
                "the row-matched fits are kept beside the decision-matched ones because the "
                "after-number alone would be one nobody could audit. More than half of what looked "
                "like a family effect was the row cut: 57 points between the SQL head and "
                "g1-matched became 30 once both were cut on whole decision groups."),
        },
        "is_the_disjoint_experience_new": load(a.redundancy),
        "signed": "Sautee (sha-ta)",
    }
    json.dump(out, open(a.out, "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print("  wrote %s" % a.out)
    for k in ("§13b", "§13c"):
        v = out[k]
        if isinstance(v, dict) and v.get("games"):
            print("  %s arms: %s" % (k, ", ".join("%s %d/%d" % (n, d["green"], d["n"])
                                                  for n, d in sorted(v["games"].items()))))


if __name__ == "__main__":
    main()
