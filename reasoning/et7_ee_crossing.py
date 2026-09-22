#!/usr/bin/env python
"""ET-VII §3d — the crossing: probe against forced judge ON THE CELLS THE THREE-WAY JUDGE TIED.

Registered by R 2026-09-22 after the 14B arm showed probe 0.833 / forced 0.500 on the 36 tied cells
and probe 0.614 / forced 0.772 on the 101 committed ones — not an ordering, a crossing. §3d scores it
per fold, paired, at matched relative depth 0.64. **Post-hoc for 7B and 14B and written down as such**;
pre-registered only for 32B, which is why the 32B arm computes it inside its own run.

It joins two halves that are produced by different scripts and must be keyed the same way:
  probe  — `per_cell_correct` from `et7_ee_depth_<judge>.json` at the chosen layer
  forced — `per_cell` from `et7_ee_forced_<judge>.json` (the patched reader)
Both are keyed by the cell's ABSOLUTE index into et7_ee_cells.json, and this script refuses to run
if the two key sets differ rather than silently scoring their intersection.
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from et7_ee_forced import K, folds_over_problems

MIN_TIE_CELLS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", required=True)
    ap.add_argument("--forced", required=True)
    ap.add_argument("--layer", type=int, required=True, help="matched relative depth 0.64 for this judge")
    ap.add_argument("--label", default="post-hoc (§3d registered for 32B; this judge is post-hoc)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dep = json.load(open(a.depth))
    fo = json.load(open(a.forced))
    if str(a.layer) not in dep["per_layer"]:
        sys.exit("layer %d not in the sweep grid %s" % (a.layer, dep["layers"]))
    L = dep["per_layer"][str(a.layer)]
    probe = {int(k): bool(v) for k, v in L["per_cell_correct"].items()}
    cells = fo.get("per_cell")
    if not cells:
        sys.exit("%s has no per_cell block — it predates the patched reader; re-emit it first" % a.forced)
    forced = {int(k): v for k, v in cells.items()}
    if set(probe) != set(forced):
        sys.exit("cell key sets differ (probe %d, forced %d, overlap %d) — refusing to score an "
                 "intersection that would silently be a different stratum"
                 % (len(probe), len(forced), len(set(probe) & set(forced))))

    probsn = {c: forced[c]["problem"] for c in forced}
    FOLDS = folds_over_problems(set(probsn.values()))
    tie = [c for c in forced if forced[c]["judge_pick"] == "TIE"]
    com = [c for c in forced if forced[c]["judge_pick"] != "TIE"]

    def block(ii):
        return {"n": len(ii),
                "probe": round(float(np.mean([probe[c] for c in ii])), 4),
                "forced_F2": round(float(np.mean([forced[c]["F2_ok_primary"] for c in ii])), 4),
                "probe_minus_forced": round(float(np.mean([probe[c] for c in ii])
                                                  - np.mean([forced[c]["F2_ok_primary"] for c in ii])), 4)}

    out = {"document": "ET-VII E-E §3d — the crossing on tied cells",
           "prereg": "docs/et7-ee-prereg.md §3d", "status_label": a.label,
           "judge": fo["judge"], "layer": a.layer,
           "relative_depth": L["relative_depth"], "n_cells": len(forced),
           "sources": {"probe": os.path.basename(a.depth), "forced": os.path.basename(a.forced)},
           "aggregate_on_tied_cells": block(tie) if tie else None,
           "aggregate_on_committed_cells": block(com) if com else None}

    if len(tie) < MIN_TIE_CELLS:
        out["per_fold"] = None
        out["verdict"] = ("REPORTED, NOT SCORED — %d tied cells is under §3d's floor of %d"
                          % (len(tie), MIN_TIE_CELLS))
    else:
        pf = []
        for k, f in enumerate(FOLDS):
            ii = [c for c in tie if probsn[c] in f]
            if not ii:
                continue
            pr = float(np.mean([probe[c] for c in ii]))
            fc = float(np.mean([forced[c]["F2_ok_primary"] for c in ii]))
            pf.append({"fold": k + 1, "tie_cells": len(ii), "probe": round(pr, 4),
                       "forced": round(fc, 4), "paired_diff": round(pr - fc, 4)})
        v = np.array([x["paired_diff"] for x in pf])
        df = len(v) - 1
        tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(df, 2.776)
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        ci = [round(float(v.mean() - tcrit * se), 4), round(float(v.mean() + tcrit * se), 4)]
        out["per_fold"] = pf
        out["paired_probe_minus_forced"] = {"folds_scored": len(v), "df": df, "t_crit": tcrit,
                                            "mean": round(float(v.mean()), 4), "se": round(se, 4),
                                            "CI95": ci}
        out["verdict"] = ("the state out-reads the output on the cells the judge abstained on"
                          if ci[0] > 0 else
                          "the output out-reads the state on the abstained cells" if ci[1] < 0 else
                          "no separation on the abstained cells at this n — the aggregate crossing "
                          "is not established per fold")
    out["signed"] = "Sautee (sha-ta)"
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  %s layer %d (%.0f%%)  tied n=%d  probe %.3f vs forced %.3f"
          % (fo["judge"].split("/")[-1], a.layer, 100 * L["relative_depth"], len(tie),
             out["aggregate_on_tied_cells"]["probe"], out["aggregate_on_tied_cells"]["forced_F2"]))
    if out.get("paired_probe_minus_forced"):
        print("  paired per fold: %+.3f  CI %s (df %d)" % (out["paired_probe_minus_forced"]["mean"],
              out["paired_probe_minus_forced"]["CI95"], out["paired_probe_minus_forced"]["df"]))
    print("  %s\n  wrote %s" % (out["verdict"], a.out))


if __name__ == "__main__":
    main()
