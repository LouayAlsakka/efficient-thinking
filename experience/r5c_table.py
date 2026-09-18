#!/usr/bin/env python3
"""R5-C: the quality-cost FRONTIER as a curve — one table, both arms (理 11395/11397).

WHY A CURVE AND NOT R5-E's POINT. R5-E showed the base at ~9.2k tokens does not reach G at ~9.2k.
That is ONE point, and its own bound said so. A curve says over WHAT RANGE the frontier is moved,
and where — if anywhere — compute substitutes for the prior.

THE READINGS WERE WRITTEN BEFORE THE RUNS (理 11395):
  head curve ABOVE base curve across the range   -> the frontier is moved over a RANGE, not a point
  curves CROSS at low budget                     -> the head helps only where it has room to act
                                                    (§7.5's bound, restated in cost terms)
  curves MERGE at high budget                    -> compute substitutes for experience there, and
                                                    the paper prints the range where it does

⚠️ EVERY POINT IS A DIFFERENT AGENT. The budget is in the prompt, so budget-24 is an agent TOLD it
has 24 actions. No point is a truncation or extension of another and the table must not be read as
a single agent given more rope.

⚠️ THE HEAD'S TOKENS ARE ALL-IN. Model tokens PLUS the controller's measured candidate passes
(R2b: 2,487.9/episode for the measured system). Comparing model-tokens-only against the base would
flatter the head by ~2.5k tokens per episode, which is exactly the accounting error R2 made in the
other direction.
"""
import glob, json, sys
import numpy as np

SP = sys.argv[1]
CTRL = 2487.9          # R2b's MEASURED controller cost for the measured system
BASE_SETS = {8: "r5c/b8_s*", 12: "v3b/v3b_s*", 14: "r5c/b14_s*", 16: "r5e300/r5e_s*",
             18: "r5c/b18_s*", 24: "r5c/b24_s*"}
HEAD_SETS = {6: "r5q/g6_s*", 8: "r5q/g8_s*", 10: "r5q/g10_s*", 12: "v3g300/v3g300_s*"}


def load(pat):
    e = {}
    for f in sorted(glob.glob("%s/%s.episodes.jsonl" % (SP, pat))):
        for l in open(f):
            r = json.loads(l)
            e[r["task_id"]] = r
    return e


EXPECT = 300


def summarise(e, controller):
    """None unless the arm is COMPLETE at EXPECT problems.

    A dry run of this table reported head@10 at 33.3% while R5-Q was still writing that arm -- a
    rate over 119 episodes printed as if it were a curve point. A partial arm is not a small
    measurement, it is an unfinished one, and a curve is exactly where such a point would be
    invisible: it sits on the plot looking like the others. Refuse it here, not at reporting time.
    """
    n = len(e)
    if n != EXPECT:
        return None
    tok = sum(x.get("tokens_in", 0) + x.get("tokens_out", 0) for x in e.values()) / n
    return {"n": n, "success_pct": round(100 * sum(x["green"] for x in e.values()) / n, 1),
            "model_tokens": round(tok, 1), "all_in_tokens": round(tok + controller, 1),
            "actions": round(sum(x["actions"] for x in e.values()) / n, 2)}


def paired(a, b, seed=11, R=10000):
    """b - a, paired per problem, with a bootstrap CI. Returns None on a size mismatch."""
    ks = sorted(set(a) & set(b))
    if len(ks) != len(a) or len(ks) != len(b):
        return None
    x = np.array([float(a[k]["green"]) for k in ks])
    y = np.array([float(b[k]["green"]) for k in ks])
    d = y - x
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(ks), size=(R, len(ks)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(ks), "delta_points": round(100 * d.mean(), 1),
            "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}


base = {b: summarise(load(p), 0.0) for b, p in BASE_SETS.items()}
head = {b: summarise(load(p), CTRL) for b, p in HEAD_SETS.items()}
incomplete = {"base": [b for b, v in base.items() if v is None],
              "head": [b for b, v in head.items() if v is None]}
base = {b: v for b, v in base.items() if v}
head = {b: v for b, v in head.items() if v}

# each head point vs the base point NEAREST it in all-in compute
rows = []
for hb, hv in sorted(head.items()):
    if not base:
        break
    nb = min(base, key=lambda b: abs(base[b]["all_in_tokens"] - hv["all_in_tokens"]))
    st = paired(load(BASE_SETS[nb]), load(HEAD_SETS[hb]))
    rows.append({"head_budget": hb, "head_tokens_all_in": hv["all_in_tokens"],
                 "head_success": hv["success_pct"], "nearest_base_budget": nb,
                 "base_tokens": base[nb]["all_in_tokens"], "base_success": base[nb]["success_pct"],
                 "compute_gap_pct": round(100 * (base[nb]["all_in_tokens"] - hv["all_in_tokens"])
                                          / hv["all_in_tokens"], 2),
                 "paired_head_minus_base": st})

out = {"document": "R5-C — the quality-cost frontier as a CURVE (理 11395/11397)",
       "controller_cost_added_to_every_head_point": CTRL,
       "base_curve": base, "head_curve": head,
       "EXCLUDED_as_incomplete": {k: v for k, v in incomplete.items() if v},
       "why_excluded": "a budget with fewer than %d finished problems is NOT plotted and NOT "
                       "quoted. A partial arm on a curve is invisible -- it looks like every other "
                       "point." % EXPECT,
       "head_points_vs_nearest_compute_base_point": rows,
       "readings_written_before_the_runs": {
           "above across the range": "the frontier is moved over a RANGE, not a point",
           "cross at low budget": "the head helps only where it has room to act (§7.5's bound in cost terms)",
           "merge at high budget": "compute substitutes for experience there; print the range"},
       "bounds": ["Each budget is a DIFFERENT AGENT (the budget is in the prompt); no point is a "
                  "truncation of another.",
                  "Head tokens are ALL-IN (model + R2b's measured controller). Model-tokens-only "
                  "would flatter the head by ~2.5k/episode.",
                  "One seed (v3 seed 21), one model, 300 problems per point, paired per problem.",
                  "A paired CI is computed only where the two arms cover the same 300 problems; "
                  "None means they did not and no interval is quoted."]}
json.dump(out, open(SP + "/r5c_table.json", "w"), indent=1, ensure_ascii=False)
if any(incomplete.values()):
    print("  ⚠️ EXCLUDED as incomplete (<%d problems): %s"
          % (EXPECT, {k: v for k, v in incomplete.items() if v}))
print("  BASE CURVE                              HEAD CURVE (all-in)")
print("  budget  tokens   success  actions       budget  tokens   success  actions")
bs, hs = sorted(base.items()), sorted(head.items())
for i in range(max(len(bs), len(hs))):
    l = ("  %-6d %8.1f %7.1f%% %7.2f" % (bs[i][0], bs[i][1]["all_in_tokens"],
         bs[i][1]["success_pct"], bs[i][1]["actions"])) if i < len(bs) else " " * 32
    r = ("       %-6d %8.1f %7.1f%% %7.2f" % (hs[i][0], hs[i][1]["all_in_tokens"],
         hs[i][1]["success_pct"], hs[i][1]["actions"])) if i < len(hs) else ""
    print(l + r)
print("\n  HEAD vs NEAREST-COMPUTE BASE")
for r in rows:
    st = r["paired_head_minus_base"]
    s = ("%+.1f pts CI [%+.1f, %+.1f]" % (st["delta_points"], st["ci95"][0], st["ci95"][1])) if st else "n/a"
    print("    head@%-3d %8.1f tok %5.1f%%   vs base@%-3d %8.1f tok %5.1f%%  (%+.2f%% compute)   %s"
          % (r["head_budget"], r["head_tokens_all_in"], r["head_success"], r["nearest_base_budget"],
             r["base_tokens"], r["base_success"], r["compute_gap_pct"], s))
