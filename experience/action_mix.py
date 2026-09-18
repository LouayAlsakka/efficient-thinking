#!/usr/bin/env python3
"""Where a paired arm's action saving actually comes from. DESCRIPTIVE — not a test, no p-values.

WHY IT EXISTS. `actions -1.64` is a scalar and it hides its own mechanism. Two very different
systems produce it: one that stops flailing (fewer repeats) and one that gets the hypothesis right
sooner (fewer of everything downstream). The paper's claim about G is the second, and a claim about
a mechanism should be shown, not asserted from a total.

WHAT IT IS NOT. No interval, no significance. Every cell is one arm's count on one task set. The
bar is in paired_stats.py; this explains a number that already passed it and cannot make one pass.

READ THE TWO REGION RATES CAREFULLY — THEY HAVE DIFFERENT DENOMINATORS.
  head_pick (from the probe fit)  = the head choosing among candidates AT the post-inspect decision.
  region_hit here                 = over EVERY hypothesize action in the episode, and the head acts
                                    at ONE decision only -- the first post-inspect one. Later
                                    hypotheses are the agent's own. The second is therefore lower by
                                    construction and the two are NOT the same quantity.
"""
from __future__ import annotations
import argparse, collections, json, sys


def load(paths):
    out = []
    for p in paths:
        out += [json.loads(l) for l in open(p)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--g", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, required=True, help="n, for per-episode rates")
    ap.add_argument("--name", default="pair")
    ap.add_argument("--out")
    a = ap.parse_args()

    b, g = load(a.base), load(a.g)
    cb = collections.Counter(x.get("action") for x in b)
    cg = collections.Counter(x.get("action") for x in g)
    tb, tg = sum(cb.values()), sum(cg.values())
    n = a.episodes

    rows = []
    for k in sorted(set(cb) | set(cg), key=lambda k: -(cb[k] + cg[k])):
        rows.append({"action": k, "base": cb[k], "G": cg[k],
                     "base_pct": round(100 * cb[k] / tb, 1), "G_pct": round(100 * cg[k] / tg, 1),
                     "per_episode_delta": round(cg[k] / n - cb[k] / n, 2)})

    def hyp(s):
        h = [x for x in s if x.get("action") == "hypothesize"]
        return len(h), sum(1 for x in h if x.get("region_hit"))
    hb, rb = hyp(b)
    hg, rg = hyp(g)

    res = {"pair": a.name, "episodes": n, "descriptive_only": True,
           "action_mix": rows,
           "total_actions": {"base": tb, "G": tg, "per_episode_delta": round(tg / n - tb / n, 2)},
           "hypotheses_that_named_the_bugs_region": {
               "base": "%d/%d = %.1f%%" % (rb, hb, 100 * rb / max(1, hb)),
               "G": "%d/%d = %.1f%%" % (rg, hg, 100 * rg / max(1, hg)),
               "denominator_warning": "over EVERY hypothesize action in the episode. The head acts "
                                      "at ONE decision (the first post-inspect one); later "
                                      "hypotheses are the agent's own. This is NOT the probe's "
                                      "head_pick, which is measured at that single decision."}}
    print("  %s   n=%d  (descriptive, not a test)" % (a.name, n))
    print("  %-20s %14s %14s %12s" % ("action", "base", "G", "per-episode"))
    for r in rows:
        print("  %-20s %5d (%4.1f%%) %5d (%4.1f%%) %+11.2f"
              % (r["action"], r["base"], r["base_pct"], r["G"], r["G_pct"], r["per_episode_delta"]))
    print("  %-20s %5d          %5d          %+11.2f" % ("TOTAL", tb, tg, res["total_actions"]["per_episode_delta"]))
    print("\n  hypotheses naming the bug's region:  base %s   G %s"
          % (res["hypotheses_that_named_the_bugs_region"]["base"],
             res["hypotheses_that_named_the_bugs_region"]["G"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
