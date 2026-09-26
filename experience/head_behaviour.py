#!/usr/bin/env python3
"""VIII-c, DESCRIPTIVE ONLY: what each head actually DID in the games, from the steps logs.

NOT A READING. No §14 verdict comes out of this file and it registers nothing: the arm's result is
problems solved on a paired run, and this is the "what was the mechanism" column beside it. It
exists because of one alternative explanation nobody has excluded: that the forced-inspection head
wins not by knowing more but by being DEGENERATE -- always naming the same region, or always the
first candidate -- in a way that happens to fit this task family's prior.

Per arm, over every decision where the head was consulted:
  n            decisions with head=true
  agree        head_pick == agent_pick (the head and the agent wanted the same region)
  hit          region_hit: the region the loop KEPT was the true bug region
  hit|agree    hit rate on the decisions where they agreed, and where they did not
  top share    the most frequent head_pick's share -- the degeneracy measure
  distinct     how many distinct regions the head ever named
"""
import collections, glob, json, math, os, sys

W = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/viiic_games89")
# ARMS ARE NAMED ON THE COMMAND LINE, and the default EXCLUDES the arm still collecting. A table
# that quietly carries a partial arm's cell is how a one-game flag became a registered reading that
# then did not fire; an arm joins this file when all three of its games are on disk.
ARMS = (os.environ.get("ARMS") or "v_b,v_g0,v_g1m,v_vb,v_fi,v_un").split(",")

def rows(arm, g):
    for s in (1, 2, 3, 4):
        p = os.path.join(W, "%s%s_s%d.steps.jsonl" % (arm, g, s))
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                yield json.loads(line)
            except ValueError:
                continue

print("  %-7s %-3s %6s %7s %7s %9s %9s %8s %5s" %
      ("arm", "g", "n", "agree%", "hit%", "hit|agree", "hit|diff", "top%", "dist"))
out = {}
for arm in ARMS:
    for g in ("1", "2", "3"):
        n = agree = hit = 0
        ha = hd = na = nd = 0
        picks = collections.Counter()
        seen_any = False
        for r in rows(arm, g):
            c = r.get("candidates") or {}
            if r.get("action") != "hypothesize":
                continue
            seen_any = True
            if not c.get("head"):
                # the base / no-head arm: only the hit rate is defined
                n += 1
                hit += bool(r.get("region_hit"))
                continue
            n += 1
            hit += bool(r.get("region_hit"))
            hp, ap = c.get("head_pick"), c.get("agent_pick")
            if hp is not None:
                picks[hp] += 1
            if hp is not None and ap is not None:
                if hp == ap:
                    agree += 1; na += 1; ha += bool(r.get("region_hit"))
                else:
                    nd += 1; hd += bool(r.get("region_hit"))
        if not seen_any:
            continue
        top = (100.0 * max(picks.values()) / sum(picks.values())) if picks else float("nan")
        row = dict(n=n, agree_pct=(100.0 * agree / n if n else None),
                   hit_pct=(100.0 * hit / n if n else None),
                   hit_given_agree=(100.0 * ha / na if na else None),
                   hit_given_differ=(100.0 * hd / nd if nd else None),
                   top_pick_share_pct=top if picks else None,
                   distinct_picks=len(picks))
        out["%s%s" % (arm, g)] = row
        f = lambda v: "   -  " if v is None or (isinstance(v, float) and math.isnan(v)) else "%6.1f" % v
        print("  %-7s %-3s %6d %7s %7s %9s %9s %8s %5d" %
              (arm, g, n, f(row["agree_pct"]), f(row["hit_pct"]), f(row["hit_given_agree"]),
               f(row["hit_given_differ"]), f(row["top_pick_share_pct"]), row["distinct_picks"]))
if len(sys.argv) > 2:
    json.dump({"document": "VIII-c — what each head did in the games. DESCRIPTIVE, registers nothing.",
               "arms": out, "signed": "Sautee (sha-ta)"}, open(sys.argv[2], "w"), indent=1)
    print("  wrote %s" % sys.argv[2])
