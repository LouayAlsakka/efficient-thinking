#!/usr/bin/env python
"""ET-8b §13b — the scorer. READINGS FIXED BY 理 BEFORE ANY ARM RAN (12417, form ruled 12436).

    g1  − g0           the confirmatory replication, scored under §14
    g1' − g1-matched   INDEPENDENCE AT EQUAL ROWS — the §13b question. Prediction, registered:
                       gen1' > gen1-matched, both intervals excluding zero and clearing the
                       largest within-head-arm draw
    within-arm sd for every head arm against base — "the head stabilises the run"

    g1 − g1-matched    DROPPED, not reported (12436): g1 is three per-decision heads on 2,824 rows
                       and g1-matched is one shared head on 1,437, so that difference varies FORM
                       and AMOUNT at once. This file refuses to print it even if asked.

§14 (理) IS TWO CONDITIONS AND BOTH ARE CHECKED SEPARATELY, because a pairing can pass the
first and fail the second and the difference is the whole point:
    (a) the 95% interval excludes zero
    (b) its LOWER BOUND exceeds the largest within-head-arm draw of this session
A reading that reports "significant" on (a) alone is the §11 mistake: the loop's own arm-to-arm
spread is wide, and an effect smaller than that spread is a draw from it.

⚠️ GAME 1 IS ASYMMETRIC IN TIME and this file says so in its own output. Its b/g0/g1 ran before
理's form ruling and its g1p/g1m about three hours later; games 2 and 3 have all five arms adjacent.
A game-1-only effect is confounded with that gap.

McNemar is a WITHIN-RUN DESCRIPTIVE on loop rows only (理's demotion). It is computed and printed
and it decides nothing.
"""
import argparse, itertools, json, math, os, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# TWO PLANS, ONE FILE. 13c has different arms from 13b and the same readings; a copy of this
# script with four names changed is how two files drift into disagreeing about what §14 means.
# The plan names the arms, their labels, the pairings to score, and the pairings that are
# FORBIDDEN because they are confounded -- the refusal travels with the plan rather than living
# in whichever copy remembered it.
ARMS = ("b", "g0", "g1", "g1p", "g1m")
# GAME 1'S ARMS ARE ON DISK UNDER THEIR ORIGINAL NAMES and are NOT renamed here: three of them ran
# before 理's form ruling under a chain that named them b1x/g0x/g1x, and the two late ones under
# g1px1/g1mx1. Renaming files a finished chain wrote is how provenance gets lost; the alias table
# is declared, printed, and carried into the artifact instead.
ALIASES = {("b", "1"): "b1x", ("g0", "1"): "g0x", ("g1", "1"): "g1x",
           ("g1p", "1"): "g1px1", ("g1m", "1"): "g1mx1"}

ARM_LABEL = {"b": "base", "g0": "gen0 head", "g1": "gen1 head (published, 3 per-decision, 2824 rows)",
             "g1p": "gen1' (SQL, one shared head, 1437 rows)",
             "g1m": "gen1-matched (gen1 subsampled to 1437, one shared head)"}
FORBIDDEN = {("g1", "g1m")}   # 理: confounded, dropped, not reported

PLANS = {
    "13b": {"arms": ARMS, "labels": None, "aliases": None,
            "wanted": [("g0", "b"), ("g1", "b"), ("g1p", "b"), ("g1m", "b"),
                       ("g1", "g0"), ("g1p", "g1m")],
            "forbidden": FORBIDDEN,
            "note": ("the g1p - g1m pair is CONFOUNDED BY THE MATCHING UNIT: g1p is an intact "
                     "collection and g1m was cut to a row budget, which shreds decision groups. "
                     "理 ruled it printed as confounded rather than re-run, because the confound "
                     "worked AGAINST g1m and it still won -- so the conclusion is conservative.")},
    "13c": {"arms": ("c_b", "c_g0", "c_g1m", "c_g1pp"),
            "labels": {"c_b": "base", "c_g0": "gen0 head",
                       "c_g1m": "gen1-matched (290 decisions, whole groups)",
                       "c_g1pp": "gen1'' — disjoint SAME-FAMILY problems (290 decisions)"},
            "aliases": {},
            "wanted": [("c_g0", "c_b"), ("c_g1m", "c_b"), ("c_g1pp", "c_b"),
                       ("c_g1pp", "c_g1m")],
            "forbidden": set(),
            "note": ("c_g1pp - c_g1m is the arm: same family, same form, same DECISION count, "
                     "origin varied. Both heads were cut on whole decision groups, so neither is "
                     "shredded. Read beside the probe line (60.7 vs 53.4) and beside the fact that "
                     "the disjoint rows are 1.3% byte-identical to gen0's but 42% "
                     "neighbour-redundant with them.")},
}


def green_rate(path):
    n = g = 0
    for line in open(path):
        e = json.loads(line)
        n += 1
        g += bool(e.get("green"))
    return g, n


def pair(base_path, g_path, name, out_json, py=sys.executable, expect=300):
    cmd = [py, os.path.join(HERE, "paired_stats.py"), "--base", base_path, "--g", g_path,
           "--expect", str(expect), "--name", name, "--out", out_json]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("STOP: paired_stats failed on %s\n%s" % (name, (r.stderr or "")[-800:]))
    return json.load(open(out_json))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of <arm><game>.episodes.jsonl")
    ap.add_argument("--games", default="1,2,3")
    ap.add_argument("--plan", default="13b", choices=sorted(PLANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    plan = PLANS[a.plan]
    arms_used = plan["arms"]
    aliases = plan["aliases"] if plan["aliases"] is not None else ALIASES
    labels = plan["labels"] or ARM_LABEL
    games = [g.strip() for g in a.games.split(",") if g.strip()]
    work = a.work or os.path.join(a.dir, "_pairs")
    os.makedirs(work, exist_ok=True)

    # ---- what is on disk, stated before anything is computed ------------------------------
    have, rates = {}, {}
    for g in games:
        for arm in arms_used:
            stem = aliases.get((arm, g), "%s%s" % (arm, g))
            p = os.path.join(a.dir, "%s.episodes.jsonl" % stem)
            if not os.path.exists(p):
                p = os.path.join(a.dir, "g13b_%s.episodes.jsonl" % stem)
            if os.path.exists(p):
                have[(arm, g)] = p
                rates[(arm, g)] = green_rate(p)
    print("  plan %s — arms found: %d of %d" % (a.plan, len(have), len(arms_used) * len(games)))
    for g in games:
        row = "  game %s: " % g
        for arm in arms_used:
            k = (arm, g)
            row += "%s %s  " % (arm, ("%d/%d" % rates[k]) if k in rates else "—")
        print(row)

    # ---- within-arm sd, ACROSS GAMES, per arm -------------------------------------------
    # This is the denominator §14(b) is measured against, so it is computed from the data and
    # never from a remembered figure.
    within = {}
    for arm in arms_used:
        pts = [100.0 * rates[(arm, g)][0] / rates[(arm, g)][1] for g in games if (arm, g) in rates]
        if len(pts) >= 2:
            within[arm] = {"draws": [round(p, 2) for p in pts],
                           "mean": round(statistics.mean(pts), 2),
                           "sd": round(statistics.stdev(pts), 3),
                           "range": round(max(pts) - min(pts), 2)}
    head_arms = [x for x in arms_used if x in within and not x.endswith("b")]
    largest_head_draw = max((within[x]["range"] for x in head_arms), default=None)
    if largest_head_draw is None:
        print("  ⚠️ §14(b) CANNOT BE EVALUATED YET: no head arm has two games on disk. The bar is "
              "the largest within-head-arm draw of THIS session and it is not measurable from one "
              "game. Every (b) verdict below reads 'not evaluable', never 'passed'.")

    # ---- the pairings ---------------------------------------------------------------------
    wanted = plan["wanted"]
    results = {}
    for hi, lo in wanted:
        if (hi, lo) in plan["forbidden"]:
            continue
        for g in games:
            if (hi, g) not in have or (lo, g) not in have:
                continue
            name = "§13b game %s: %s vs %s" % (g, hi, lo)
            j = pair(have[(lo, g)], have[(hi, g)], name,
                     os.path.join(work, "%s%s_vs_%s%s.json" % (hi, g, lo, g)), py=a.python)
            ci = j["success_95CI"]
            excl0 = ci[0] > 0.0
            clears = None if largest_head_draw is None else ci[0] > largest_head_draw
            results["%s_vs_%s_game%s" % (hi, lo, g)] = {
                "delta_points": j["success_delta_points"], "ci": ci,
                "base_pct": j["base_success_pct"], "g_pct": j["G_success_pct"],
                "§14a_interval_excludes_zero": excl0,
                "§14b_lower_bound_clears_largest_within_head_draw": (
                    "not evaluable — needs two games of a head arm" if clears is None else clears),
                "§14b_bar_points": largest_head_draw,
                "mcnemar_DESCRIPTIVE_ONLY": j.get("McNemar"),
            }
            print("  %-34s %+5.1f  [%+.1f, %+.1f]  §14a %-5s §14b %s"
                  % (name, j["success_delta_points"], ci[0], ci[1], excl0,
                     results["%s_vs_%s_game%s" % (hi, lo, g)]
                     ["§14b_lower_bound_clears_largest_within_head_draw"]))

    out = {
        "document": "ET-8b §13b — scored under the readings 理 fixed before any arm ran",
        "readings": {
            "g1_minus_g0": "the confirmatory replication, §14 (a) AND (b)",
            "g1p_minus_g1m": ("INDEPENDENCE AT EQUAL ROWS — the §13b question. Registered "
                              "prediction: gen1' > gen1-matched, both intervals excluding zero and "
                              "clearing the largest within-head-arm draw."),
            "within_arm_sd": "the head stabilises the run",
            "g1_minus_g1m": "DROPPED (理) — confounds form with amount. Not computed here."},
        "§14": {"a": "the 95% interval excludes zero",
                "b": "its lower bound exceeds the largest within-head-arm draw of this session",
                "bar_points": largest_head_draw,
                "why_both": ("a pairing can pass (a) and fail (b); an effect smaller than the "
                             "loop's own arm-to-arm spread is a draw from that spread")},
        "⚠️_game_1_is_asymmetric_in_time": (
            "game 1's b/g0/g1 ran before 理's form ruling and its g1p/g1m about three hours later; "
            "games 2 and 3 have all five arms adjacent. A game-1-only effect is confounded with "
            "that gap."),
        "mcnemar": "within-run descriptive on loop rows only (理's demotion). Decides nothing.",
        "plan": a.plan,
        "plan_note": plan["note"],
        "arm_labels": labels,
        "file_aliases": {"%s game %s" % k: v for k, v in aliases.items()},
        "green_counts": {"%s%s" % (k[0], k[1]): {"green": v[0], "n": v[1]} for k, v in rates.items()},
        "within_arm": within,
        "pairings": results,
        "signed": "Sautee (sha-ta)"}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("\n  wrote %s" % a.out)


if __name__ == "__main__":
    main()
