#!/usr/bin/env python
"""ET-VII E-E stage 1 — build the DECISIVE PAIRS. No model, no GPU, no spend.

Pre-registered in docs/et7-ee-prereg.md §1. A cell is a pair of policy answers to the same MATH
problem where EXACTLY ONE grades correct against `gold`, by the III harness's own grader
(reason_math_sweep.extract_boxed + normalize) rather than a new one.

Ties and both-right/both-wrong pairs are excluded AND COUNTED. A judge cannot be wrong on a pair
with no right answer; including them would dilute A and A* by the same unknown amount and make Δ
look smaller than it is for a reason that has nothing to do with the judge.

Run before any GPU is committed, because the arm's n is a fact about the data and not about the
plan: if there are too few decisive pairs the arm needs rethinking, not running.
"""
import argparse, collections, itertools, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reason_math_sweep import extract_boxed, normalize


def graded(ans, gold):
    """True/False, or None when the answer has no extractable boxed value at all."""
    b = extract_boxed(ans or "")
    if b is None:
        return None
    return normalize(b) == normalize(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=os.path.join(HERE, "arena_answers_v2.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "et7_ee_cells.json"))
    # E-D's intransitivity half needs the pairs this builder excludes: a 3-cycle spans three
    # policies, and the decisive filter removes the closing edge -- MEASURED, zero closed
    # triangles exist across all 205 problems of the decisive corpus. REQUIRED with no
    # default, because which population is built is a registered choice.
    ap.add_argument("--keep", choices=("decisive", "all"), required=True,
                    help="REQUIRED, no default: 'decisive' is E-E stage 1 as registered -- "
                         "exactly one side right. 'all' keeps the both-right and both-wrong "
                         "pairs too, for E-D's round-robin. Writing 'all' to the decisive "
                         "corpus's own path is refused.")
    a = ap.parse_args()
    if a.keep == "all" and os.path.abspath(a.out) == os.path.join(HERE, "et7_ee_cells.json"):
        raise SystemExit("refusing to write the round-robin over et7_ee_cells.json: that file "
                         "is the provenance of every E-E reading. Name a new --out.")
    d = json.load(open(a.answers))
    qs, ans = d["questions"], d["answers"]
    models = sorted(ans)
    print("  %d problems · %d policies: %s" % (len(qs), len(models), ", ".join(m.split("-")[1] for m in models)))

    per_model_correct = collections.Counter()
    unreadable = collections.Counter()
    cells, tally = [], collections.Counter()
    for i, q in enumerate(qs):
        gold = q["gold"]
        g = {}
        for m in models:
            rows = ans[m]
            if i >= len(rows):
                continue
            r = rows[i]
            txt = r if isinstance(r, str) else (r.get("text") or r.get("answer") or "")
            v = graded(txt, gold)
            g[m] = (v, txt)
            if v is None: unreadable[m] += 1
            elif v: per_model_correct[m] += 1
        for x, y in itertools.combinations([m for m in models if m in g], 2):
            vx, tx = g[x]; vy, ty = g[y]
            if vx is None or vy is None:
                tally["excluded_unreadable"] += 1; continue
            if vx == vy:
                tally["excluded_both_right" if vx else "excluded_both_wrong"] += 1
                if a.keep != "all":
                    continue
                # kept only for --keep all, and MARKED so no downstream reading can mistake a
                # both-right or both-wrong pair for a decisive one
                cells.append({"problem_index": i, "problem": q["problem"], "gold": gold,
                              "level": q.get("level"), "model_A": x, "model_B": y,
                              "answer_A": tx, "answer_B": ty,
                              "decisive": False,
                              "both": "right" if vx else "wrong"})
                continue
            tally["DECISIVE"] += 1
            cells.append({"problem_index": i, "problem": q["problem"], "gold": gold,
                          "level": q.get("level"), "model_A": x, "model_B": y,
                          "answer_A": tx, "answer_B": ty,
                          "correct_side": "A" if vx else "B"})
    print("\n  per-policy accuracy on these problems (the grader's own view):")
    for m in models:
        n = len(qs) - unreadable[m]
        print("    %-34s %3d/%3d = %5.1f%%   unreadable %d" % (m, per_model_correct[m], n,
              100.0*per_model_correct[m]/max(1,n), unreadable[m]))
    print("\n  pair census:")
    for k in ("DECISIVE", "excluded_both_right", "excluded_both_wrong", "excluded_unreadable"):
        print("    %-24s %5d" % (k, tally[k]))
    tot = sum(tally.values())
    print("    %-24s %5d  (%.1f%% decisive)" % ("total pairs", tot, 100.0*tally["DECISIVE"]/max(1,tot)))
    probs = len({c["problem_index"] for c in cells})
    print("\n  DECISIVE CELLS: %d, over %d distinct problems" % (len(cells), probs))
    print("  (held out BY PROBLEM, per the prereg — so the usable split is %d problems, not %d pairs)"
          % (probs, len(cells)))
    # ⚖️ --keep decisive must reproduce the existing file BYTE-IDENTICALLY, so NOTHING
    # new may appear in that mode: the extra keys exist only for --keep all.
    rec = {"document": "ET-VII E-E stage 1 — decisive pairs",
           "prereg": "docs/et7-ee-prereg.md §1",
           "source": os.path.basename(a.answers),
           "census": dict(tally), "per_policy_correct": dict(per_model_correct),
           "unreadable_per_policy": dict(unreadable),
           "decisive_cells": len(cells), "distinct_problems": probs,
           "cells": cells}
    if a.keep == "all":
        # 🔴 THE PRESENTATION ORDER MUST BE A PROPERTY OF THE CELL, NOT OF ITS POSITION.
        # et7_ee_probe.py draws one flip per cell from a single sequential Random(0) in
        # FILE ORDER, so interleaving the excluded pairs with the decisive ones would give
        # every later decisive cell a different presentation than E-E used -- its states
        # would not be bitwise identical and its forced pick would be read under a different
        # presentation. Recording the flip faithfully would not fix that; it would only
        # record the wrong order. So the builder assigns it HERE, by cell:
        #   decisive cells      replay Random(0) over them IN CANONICAL ORDER -> exactly the
        #                       flips E-E used (verified: 1050 of 1050 against stored meta)
        #   non-decisive cells  a SEPARATE registered stream, Random(101), over them in their
        #                       own order -- so adding or removing them cannot disturb the
        #                       decisive flips
        # Only --keep all carries the field, so --keep decisive stays byte-identical.
        r_dec, r_non = random.Random(0), random.Random(101)
        for c in cells:
            c["flipped"] = bool((r_non if c.get("decisive") is False else r_dec).random() < 0.5)
        n_dec = tally["DECISIVE"]
        rec["document"] = ("ET-VII E-D — the ROUND-ROBIN corpus: every pair, decisive and not. "
                           "A decisive-only reading must filter on cells[].decisive")
        rec["keep"] = "all"
        rec["decisive_cells"] = n_dec
        rec["non_decisive_cells"] = len(cells) - n_dec
        rec["why"] = ("E-D's intransitivity clause is unmeasurable on the decisive corpus: "
                      "zero closed triangles exist in it, because the filter removes the "
                      "edge a triangle needs. These pairs restore it.")
    json.dump(rec, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
