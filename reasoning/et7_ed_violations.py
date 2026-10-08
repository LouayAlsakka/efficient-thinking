#!/usr/bin/env python
"""ET-VII E-D — the judge's OWN order-invariance violations, extracted. No external labels.

    python reasoning/et7_ed_violations.py --forced experience/results/et7_ee_forced_14B.json \
        --cells reasoning/et7_ee_cells.json --out experience/results/et7_ed_violations_14B.json

E-D is registered in the concept (efficient-thinking-7-concept.md:79-83): train a judge against
its own intransitivity and order-invariance violations, no external labels; measure q
before/after against ground truth. The cross-registration (:43-45) is that realised gains must
not exceed the probed delta from E-E, and finding out which side broke "is itself the result".

WHAT THIS DOES, AND THE TWO THINGS IT REFUSES TO DO
  * extracts the ORDER-INVARIANCE violations only: cells where the forced pick changes when the
    candidates are presented in the other order. That is the judge disagreeing with itself, so
    it is label-free by construction.
  * it does NOT touch intransitivity. MEASURED: the corpus contains ZERO closed triangles
    across all 205 problems (every problem is 5 policies with 4 or 6 pairs, and in none do three
    pairs close a triangle), so no cycle is detectable from it however much is judged. The
    decisive-pair filter that defines this corpus removes the edge a triangle needs. E-D's
    intransitivity half needs a round-robin corpus that includes the both-right and both-wrong
    pairs this one excludes -- a new acquisition, registered separately.
  * it does NOT construct training pairs. The target for a violated cell is a design choice
    with no label available: the saved per-cell block carries no margin (the producer computes
    one and does not write it), so either the reader re-emits margins or the objective is
    agreement between the two orders with no target at all. That rule belongs in a signed
    registration before any fit, not in this reader.

GROSS, NOT NET. The aggregate `swap_gap_points` is a DIFFERENCE of two accuracies: 5.01 points
for this judge. The gross flip rate is 37%. Training consumes the gross violations; the net
number understates the available signal by about sevenfold, and reading one for the other is
the mistake this file exists to make impossible.
"""
import argparse
import collections
import json
import os
import sys


def violations(per_cell):
    """-> (flipped cell ids, stable cell ids, skipped). A flip is the judge's own disagreement.

    A cell is DECISIVE (exactly one side right), so the forced pick is recoverable from whether
    it was correct: correctness differing between the two orders means the pick changed.
    A cell missing either order is SKIPPED and counted, never silently treated as stable.
    """
    flipped, stable, skipped = [], [], []
    for cid, r in sorted(per_cell.items(), key=lambda kv: int(kv[0])):
        p, s = r.get("F2_ok_primary"), r.get("F2_ok_swapped")
        if p is None or s is None:
            skipped.append(int(cid))
        elif bool(p) != bool(s):
            flipped.append(int(cid))
        else:
            stable.append(int(cid))
    return flipped, stable, skipped


def by_pair(ids, cells):
    """Violations grouped by policy pair, so a rate is never read across pairs it mixes."""
    out = collections.Counter()
    for i in ids:
        c = cells[i]
        out[" vs ".join(sorted((c["model_A"], c["model_B"])))] += 1
    return dict(out)


def self_test():
    """Controls for the two pure functions. A reader that cannot fail is not a reader."""
    ran, bad = [], []

    def ck(name, got, want):
        ran.append(name)
        ok = got == want
        print("  %-4s %-62s %s  want %s" % ("ok" if ok else "FAIL", name, got, want))
        if not ok:
            bad.append(name)

    pc = {"0": {"F2_ok_primary": True, "F2_ok_swapped": False},     # flipped
          "1": {"F2_ok_primary": True, "F2_ok_swapped": True},      # stable
          "2": {"F2_ok_primary": False, "F2_ok_swapped": False},    # stable, both wrong
          "3": {"F2_ok_primary": False, "F2_ok_swapped": True},     # flipped the other way
          "4": {"F2_ok_primary": True}}                             # one order MISSING
    f, st, sk = violations(pc)
    ck("a pick that changes with order is a violation", f, [0, 3])
    ck("agreeing in both orders is stable, right or wrong", st, [1, 2])
    ck("a cell missing an order is SKIPPED, never counted stable", sk, [4])
    ck("and skipped cells are not in either other list",
       set(sk) & (set(f) | set(st)), set())
    ck("the denominator excludes the skipped cell", len(f) + len(st), 4)

    cells = [{"model_A": "P", "model_B": "Q"}, {"model_A": "Q", "model_B": "P"},
             {"model_A": "P", "model_B": "R"}, {"model_A": "R", "model_B": "P"}]
    ck("by_pair groups the two orderings of one pair together",
       by_pair([0, 1], cells), {"P vs Q": 2})
    ck("...and keeps different pairs apart",
       by_pair([0, 2], cells), {"P vs Q": 1, "P vs R": 1})
    ck("an empty violation set groups to nothing", by_pair([], cells), {})

    print("\n%s" % ("ok   all %d checks pass" % len(ran) if not bad
                    else "FAIL %r" % bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--forced", help="a per-judge forced file WITH per_cell")
    ap.add_argument("--cells")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.forced or not a.cells:
        sys.exit("--forced and --cells are required (or --self-test)")

    fo = json.load(open(a.forced, encoding="utf-8"))
    pc = fo.get("per_cell")
    if not pc:
        sys.exit("%s has no per_cell block -- it predates the patched reader, and a violation "
                 "set cannot be built from aggregates. Re-emit it first." % a.forced)
    cells = json.load(open(a.cells, encoding="utf-8"))["cells"]

    flipped, stable, skipped = violations(pc)
    n = len(flipped) + len(stable)
    print("=== E-D order-invariance violations: %s ===" % os.path.basename(a.forced))
    print("  judge            %s" % fo.get("judge", "(not recorded in this file)"))
    print("  cells with both orders read   %d" % n)
    print("  FLIPPED (the judge's own disagreement)  %d  (%.1f%%)"
          % (len(flipped), 100.0 * len(flipped) / n if n else 0.0))
    print("  stable                                  %d" % len(stable))
    if skipped:
        print("  ⛔ SKIPPED, one order missing            %d  %r" % (len(skipped), skipped[:8]))
    print("\n  by policy pair (a rate across pairs would mix populations):")
    for k, v in sorted(by_pair(flipped, cells).items(), key=lambda kv: -kv[1]):
        print("    %-42s %d" % (k, v))

    rec = {"document": "ET-VII E-D -- order-invariance violations, label-free",
           "registration": "efficient-thinking-7-concept.md:79-83 (arm) and :43-45 "
                           "(cross-registration against E-E's probed delta)",
           "source": os.path.basename(a.forced),
           "judge": fo.get("judge"),
           "n_cells_both_orders": n,
           "flipped": flipped, "stable_n": len(stable), "skipped": skipped,
           "flip_rate": round(len(flipped) / float(n), 4) if n else None,
           "flipped_by_pair": by_pair(flipped, cells),
           "gross_not_net": ("this flip rate is GROSS. The aggregate swap_gap_points in the "
                             "source file is a difference of two accuracies and is far "
                             "smaller; training consumes the gross violations."),
           "intransitivity": ("NOT INCLUDED and not measurable here: zero closed triangles "
                              "exist across all 205 problems of this corpus, so no 3-cycle is "
                              "detectable however much is judged. A round-robin corpus "
                              "including the both-right and both-wrong pairs this one excludes "
                              "is a separate registered acquisition."),
           "no_training_pairs": ("deliberately absent: the per-cell block carries no margin, "
                                 "so the target for a violated cell is a design choice that "
                                 "belongs in a signed registration, not in this reader.")}
    if a.out:
        json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
        print("\n  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
