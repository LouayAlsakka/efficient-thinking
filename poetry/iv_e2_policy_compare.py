#!/usr/bin/env python
"""ET-IV E2 — is the Pareto collapse a property of the POLICY or of the CHECKER?

On the 7B, 53.3% of briefs collapse to a single Pareto candidate, so E2's realizable n is 140 of
300. Two readings, and they point at different fixes:

  POLICY   a weak policy produces candidates the checker can totally order, so one dominates.
           A stronger policy would produce more comparable candidates and collapse less, and E2's
           n is a fact about which generator it rates.
  CHECKER  the dimension set itself forces domination -- `score` is n_meter + rhyme_ok + iamb/10,
           a COMPOSITE of `valid` and `meter`, so including it beside its own components makes the
           checker's own ranking a domination axis. Then collapse is high at EVERY policy and the
           fix is the dimension set, not the generator.

This measures both policies under every dimension set, so the answer is read off a table instead of
argued. It costs no API spend: candidates are already generated and the checker is local.
"""
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
POETRY = os.path.expanduser("~/github/efficient-thinking/poetry")
sys.path.insert(0, POETRY)
import iv_e2 as E2                      # the arm itself -- never a reimplementation of it
import human_session as HS

SETS = [("all four (valid,meter,score,clean)", ["valid", "meter", "score", "clean"]),
        ("no composite (valid,meter,clean)",   ["valid", "meter", "clean"]),
        ("craft only (valid,meter)",           ["valid", "meter"]),
        ("with composite, no clean",           ["valid", "meter", "score"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True, metavar="TAG=PATH")
    ap.add_argument("--prompts", default=os.path.join(POETRY, "data", "e1_prompts.jsonl"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = {}
    for spec in a.caches:
        tag, path = spec.split("=", 1)
        if not os.path.exists(path):
            sys.exit("STOP: no cache at %s (tag %s)" % (path, tag))
        items = E2.load_candidates(path, a.prompts)
        row = {"briefs": len(items),
               "candidates_per_brief": round(sum(len(i["candidates"]) for i in items) / max(1, len(items)), 1)}
        # how often is each dimension even able to separate?
        varies = collections.Counter()
        for it in items:
            for d in ("valid", "meter", "score", "clean"):
                if len({round(c["scores"][d], 4) for c in it["candidates"]}) > 1:
                    varies[d] += 1
        row["dimension_varies_in_n_briefs"] = dict(varies)
        row["briefs_with_no_form_valid_candidate"] = sum(
            1 for it in items if all(c["scores"]["valid"] == 0 for c in it["candidates"]))
        row["collapse_by_dimension_set"] = {}
        for name, dims in SETS:
            c = sum(1 for it in items if len(HS.pareto_prune(it["candidates"], dims)) < 2)
            row["collapse_by_dimension_set"][name] = {
                "collapse_pct": round(100.0 * c / max(1, len(items)), 1),
                "realizable_n": len(items) - c}
        res[tag] = row
        print("\n  %s — %d briefs, %.1f candidates each, %d with NO form-valid candidate"
              % (tag, row["briefs"], row["candidates_per_brief"],
                 row["briefs_with_no_form_valid_candidate"]))
        for name, _ in SETS:
            v = row["collapse_by_dimension_set"][name]
            print("      %-38s collapse %5.1f%%   n = %d" % (name, v["collapse_pct"], v["realizable_n"]))

    # the reading, computed rather than asserted
    verdict = None
    if len(res) >= 2:
        tags = list(res)
        cur = [res[t]["collapse_by_dimension_set"]["all four (valid,meter,score,clean)"]["collapse_pct"] for t in tags]
        nc = [res[t]["collapse_by_dimension_set"]["no composite (valid,meter,clean)"]["collapse_pct"] for t in tags]
        spread_policy = max(cur) - min(cur)
        drop = [c - n for c, n in zip(cur, nc)]
        verdict = {
            "collapse_under_the_current_set_across_policies": dict(zip(tags, cur)),
            "spread_across_policies_points": round(spread_policy, 1),
            "drop_from_removing_the_composite_points": dict(zip(tags, [round(d, 1) for d in drop])),
            "reading": ("the composite explains more than the policy does — removing it drops "
                        "collapse by %.1f-%.1f points while changing policy moves it %.1f"
                        % (min(drop), max(drop), spread_policy))
            if min(drop) > spread_policy else
                       ("the policy explains more than the composite does — changing policy moves "
                        "collapse %.1f points while removing the composite drops it %.1f-%.1f"
                        % (spread_policy, min(drop), max(drop)))}
        print("\n  READING: %s" % verdict["reading"])

    json.dump({"document": "ET-IV E2 — Pareto collapse across policies and dimension sets",
               "why": "53.3% collapse on the 7B halves E2's n. Whether that is the policy or the "
                      "checker's dimension set decides whether the fix is a different generator or "
                      "a corrected DIMS.",
               "no_API_spend": True, "policies": res, "verdict": verdict,
               "⚠️_method": "these numbers are for 理 to rule on. Choosing a dimension set by the n "
                            "it yields would be selecting an instrument on its result; the argument "
                            "for dropping `score` is that it is a COMPOSITE of its own "
                            "co-dimensions, and it holds regardless of which way the n moves.",
               "signed": "Sautee (sha-ta)"}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("\n  wrote %s" % a.out)


if __name__ == "__main__":
    main()
