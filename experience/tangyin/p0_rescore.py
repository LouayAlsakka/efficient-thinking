#!/usr/bin/env python3
"""Re-score a P0 run from its saved generations — no GPU, no resampling.

    python3 p0_rescore.py --in results/p0_base_7b.jsonl --out results/p0_base_7b_rescored.jsonl \
        --report results/p0_base_7b_rescored.json

The generations are fixed; only the INSTRUMENT changes. Re-running p0_baseline.py would sample
new poems and confound an instrument change with a sampling change, so the raw text is re-read
and re-scored instead. Prints the old and new rate side by side, per form.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify_form import Rime, verify                                   # noqa: E402
from p0_baseline import extract, FORMS                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--rime", default=os.path.join(HERE, "rime", "pingshui.json"))
    ap.add_argument("--no-simplified", action="store_true")
    a = ap.parse_args()

    rime = Rime(a.rime, simplified=not a.no_simplified)
    print("  normalisation:", rime.normaliser or "NONE (opencc absent)")
    rows = [json.loads(l) for l in open(a.inp)]
    out = []
    with open(a.out, "w") as fh:
        for r in rows:
            poem, bucket = extract(r["raw"], *[(p, n) for f, p, n in FORMS if f == r["form"]][0])
            v = verify(poem, rime) if bucket == "EXTRACTED" else None
            n = dict(r)
            n.update({"poem": poem, "bucket": bucket, "verify": v,
                      "pass": bool(v and v["pass"]), "pass_before": r["pass"]})
            out.append(n)
            fh.write(json.dumps(n, ensure_ascii=False) + "\n")

    rep = {"source": a.inp, "normalisation": rime.normaliser, "by_form": {}}
    print(f"\n  {'form':6s} {'n':>3s} {'was':>7s} {'now':>7s} {'unknown positions':>18s} "
          f"{'reached by mapping':>19s}")
    for form, per, n_ in FORMS:
        d = [x for x in out if x["form"] == form]
        ex = [x for x in d if x["bucket"] == "EXTRACTED"]
        was = sum(1 for x in d if x["pass_before"]) / len(d)
        now = sum(1 for x in d if x["pass"]) / len(d)
        ep = sum(x["verify"]["enforced_positions"] for x in ex)
        eu = sum(x["verify"]["enforced_undecided"] for x in ex)
        es = sum(x["verify"].get("enforced_simplified", 0) for x in ex)
        rep["by_form"][form] = {
            "n": len(d), "extracted": len(ex),
            "pass_over_all_attempts_before": was, "pass_over_all_attempts": now,
            "pass_over_extracted": (sum(1 for x in ex if x["pass"]) / len(ex)) if ex else None,
            "enforced_positions": ep, "enforced_undecided": eu,
            "enforced_reached_by_mapping": es,
            "flipped_to_fail": [x["topic"] for x in d if x["pass_before"] and not x["pass"]],
            "flipped_to_pass": [x["topic"] for x in d if x["pass"] and not x["pass_before"]],
        }
        print(f"  {form:6s} {len(d):3d} {was*100:6.1f}% {now*100:6.1f}% "
              f"{eu:10d}/{ep:<7d} {es:19d}")
    json.dump(rep, open(a.report, "w"), ensure_ascii=False, indent=1)
    print("\n ->", a.out, a.report)


if __name__ == "__main__":
    main()
