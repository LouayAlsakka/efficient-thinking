#!/usr/bin/env python
"""ET-IV 7d — selection value, stage 1: the three selectors' PICKS (local, no API spend).

理's run order is E2 -> 7d -> E5 -> E3 -> re-rate. This is 7d, split the same way E2 is: picks are
local and GPU-bound, rating is the API arm. Splitting them means the rating can start the instant a
Bedrock principal lands, and the picks can be inspected before a cent is spent.

THE THREE SELECTORS come from poetry/canon/selection_7d.py and are NOT reimplemented here:
  random         the no-selection floor
  checker        first form-valid candidate — form without taste (checker 1a)
  canon-judge    the 7b LoRA's pick via single-elim pairwise tournament

WHAT CHANGES UNDER AMENDMENT 2: selection_7d.py scored the picks with a 7B persona oracle on a 1-10
scale. Absolute small-scale ratings are the weakest instrument in the drawer, and the amendment
replaces that judge with the frontier model — so stage 2 rates the picks PAIRWISE (3 pairs per
field: random-vs-checker, random-vs-canon, checker-vs-canon), which is also what the cost line
budgets ("100 fields -> 3 pairs each").

⚠️ THIS IS A NEW MEASUREMENT, NOT A REPRODUCTION. 理 11768 §3: selection_7d.json records its scores
and not its instrument — no adapter hash — so identity with the run that produced canon 7.62 cannot
be established. The adapter used here is hashed into the output.

⛔ AND THE PROTOTYPE BOUND STANDS, from selection_7d.py's own header: the 7b canon judge is a
noise-level prototype (n=34 corpus). This measures the PIPELINE, not G2p.
"""
import argparse, hashlib, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CANON = os.path.join(HERE, "canon")
sys.path.insert(0, CANON)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "checkers"))


def lines_of(sample, n=4):
    return [x.strip() for x in (sample or "").splitlines()
            if x.strip() and not x.strip().startswith("(")][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache", "e1_7B.jsonl"))
    ap.add_argument("--task", default="sonnet")
    ap.add_argument("--fields", type=int, default=100, help="the cost line budgets 100")
    ap.add_argument("--n", type=int, default=4, help="candidates per field")
    ap.add_argument("--adapter", default=os.path.join(CANON, "adapter"))
    ap.add_argument("--canon-model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from prep_7b import fmt_prompt
    from meter_rhyme import check_iambic_pentameter
    from mlx_lm import load, generate

    ad = os.path.join(a.adapter, "adapters.safetensors")
    if not os.path.exists(ad):
        sys.exit("STOP: no adapter at %s. The canon judge's weights were missing from this repo "
                 "until c1b1ec8; see poetry/canon/adapter/PROVENANCE.json." % a.adapter)
    adapter_md5 = hashlib.md5(open(ad, "rb").read()).hexdigest()
    print("  canon adapter md5 %s" % adapter_md5)

    fields = []
    for line in open(a.cache):
        r = json.loads(line)
        if r.get("task") != a.task:
            continue
        cands = [lines_of(s, r.get("n_lines", 4)) for s in r["samples"][:a.n]]
        cands = [c for c in cands if len(c) == r.get("n_lines", 4)]
        if len(cands) >= 2:
            fields.append({"id": r["id"], "brief": r.get("prompt", ""), "cands": cands})
        if len(fields) >= a.fields:
            break
    print("  fields with >= 2 usable candidates: %d (task=%s)" % (len(fields), a.task))
    if len(fields) < a.fields:
        print("  ⚠️  fewer than the %d the cost line budgets — n is reported, not padded" % a.fields)

    canon, tokc = load(a.canon_model, adapter_path=a.adapter)

    def canon_pick(cands):
        idx = list(range(len(cands)))
        while len(idx) > 1:
            nxt = []
            for i in range(0, len(idx), 2):
                if i + 1 >= len(idx):
                    nxt.append(idx[i]); continue
                x, y = idx[i], idx[i + 1]
                pr = tokc.apply_chat_template([{"role": "user",
                                                "content": fmt_prompt(cands[x], cands[y])}],
                                              add_generation_prompt=True)
                out = generate(canon, tokc, prompt=pr, max_tokens=4, verbose=False).upper()
                mm = re.search(r"[AB]", out)
                nxt.append(x if (mm.group(0) if mm else "A") == "A" else y)
            idx = nxt
        return idx[0]

    rng = random.Random(a.seed)
    rows, agree = [], {"random_eq_checker": 0, "random_eq_canon": 0, "checker_eq_canon": 0}
    for i, f in enumerate(fields):
        c = f["cands"]
        pr = rng.randrange(len(c))
        pc = next((j for j, x in enumerate(c)
                   if all(check_iambic_pentameter(l)["ok"] for l in x)), 0)
        pk = canon_pick(c)
        agree["random_eq_checker"] += pr == pc
        agree["random_eq_canon"] += pr == pk
        agree["checker_eq_canon"] += pc == pk
        rows.append({"field_id": f["id"], "brief": f["brief"],
                     "candidates": c, "picks": {"random": pr, "checker": pc, "canon": pk}})
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(fields)), file=sys.stderr)

    # A SELECTOR THAT AGREES WITH ANOTHER ON EVERY FIELD CONTRIBUTES NO PAIRS TO RATE.
    n = len(rows)
    print("\n  selector agreement (how often two selectors pick the SAME candidate):")
    for k, v in agree.items():
        print("    %-20s %3d/%d = %.1f%%" % (k, v, n, 100.0 * v / max(1, n)))
    print("  pairs worth rating are the DISAGREEMENTS: %d of a possible %d"
          % (3 * n - sum(agree.values()), 3 * n))

    json.dump({"document": "ET-IV 7d stage 1 — the three selectors' picks, local, $0.00",
               "⚠️_new_measurement_not_a_reproduction":
                   "selection_7d.json records its scores and not its instrument (理 11768 §3), so "
                   "identity with the run that produced canon 7.62 cannot be established.",
               "prototype_bound": "the 7b canon judge is a noise-level prototype (n=34 corpus); "
                                  "this measures the PIPELINE, not G2p (selection_7d.py's own header)",
               "instrument": {"canon_model": a.canon_model, "canon_adapter_md5": adapter_md5,
                              "candidate_cache": os.path.abspath(a.cache), "task": a.task,
                              "n_per_field": a.n, "seed": a.seed},
               "fields": n, "selector_agreement": agree,
               "pairs_to_rate": 3 * n - sum(agree.values()),
               "rows": rows}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("\n  wrote %s" % a.out)


if __name__ == "__main__":
    main()
