#!/usr/bin/env python3
"""Mechanism E, form-preserving arm: the voice corpus MIXED with a form anchor.

    python3 make_mixed_lora_data.py --out lora_data_mixed/

P2 measured the cost of the plain arm: 3 epochs on 63 poems HALVES form compliance, 1 epoch is
neutral, neither helps. 理's item 1 is the arm that tries to keep form while moving voice.

WHERE THE ANCHOR COMES FROM, AND WHY IT IS NOT AN OUTSIDE DATASET
-----------------------------------------------------------------
The anchor is the BASE MODEL'S OWN generations that PASSED the form verifier — self-distillation
against the very behaviour the plain LoRA destroys. It needs no new data, no new prompt shape and
no judgement about what "good instruction data" is: it is this model, doing this task, correctly,
as measured by the same instrument that scores the arms.

The anchor is drawn EQUALLY ACROSS THE FOUR FORMS, matching the prompt distribution P0 and P2 are
scored on. Weighting it toward 七絕 would train a 七絕 booster rather than an instruction anchor,
and the scored form would then be the one form the anchor had over-taught.
"""
import argparse, json, os, random, re

CJK = re.compile(r"[㐀-鿿]")
FORMS = ("五絕", "七絕", "五律", "七律")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--poems", default=os.path.join(here, "lora_data", "train.jsonl"))
    ap.add_argument("--poems-valid", default=os.path.join(here, "lora_data", "valid.jsonl"))
    ap.add_argument("--baseline", nargs="+", default=[
        os.path.join(here, "results", f + ".jsonl")
        for f in ("p0_base_7b_rescored", "p0_base_7b_s2", "p0_base_7b_s3")])
    ap.add_argument("--anchor", type=int, default=63, help="anchor examples, split evenly by form")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(here, "lora_data_mixed"))
    a = ap.parse_args()

    poems = [json.loads(l) for l in open(a.poems)]
    pvalid = [json.loads(l) for l in open(a.poems_valid)]

    pool = {f: [] for f in FORMS}
    for p in a.baseline:
        for l in open(p):
            r = json.loads(l)
            if r["pass"] and r["form"] in pool:
                pool[r["form"]].append(r)
    rnd = random.Random(a.seed)
    per = a.anchor // len(FORMS)
    anchor = []
    for f in FORMS:
        rnd.shuffle(pool[f])
        anchor += pool[f][:per]
    # the remainder goes to the form with the most spare, deterministically
    extra = a.anchor - len(anchor)
    if extra:
        anchor += pool[max(FORMS, key=lambda f: len(pool[f]))][per:per + extra]

    def as_example(r):
        return {"messages": [{"role": "user", "content": r["prompt"]},
                             {"role": "assistant", "content": r["poem"]}]}

    avalid = anchor[:7]
    atrain = anchor[7:]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.jsonl"), "w") as f:
        for e in poems:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        for r in atrain:
            f.write(json.dumps(as_example(r), ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "valid.jsonl"), "w") as f:
        for e in pvalid:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        for r in avalid:
            f.write(json.dumps(as_example(r), ensure_ascii=False) + "\n")

    man = {"arm": "form-preserving: voice corpus + self-distilled form anchor",
           "poems_train": len(poems), "poems_valid": len(pvalid),
           "anchor_total": len(anchor), "anchor_train": len(atrain), "anchor_valid": len(avalid),
           "anchor_by_form": {f: sum(1 for r in anchor if r["form"] == f) for f in FORMS},
           "anchor_source": "base-model generations that PASSED verify_form.py, across three seeds",
           "anchor_pool_available": {f: len(pool[f]) for f in FORMS},
           "seed": a.seed,
           "train_total": len(poems) + len(atrain), "valid_total": len(pvalid) + len(avalid),
           "why_even_across_forms": "matches the prompt distribution P0/P2 are scored on; weighting "
                                    "toward 七絕 would make it a 七絕 booster, and 七絕 is the scored form",
           "what_it_does_not_do": "this is NOT KL-to-base. It anchors by DATA, not by a loss term — "
                                  "cheaper and available in mlx-lm, but it constrains the behaviour "
                                  "only where the anchor examples reach."}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print(f"  train {man['train_total']} ({len(poems)} poems + {len(atrain)} anchor) · "
          f"valid {man['valid_total']}")
    print(f"  anchor by form: {man['anchor_by_form']}   pool: {man['anchor_pool_available']}")


if __name__ == "__main__":
    main()
