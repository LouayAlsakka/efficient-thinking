#!/usr/bin/env python
"""ET-IV item 7b (prep) — format the canon contrast into a LoRA preference-judge dataset.

Each pairwise record becomes a binary judge example: two poems A/B (canon side randomized), the model
learns to pick the canon one. Held-out split is stratified by tier so 7b's discrimination eval reports
canon-vs-corrupted (easy) and canon-vs-model (hard) separately. Writes mlx_lm-lora format
(prompt/completion) to poetry/canon/lora_data/{train,valid}.jsonl and the held-out pairs to heldout.jsonl.
"""
import json, os, random

SYS = ("You are a discerning poetry critic. Two poems follow, A and B. Exactly one is canonical, "
       "published verse; the other is weaker. Reply with only the letter of the better poem.")


def fmt_prompt(a_lines, b_lines):
    return (f"{SYS}\n\n[A]\n" + "\n".join(a_lines) + "\n\n[B]\n" + "\n".join(b_lines) +
            "\n\nWhich is the better poem, A or B? Answer with a single letter.")


def main():
    rng = random.Random(1)
    rows = [json.loads(l) for l in open("poetry/canon/contrast.jsonl")]
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    train, heldout = [], []
    for tier, rs in by_tier.items():
        rng.shuffle(rs)
        k = max(3, len(rs) // 4)                       # ~25% held out per tier for the discrimination eval
        heldout += [{**r} for r in rs[:k]]
        train += rs[k:]
    rng.shuffle(train)

    def to_example(r):
        canon_is_a = rng.random() < 0.5
        a, b = (r["chosen"], r["rejected"]) if canon_is_a else (r["rejected"], r["chosen"])
        return {"prompt": fmt_prompt(a, b), "completion": " A" if canon_is_a else " B"}

    exs = [to_example(r) for r in train]
    nval = max(2, len(exs) // 6)
    os.makedirs("poetry/canon/lora_data", exist_ok=True)
    with open("poetry/canon/lora_data/valid.jsonl", "w") as f:
        for e in exs[:nval]:
            f.write(json.dumps(e) + "\n")
    with open("poetry/canon/lora_data/train.jsonl", "w") as f:
        for e in exs[nval:]:
            f.write(json.dumps(e) + "\n")
    with open("poetry/canon/heldout.jsonl", "w") as f:
        for r in heldout:
            f.write(json.dumps(r) + "\n")
    print(f"[7b-prep] train={len(exs)-nval} valid={nval} heldout={len(heldout)} "
          f"(tiers: {[(t, sum(1 for h in heldout if h['tier']==t)) for t in by_tier]})")


if __name__ == "__main__":
    main()
