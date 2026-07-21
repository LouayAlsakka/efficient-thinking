#!/usr/bin/env python
"""ET-IV item 7b (eval) — tiered canon-judge discrimination (scores G1p, prototype scale).

Loads the base Qwen judge and the canon LoRA adapter, runs each held-out pair (canon side randomized)
through both, and reports discrimination accuracy per tier: easy (canon vs corrupted) and hard (canon vs
model-generated). G1p predicts near-ceiling on corrupted, weakest on model-generated. NOTE: at n=34 this
is a pipeline prototype (heldout is a handful of pairs) — the pattern, not the point estimate, is the
deliverable; corpus expansion is the scale-up before any claim.

  PYTHONPATH=poetry/canon python poetry/canon/eval_7b.py
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_7b import fmt_prompt
from mlx_lm import load, generate


def judge(model, tok, a_lines, b_lines):
    msgs = [{"role": "user", "content": fmt_prompt(a_lines, b_lines)}]
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
    out = generate(model, tok, prompt=pr, max_tokens=4, verbose=False)
    m = re.search(r"[AB]", out.upper())
    return m.group(0) if m else "A"


def run(model, tok, held, rng):
    tiers = {}
    for r in held:
        canon_a = rng.random() < 0.5
        a, b = (r["chosen"], r["rejected"]) if canon_a else (r["rejected"], r["chosen"])
        correct = (judge(model, tok, a, b) == "A") == canon_a
        tiers.setdefault(r["tier"], []).append(int(correct))
    return {t: round(100 * sum(v) / len(v), 1) for t, v in tiers.items()}, {t: len(v) for t, v in tiers.items()}


def main():
    held = [json.loads(l) for l in open("poetry/canon/heldout.jsonl")]
    base, tok = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    b_acc, ns = run(base, tok, held, random.Random(7))
    lora, tokl = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit", adapter_path="poetry/canon/adapter")
    l_acc, _ = run(lora, tokl, held, random.Random(7))
    print(f"[7b] held-out sizes: {ns}")
    print(f"[7b] base discrimination: {b_acc}")
    print(f"[7b] LoRA discrimination: {l_acc}")
    json.dump({"heldout_sizes": ns, "base": b_acc, "lora": l_acc, "scale": "prototype n=34"},
              open("poetry/canon/eval_7b.json", "w"), indent=2)
    print("[7b] wrote poetry/canon/eval_7b.json")


if __name__ == "__main__":
    main()
