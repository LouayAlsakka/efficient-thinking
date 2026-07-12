#!/usr/bin/env python
"""Arm B (LLM reasoning) — the chess framework in language.

Measures reasoning strength "like chess": a verifiable domain (math) is the perfect oracle; pass@1 is
the open-loop policy; self-consistency@N (majority vote over N samples) is inference-time search. This
is the seed harness — a small GSM8K-style set to prove the pipeline; scales to a full benchmark next.

  /Users/lab/llm/.venv/bin/python reasoning/reason_eval.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
      --samples 8 --temp 0.8
"""
import argparse, re, time
from collections import Counter
from mlx_lm import load, generate

# (question, gold answer) — GSM8K-style, verifiable exact integers
PROBLEMS = [
    ("Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell altogether?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?", 10),
    ("Betty needs $100 for a wallet. She has half of it. Her parents give her $15, and her grandparents give twice as much as her parents. How much more money does Betty need?", 5),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write in a year?", 624),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", 3),
    ("Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. Seattle has 20 sheep. How many sheep do they have together?", 260),
    ("There are 15 trees. Workers plant trees so that there will be 21 trees. How many trees did they plant?", 6),
    ("Ken had 40 apples. He gave 12 to Ann and ate 5. How many does Ken have left?", 23),
]


def extract(text):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text)              # fallback: last number in the text
    return nums[-1].replace(",", "") if nums else None


def solve(model, tok, q, temp, max_tokens):
    msgs = [{"role": "user", "content": q + "\nThink step by step, then end your answer with: #### <number>"}]
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
    kw = {"max_tokens": max_tokens, "verbose": False}
    if temp > 0:                                        # temperature sampling for the search arm
        try:
            from mlx_lm.sample_utils import make_sampler
            kw["sampler"] = make_sampler(temp=temp)
        except Exception:
            kw["temp"] = temp
    return extract(generate(model, tok, prompt=pr, **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--samples", type=int, default=8, help="N for self-consistency (search)")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    t = time.time()
    model, tok = load(args.model)
    print(f"[load] {args.model}  {time.time()-t:.1f}s", flush=True)

    p1 = sc = 0
    for q, gold in PROBLEMS:
        a1 = solve(model, tok, q, 0.0, args.max_tokens)                     # open loop: greedy pass@1
        votes = [solve(model, tok, q, args.temp, args.max_tokens) for _ in range(args.samples)]
        maj = Counter([v for v in votes if v is not None]).most_common(1)   # search: self-consistency
        scv = maj[0][0] if maj else None
        ok1, okc = (str(a1) == str(gold)), (str(scv) == str(gold))
        p1 += ok1; sc += okc
        print(f"  gold={gold:<5} pass@1={a1}{' ✓' if ok1 else ' ✗'}   sc@{args.samples}={scv}{' ✓' if okc else ' ✗'}", flush=True)

    n = len(PROBLEMS)
    print(f"\n>>> pass@1 (open loop): {p1}/{n} = {100*p1/n:.0f}%", flush=True)
    print(f">>> self-consistency@{args.samples} (search): {sc}/{n} = {100*sc/n:.0f}%   "
          f"[search gain: {100*(sc-p1)/n:+.0f} pts]", flush=True)


if __name__ == "__main__":
    main()
