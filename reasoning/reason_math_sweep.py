#!/usr/bin/env python
"""MATH difficulty-tiered accuracy sweep — the calibrated 'opponent ladder' for reasoning-GELO.

MATH-500 is labeled by difficulty level 1-5 (the analog of a Stockfish rung / ab-depth). We measure a
model's accuracy per level (greedy + self-consistency@N), which (a) is the tiered ladder and (b) feeds
the IRT fit that places the model's ability on the GELO scale. Saves per-problem correctness so
multiple models can be co-calibrated.

  /Users/lab/llm/.venv/bin/python reasoning/reason_math_sweep.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
      --per-level 30 --nmax 4
"""
import argparse, json, re, time, random
from collections import Counter


def extract_boxed(text):
    i = text.rfind("\\boxed")
    if i == -1:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "") if nums else None
    j = text.find("{", i)
    if j == -1:
        return None
    depth, out = 0, []
    for c in text[j:]:
        if c == "{":
            depth += 1
            if depth == 1:
                continue
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
    return "".join(out)


def normalize(s):
    if s is None:
        return None
    s = s.strip()
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\ ", ""),
                 (" ", ""), ("$", ""), ("dfrac", "frac"), ("tfrac", "frac"), ("\\cdot", ""),
                 ("\\{", ""), ("\\}", ""), ("^{\\circ}", ""), ("^\\circ", "")]:
        s = s.replace(a, b)
    return s.rstrip(".").strip()


def solve(model, tok, q, temp, max_tokens):
    from mlx_lm import generate
    msgs = [{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
    kw = {"max_tokens": max_tokens, "verbose": False}
    if temp > 0:
        try:
            from mlx_lm.sample_utils import make_sampler
            kw["sampler"] = make_sampler(temp=temp)
        except Exception:
            kw["temp"] = temp
    return normalize(extract_boxed(generate(model, tok, prompt=pr, **kw)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--data", default="reasoning/data/math500.jsonl")
    ap.add_argument("--per-level", type=int, default=30, help="problems sampled per difficulty level")
    ap.add_argument("--nmax", type=int, default=4, help="self-consistency samples")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    out = args.out or f"reasoning/math_{tag}.json"

    rows = [json.loads(l) for l in open(args.data)]
    by_level = {}
    rng = random.Random(args.seed)
    for r in rows:
        by_level.setdefault(r["level"], []).append(r)
    picked = []
    for lv in sorted(by_level):
        rng.shuffle(by_level[lv])
        picked += [(lv, r) for r in by_level[lv][:args.per_level]]
    print(f"[math] {args.model} | {len(picked)} problems ({args.per_level}/level) | N={args.nmax}", flush=True)

    from mlx_lm import load
    model, tok = load(args.model)
    t0 = time.time()
    per_problem = []                                    # (level, greedy_correct, sc_correct)
    for i, (lv, r) in enumerate(picked):
        gold = normalize(r["answer"])
        g = solve(model, tok, r["problem"], 0.0, args.max_tokens)
        votes = [solve(model, tok, r["problem"], args.temp, args.max_tokens) for _ in range(args.nmax)]
        maj = Counter([v for v in votes if v]).most_common(1)
        sc = maj[0][0] if maj else None
        per_problem.append({"level": lv, "greedy": g == gold, "sc": sc == gold})
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(picked)}  ({el/(i+1):.1f}s/prob, ETA {el/(i+1)*(len(picked)-i-1)/60:.0f}min)", flush=True)

    levels = sorted(set(p["level"] for p in per_problem))
    print("\n>>> accuracy by difficulty level:", flush=True)
    summary = {}
    for lv in levels:
        ps = [p for p in per_problem if p["level"] == lv]
        gacc = 100.0 * sum(p["greedy"] for p in ps) / len(ps)
        sacc = 100.0 * sum(p["sc"] for p in ps) / len(ps)
        summary[lv] = {"n": len(ps), "greedy": round(gacc, 1), "sc": round(sacc, 1)}
        print(f"  L{lv}: greedy={gacc:5.1f}%  sc@{args.nmax}={sacc:5.1f}%  (n={len(ps)})", flush=True)

    json.dump({"model": args.model, "per_level": summary, "per_problem": per_problem,
               "nmax": args.nmax}, open(out, "w"), indent=2)
    print(f"[math] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
