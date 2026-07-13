#!/usr/bin/env python
"""Arm B evaluator-quality ablation — how much does the EVALUATOR gate search in reasoning?

From N samples/problem, compute two search strategies that differ ONLY in evaluator quality:
  * self-consistency@N  — majority vote (verifier-FREE consensus; a weak, implicit evaluator)
  * oracle-best-of-N@N  — correct if the gold answer is among the N (a PERFECT verifier / pass@N)
The gap (oracle-best-of-N − self-consistency) is the accuracy a *perfect evaluator* would unlock over
consensus — the reasoning analog of 'the evaluator is the bottleneck'. If the gap is large, search is
being held back by the evaluator, not by the policy's ability to generate a correct chain.

  /Users/lab/llm/.venv/bin/python reasoning/reason_ablation.py --problems 120 --nmax 32 --temp 0.8
"""
import argparse, json, random, time
from collections import Counter
from mlx_lm import load
from reason_sweep import sample_once, extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--data", default="reasoning/data/gsm8k_test.jsonl")
    ap.add_argument("--problems", type=int, default=120)
    ap.add_argument("--nmax", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reasoning/ablation_results.json")
    args = ap.parse_args()

    probs = [json.loads(l) for l in open(args.data)]
    random.Random(args.seed).shuffle(probs)
    probs = probs[:args.problems]
    golds = [p["answer"].split("####")[-1].strip().replace(",", "") for p in probs]
    print(f"[ablation] {args.model} | {len(probs)} problems | N<={args.nmax} | temp={args.temp}", flush=True)

    model, tok = load(args.model)
    t0 = time.time()
    samples = []
    for i, p in enumerate(probs):
        samples.append([sample_once(model, tok, p["question"], args.temp, args.max_tokens)
                        for _ in range(args.nmax)])
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(probs)}  ({el/(i+1):.1f}s/prob, ETA {el/(i+1)*(len(probs)-i-1)/60:.0f}min)", flush=True)

    Ns = [n for n in [1, 2, 4, 8, 16, 32] if n <= args.nmax]
    sc_curve, bo_curve = [], []
    for N in Ns:
        sc = bo = 0
        for smp, g in zip(samples, golds):
            sub = smp[:N]
            votes = [s for s in sub if s is not None]
            maj = Counter(votes).most_common(1)
            sc += bool(maj and str(maj[0][0]) == str(g))            # verifier-free consensus
            bo += any(str(s) == str(g) for s in sub)                # perfect verifier (pass@N)
        sc_a, bo_a = 100.0 * sc / len(probs), 100.0 * bo / len(probs)
        sc_curve.append((N, round(sc_a, 1))); bo_curve.append((N, round(bo_a, 1)))
        print(f"  N={N:<3d}  self-consistency={sc_a:5.1f}%   oracle-best-of-N={bo_a:5.1f}%   "
              f"evaluator-gap=+{bo_a-sc_a:.1f}", flush=True)

    json.dump({"model": args.model, "problems": len(probs), "temp": args.temp,
               "self_consistency": sc_curve, "oracle_best_of_n": bo_curve}, open(args.out, "w"), indent=2)
    print(f"[ablation] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
