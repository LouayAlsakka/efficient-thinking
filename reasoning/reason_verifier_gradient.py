#!/usr/bin/env python
"""Verifier-quality GRADIENT — accuracy as a function of evaluator quality (turns the +14.2 two-point
ablation into the full evaluator-quality -> capability curve).

Generate N samples/problem, then for a grid of verifier accuracies q, simulate a noisy verifier that
labels each candidate correct/incorrect with per-item accuracy q, keep the verifier-approved
candidates, and take the majority among them (fall back to overall majority if none approved).
  q = 0.5  -> verifier is pure noise            -> reduces to self-consistency (consensus)
  q = 1.0  -> perfect verifier                  -> reduces to oracle best-of-N (pass@N)
The curve between is how much capability a better evaluator buys — the strongest form of the
'evaluator is the bottleneck' result. Saves raw samples so the gradient can be recomputed offline.

  /Users/lab/llm/.venv/bin/python reasoning/reason_verifier_gradient.py --problems 120 --nmax 16
"""
import argparse, json, random, time
from collections import Counter
import numpy as np
from mlx_lm import load
from reason_sweep import sample_once, extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--data", default="reasoning/data/gsm8k_test.jsonl")
    ap.add_argument("--problems", type=int, default=120)
    ap.add_argument("--nmax", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=25, help="random noisy-verifier draws to average")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reasoning/vgrad_results.json")
    ap.add_argument("--samples-out", default="reasoning/vgrad_samples.json")
    args = ap.parse_args()

    probs = [json.loads(l) for l in open(args.data)]
    random.Random(args.seed).shuffle(probs)
    probs = probs[:args.problems]
    golds = [p["answer"].split("####")[-1].strip().replace(",", "") for p in probs]
    print(f"[vgrad] {args.model} | {len(probs)} problems | N={args.nmax} | temp={args.temp}", flush=True)

    model, tok = load(args.model)
    t0 = time.time()
    data = []                                       # per problem: list of (answer_str, is_correct)
    for i, (p, g) in enumerate(zip(probs, golds)):
        samples = [sample_once(model, tok, p["question"], args.temp, args.max_tokens) for _ in range(args.nmax)]
        data.append([(s, str(s) == str(g)) for s in samples])
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(probs)}  ({el/(i+1):.1f}s/prob, ETA {el/(i+1)*(len(probs)-i-1)/60:.0f}min)", flush=True)

    json.dump({"golds": golds, "samples": [[(s or "", bool(c)) for s, c in d] for d in data]},
              open(args.samples_out, "w"))

    def majority(pool):
        pool = [a for a in pool if a]
        return Counter(pool).most_common(1)[0][0] if pool else None

    rng = np.random.default_rng(args.seed)
    qs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    curve = []
    for q in qs:
        accs = []
        for _ in range(args.trials):
            correct = 0
            for d, g in zip(data, golds):
                approved = [ans for ans, is_c in d if (is_c if rng.random() < q else (not is_c))]
                pick = majority(approved) if approved else majority([a for a, _ in d])
                correct += (str(pick) == str(g))
            accs.append(100.0 * correct / len(golds))
        m, s = float(np.mean(accs)), float(np.std(accs))
        curve.append((q, round(m, 1), round(s, 1)))
        print(f"  verifier q={q:.1f}  accuracy={m:5.1f}% ± {s:.1f}", flush=True)

    json.dump({"model": args.model, "problems": len(probs), "nmax": args.nmax,
               "verifier_gradient": curve}, open(args.out, "w"), indent=2)
    print(f"[vgrad] wrote {args.out}  (q=0.5≈consensus, q=1.0≈oracle best-of-N)", flush=True)


if __name__ == "__main__":
    main()
