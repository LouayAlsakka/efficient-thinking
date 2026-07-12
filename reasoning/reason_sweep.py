#!/usr/bin/env python
"""Arm B — accuracy vs inference-compute sweep (the chess sims-sweep analog).

Generate N_max samples per math problem once, then read off self-consistency@N accuracy for a sweep
of N. Tests whether reasoning accuracy scales with inference compute (search) the way chess Elo scaled
log-linearly with MCTS sims — and by how much search lifts a fixed model (the +286 analog).

  /Users/lab/llm/.venv/bin/python reasoning/reason_sweep.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
      --problems 50 --nmax 16 --temp 0.8
"""
import argparse, json, re, time, random
from collections import Counter
from mlx_lm import load, generate


def extract(text):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else None


def sample_once(model, tok, q, temp, max_tokens):
    msgs = [{"role": "user", "content": q + "\nThink step by step, then end with: #### <number>"}]
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
    kw = {"max_tokens": max_tokens, "verbose": False}
    if temp > 0:
        try:
            from mlx_lm.sample_utils import make_sampler
            kw["sampler"] = make_sampler(temp=temp)
        except Exception:
            kw["temp"] = temp
    return extract(generate(model, tok, prompt=pr, **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--data", default="reasoning/data/gsm8k_test.jsonl")
    ap.add_argument("--problems", type=int, default=50)
    ap.add_argument("--nmax", type=int, default=16, help="samples/problem; sweep N reads off subsets")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reasoning/sweep_results.json")
    args = ap.parse_args()

    probs = [json.loads(l) for l in open(args.data)]
    random.Random(args.seed).shuffle(probs)
    probs = probs[:args.problems]
    golds = [p["answer"].split("####")[-1].strip().replace(",", "") for p in probs]
    print(f"[sweep] {args.model} | {len(probs)} problems | N≤{args.nmax} | temp={args.temp}", flush=True)

    model, tok = load(args.model)
    t0 = time.time()
    all_samples = []
    for i, p in enumerate(probs):
        all_samples.append([sample_once(model, tok, p["question"], args.temp, args.max_tokens)
                            for _ in range(args.nmax)])
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(probs)}  ({el/(i+1):.1f}s/prob, ETA {el/(i+1)*(len(probs)-i-1)/60:.0f}min)", flush=True)

    Ns = [n for n in [1, 2, 4, 8, 16, 32, 64] if n <= args.nmax]
    print("\n>>> accuracy vs inference compute (self-consistency@N):", flush=True)
    curve = []
    for N in Ns:
        correct = 0
        for samples, g in zip(all_samples, golds):
            votes = [s for s in samples[:N] if s is not None]
            maj = Counter(votes).most_common(1)
            correct += (maj and str(maj[0][0]) == str(g))
        acc = 100.0 * correct / len(probs)
        curve.append((N, round(acc, 1)))
        print(f"  N={N:<3d}  accuracy={acc:.1f}%", flush=True)
    if len(curve) > 1:
        print(f"\n>>> pass@1 → sc@{Ns[-1]}: {curve[0][1]:.1f}% → {curve[-1][1]:.1f}%  "
              f"(+{curve[-1][1]-curve[0][1]:.1f} pts from search)", flush=True)
    json.dump({"model": args.model, "problems": len(probs), "temp": args.temp, "curve": curve},
              open(args.out, "w"), indent=2)
    print(f"[sweep] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
