#!/usr/bin/env python
"""Frontier at scale — size × search across 0.5B → 14B, well-powered, with the oracle gap from the same
samples. Serves step 2 (power: larger n than the original ~120) AND step 3 (scale: 7B/14B points that,
per Snell et al., are where search should start winning — the above-threshold half of our competence
threshold). One sweep gives: greedy / sc@4 / sc@16 (the frontier) and oracle-best-of-N (the +14.2 gap)
for each model. Saves incrementally per model so partial results survive.

  python reasoning/reason_frontier_scale.py --problems 200 --nmax 16 --temp 0.8
"""
import argparse, json, re, random, time
from collections import Counter

MODELS = [("0.5B", 0.5, "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
          ("1.5B", 1.5, "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
          ("3B",   3.0, "mlx-community/Qwen2.5-3B-Instruct-4bit"),
          ("7B",   7.0, "mlx-community/Qwen2.5-7B-Instruct-4bit"),
          ("14B", 14.0, "mlx-community/Qwen2.5-14B-Instruct-4bit")]


def extract(text):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", text or "")
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text or "")
    return nums[-1].replace(",", "") if nums else None


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--nmax", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reasoning/frontier_scale_results.json")
    a = ap.parse_args()
    from mlx_lm import load, generate as gen
    from mlx_lm.sample_utils import make_sampler

    probs = [json.loads(l) for l in open("reasoning/data/gsm8k_test.jsonl")]
    random.Random(a.seed).shuffle(probs); probs = probs[:a.problems]
    golds = [gold(p["answer"]) for p in probs]
    sampler = make_sampler(temp=a.temp)
    n = len(probs)
    print(f"[frontier] {n} problems, nmax {a.nmax}, temp {a.temp} | models: {', '.join(m[0] for m in MODELS)}", flush=True)
    rows, t0 = [], time.time()
    for name, params, mid in MODELS:
        try:
            model, tok = load(mid)
        except Exception as e:
            print(f"[frontier] SKIP {name} ({mid}): {str(e)[:100]}", flush=True); continue
        greedy_ok = 0; per = []
        for i, (p, g) in enumerate(zip(probs, golds)):
            msgs = [{"role": "user", "content": p["question"] + "\nThink step by step, then end with: #### <number>"}]
            pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
            greedy_ok += (extract(gen(model, tok, prompt=pr, max_tokens=a.max_tokens, verbose=False)) == g)
            ans = [extract(gen(model, tok, prompt=pr, max_tokens=a.max_tokens, sampler=sampler, verbose=False))
                   for _ in range(a.nmax)]
            per.append(ans)
            if (i + 1) % 50 == 0:
                print(f"  {name} {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
        row = {"model": name, "params": params, "greedy": round(100 * greedy_ok / n, 1)}
        for N in (4, 16):
            if N > a.nmax:
                continue
            sc = 0
            for ans, g in zip(per, golds):
                maj = Counter([x for x in ans[:N] if x]).most_common(1)
                sc += bool(maj and str(maj[0][0]) == g)
            row[f"sc{N}"] = round(100 * sc / n, 1)
        orc = sum(any(str(x) == g for x in ans) for ans, g in zip(per, golds))
        row[f"oracle{a.nmax}"] = round(100 * orc / n, 1)
        rows.append(row)
        json.dump({"problems": n, "nmax": a.nmax, "rows": rows}, open(a.out, "w"), indent=2)
        print(f"[frontier] {name} ({params}B): greedy={row['greedy']}  sc4={row.get('sc4')}  "
              f"sc16={row.get('sc16')}  oracle{a.nmax}={row[f'oracle{a.nmax}']}  ({time.time()-t0:.0f}s)", flush=True)
    print("[frontier] DONE", flush=True)


if __name__ == "__main__":
    main()
