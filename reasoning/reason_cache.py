#!/usr/bin/env python
"""Sample once, select many. Generate N full completions per problem per model, cache them, then compute
sc@{1,4,16,32}, oracle-best-of-N, and the whole graded-verifier curve post-hoc over the same cache — three
headline experiments as byproducts of one generation run. Resumable (append-only); one cache file per model.

  # generate (llm2 / llm1, splittable by --model and --data)
  python reasoning/reason_cache.py generate --model mlx-community/Qwen2.5-3B-Instruct-4bit \
      --problems 300 --nmax 32 --out reasoning/cache/gsm8k_3B.jsonl
  # score (any machine, mlx-free)
  python reasoning/reason_cache.py score --glob 'reasoning/cache/gsm8k_*.jsonl'
"""
import argparse, glob, json, os, random, re
from collections import Counter


def extract(text):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", text or "")
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text or "")
    return nums[-1].replace(",", "") if nums else None


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def generate(a):
    from mlx_lm import load, generate as gen
    from mlx_lm.sample_utils import make_sampler
    probs = [json.loads(l) for l in open(a.data)]
    random.Random(a.seed).shuffle(probs); probs = probs[:a.problems]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = sum(1 for _ in open(a.out)) if os.path.exists(a.out) else 0
    print(f"[cache] {a.model} | {len(probs)} problems × {a.nmax} samples → {a.out} (resume @ {done})", flush=True)
    model, tok = load(a.model)
    sampler = make_sampler(temp=a.temp)
    for i, p in enumerate(probs):
        if i < done:
            continue
        msgs = [{"role": "user", "content": p["question"] + "\nThink step by step, then end with: #### <number>"}]
        pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
        samples = [gen(model, tok, prompt=pr, max_tokens=a.max_tokens, sampler=sampler, verbose=False)
                   for _ in range(a.nmax)]
        with open(a.out, "a") as f:
            f.write(json.dumps({"gold": gold(p["answer"]), "samples": samples}) + "\n")
        if (i + 1) % 25 == 0:
            print(f"  {os.path.basename(a.out)} {i+1}/{len(probs)}", flush=True)
    print(f"[cache] DONE {a.out}", flush=True)


def score(a):
    files = sorted(glob.glob(a.glob))
    print(f"[score] {len(files)} caches | pass@1 · sc@N (majority) · oracle@N (coverage) · graded-verifier", flush=True)
    out = []
    for fn in files:
        items = [json.loads(l) for l in open(fn)]
        nmax = min(len(it["samples"]) for it in items)
        ext = [([extract(s) for s in it["samples"][:nmax]], str(it["gold"])) for it in items]
        n = len(ext)
        row = {"cache": os.path.basename(fn), "n": n, "nmax": nmax}
        # pass@1 = mean single-sample accuracy; sc@N = majority; oracle@N = any-correct
        row["pass1"] = round(100 * sum(sum(str(x) == g for x in ans) / len(ans) for ans, g in ext) / n, 1)
        for N in (4, 16, 32):
            if N > nmax:
                continue
            sc = orc = 0
            for ans, g in ext:
                maj = Counter([x for x in ans[:N] if x]).most_common(1)
                sc += bool(maj and str(maj[0][0]) == g)
                orc += any(str(x) == g for x in ans[:N])
            row[f"sc{N}"] = round(100 * sc / n, 1)
            row[f"oracle{N}"] = round(100 * orc / n, 1)
        # graded-verifier curve at N=nmax: a verifier with per-item accuracy q picks correct-if-present w.p. q
        grade = {}
        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            acc = 0.0
            for ans, g in ext:
                has = any(str(x) == g for x in ans)
                # q picks a correct sample if present, else a wrong one; consensus fallback at q=0.5 baseline
                acc += (q if has else 0.0)
            grade[f"q{q}"] = round(100 * acc / n, 1)
        row["graded"] = grade
        out.append(row)
        print(f"  {row['cache']}: pass1={row['pass1']} sc4={row.get('sc4')} sc16={row.get('sc16')} "
              f"sc32={row.get('sc32')} oracle{nmax}={row.get(f'oracle{nmax}')}", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[score] wrote {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--data", default="reasoning/data/gsm8k_test.jsonl")
    g.add_argument("--problems", type=int, default=300); g.add_argument("--nmax", type=int, default=32)
    g.add_argument("--temp", type=float, default=0.8); g.add_argument("--max-tokens", type=int, default=1024)
    g.add_argument("--seed", type=int, default=0); g.add_argument("--out", required=True)
    g.set_defaults(func=generate)
    s = sub.add_parser("score")
    s.add_argument("--glob", default="reasoning/cache/gsm8k_*.jsonl")
    s.add_argument("--out", default="reasoning/cache_scores.json")
    s.set_defaults(func=score)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
