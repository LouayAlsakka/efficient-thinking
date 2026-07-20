#!/usr/bin/env python
"""ET-IV E1 — sample-once poetry generation (batched, MLX). Mirrors reason_cache: N temperature samples
per prompt per model, cached raw; scoring (e1_score.py) selects post-hoc so best-of-N at any N ≤ nmax
costs no extra generation. Per-prompt logging from day one (the ET-III lesson).

  python poetry/poetry_gen.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --tag 3B \
      --nmax 16 --temp 0.8 --max-tokens 96 --out poetry/cache/e1_3B.jsonl
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="poetry/data/e1_prompts.jsonl")
    ap.add_argument("--nmax", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="cap #prompts (smoke tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mlx_lm import load, batch_generate
    from mlx_lm.sample_utils import make_sampler

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    prompts = [json.loads(l) for l in open(a.prompts)]
    if a.limit:
        prompts = prompts[:a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    model, tok = load(a.model)
    sampler = make_sampler(temp=a.temp)
    n_new = 0
    with open(a.out, "a") as f:
        for p in prompts:
            if p["id"] in done:
                continue
            msgs = [{"role": "user", "content": p["prompt"]}]
            pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
            r = batch_generate(model, tok, prompts=[pr] * a.nmax, max_tokens=a.max_tokens,
                               sampler=sampler, verbose=False)
            rec = {k: p[k] for k in p}
            rec["samples"] = r.texts
            f.write(json.dumps(rec) + "\n"); f.flush()
            n_new += 1
            if n_new % 20 == 0:
                print(f"[gen {a.tag}] {n_new} prompts done", flush=True)
    print(f"[gen {a.tag}] wrote {a.out} (+{n_new} prompts)", flush=True)


if __name__ == "__main__":
    main()
