#!/usr/bin/env python
"""ET-IV E6 — serial self-revision (closed loop), for the P6 test: does draft→self-critique→revise
beat parallel sampling + external checker selection at a matched token budget? The framework predicts
the closed loop underperforms (no external signal enters). This script produces ONLY the closed-loop
arm; the parallel arm reuses E1's cached samples. Every intermediate draft + its token count is logged
so scoring can match budgets exactly (select the parallel N whose cumulative tokens ≈ serial's).

  python poetry/e6_revise.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --tag 3B \
      --rounds 8 --limit 50 --out poetry/cache/e6_3B.jsonl
"""
import argparse, json, os

CRITIQUE = ("Here is your draft:\n\n{draft}\n\nCritique it for {form}, then output an improved version "
            "that fixes any problems. Output ONLY the revised poem, nothing else.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="poetry/data/e1_prompts.jsonl")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    prompts = [json.loads(l) for l in open(a.prompts)][:a.limit] if a.limit else \
        [json.loads(l) for l in open(a.prompts)]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    model, tok = load(a.model)
    sampler = make_sampler(temp=a.temp)
    form = {"sonnet": "iambic pentameter and ABAB rhyme", "villanelle": "iambic pentameter and ABA rhyme",
            "lyric": "the requested syllable-stress pattern"}
    n_new = 0
    with open(a.out, "a") as f:
        for p in prompts:
            if p["id"] in done:
                continue
            drafts, toks = [], []
            msgs = [{"role": "user", "content": p["prompt"]}]
            pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
            d = generate(model, tok, prompt=pr, max_tokens=a.max_tokens, sampler=sampler, verbose=False)
            drafts.append(d); toks.append(len(tok.encode(d)))
            for _ in range(a.rounds):
                msgs = [{"role": "user", "content": p["prompt"]},
                        {"role": "assistant", "content": drafts[-1]},
                        {"role": "user", "content": CRITIQUE.format(draft=drafts[-1], form=form[p["task"]])}]
                pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
                d = generate(model, tok, prompt=pr, max_tokens=a.max_tokens, sampler=sampler, verbose=False)
                drafts.append(d); toks.append(len(tok.encode(d)))
            rec = {k: p[k] for k in p}
            rec["drafts"] = drafts; rec["draft_tokens"] = toks
            f.write(json.dumps(rec) + "\n"); f.flush()
            n_new += 1
            if n_new % 20 == 0:
                print(f"[e6 {a.tag}] {n_new} prompts x {a.rounds} rounds done", flush=True)
    print(f"[e6 {a.tag}] wrote {a.out} (+{n_new} prompts)", flush=True)


if __name__ == "__main__":
    main()
