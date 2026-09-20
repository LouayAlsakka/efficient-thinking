#!/usr/bin/env python
"""B1 — judge-asymmetry recompute with a LOCAL 32B judge (§5, previously exploratory).

§5's claim: search buys a POLICY +8-11 pts but a JUDGE only ~+2.5 — a weak/uncertain evaluator's
repeated judgments concentrate on its own systematic error, so self-consistency can't buy it back.
The shipped number used the cloud Kimi judge at exploratory power. Here we recompute with a strong
LOCAL judge (Qwen2.5-32B-Instruct-4bit, mlx) over more decisive pairs, so the asymmetry is powered
and reproducible offline. Judge is BLINDED (never sees model names) and pair order RANDOMIZED.

  ./.venv/bin/python reasoning/reason_judge_scaling_local.py --answers reasoning/arena_answers_v2.json \
      --judge mlx-community/Qwen2.5-32B-Instruct-4bit --pairs 200 --votes 9
"""
import argparse, json, random, sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reason_math_sweep import extract_boxed, normalize

SYS = ("You are a strict, impartial grader. Two assistants answered the same problem. Decide whose "
       "FINAL answer is correct. Reply with EXACTLY one token: A, B, or TIE.")


def jcall_local(model, tok, gen, sampler, problem, a, b):
    user = f"[Problem]\n{problem}\n\n[Assistant A]\n{a}\n\n[Assistant B]\n{b}\n\nWhich is correct? A, B, or TIE."
    pr = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                 add_generation_prompt=True)
    t = (gen(model, tok, prompt=pr, max_tokens=4, sampler=sampler, verbose=False) or "").strip().upper()
    return "A" if t.startswith("A") else ("B" if t.startswith("B") else "TIE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="reasoning/arena_answers_v2.json")
    ap.add_argument("--judge", default="mlx-community/Qwen2.5-32B-Instruct-4bit")
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--votes", type=int, default=9)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reasoning/b1_judge_scaling_32B.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    d = json.load(open(a.answers)); Q = d["questions"]; A = d["answers"]
    names = [n for n in A if "kimi" not in n.lower()]              # judge grades the small contestants, not the master
    decisive = []
    for qi, q in enumerate(Q):
        gold = normalize(q.get("gold"))
        if gold is None:
            continue
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if qi >= len(A[names[i]]) or qi >= len(A[names[j]]):
                    continue
                ai, aj = A[names[i]][qi], A[names[j]][qi]
                ci, cj = normalize(extract_boxed(ai)), normalize(extract_boxed(aj))
                if ci is None or cj is None:
                    continue
                ri, rj = (ci == gold), (cj == gold)
                if ri != rj:                                       # exactly one correct → decisive
                    if rng.random() < 0.5:
                        pa, pb, true_slot = ai, aj, ("A" if ri else "B")
                    else:
                        pa, pb, true_slot = aj, ai, ("A" if rj else "B")
                    decisive.append((q["problem"], pa, pb, true_slot))
    rng.shuffle(decisive); decisive = decisive[:a.pairs]
    print(f"[b1] {len(decisive)} clean decisive pairs x up to {a.votes} votes | judge={a.judge}", flush=True)

    from mlx_lm import load, generate as gen
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(a.judge)
    sampler = make_sampler(temp=a.temp)

    votes = []
    for k, (prob, pa, pb, true_slot) in enumerate(decisive):
        vs = [jcall_local(model, tok, gen, sampler, prob, pa, pb) for _ in range(a.votes)]
        votes.append((vs, true_slot))
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(decisive)}", flush=True)
            json.dump({"pairs": len(votes), "partial": True}, open(a.out, "w"))

    curve = []
    for N in [n for n in (1, 3, 5, 7, 9) if n <= a.votes]:
        agree = 0
        for vs, true_slot in votes:
            sub = [v for v in vs[:N] if v != "TIE"]
            maj = Counter(sub).most_common(1)
            agree += ((maj[0][0] if maj else "TIE") == true_slot)
        acc = round(100.0 * agree / len(votes), 1)
        curve.append({"N": N, "agree": acc})
        print(f"  judge@{N}: agrees with verifier {acc}%", flush=True)
    lift = round(curve[-1]["agree"] - curve[0]["agree"], 1)
    json.dump({"judge": a.judge, "pairs": len(votes), "votes": a.votes, "temp": a.temp,
               "curve": curve, "sc_lift_N1_to_Nmax": lift}, open(a.out, "w"), indent=2)
    print(f"[b1] wrote {a.out} — judge self-consistency lift N=1→{a.votes}: {lift:+.1f} pts "
          f"(§5 asymmetry: a judge's search lift stays small vs a policy's +8–11)", flush=True)


if __name__ == "__main__":
    main()
