#!/usr/bin/env python
"""ET-III — the policy × judge allocation grid over frozen caches (E3, subsuming on-distribution E2).

For each policy cache (0.5B→72B, 32 samples/problem from ET-II), a *local* Qwen judge does pick-best among
the first N candidates. We record, per (policy, judge, N) cell:
  - pick_best_acc  : judge selection accuracy = the ON-DISTRIBUTION judge quality q that actually drives
                     the allocation (the judge faces the policy's own plausible-but-wrong samples)
  - majority_acc   : verifier-free baseline (j = 0)
  - oracle_acc     : pass@N ceiling (s-independent upper envelope)
  - mean_gen_tok / mean_judge_in_tok : token counts → FLOP accounting (policy p·gen ; judge j·prefill)
so cells project onto iso-budget curves. One judge per invocation; loop judges via a launcher.

  python judging/et3_judge.py --judge mlx-community/Qwen2.5-7B-Instruct-4bit --judge-params 7 \
      --policies 1.5B:1.5,3B:3,7B:7 --nlist 4,16 --problems 200 --out judging/e3_judge7B.json
"""
import argparse, json, os, random, re
from collections import Counter


def extract(text):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", text or "")
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text or "")
    return nums[-1].replace(",", "") if nums else None


def load_problems(data, n, seed=0):
    probs = [json.loads(l) for l in open(data)]
    random.Random(seed).shuffle(probs)           # SAME shuffle as reason_cache generate → aligns by index
    return probs[:n]


def pick_best(model, tok, gen, problem, cands, max_new=8):
    listing = "\n\n".join(f"[Candidate {k+1}]\n{c}" for k, c in enumerate(cands))
    msgs = [{"role": "system", "content": "You are a strict math grader. Several candidate solutions to one "
             "problem follow, numbered. Reply with ONLY the number of the candidate whose FINAL answer is "
             "correct. If several are correct, reply any correct one."},
            {"role": "user", "content": f"[Problem]\n{problem}\n\n{listing}\n\n"
             f"Which candidate (1-{len(cands)}) is correct? Reply with just the number."}]
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
    out = gen(model, tok, prompt=pr, max_tokens=max_new, verbose=False)
    for t in re.findall(r"\d+", out):
        i = int(t)
        if 1 <= i <= len(cands):
            return i - 1, len(pr)
    return 0, len(pr)


def judge_pair(model, tok, gen, problem, ca, cb):
    """One pairwise comparison — small context (2 candidates), avoids the long-list position bias."""
    msgs = [{"role": "system", "content": "You are a strict math grader. Two candidate solutions to one "
             "problem follow. Reply with ONLY 'A' or 'B' — whichever has the correct final answer."},
            {"role": "user", "content": f"[Problem]\n{problem}\n\n[A]\n{ca}\n\n[B]\n{cb}\n\nWhich is correct, A or B?"}]
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
    out = gen(model, tok, prompt=pr, max_tokens=4, verbose=False).strip().upper()
    return (0 if out.startswith("A") else 1), len(pr)


def pairwise_best(model, tok, gen, problem, cands, rng):
    """Single-elimination pairwise tournament: N-1 small-context calls vs pick-best's one huge-context call.
    Returns (winning_index, total_judge_input_tokens)."""
    idx = list(range(len(cands))); rng.shuffle(idx); jin = 0
    while len(idx) > 1:
        nxt = []
        for i in range(0, len(idx), 2):
            if i + 1 >= len(idx):
                nxt.append(idx[i]); continue
            a, b = idx[i], idx[i + 1]
            w, t = judge_pair(model, tok, gen, problem, cands[a], cands[b]); jin += t
            nxt.append(a if w == 0 else b)
        idx = nxt
    return idx[0], jin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True); ap.add_argument("--judge-params", type=float, required=True)
    ap.add_argument("--policies", required=True, help="comma list tag:params, caches at reasoning/cache/gsm8k_<tag>.jsonl")
    ap.add_argument("--data", default="reasoning/data/gsm8k_test.jsonl")
    ap.add_argument("--nlist", default="4,16"); ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--mode", default="pick-best", choices=["pick-best", "pairwise"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mlx_lm import load, generate as gen
    probs = load_problems(a.data, a.problems)
    golds = [str(p["answer"].split("####")[-1].strip().replace(",", "")) for p in probs]
    Ns = [int(x) for x in a.nlist.split(",")]
    model, tok = load(a.judge)
    rng = random.Random(0)
    rows = []
    for spec in a.policies.split(","):
        ptag, pp = spec.split(":")
        cache = f"reasoning/cache/gsm8k_{ptag}.jsonl"
        items = [json.loads(l) for l in open(cache)][:a.problems]
        m = min(len(items), len(probs))
        for N in Ns:
            pb = maj = orc = 0; jin = gtok = 0
            for i in range(m):
                subs = items[i]["samples"][:N]; g = golds[i]
                ans = [extract(s) for s in subs]
                gtok += sum(len(tok.encode(s)) for s in subs) / N        # avg policy gen tokens/sample
                mc = Counter([x for x in ans if x]).most_common(1)
                maj += bool(mc and str(mc[0][0]) == g)
                orc += any(str(x) == g for x in ans)
                if a.mode == "pairwise":
                    pick, plen = pairwise_best(model, tok, gen, probs[i]["question"], subs, rng)
                else:
                    pick, plen = pick_best(model, tok, gen, probs[i]["question"], subs)
                jin += plen
                pb += (str(ans[pick]) == g)
            rows.append({"policy": ptag, "policy_params": float(pp), "judge": a.judge.split("/")[-1],
                         "judge_params": a.judge_params, "mode": a.mode, "N": N, "n": m,
                         "pick_best_acc": round(100 * pb / m, 1), "majority_acc": round(100 * maj / m, 1),
                         "oracle_acc": round(100 * orc / m, 1),
                         "mean_gen_tok": round(gtok / m), "mean_judge_in_tok": round(jin / m)})
            print(f"[e3] policy={ptag} judge={a.judge_params}B N={N}: pick-best={rows[-1]['pick_best_acc']} "
                  f"maj={rows[-1]['majority_acc']} oracle={rows[-1]['oracle_acc']} "
                  f"(judge_in≈{rows[-1]['mean_judge_in_tok']}tok)", flush=True)
            json.dump(rows, open(a.out, "w"), indent=2)
    print(f"[e3] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
