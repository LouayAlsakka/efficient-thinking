#!/usr/bin/env python
"""ET-IV item 7d — canon-judge selection value (scores G2p; prototype scale, guardrailed).

Over E1 sonnet samples, three selectors pick one of N candidates:
  random          : baseline
  checker-only    : first form-valid candidate (checker 1a) — form without taste
  canon-judge     : the 7b LoRA judge's pick via single-elim pairwise tournament
The selected poems are scored by an INDEPENDENT model (7B persona judge) — never the canon judge
(Goodhart circularity guardrail). G2p predicts canon-judge selection beats random and checker-only.
NOTE: the 7b canon judge is a noise-level prototype (n=34 corpus), so this measures the PIPELINE, not
a claim; corpus expansion is the prerequisite for a real G2p.

  PYTHONPATH=poetry/canon python poetry/canon/selection_7d.py
"""
import glob, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prep_7b import fmt_prompt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkers"))
from meter_rhyme import check_iambic_pentameter
from mlx_lm import load, generate

PERSONA = ("You are a discerning poetry critic. Rate the following poem from 1 to 10 on overall "
           "aesthetic quality. Reply with ONLY the number.")


def lines_of(sample):
    return [x.strip() for x in (sample or "").splitlines() if x.strip() and not x.strip().startswith("(")][:4]


def canon_pick(model, tok, cands):
    idx = list(range(len(cands)))
    while len(idx) > 1:
        nxt = []
        for i in range(0, len(idx), 2):
            if i + 1 >= len(idx):
                nxt.append(idx[i]); continue
            a, b = idx[i], idx[i + 1]
            msgs = [{"role": "user", "content": fmt_prompt(cands[a], cands[b])}]
            pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
            out = generate(model, tok, prompt=pr, max_tokens=4, verbose=False).upper()
            m = re.search(r"[AB]", out)
            nxt.append(a if (m.group(0) if m else "A") == "A" else b)
        idx = nxt
    return idx[0]


def persona_score(model, tok, poem_lines):
    msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": "\n".join(poem_lines)}]
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
    out = generate(model, tok, prompt=pr, max_tokens=4, verbose=False)
    m = re.findall(r"10|[1-9]", out)
    return int(m[0]) if m else 5


def main():
    N = 4
    items = []
    for r in (json.loads(l) for l in open("poetry/cache/e1_3B.jsonl")):
        if r.get("task") == "sonnet":
            cands = [lines_of(s) for s in r["samples"][:N]]
            cands = [c for c in cands if len(c) == 4]
            if len(cands) >= 2:
                items.append(cands)
    items = items[:40]
    rng = random.Random(0)
    canon, tokc = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit", adapter_path="poetry/canon/adapter")
    scorer, toks = load("mlx-community/Qwen2.5-7B-Instruct-4bit")   # INDEPENDENT scorer
    res = {"random": [], "checker": [], "canon": []}
    for cands in items:
        res["random"].append(persona_score(scorer, toks, cands[rng.randrange(len(cands))]))
        chk = next((c for c in cands if check_iambic_pentameter(" ".join(c)) or all(check_iambic_pentameter(l)["ok"] for l in c)), cands[0])
        res["checker"].append(persona_score(scorer, toks, chk))
        res["canon"].append(persona_score(scorer, toks, cands[canon_pick(canon, tokc, cands)]))
    out = {k: round(sum(v) / len(v), 2) for k, v in res.items()}
    print(f"[7d] independent persona score of selected poem (n={len(items)}): {out}")
    g2p = out["canon"] > out["random"] and out["canon"] > out["checker"]
    print(f"[7d] G2p (canon > random and > checker): {'holds' if g2p else 'NOT shown'}  "
          f"[prototype — canon judge is noise-level at n=34; pipeline, not claim]")
    json.dump({"n": len(items), "mean_persona": out, "g2p_holds": g2p, "scale": "prototype n=34"},
              open("poetry/canon/selection_7d.json", "w"), indent=2)
    print("[7d] wrote poetry/canon/selection_7d.json")


if __name__ == "__main__":
    main()
