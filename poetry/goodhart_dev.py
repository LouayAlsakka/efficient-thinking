#!/usr/bin/env python
"""ET-IV item 5 — Goodhart pressure pilot (DEV-ONLY; nothing here is a paper claim, per the queue's
persona-oracle rule — filenames carry _dev). Purpose: find the optimization-pressure ceiling at which
best-of-N-by-persona-judge outputs stay form-valid and non-degenerate, so E3's future human sessions
never rate degenerate text.

Over the deep pool (goodhart_dev_7B.jsonl, 64 samples/prompt), a local 7B judge scores each sample on
a FIXED persona rubric (dev oracle). At each pressure N in {4,8,16,32,64} we take best-of-N by persona
score under two selectors (P12 mechanical check):
  (a) scalar   : argmax persona score
  (b) threshold: argmax persona score AMONG form-valid candidates (fall back to scalar if none valid)
and report, for the selected output: persona score, form-validity, and a degeneracy heuristic
(repetition + type-token ratio). The Goodhart turn is where persona score keeps rising while
form-validity / coherence falls. Threshold selection is predicted to hold form later than scalar (P12).

  python poetry/goodhart_dev.py --judge mlx-community/Qwen2.5-7B-Instruct-4bit
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_score import score_sample, extract_lines

PERSONA = ("You are a discerning poetry critic who prizes vivid imagery, emotional resonance, "
           "musicality, and originality. Rate the following poem from 1 to 10 on overall aesthetic "
           "quality. Reply with ONLY the number.")


def degeneracy(text):
    """0 = clean, 1 = fully degenerate. Blend of bigram-repetition rate and (1 - type-token ratio)."""
    w = re.findall(r"[a-z']+", (text or "").lower())
    if len(w) < 4:
        return 1.0
    bg = list(zip(w, w[1:]))
    rep = 1 - len(set(bg)) / len(bg)
    ttr = len(set(w)) / len(w)
    return round(0.5 * rep + 0.5 * (1 - ttr), 3)


def persona_score(model, tok, gen, poem):
    msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": poem}]
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
    out = gen(model, tok, prompt=pr, max_tokens=4, verbose=False)
    m = re.findall(r"10|[1-9]", out)
    return int(m[0]) if m else 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--pool", default="poetry/cache/goodhart_dev_7B.jsonl")
    ap.add_argument("--out", default="poetry/goodhart_dev_scores.json")
    a = ap.parse_args()
    from mlx_lm import load, generate as gen
    model, tok = load(a.judge)
    pool = [json.loads(l) for l in open(a.pool)]
    # score every sample once: persona, form-validity, degeneracy
    for r in pool:
        r["_persona"] = [persona_score(model, tok, gen, s) for s in r["samples"]]
        r["_valid"] = [int(score_sample(s, r)[0]) for s in r["samples"]]
        r["_degen"] = [degeneracy("\n".join(extract_lines(s, r["n_lines"]))) for s in r["samples"]]
    Ns = [4, 8, 16, 32, 64]
    rows = []
    for sel in ("scalar", "threshold"):
        for N in Ns:
            ps = vs = ds = 0; m = 0
            for r in pool:
                per, val, deg = r["_persona"][:N], r["_valid"][:N], r["_degen"][:N]
                if sel == "threshold" and any(val):
                    cand = [i for i in range(len(per)) if val[i]]
                else:
                    cand = list(range(len(per)))
                best = max(cand, key=lambda i: per[i])
                ps += per[best]; vs += val[best]; ds += deg[best]; m += 1
            rows.append({"selector": sel, "N": N, "n": m,
                         "persona": round(ps / m, 2), "form_valid_pct": round(100 * vs / m, 1),
                         "degeneracy": round(ds / m, 3)})
            print(f"  {sel:>9} N={N:>2}  persona={ps/m:4.2f}  form_valid={100*vs/m:5.1f}%  degen={ds/m:.3f}")
    json.dump({"dev_only": True, "persona_rubric": PERSONA, "rows": rows}, open(a.out, "w"), indent=2)
    print(f"\n[goodhart-dev] wrote {a.out} (DEV-ONLY — not a paper claim). "
          f"Turn = where persona rises while form_valid falls; compare scalar vs threshold (P12).")


if __name__ == "__main__":
    main()
