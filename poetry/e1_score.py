#!/usr/bin/env python
"""ET-IV E1 scoring — the form-validity frontier (scores P1).

Over the frozen poetry caches (poetry/cache/e1_<tag>.jsonl, N samples/prompt), score each sample's
form validity with the committed checkers (1a meter/rhyme for sonnet/villanelle, 1b lyric-fit for
lyric), then per (policy, task, N) cell report:
  form_valid@1        : first sample is fully form-valid (no search)
  verifier_selected@N : checker picks the best-scoring of N; is it valid (best-of-N through the checker)
  oracle_any@N        : any of N valid (coverage ceiling)
P1 predicts verifier_selected@N >> form_valid@1 for weak policies and the lift collapsing as base
competence grows (the ET-II crossover). Per-prompt outcomes are logged for exact McNemar on any flip.

  python poetry/e1_score.py
"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkers"))
from meter_rhyme import check_iambic_pentameter, check_rhyme_scheme
from lyric_fit import check_lyric_fit

_JUNK = re.compile(r"end of (sentence|text)|^note:|^here('s| is)|^sure[,!]|^i hope|^this |^\(", re.I)


def extract_lines(text, n_lines):
    """Pull the first n_lines verse-looking lines from a completion (strip numbering/quotes/markdown/junk)."""
    out = []
    for raw in (text or "").splitlines():
        s = raw.strip().strip('"').strip("*").strip()
        s = re.sub(r"^\s*\d+[\.\)]\s*", "", s)                 # leading "1." / "1)"
        s = re.sub(r"\s*\(end of (sentence|text)\)\s*$", "", s, flags=re.I)
        if not s or _JUNK.search(s):
            continue
        out.append(s)
        if len(out) == n_lines:
            break
    return out


def score_sample(sample, rec):
    """Return (valid, score, meter_valid) for one sample. `valid` = full form (meter AND rhyme);
    `meter_valid` = meter-only (ignore rhyme) — a robust secondary frontier when rhyme is scarce."""
    task = rec["task"]
    lines = extract_lines(sample, rec["n_lines"])
    if len(lines) < rec["n_lines"]:
        return False, 0.0, False                               # wrong number of lines = malformed
    if task in ("sonnet", "villanelle"):
        checks = [check_iambic_pentameter(ln) for ln in lines]
        n_meter = sum(c["ok"] for c in checks)
        meter_valid = n_meter == rec["n_lines"]
        rhyme = check_rhyme_scheme(lines, rec["scheme"])
        valid = meter_valid and rhyme["ok"]
        score = n_meter + (1.0 if rhyme["ok"] else 0.0) + sum(c["iamb_score"] for c in checks) / 10.0
        return valid, score, meter_valid
    else:  # lyric — single line, "meter" and "form" coincide
        c = check_lyric_fit(lines[0], rec["template"])
        v = bool(c["ok"])
        return v, (1.0 if v else 0.0) + c["align_score"] / 10.0, v


def main():
    Ns = [1, 4, 16]
    cells = []
    for path in sorted(glob.glob("poetry/cache/e1_*.jsonl")):
        tag = os.path.basename(path)[3:-6]                     # e1_<tag>.jsonl
        recs = [json.loads(l) for l in open(path)]
        by_task = {}
        for r in recs:
            by_task.setdefault(r["task"], []).append(r)
        for task, rs in sorted(by_task.items()):
            scored = []
            for r in rs:
                vs = [score_sample(s, r) for s in r["samples"]]
                scored.append(vs)                              # list of (valid,score) per sample
            for N in Ns:
                v1 = sel = orc = mv1 = msel = morc = 0; m = len(scored)
                per = {"valid1": [], "sel": [], "orc": []}
                for vs in scored:
                    sub = vs[:N]
                    valid1 = sub[0][0]
                    best = max(range(len(sub)), key=lambda i: sub[i][1])
                    selv = sub[best][0]
                    orcv = any(v for v, _, _ in sub)
                    v1 += valid1; sel += selv; orc += orcv
                    mv1 += sub[0][2]; msel += sub[best][2]; morc += any(mv for _, _, mv in sub)
                    per["valid1"].append(int(valid1)); per["sel"].append(int(selv)); per["orc"].append(int(orcv))
                cells.append({"policy": tag, "task": task, "N": N, "n": m,
                              "form_valid_at1": round(100 * v1 / m, 1),
                              "verifier_selected_atN": round(100 * sel / m, 1),
                              "oracle_any_atN": round(100 * orc / m, 1),
                              "meter_valid_at1": round(100 * mv1 / m, 1),
                              "meter_selected_atN": round(100 * msel / m, 1),
                              "meter_oracle_atN": round(100 * morc / m, 1),
                              "per_prompt": per})
                print(f"  {tag:>4} {task:<10} N={N:>2}  valid@1={100*v1/m:5.1f} sel@N={100*sel/m:5.1f} "
                      f"orc@N={100*orc/m:5.1f} | meter v@1={100*mv1/m:5.1f} sel@N={100*msel/m:5.1f} (n={m})")
    json.dump(cells, open("poetry/e1_scores.json", "w"), indent=2)
    print(f"\n[E1] wrote poetry/e1_scores.json ({len(cells)} cells)")


if __name__ == "__main__":
    main()
