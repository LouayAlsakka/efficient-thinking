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

MEMORIZATION_THRESHOLD = 0.5      # committed before results: >50% of a generation's 4-grams in canon = memorized


def _ngrams(text, n=4):
    w = re.findall(r"[a-z']+", (text or "").lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= n else set()


_CANON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "canon_reference.txt")
_CANON_4GRAMS = _ngrams(open(_CANON_PATH).read()) if os.path.exists(_CANON_PATH) else set()


def canon_overlap(text):
    """Fraction of a generation's 4-grams that appear verbatim in the canon corpus (memorization signal)."""
    g = _ngrams(text)
    return (len(g & _CANON_4GRAMS) / len(g)) if g else 0.0


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
    """Return (valid, score, meter_valid, oov_rate) for one sample. `valid` = full form (meter AND
    rhyme); `meter_valid` = meter-only (ignore rhyme, robust secondary frontier)."""
    task = rec["task"]
    lines = extract_lines(sample, rec["n_lines"])
    if len(lines) < rec["n_lines"]:
        return False, 0.0, False, 0.0                          # wrong number of lines = malformed
    if task in ("sonnet", "villanelle"):
        checks = [check_iambic_pentameter(ln) for ln in lines]
        n_meter = sum(c["ok"] for c in checks)
        meter_valid = n_meter == rec["n_lines"]
        rhyme = check_rhyme_scheme(lines, rec["scheme"])
        valid = meter_valid and rhyme["ok"]
        score = n_meter + (1.0 if rhyme["ok"] else 0.0) + sum(c["iamb_score"] for c in checks) / 10.0
        oov = sum(c["oov_rate"] for c in checks) / len(checks)
        return valid, score, meter_valid, oov
    else:  # lyric — single line, meter and form coincide
        c = check_lyric_fit(lines[0], rec["template"])
        v = bool(c["ok"])
        return v, (1.0 if v else 0.0) + c["align_score"] / 10.0, v, 0.0


def _by_id(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}


def main():
    cells = []
    for temp_path in sorted(glob.glob("poetry/cache/e1_*.jsonl")):
        base = os.path.basename(temp_path)
        if base.endswith("_greedy.jsonl"):
            continue
        tag = base[3:-6]                                       # e1_<tag>.jsonl
        temp = _by_id(temp_path)
        greedy = _by_id(f"poetry/cache/e1_{tag}_greedy.jsonl")  # true-greedy @1 (may be absent → falls back)
        tasks = sorted({r["task"] for r in temp.values()})
        for task in tasks:
            ids = [i for i, r in temp.items() if r["task"] == task]
            row = {"policy": tag, "task": task, "n": len(ids),
                   "memorization_threshold": MEMORIZATION_THRESHOLD}
            # --- @1: true greedy, memorized items excluded from validity ---
            g_valid, g_meter, g_ov, oov_acc, n_g, n_mem = [], [], [], 0.0, 0, 0
            for i in ids:
                grec = greedy.get(i, {"samples": [temp[i]["samples"][0]], **temp[i]})  # fallback: 1st temp sample
                s = grec["samples"][0]
                v, _, mv, oov = score_sample(s, temp[i])
                ov = canon_overlap(s); mem = ov > MEMORIZATION_THRESHOLD
                n_g += 1; n_mem += mem; oov_acc += oov; g_ov.append(ov)
                g_valid.append(None if mem else int(v)); g_meter.append(None if mem else int(mv))
            kept = [x for x in g_valid if x is not None]
            row["form_valid_at1"] = round(100 * sum(kept) / len(kept), 1) if kept else 0.0
            mkept = [x for x in g_meter if x is not None]
            row["meter_valid_at1"] = round(100 * sum(mkept) / len(mkept), 1) if mkept else 0.0
            row["oov_rate"] = round(oov_acc / n_g, 3) if n_g else 0.0
            row["oov_flag"] = row["oov_rate"] > 0.10
            row["memorized_frac"] = round(n_mem / n_g, 3) if n_g else 0.0
            row["greedy_used"] = bool(greedy)
            # --- @N: temperature samples, memorized samples excluded from the candidate pool ---
            per = {"g_valid": [x if x is not None else 0 for x in g_valid]}
            for N in (4, 16):
                sel = orc = msel = 0
                selv_arr, orcv_arr = [], []
                for i in ids:
                    subs = temp[i]["samples"][:N]
                    scored = [(*score_sample(s, temp[i]), canon_overlap(s)) for s in subs]
                    pool = [x for x in scored if x[4] <= MEMORIZATION_THRESHOLD] or scored
                    best = max(range(len(pool)), key=lambda k: pool[k][1])
                    selv = int(pool[best][0]); orcv = int(any(x[0] for x in pool))
                    msel += int(pool[best][2])
                    sel += selv; orc += orcv; selv_arr.append(selv); orcv_arr.append(orcv)
                m = len(ids)
                row[f"verifier_selected_at{N}"] = round(100 * sel / m, 1)
                row[f"oracle_any_at{N}"] = round(100 * orc / m, 1)
                row[f"meter_selected_at{N}"] = round(100 * msel / m, 1)
                per[f"sel{N}"] = selv_arr; per[f"orc{N}"] = orcv_arr
            row["per_prompt"] = per
            cells.append(row)
            print(f"  {tag:>4} {task:<10} valid@1={row['form_valid_at1']:5.1f} "
                  f"sel@4={row['verifier_selected_at4']:5.1f} sel@16={row['verifier_selected_at16']:5.1f} "
                  f"orc@16={row['oracle_any_at16']:5.1f} | meter@1={row['meter_valid_at1']:5.1f} "
                  f"oov={row['oov_rate']:.2f}{'!' if row['oov_flag'] else ''} mem={row['memorized_frac']:.2f} (n={m})")
    json.dump(cells, open("poetry/e1_scores.json", "w"), indent=2)
    print(f"\n[E1] wrote poetry/e1_scores.json ({len(cells)} rows). Greedy@1, memorized excluded (>{MEMORIZATION_THRESHOLD} 4-gram overlap).")


if __name__ == "__main__":
    main()
