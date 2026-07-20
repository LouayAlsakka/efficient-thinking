#!/usr/bin/env python
"""ET-IV E4 scoring — counterpoint form-validity via checker 1c (scores P1 in music, the cleanest
verifier cell). Parses the counter-voice note names from each generation, scores against its cantus
firmus with the first-species checker, and reports per (policy, N):
  violation_free@1     : greedy counter-voice has zero rule violations (true-greedy)
  checker_selected@N   : checker picks the fewest-violation candidate; is it violation-free
  oracle_any@N         : any of N violation-free (coverage)
  format_failure_rate  : fraction of generations with fewer parseable notes than the cantus
  mean_violations      : violations-per-exercise distribution
If format failure is near-total, that is a scoping result (log it, don't hide it).

  python music/e4_score.py
"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkers"))
from counterpoint import check_first_species

_NOTE = re.compile(r"\b([A-Ga-g][#b\-]?[0-9])\b")


def parse_notes(text, n):
    """First n parseable note names from a completion; None if fewer than n (format failure)."""
    found = _NOTE.findall(text or "")
    norm = [m[0].upper() + m[1:] for m in found]
    return norm[:n] if len(norm) >= n else None


def score_sample(sample, rec):
    """Return (violation_free: bool, n_violations: int|None, format_ok: bool)."""
    notes = parse_notes(sample, rec["n_notes"])
    if notes is None:
        return False, None, False
    try:
        res = check_first_species(rec["cantus"], notes)
    except Exception:
        return False, None, False                              # unparseable note (e.g. bad octave) = format fail
    return res["ok"], len(res["violations"]), True


def _by_id(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}


def main():
    cells = []
    for temp_path in sorted(glob.glob("music/cache/e4_*.jsonl")):
        base = os.path.basename(temp_path)
        if base.endswith("_greedy.jsonl"):
            continue
        tag = base[3:-6]
        temp = _by_id(temp_path)
        greedy = _by_id(f"music/cache/e4_{tag}_greedy.jsonl")
        ids = list(temp)
        m = len(ids)
        # @1 greedy
        vf1 = fmt_fail = 0
        g_arr = []
        for i in ids:
            grec = greedy.get(i, {"samples": [temp[i]["samples"][0]]})
            vf, nv, ok = score_sample(grec["samples"][0], temp[i])
            vf1 += vf; fmt_fail += (not ok); g_arr.append(int(vf))
        row = {"policy": tag, "task": "counterpoint", "n": m,
               "violation_free_at1": round(100 * vf1 / m, 1),
               "format_failure_rate": round(100 * fmt_fail / m, 1),
               "greedy_used": bool(greedy)}
        per = {"g_vf": g_arr}
        for N in (4, 16):
            sel = orc = 0; viol_acc = 0; viol_n = 0
            sarr, oarr = [], []
            for i in ids:
                subs = temp[i]["samples"][:N]
                scored = [score_sample(s, temp[i]) for s in subs]
                valids = [(vf, nv) for vf, nv, ok in scored if ok]
                if valids:
                    best_nv = min(nv for _, nv in valids)
                    selv = int(any(vf for vf, nv in valids if nv == best_nv))
                    orcv = int(any(vf for vf, _ in valids))
                    viol_acc += best_nv; viol_n += 1
                else:
                    selv = orcv = 0
                sel += selv; orc += orcv; sarr.append(selv); oarr.append(orcv)
            row[f"checker_selected_at{N}"] = round(100 * sel / m, 1)
            row[f"oracle_any_at{N}"] = round(100 * orc / m, 1)
            per[f"sel{N}"] = sarr; per[f"orc{N}"] = oarr
            if N == 16:
                row["mean_violations_bestof16"] = round(viol_acc / viol_n, 2) if viol_n else None
        row["per_prompt"] = per
        cells.append(row)
        print(f"  {tag:>4} counterpoint  vf@1={row['violation_free_at1']:5.1f} "
              f"sel@4={row['checker_selected_at4']:5.1f} sel@16={row['checker_selected_at16']:5.1f} "
              f"orc@16={row['oracle_any_at16']:5.1f} | fmt_fail={row['format_failure_rate']:5.1f}% (n={m})")
    json.dump(cells, open("music/e4_scores.json", "w"), indent=2)
    print(f"\n[E4] wrote music/e4_scores.json ({len(cells)} rows).")


if __name__ == "__main__":
    main()
