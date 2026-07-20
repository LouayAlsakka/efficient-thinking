#!/usr/bin/env python
"""ET-IV checker 1a — meter/rhyme validator (CMUdict).

Form component for the poetry arm (E1). Checkable-where-possible: syllable count and stress alignment
from a pronunciation dictionary, rhyme from terminal-rime match. OOV words fall back to a vowel-group
syllabifier and are marked stress-unknown (flexible); the OOV rate is reported so a caller can flag a
line whose form verdict rests on guesses.

Design stance (calibrated against real iambic pentameter, which is NOT perfectly regular):
- syllable count is the hard gate (a real pentameter line is 10, or 11 with a feminine ending);
- meter is a *soft* iambic-alternation score — mean stress on the five strong (even) positions minus
  mean stress on the weak (odd) positions — using only syllables whose stress the dictionary knows.
  Real lines score clearly positive; a stress-inverted (trochaic) line scores negative. Monosyllabic
  function words carry unreliable lexical stress, so the score is computed over dictionary-known
  content stress and thresholded leniently, not demanded foot-by-foot.
Every check returns a structured dict; nothing here calls a model.
"""
import re
import cmudict

_CMU = cmudict.dict()
_VOWELS = "aeiouy"


def _clean(word):
    return re.sub(r"[^a-z']", "", word.lower())


def _fallback_syllables(word):
    """Vowel-group count for OOV words: runs of vowels = 1 syllable, minus common silent-e."""
    w = _clean(word)
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and w[-2] not in _VOWELS:   # silent terminal e (but not 'the','she')
        n -= 1
    return max(1, n)


def word_stresses(word):
    """Return (stress_list, oov). stress_list: 1 for stressed (primary/secondary), 0 unstressed,
    None where unknown (OOV syllable, stress cannot be asserted)."""
    w = _clean(word)
    if not w:
        return [], False
    prons = _CMU.get(w)
    if not prons:
        return [None] * _fallback_syllables(w), True
    phones = prons[0]
    stresses = [int(p[-1] != "0") for p in phones if p[-1].isdigit()]
    return stresses, False


def word_syllable_range(word):
    """(min, max) syllable count over all CMUdict pronunciation variants (poetic license: a word like
    'temperate' or 'heaven' scans as 2 or 3). OOV → fallback count as a point range."""
    w = _clean(word)
    if not w:
        return (0, 0)
    prons = _CMU.get(w)
    if not prons:
        n = _fallback_syllables(w)
        return (n, n)
    counts = [sum(1 for p in ph if p[-1].isdigit()) for ph in prons]
    return (min(counts), max(counts))


def line_syllable_range(line):
    lo = hi = 0
    for tok in line.split():
        if _clean(tok):
            a, b = word_syllable_range(tok)
            lo += a; hi += b
    return lo, hi


def line_scan(line):
    """Scan a line → per-syllable stress marks (0/1/None) and OOV word count."""
    marks, oov_words, n_words = [], 0, 0
    for tok in line.split():
        if not _clean(tok):
            continue
        n_words += 1
        st, oov = word_stresses(tok)
        marks.extend(st)
        oov_words += oov
    return marks, oov_words, n_words


def check_iambic_pentameter(line, allow_feminine=True):
    """Verdict dict for a single iambic-pentameter line. `ok` = syllable gate AND meter gate."""
    marks, oov_words, n_words = line_scan(line)
    n_syll = len(marks)
    # syllable gate: a target count must be reachable under some pronunciation variant (poetic license)
    smin, smax = line_syllable_range(line)
    targets = {10, 11} if allow_feminine else {10}
    syll_ok = any(smin <= t <= smax for t in targets)
    # iambic alternation over the first 10 metrical positions, dictionary-known stress only
    strong = [marks[i] for i in range(1, min(n_syll, 10), 2) if marks[i] is not None]   # positions 2,4,6,8,10
    weak = [marks[i] for i in range(0, min(n_syll, 10), 2) if marks[i] is not None]      # positions 1,3,5,7,9
    strong_mean = sum(strong) / len(strong) if strong else 0.0
    weak_mean = sum(weak) / len(weak) if weak else 0.0
    iamb_score = strong_mean - weak_mean
    meter_ok = iamb_score >= 0.0                     # weak positions must not out-stress strong (inverted → <0)
    oov_rate = oov_words / n_words if n_words else 0.0
    return {"line": line, "n_syllables": n_syll, "syllable_range": [smin, smax], "syllable_ok": syll_ok,
            "iamb_score": round(iamb_score, 3), "meter_ok": meter_ok,
            "ok": bool(syll_ok and meter_ok), "oov_words": oov_words, "oov_rate": round(oov_rate, 3)}


def rhyme_key(word):
    """Terminal rime = from the last stressed vowel to the end (CMUdict). None if OOV."""
    w = _clean(word)
    prons = _CMU.get(w)
    if not prons:
        return None
    phones = prons[0]
    idx = [i for i, p in enumerate(phones) if p[-1].isdigit()]
    if not idx:
        return None
    last_stressed = next((i for i in reversed(idx) if phones[i][-1] in "12"), idx[-1])
    return " ".join(re.sub(r"\d", "", p) for p in phones[last_stressed:])


def rhymes(w1, w2):
    """True iff the two words share a terminal rime and are not identical."""
    k1, k2 = rhyme_key(w1), rhyme_key(w2)
    if k1 is None or k2 is None:
        return False
    return k1 == k2 and _clean(w1) != _clean(w2)


def _last_word(line):
    toks = [t for t in line.split() if _clean(t)]
    return toks[-1] if toks else ""


def check_rhyme_scheme(lines, scheme):
    """scheme e.g. 'ABAB': lines with the same letter must rhyme, different letters must not.
    Returns {ok, violations:[...]}. Ground truth for the form gate on stanza rhyme."""
    groups = {}
    for ln, lab in zip(lines, scheme):
        groups.setdefault(lab, []).append(_last_word(ln))
    violations = []
    labs = list(groups)
    for lab in labs:                                   # same-label lines must all rhyme
        ws = groups[lab]
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                if not rhymes(ws[i], ws[j]):
                    violations.append(f"{lab}: '{ws[i]}' !~ '{ws[j]}' (should rhyme)")
    for a in range(len(labs)):                         # different-label lines must not rhyme
        for b in range(a + 1, len(labs)):
            for wa in groups[labs[a]]:
                for wb in groups[labs[b]]:
                    if rhymes(wa, wb):
                        violations.append(f"{labs[a]}/{labs[b]}: '{wa}' ~ '{wb}' (should not rhyme)")
    return {"ok": not violations, "violations": violations, "scheme": scheme}


if __name__ == "__main__":                             # quick manual calibration
    import sys
    for ln in sys.stdin:
        ln = ln.strip()
        if ln:
            print(check_iambic_pentameter(ln))
