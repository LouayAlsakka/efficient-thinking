#!/usr/bin/env python
"""ET-IV checker 1b — lyric-fit validator.

Given a text line and a melodic stress template (a sequence of beat strengths, 1 = strong/downbeat,
0 = weak), decide whether the line can be sung to that template: the syllable count must match the
number of notes, and stressed syllables must land on strong beats rather than fighting them. Reuses
1a's CMUdict scansion; monosyllables (flexible stress) never force a clash, polysyllabic lexical
stress must not sit against a strong beat while an unstressed syllable takes it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meter_rhyme import line_scan, line_syllable_range


def check_lyric_fit(line, template):
    """template: list of 0/1 beat strengths. Returns verdict dict; `ok` = count match AND stress align."""
    marks, oov_words, n_words = line_scan(line)
    n_syll = len(marks)
    smin, smax = line_syllable_range(line)
    count_ok = smin <= len(template) <= smax          # some pronunciation variant matches the note count
    # alignment: over syllables whose stress is known, strong beats should not be given to unstressed
    # syllables while a stressed syllable sits on a weak beat. Score = mean(stress) on strong beats −
    # mean(stress) on weak beats, using the template as the metrical grid.
    m = min(n_syll, len(template))
    strong = [marks[i] for i in range(m) if template[i] == 1 and marks[i] is not None]
    weak = [marks[i] for i in range(m) if template[i] == 0 and marks[i] is not None]
    smean = sum(strong) / len(strong) if strong else 0.0
    wmean = sum(weak) / len(weak) if weak else 0.0
    align = smean - wmean
    align_ok = align >= 0.0
    return {"line": line, "n_syllables": n_syll, "syllable_range": [smin, smax],
            "template_len": len(template), "count_ok": count_ok,
            "align_score": round(align, 3), "align_ok": align_ok,
            "ok": bool(count_ok and align_ok), "oov_words": oov_words}
