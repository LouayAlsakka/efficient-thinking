#!/usr/bin/env python
"""ET-IV checker 1c — first-species counterpoint validator (music21).

Note-against-note (first species): given a cantus firmus and a counter-voice as equal-length lists of
note names (e.g. ['C4','D4',...]), check the textbook rules and name every violation:
  - all harmonic intervals consonant (P1/P5/P8, M/m 3/6; NOT 2nds, 4ths, 7ths, tritone);
  - no parallel or direct perfect consonances (consecutive P5→P5, P8→P8, P1→P1) in similar motion;
  - no voice crossing (counter voice stays above the cantus throughout);
  - a proper cadence (penultimate an imperfect consonance — M6 or m3 — resolving to P8/P1).
This is the complete form evaluator for the music arm's cleanest cell; it is exact and model-free."""
from music21 import interval, note

_CONSONANT = {"P1", "m3", "M3", "P5", "m6", "M6", "P8"}
_PERFECT = {"P1", "P5", "P8"}


def _harmonic(cf, ct):
    """(simple interval name, crossing?) for the vertical pair; name taken on the ascending interval
    so octaves read 'P8' not 'P1'."""
    pcf, pct = note.Note(cf).pitch, note.Note(ct).pitch
    crossing = pct.midi < pcf.midi
    semis = abs(pct.midi - pcf.midi)
    if semis == 0:
        name = "P1"
    elif semis % 12 == 0:
        name = "P8"                                   # simpleName collapses octave→P1; keep it P8
    else:
        lo, hi = (pcf, pct) if pcf.midi <= pct.midi else (pct, pcf)
        name = interval.Interval(noteStart=note.Note(lo), noteEnd=note.Note(hi)).simpleName
    return name, crossing


def check_first_species(cantus, counter):
    """cantus, counter: equal-length lists of note names (counter is the upper voice). Returns
    {ok, violations:[...]} with each violation named and located (0-indexed note position)."""
    v = []
    if len(cantus) != len(counter):
        return {"ok": False, "violations": [f"length mismatch: cantus {len(cantus)} vs counter {len(counter)}"]}
    n = len(cantus)
    names = []
    for i, (cf, ct) in enumerate(zip(cantus, counter)):
        name, crossing = _harmonic(cf, ct)
        names.append(name)
        if crossing:
            v.append(f"pos {i}: voice crossing (counter '{ct}' below cantus '{cf}')")
        if name not in _CONSONANT:
            v.append(f"pos {i}: dissonant harmonic interval {name}")
    # parallel perfects: consecutive identical perfect interval with both voices moving
    for i in range(1, n):
        if names[i - 1] in _PERFECT and names[i - 1] == names[i]:
            moved_ct = note.Note(counter[i]).pitch.midi - note.Note(counter[i - 1]).pitch.midi
            moved_cf = note.Note(cantus[i]).pitch.midi - note.Note(cantus[i - 1]).pitch.midi
            if moved_ct != 0 and moved_cf != 0:
                v.append(f"pos {i-1}->{i}: parallel {names[i-1]}")
    # cadence: penultimate an imperfect consonance (M6 or m3) resolving to a final P8 or P1
    if n >= 2:
        if names[-1] not in {"P8", "P1"}:
            v.append(f"cadence: final interval {names[-1]} is not an octave/unison")
        elif names[-2] not in {"M6", "m3"}:
            v.append(f"cadence: penultimate {names[-2]} does not resolve by step to the final (expect M6 or m3)")
    return {"ok": not v, "violations": v, "n": n}
