#!/usr/bin/env python
"""Unit tests for checker 1c (first-species counterpoint). 10 valid exercises must pass; 10 with one
planted violation each must fail, and the named violation must appear in the report. Valid examples are
transpositions of a hand-verified first-species pair (transposition preserves every interval relation,
so all are valid by construction); broken examples plant one rule violation apiece."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from counterpoint import check_first_species
from music21 import note

CF = ["C4", "D4", "E4", "F4", "D4", "C4"]
CT = ["C5", "A4", "G4", "A4", "B4", "C5"]


def transpose(seq, dia):
    """Diatonic transposition by a named interval (e.g. 'P5') — preserves every interval spelling,
    so a valid exercise stays valid (unlike raw-semitone transposition, which enharmonically
    respells m3 as A2 etc.)."""
    return [note.Note(n).transpose(dia).nameWithOctave for n in seq]


# 10 valid: diatonic transpositions of the verified base (all interval names preserved exactly)
VALID = [(transpose(CF, d), transpose(CT, d))
         for d in ("P1", "M2", "M3", "P4", "P5", "P8", "m3", "m6", "M6", "m2")]

# 10 broken: one planted violation each, with the substring the report must name
BROKEN = [
    (["C4", "D4", "C4"], ["C5", "D5", "C5"], "parallel P8"),          # parallel octaves (both rise)
    (["C4", "D4", "E4"], ["G4", "A4", "B4"], "parallel P5"),          # parallel fifths (both rise)
    (CF, ["C5", "A4", "G4", "G4", "B4", "C5"], "dissonant"),          # F4 vs G4 = M2 dissonance at pos 3
    (CF, ["C5", "A4", "G4", "A4", "C4", "C5"], "voice crossing"),     # counter C4 below D4 at pos 4
    (["C4", "D4", "C4"], ["E4", "F4", "G4"], "not an octave/unison"),  # final P5, bad cadence
    (["C4", "F4", "C4"], ["E4", "C5", "C5"], "penultimate"),          # final P8, penultimate P5 (not M6/m3)
    (["C4", "D4", "C4"], ["C5", "G4", "C5"], "dissonant"),            # D4-G4 = P4 dissonance at pos 1
    (["C4", "D4", "E4"], ["D5", "E5", "F5"], "dissonant"),            # C4-D5 = M9(M2), all seconds
    (["C4", "D4", "C4"], ["F4", "G4", "F4"], "dissonant"),            # parallel fourths read as dissonant P4
    (["C4", "D4", "C4"], ["B4", "C5", "B4"], "dissonant"),            # C4-B4 = M7 dissonance at pos 0
]


def main():
    fails = []
    for cf, ct in VALID:
        r = check_first_species(cf, ct)
        if not r["ok"]:
            fails.append(("VALID misflagged", cf, ct, r["violations"]))
    for cf, ct, expect in BROKEN:
        r = check_first_species(cf, ct)
        if r["ok"]:
            fails.append(("BROKEN passed", cf, ct, expect))
        elif not any(expect in viol for viol in r["violations"]):
            fails.append(("BROKEN wrong-reason", cf, ct, f"expected '{expect}', got {r['violations']}"))
    print(f"[1c] VALID {len(VALID)}/{len(VALID)} pass, BROKEN {len(BROKEN)}/{len(BROKEN)} named-and-failed")
    if fails:
        print(f"[1c] FAILURES ({len(fails)}):")
        for f in fails:
            print("   ", f[0], "|", f[1:])
        sys.exit(1)
    print("[1c] ALL GREEN")


if __name__ == "__main__":
    main()
