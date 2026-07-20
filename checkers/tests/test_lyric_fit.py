#!/usr/bin/env python
"""Unit tests for checker 1b (lyric-fit). 10 aligned (line, template) pairs must fit; 10 misaligned
(stress fighting the beat, or wrong note count) must fail. Templates use polysyllabic words so lexical
stress is decisive; 1 = strong beat, 0 = weak."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lyric_fit import check_lyric_fit

# (line, template) — stressed syllable on the strong beat
ALIGNED = [
    ("wonderful", [1, 0, 0]),          # WON-der-ful
    ("beautiful", [1, 0, 0]),          # BEAU-ti-ful
    ("elephant", [1, 0, 0]),           # EL-e-phant
    ("banana", [0, 1, 0]),             # ba-NA-na
    ("computer", [0, 1, 0]),           # com-PU-ter
    ("remember", [0, 1, 0]),           # re-MEM-ber
    ("tomorrow", [0, 1, 0]),           # to-MOR-row
    ("photography", [0, 1, 0, 0]),     # pho-TOG-ra-phy
    ("celebration", [0, 0, 1, 0]),     # ce-le-BRA-tion
    ("happy little bird", [1, 0, 1, 0, 1]),  # HAP-py LIT-tle BIRD
]
# same words, beat now fights the stress, or wrong length
MISALIGNED = [
    ("wonderful", [0, 0, 1]),          # strong beat on unstressed final
    ("beautiful", [0, 1, 0]),          # strong beat on unstressed middle
    ("elephant", [0, 1, 0]),
    ("banana", [1, 0, 0]),             # strong beat on unstressed first
    ("computer", [1, 0, 0]),
    ("remember", [1, 0, 0]),            # strong beat on unstressed first
    ("tomorrow", [1, 0, 0]),
    ("photography", [1, 0, 0, 0]),     # strong beat on unstressed first
    ("celebration", [0, 1, 0, 0]),     # strong beat on unstressed 2nd; stress at pos2 on weak
    ("wonderful", [1, 0]),             # wrong note count (2 vs 3)
]


def main():
    fails = []
    for line, tmpl in ALIGNED:
        r = check_lyric_fit(line, tmpl)
        if not r["ok"]:
            fails.append(("ALIGNED misflagged", line, tmpl, r))
    for line, tmpl in MISALIGNED:
        r = check_lyric_fit(line, tmpl)
        if r["ok"]:
            fails.append(("MISALIGNED passed", line, tmpl, r))
    print(f"[1b] ALIGNED {len(ALIGNED)}/{len(ALIGNED)} fit, MISALIGNED {len(MISALIGNED)}/{len(MISALIGNED)} rejected")
    if fails:
        print(f"[1b] FAILURES ({len(fails)}):")
        for f in fails:
            print("   ", f[0], "|", f[1], f[2], "|", f[3])
        sys.exit(1)
    print("[1b] ALL GREEN")


if __name__ == "__main__":
    main()
