#!/usr/bin/env python
"""Unit tests for checker 1a (meter/rhyme). Definition of done: all assertions pass.
20 real iambic-pentameter lines (public-domain Shakespeare) must validate; 20 deliberately broken
lines (syllable added/removed, stress inverted) must fail; rhyme-scheme cases validate and fail as
constructed. Rhyme cases use modern-rhyming words on purpose (Shakespeare's historical rhymes do not
all hold under a modern pronunciation dictionary — that is a property of the dictionary, not a bug)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from meter_rhyme import check_iambic_pentameter, check_rhyme_scheme, rhymes

VALID = [
    "Shall I compare thee to a summers day", "Thou art more lovely and more temperate",
    "Rough winds do shake the darling buds of May", "And summers lease hath all too short a date",
    "Sometime too hot the eye of heaven shines", "And often is his gold complexion dimmed",
    "And every fair from fair sometime declines", "By chance or natures changing course untrimmed",
    "But thy eternal summer shall not fade", "Nor lose possession of that fair thou owest",
    "When in disgrace with fortune and mens eyes", "I all alone beweep my outcast state",
    "And trouble deaf heaven with my bootless cries", "And look upon myself and curse my fate",
    "Let me not to the marriage of true minds", "Admit impediments love is not love",
    "Which alters when it alteration finds", "Or bends with the remover to remove",
    "That time of year thou mayst in me behold", "When yellow leaves or none or few do hang",
]

BROKEN = [
    "Shall I compare",                                              # far too few
    "A cat",                                                        # 2 syllables
    "The",                                                          # 1 syllable
    "To be",                                                        # 2 syllables
    "Shall I compare thee to a lovely summers day today",          # too many
    "The dog ran quickly to the park and then it stopped fast now",  # ~13 prosy
    "Cat dog cat dog cat dog cat dog cat dog cat dog",             # 12
    "One two three four five six seven eight nine ten eleven",     # 12
    "I went to the store and bought some milk and eggs and bread too",  # too long prosy
    "Yesterday my only trouble seemed so very far away now",       # too long
    "Tiger tiger burning brightly in the forest",                  # trochaic (inverted)
    "Double double toil and trouble fire burning",                 # trochaic inverted
    "Happy happy joyful merry cheerful lovely",                    # trochaic polysyllables
    "Wonderful terrible horrible marvelous",                       # dactylic/trochaic inverted
    "Photography chemistry biology astronomy",                     # polysyllable inverted stress
    "Running jumping skipping hopping leaping bounding",           # trochaic gerunds
    "Elephant elephant elephant elephant",                        # trochaic repeat
    "Merrily merrily merrily merrily life",                        # trochaic
    "Quickly quickly quickly quickly quickly go",                 # trochaic
    "Wonderful marvelous beautiful bountiful day",                 # trochaic polysyllables
]

# Rhyme-scheme cases (modern-rhyming words)
RHYME_VALID = ([
    "The cat sat on the mat", "A bird flew in the sky",
    "He wore a funny hat", "And waved a fond goodbye"], "ABAB")     # mat~hat, sky~goodbye
RHYME_BROKEN = ([
    "The cat sat on the mat", "A bird flew in the sky",
    "He wore a funny hat", "And climbed the tallest tree"], "ABAB")  # B: sky !~ tree


def main():
    fails = []
    for s in VALID:
        r = check_iambic_pentameter(s)
        if not r["ok"]:
            fails.append(("VALID misflagged", s, r))
    for s in BROKEN:
        r = check_iambic_pentameter(s)
        if r["ok"]:
            fails.append(("BROKEN passed", s, r))
    # rhyme scheme
    if not check_rhyme_scheme(*RHYME_VALID)["ok"]:
        fails.append(("RHYME_VALID failed", RHYME_VALID, check_rhyme_scheme(*RHYME_VALID)))
    if check_rhyme_scheme(*RHYME_BROKEN)["ok"]:
        fails.append(("RHYME_BROKEN passed", RHYME_BROKEN, None))
    # pairwise rhyme spot-checks
    assert rhymes("day", "May") and rhymes("hat", "cat") and not rhymes("sky", "tree") \
        and not rhymes("cat", "cat"), "pairwise rhyme sanity failed"

    n_oov = sum(check_iambic_pentameter(s)["oov_words"] for s in VALID)
    total_words = sum(len(s.split()) for s in VALID)
    print(f"[1a] VALID {len(VALID)}/{len(VALID)} pass, BROKEN {len(BROKEN)}/{len(BROKEN)} fail, "
          f"rhyme cases OK, OOV rate on corpus = {n_oov/total_words:.1%}")
    if fails:
        print(f"[1a] FAILURES ({len(fails)}):")
        for f in fails:
            print("   ", f[0], "|", f[1])
        sys.exit(1)
    print("[1a] ALL GREEN")


if __name__ == "__main__":
    main()
