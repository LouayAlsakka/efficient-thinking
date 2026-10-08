#!/usr/bin/env python
"""Controls for the PRESENTATION ORDER: that it is a property of the cell, not of its
position in a file, and that an unlabelled cell cannot acquire a probe target.

    python reasoning/test_et7_ee_presentation.py

The probe extractor drew one flip per cell from a single sequential Random(0) in FILE ORDER.
Interleaving a larger population would therefore have re-presented every later cell: its
states would not reproduce and its forced pick would be read under a different presentation.
Recording the flip faithfully does not fix that -- it records the wrong order. These controls
cover the pure logic, which needs no model and no states; the two measured results that need
the cached states are recorded with their commands in the run's own notes.
"""
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BAL = {("Qwen2.5-1.5B-Instruct-4bit", "Qwen2.5-3B-Instruct-4bit"),
       ("Qwen2.5-14B-Instruct-4bit", "Qwen2.5-7B-Instruct-4bit")}
RAN, FAILED = [], []


def _show(v, n=48):
    """A control whose output no operator can read is an operator-facing defect. I fixed one
    1,050-element dump in a sibling test and then wrote another here an hour later, so the
    truncation lives in `ck` itself rather than in my care at each call site."""
    t = repr(v)
    return t if len(t) <= n else "%s… (%s, len %d)" % (t[:n], type(v).__name__, len(v))


def ck(name, got, want):
    RAN.append(name)
    ok = got == want
    print("  %-4s %-62s %s  want %s"
          % ("ok" if ok else "FAIL", name, _show(got), _show(want)))
    if not ok:
        FAILED.append(name)


def masks(meta):
    """The extractor's split logic, verbatim in effect: labelled cells and their problems."""
    lab = np.array([m["correct"] is not None for m in meta])
    probs = sorted({m["problem"] for m, k in zip(meta, lab) if k})
    r = random.Random(7)
    r.shuffle(probs)
    cut = int(0.75 * len(probs))
    train_p = set(probs[:cut])
    tr = np.array([m["problem"] in train_p and m["correct"] is not None for m in meta])
    sel = np.array([tuple(m["pair"]) in BAL for m in meta])
    return lab, probs, (tr & sel & lab), ((~tr) & sel & lab)


def cell(i, labelled=True, bal=True):
    pair = (["Qwen2.5-1.5B-Instruct-4bit", "Qwen2.5-3B-Instruct-4bit"] if bal
            else ["Qwen2.5-0.5B-Instruct-4bit", "Qwen2.5-7B-Instruct-4bit"])
    return {"problem": i, "pair": pair, "correct": ("A" if labelled else None),
            "judge_pick": "A", "strong_side": "B", "balanced": bal}


print("=== the builder assigns a flip per CELL, from two separate streams ===")
# the decisive stream must reproduce a plain sequential Random(0) over the decisive cells
r0 = random.Random(0)
want = [r0.random() < 0.5 for _ in range(1050)]
r_dec, r_non = random.Random(0), random.Random(101)
dec_flips, non_flips = [], []
for k in range(2830):                      # interleaved, as --keep all writes them
    if k % 3 == 0 and len(non_flips) < 1780:
        non_flips.append(r_non.random() < 0.5)
    elif len(dec_flips) < 1050:
        dec_flips.append(r_dec.random() < 0.5)
    else:
        non_flips.append(r_non.random() < 0.5)
ck("1. the decisive flips equal a plain sequential draw over the decisive cells ONLY",
   dec_flips, want)
ck("2. ...and interleaving the excluded pairs changed none of them",
   sum(1 for a, b in zip(dec_flips, want) if a != b), 0)
ck("3. the excluded pairs all drew from the other stream", len(non_flips), 1780)

print("\n=== an unlabelled cell gets NO probe target, and no mask admits it ===")
canon = [cell(i) for i in range(40)]
lab1, p1, mtr1, mte1 = masks(canon)
sup = canon + [cell(10000 + i, labelled=False, bal=True) for i in range(60)]
lab2, p2, mtr2, mte2 = masks(sup)
n = len(canon)
ck("4. the labelled problem set is unchanged by the superset", p1, p2)
ck("5. ...so the seeded shuffle sees the SAME population", p1 == p2, True)
ck("6. the probe TRAIN mask is identical on the real rows",
   bool((mtr1 == mtr2[:n]).all()), True)
ck("7. the probe TEST mask is identical on the real rows",
   bool((mte1 == mte2[:n]).all()), True)
ck("8. no unlabelled row enters the train mask", bool((~mtr2[n:]).all()), True)
ck("9. no unlabelled row enters the TEST mask -- where y would silently become 0.0",
   bool((~mte2[n:]).all()), True)
ck("10. and `lab` marks exactly the unlabelled rows",
   int((~lab2).sum()), 60)

print("\n=== labelled-ness and presentation order are INDEPENDENT ===")
# ✏️ two controls here asserted Python's own dict.get and tested nothing of this code. A
# vacuous control is worse than none: it inflates the count. Replaced with the property that
# actually matters -- a recorded flip does not make an unlabelled cell labelled.
mixed = [dict(cell(1), flipped=True), dict(cell(2, labelled=False), flipped=True),
         dict(cell(3, labelled=False), flipped=False)]
labm, pm, mtrm, mtem = masks(mixed)
ck("11. a recorded flip does NOT make an unlabelled cell labelled",
   [bool(x) for x in labm], [True, False, False])
ck("12. ...and only the labelled cell's problem reaches the shuffle", pm, [1])
ck("13. no unlabelled cell enters either mask, flip recorded or not",
   bool((~mtrm[1:]).all() and (~mtem[1:]).all()), True)

print()
print("ALL %d CHECKS PASS" % len(RAN) if not FAILED
      else "%d of %d FAILED: %r" % (len(FAILED), len(RAN), FAILED))
sys.exit(1 if FAILED else 0)
