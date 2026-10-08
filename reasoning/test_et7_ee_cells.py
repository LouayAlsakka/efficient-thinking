#!/usr/bin/env python
"""Controls for the cells builder, codified.

    python reasoning/test_et7_ee_cells.py

The builder gained `--keep decisive|all` so E-D's intransitivity half has a corpus at all: a
3-cycle spans three policies, and keeping only pairs where exactly one side is right removes
the edge a triangle needs -- measured, zero closed triangles exist in the decisive corpus.

These controls were run once by hand when the flag landed. A one-time run is not a control: a
later edit to the builder could break byte-identity with the existing corpus and nothing would
catch it. They live beside the builder, in the repo that holds the cells it produces, so anyone
can re-run them against the instrument rather than trusting a report of them.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDER = os.path.join(HERE, "et7_ee_cells.py")
CANON = os.path.join(HERE, "et7_ee_cells.json")
RAN, FAILED = [], []


def ck(name, got, want):
    RAN.append(name)
    ok = got == want
    print("  %-4s %-64s %s  want %s" % ("ok" if ok else "FAIL", name, got, want))
    if not ok:
        FAILED.append(name)


def run(*args):
    p = subprocess.run([sys.executable, BUILDER] + list(args),
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


d = tempfile.mkdtemp(prefix="cellsctl_")
dec = os.path.join(d, "dec.json")
alls = os.path.join(d, "all.json")

print("=== CONTROL 1 — --keep decisive reproduces the canonical file BYTE-IDENTICALLY ===")
rc, out = run("--keep", "decisive", "--out", dec)
ck("1. the build succeeds", rc, 0)
ck("1b. and its bytes are the canonical file's", sha(dec), sha(CANON))

print("\n=== CONTROL 2 — --keep all is the canonical corpus PLUS exactly the excluded pairs ===")
rc, out = run("--keep", "all", "--out", alls)
ck("2. the build succeeds", rc, 0)
A = json.load(open(alls))
C = json.load(open(CANON))
cells, canon = A["cells"], C["cells"]
cen = C["census"]
n_dec = sum(1 for c in cells if c.get("decisive") is not False)
non = [c for c in cells if c.get("decisive") is False]
ck("2b. total cells == decisive + both-wrong + both-right, recomputed",
   len(cells), cen["DECISIVE"] + cen["excluded_both_wrong"] + cen["excluded_both_right"])
ck("2c. the decisive count is unchanged", n_dec, cen["DECISIVE"])
ck("2d. and exactly the excluded pairs were added",
   len(non), cen["excluded_both_wrong"] + cen["excluded_both_right"])
both = {}
for c in non:
    both[c.get("both")] = both.get(c.get("both"), 0) + 1
ck("2e. both-wrong count matches the census", both.get("wrong"), cen["excluded_both_wrong"])
ck("2f. both-right count matches the census", both.get("right"), cen["excluded_both_right"])
key = lambda c: (c["problem_index"], c["model_A"], c["model_B"])
amap = {key(c): c for c in cells if c.get("decisive") is not False}
missing = [k for k in (key(c) for c in canon) if k not in amap]
ck("2g. every canonical decisive cell is present", len(missing), 0)
# ✏️ this asserted the cells were IDENTICAL. --keep all now adds `flipped` to every cell --
# the presentation order, made a property of the cell so that interleaving cannot change it --
# so a decisive cell is the canonical cell PLUS one registered key. Asserting identity would
# have been wrong; asserting "agrees on every canonical field AND adds only the registered
# key" is stricter than either, because a stray extra field now fails too.
differ = [key(c) for c in canon
          if key(c) in amap and any(amap[key(c)].get(k) != v for k, v in c.items())]
ck("2h. ...agreeing on every canonical field", len(differ), 0)
added = set()
for c in canon:
    added |= set(amap[key(c)]) - set(c)
ck("2h2. ...and adding ONLY the registered presentation field", sorted(added), ["flipped"])
ck("2i. every non-decisive cell is MARKED, so none can be mistaken for decisive",
   all(c.get("decisive") is False and c.get("both") in ("right", "wrong") for c in non), True)
ck("2j. the canonical file itself carries no such marks (it is decisive-only)",
   any("decisive" in c for c in canon), False)

print("\n=== CONTROL 2b — the presentation order is a property of the CELL, not of the probe ===")
# 🔴 et7_ee_probe.py draws one flip per cell from a single sequential Random(0) in FILE
# ORDER, so interleaving the excluded pairs would have given every later decisive cell a
# different presentation than E-E used. These assert the builder's assignment instead.
import random as _r
_rd = _r.Random(0)
_canon_flips = [_rd.random() < 0.5 for _ in canon]
_amap = {key(c): c for c in cells if c.get("decisive") is not False}
_eq = sum(1 for i, c in enumerate(canon)
          if _amap[key(c)]["flipped"] == _canon_flips[i])
ck("2k. every decisive cell's flip is EXACTLY the one E-E's sequential Random(0) gave it",
   _eq, len(canon))
ck("2l. every non-decisive cell carries a flip too",
   sum(1 for c in cells if c.get("decisive") is False and "flipped" in c), len(non))
# and the decisive flips must not depend on the excluded pairs being there at all
rc2, _ = run("--keep", "all", "--out", os.path.join(d, "all2.json"))
_c2 = json.load(open(os.path.join(d, "all2.json")))["cells"]
_m2 = {key(c): c for c in _c2 if c.get("decisive") is not False}
ck("2m. the assignment is deterministic across builds",
   all(_m2[k]["flipped"] == v["flipped"] for k, v in _amap.items()), True)
# ✏️ this compared two 1,050-element lists and printed BOTH on failure AND on success --
# a control whose output no operator can read. It is a count now, and it duplicated 2k's
# assertion anyway; what it adds is that the excluded pairs' PRESENCE changes nothing.
ck("2n. the excluded pairs' presence disturbs no decisive flip",
   sum(1 for i, c in enumerate(canon) if _amap[key(c)]["flipped"] != _canon_flips[i]), 0)

print("\n=== CONTROL 3 — the refusals ===")
rc, out = run("--out", os.path.join(d, "x.json"))
ck("3. --keep omitted REFUSES", rc, 2)
ck("3b. ...naming the flag", "--keep" in out, True)
rc, out = run("--keep", "all", "--out", CANON)
ck("3c. --keep all onto the canonical path REFUSES", rc != 0, True)
ck("3d. ...saying why that file may not be overwritten", "provenance" in out, True)
ck("3e. and the canonical file is untouched by the attempt", sha(CANON), sha(dec))

print()
print("ALL %d CHECKS PASS" % len(RAN) if not FAILED
      else "%d of %d FAILED: %r" % (len(FAILED), len(RAN), FAILED))
sys.exit(1 if FAILED else 0)
