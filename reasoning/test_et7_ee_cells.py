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
differ = [key(c) for c in canon if key(c) in amap and amap[key(c)] != c]
ck("2h. ...and identical in content, field for field", len(differ), 0)
ck("2i. every non-decisive cell is MARKED, so none can be mistaken for decisive",
   all(c.get("decisive") is False and c.get("both") in ("right", "wrong") for c in non), True)
ck("2j. the canonical file itself carries no such marks (it is decisive-only)",
   any("decisive" in c for c in canon), False)

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
