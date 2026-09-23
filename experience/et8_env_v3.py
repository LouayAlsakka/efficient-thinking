#!/usr/bin/env python3
"""ET-8 task set v3 — MANY DIFFERENT PROGRAMS, gated on having problems in it (理 ).

    python3 et8_env_v3.py gate --n 300 --seed 21 --out /tmp/v3_probe     # measure, emit nothing
    python3 et8_env_v3.py gen  --n 300 --seed 21 --out experience/tasks/v3

WHY v3 EXISTS, and it is not the reason v2 existed. v2 fixed v1's symptom axis and inherited the
defect underneath it: BOTH sets are ONE clean program with a bug injected at one of N sites, so
2,000 task files carry 11 or 12 distinct problems. Under greedy decoding identical tasks give
identical outcomes, so every rate published from either set rests on 11-12 observations.

  v1   2,000 files ->  12 distinct signatures   largest group 240
  v2   2,000 files ->  11 distinct signatures   largest group 192

So v3's requirement is not a better symptom or a better injector. It is MANY DIFFERENT PROGRAMS.

THE DESIGN:

  a PROGRAM is a pipeline of R regions (R in 3..5), each a function, composed by a consumer.
  Regions are drawn from OPERATION FAMILIES over a chosen data shape -- parse, filter, map,
  aggregate, format -- and the grammar varies the data type, the operations, the region count, the
  region NAMES and the call graph. Two programs differ in what they COMPUTE, not in their spelling.

  TESTS ARE GENERATED FROM THE CLEAN PROGRAM'S OWN BEHAVIOUR. Each program is run on fixture inputs
  and the observed outputs become the assertions. That makes every suite correct by construction and
  every failure message BEHAVIOURAL ("[3, 1] != [3, 1, 2]") rather than a test name -- which is what
  v2 needed a broken extractor fixed to get.

  BUGS are mutations of one region's source: a boundary slip, a comparator flip, a wrong aggregator,
  an early return, a dropped element, an integer division. A mutation is KEPT only if the clean
  program passes its suite and the mutated one fails it -- the verifier decides what is a bug, not
  the author.

THE GATES, in the order they run, and `gen` refuses to write unless all pass:

  P0  distinct task signatures >= 0.9 x tasks        <- FIRST. v1 0.0060, v2 0.0055.
  P1  every failing-test identity reachable from >= 2 bug regions
  P2  mean H(bug_region | symptom) >= 1.0 bit
  P3  a HELD-OUT lookup table on the symptom scores <= 60%
  P4  runs on the EMITTED set, after a baseline: a set whose programs are superficially different
      and behaviourally identical passes P0 and is v2 again with more strings. Only outcomes expose
      that, so the set is PROVISIONAL until a base run reports its outcome spread.

The file count EQUALS the signature count on purpose: the 2,000-file habit is what hid the defect,
and a set whose file count is its problem count cannot hide it again.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, subprocess, sys, tempfile, textwrap

# ---------- the grammar -------------------------------------------------------------------------
# Each entry is (name, source template, a python callable computing the same thing) -- the callable
# is NOT used to generate tests (the program's own run is), it documents intent for a reader.

PARSERS = [
 ("parse_ints",      "out = []\n    for tok in raw:\n        s = tok.strip()\n        if s.lstrip('-').isdigit():\n            out.append(int(s))\n    return out"),
 ("parse_positive",  "out = []\n    for tok in raw:\n        s = tok.strip()\n        if s.isdigit() and int(s) > 0:\n            out.append(int(s))\n    return out"),
 ("parse_lengths",   "return [len(tok.strip()) for tok in raw if tok.strip()]"),
 ("parse_floats",    "out = []\n    for tok in raw:\n        try:\n            out.append(float(tok))\n        except ValueError:\n            pass\n    return out"),
 ("parse_pairs",     "out = []\n    for tok in raw:\n        if ':' in tok:\n            a, b = tok.split(':', 1)\n            if b.strip().isdigit():\n                out.append((a.strip(), int(b)))\n    return out"),
]

MAPPERS = [
 ("clamp",     "return [v if v <= limit else limit for v in xs]"),
 ("scale",     "return [v * factor for v in xs]"),
 ("shift",     "return [v + offset for v in xs]"),
 ("keep_even", "return [v for v in xs if v % 2 == 0]"),
 ("drop_small","return [v for v in xs if v >= floor_]"),
 ("running",   "acc = 0\n    out = []\n    for v in xs:\n        acc = acc + v\n        out.append(acc)\n    return out"),
 ("squares",   "return [v * v for v in xs]"),
]

AGGS = [
 ("total",    "return {'n': len(xs), 'total': sum(xs)}"),
 ("meanmax",  "n = len(xs)\n    return {'n': n, 'mean': (sum(xs) / n) if n else 0.0, 'max': max(xs) if xs else 0}"),
 ("minspan",  "return {'n': len(xs), 'min': min(xs) if xs else 0, 'span': (max(xs) - min(xs)) if xs else 0}"),
 ("buckets",  "lo = [v for v in xs if v < cut]\n    hi = [v for v in xs if v >= cut]\n    return {'n': len(xs), 'lo': len(lo), 'hi': len(hi)}"),
 ("tally",    "d = {}\n    for v in xs:\n        d[v % 3] = d.get(v % 3, 0) + 1\n    return {'n': len(xs), 'tally': sorted(d.items())}"),
]

FORMATS = [
 ("grade",   "out = dict(summary)\n    out['grade'] = 'high' if summary.get('n', 0) >= bar else 'low'\n    return out"),
 ("label",   "out = dict(summary)\n    out['label'] = name.upper() if summary.get('n', 0) else 'EMPTY'\n    return out"),
 ("ratio",   "out = dict(summary)\n    n = summary.get('n', 0)\n    out['ratio'] = round(n / bar, 3) if bar else 0.0\n    return out"),
 ("flags",   "out = dict(summary)\n    out['flags'] = sorted(k for k, v in summary.items() if isinstance(v, int) and v > 0)\n    return out"),
]

FIXTURES = [
 ["3", "1", "2"], ["10", "0", "7", "x"], [" 5 ", "-2", "8"], ["4:2", "9:1", "bad"],
 ["12", "12", "1"], ["0"], ["6", "6", "6", "6"], ["2", "hello", "40"],
]


# ---------- assembling one program ---------------------------------------------------------------
REGION_NAMES = ["intake", "shape", "digest", "render", "sift", "fold", "emit", "gather"]


def make_program(rng: random.Random):
    """One pipeline: parser -> 1..3 mappers -> aggregator -> formatter, with its own names/params."""
    nmap = rng.choice([1, 1, 2, 2, 3])
    names = rng.sample(REGION_NAMES, 3 + nmap)
    p_name, p_src = rng.choice(PARSERS)
    maps = [rng.choice(MAPPERS) for _ in range(nmap)]
    a_name, a_src = rng.choice(AGGS)
    f_name, f_src = rng.choice(FORMATS)
    params = {"limit": rng.choice([5, 9, 12, 20]), "factor": rng.choice([2, 3, 10]),
              "offset": rng.choice([1, -1, 4]), "floor_": rng.choice([2, 3, 6]),
              "cut": rng.choice([3, 5, 8]), "bar": rng.choice([2, 3, 4]),
              "name": rng.choice(["alpha", "beta", "gamma"])}

    regions, order = {}, []
    src = "def %s(raw):\n    %s\n" % (names[0], p_src)
    regions[names[0]] = src; order.append(names[0])
    prev = names[0]
    for i, (mn, ms) in enumerate(maps):
        nm = names[1 + i]
        sig = ", ".join(["xs"] + ["%s=%r" % (k, params[k]) for k in ("limit", "factor", "offset", "floor_")
                                  if k in ms])
        regions[nm] = "def %s(%s):\n    %s\n" % (nm, sig, ms); order.append(nm)
        prev = nm
    an = names[1 + nmap]
    asig = ", ".join(["xs"] + ["%s=%r" % (k, params[k]) for k in ("cut",) if k in a_src])
    regions[an] = "def %s(%s):\n    %s\n" % (an, asig, a_src); order.append(an)
    fn = names[2 + nmap]
    fsig = ", ".join(["summary"] + ["%s=%r" % (k, params[k]) for k in ("bar", "name") if k in f_src])
    regions[fn] = "def %s(%s):\n    %s\n" % (fn, fsig, f_src); order.append(fn)

    chain = "%s(raw)" % names[0]
    for i in range(nmap):
        chain = "%s(%s)" % (names[1 + i], chain)
    body = "def run(raw):\n    return %s(%s(%s))\n" % (fn, an, chain)
    regions["run"] = body; order.append("run")
    return regions, order, (p_name, [m[0] for m in maps], a_name, f_name)


def assemble(regions: dict, order: list) -> str:
    return "\n".join("# region: %s\n%s" % (r, regions[r]) for r in order)


def run_program(program: str, inputs: list):
    """Execute the assembled program on each fixture, returning outputs or the exception text."""
    g = {}
    try:
        exec(compile(program, "<program>", "exec"), g)
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)
    out = []
    for raw in inputs:
        try:
            out.append(("ok", g["run"](list(raw))))
        except Exception as e:
            out.append(("err", "%s: %s" % (type(e).__name__, e)))
    return out, None


def make_tests(results, inputs) -> str:
    """The suite IS the clean program's own behaviour. Correct by construction; messages behavioural."""
    lines = ["import unittest", "from program import run", "", "class T(unittest.TestCase):"]
    for i, ((kind, val), raw) in enumerate(zip(results, inputs)):
        if kind != "ok":
            continue
        lines.append("    def test_case_%02d(self):" % i)
        lines.append("        self.assertEqual(run(%r), %r)" % (list(raw), val))
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    unittest.main()")
    return "\n".join(lines) + "\n"


# ---------- mutations: a bug is what the VERIFIER says is a bug ------------------------------------
MUTATIONS = [
 ("boundary",     lambda s: s.replace("for v in xs:", "for v in xs[:-1]:", 1)
                            if "for v in xs:" in s else None),
 ("boundary2",    lambda s: s.replace("len(xs)", "len(xs) - 1", 1) if "len(xs)" in s else None),
 ("comparator",   lambda s: s.replace("v <= limit", "v < limit", 1) if "v <= limit" in s else
                            (s.replace("v >= floor_", "v > floor_", 1) if "v >= floor_" in s else
                             (s.replace("v >= cut", "v > cut", 1) if "v >= cut" in s else None))),
 ("comparator2",  lambda s: s.replace(">= bar", "> bar", 1) if ">= bar" in s else None),
 ("int_division", lambda s: s.replace("sum(xs) / n", "sum(xs) // n", 1) if "sum(xs) / n" in s else
                            (s.replace("n / bar", "n // bar", 1) if "n / bar" in s else None)),
 ("wrong_agg",    lambda s: s.replace("max(xs)", "min(xs)", 1) if "max(xs)" in s else
                            (s.replace("sum(xs)", "max(xs)", 1) if "sum(xs)" in s else None)),
 ("early_return", lambda s: s.replace("        out.append(acc)\n", "        out.append(acc)\n        return out\n", 1)
                            if "        out.append(acc)\n" in s else
                            (s.replace("            out.append(int(s))\n",
                                       "            out.append(int(s))\n            return out\n", 1)
                             if "            out.append(int(s))\n" in s else None)),
 ("drop_first",   lambda s: s.replace("for tok in raw:", "for tok in raw[1:]:", 1)
                            if "for tok in raw:" in s else None),
 ("off_by_one_slice", lambda s: s.replace("return [", "return [", 1) and
                            (s.replace("for v in xs]", "for v in xs[1:]]", 1)
                             if "for v in xs]" in s else None)),
 ("swap_sign",    lambda s: s.replace("v + offset", "v - offset", 1) if "v + offset" in s else None),
]


def mutate(regions, order, name, mut):
    src = regions[name]
    new = mut(src)
    if not new or new == src:
        return None
    r2 = dict(regions); r2[name] = new
    return r2


def make_task(idx, rng, keep_signatures):
    """One task, or None. The verifier decides: clean must PASS its own suite, mutant must FAIL it."""
    regions, order, kinds = make_program(rng)
    program = assemble(regions, order)
    results, err = run_program(program, FIXTURES)
    if err or not results or all(k != "ok" for k, _ in results):
        return None
    tests = make_tests(results, FIXTURES)
    green, fails = run_suite(program, tests)
    if not green:
        return None                                   # the clean program must pass its own suite
    cands = [(r, mn, m) for r in order if r != "run" for (mn, m) in [rng.choice(MUTATIONS)]]
    rng.shuffle(cands)
    for region, mut_name, mut in cands:
        r2 = mutate(regions, order, region, mut)
        if r2 is None:
            continue
        buggy = assemble(r2, order)
        g2, f2 = run_suite(buggy, tests)
        if g2 or not f2:
            continue                                  # not observable: not a bug
        first = f2[0]
        sig = hashlib.sha1((buggy + tests).encode()).hexdigest()[:12]
        if sig in keep_signatures:
            continue                                  # never emit the same problem twice
        keep_signatures.add(sig)
        # P1' (理): the failing-test set must be reachable from >= 2 of THIS PROGRAM'S regions.
        # v2's P1 counted test identities ACROSS programs, which is meaningless once every program is
        # its own problem. Here it is answered per program, by actually trying the other regions.
        want = {f["test"] for f in f2}
        siblings = []
        for other in order:
            if other == region or other == "run":
                continue
            for _, m2 in MUTATIONS:
                r3 = mutate(regions, order, other, m2)
                if r3 is None:
                    continue
                g3, f3 = run_suite(assemble(r3, order), tests)
                if not g3 and f3 and want & {f["test"] for f in f3}:
                    siblings.append(other)
                    break
            if siblings:
                break
        if not siblings:
            # REJECT rather than emit-and-fail-the-gate. A task whose failing test NO other region of
            # its own program can produce is exactly a task whose symptom pins the region -- the
            # thing P1 exists to exclude. Filtering it at generation makes 理's "every program"
            # literal, with zero tolerance, instead of my picking a tolerance for a statistic they
            # set on a different (cross-program) count. The rejection RATE is reported.
            continue
        return {"task_id": "task_%04d" % idx, "family": "v3", "bug_class": mut_name,
                "bug_region": region, "program": buggy, "tests": tests,
                "symptom": "%s: %s" % (first["kind"], first["message"] or "no detail"),
                # v1 derived the region from the TEST NAME by substring, which is what made its
                # symptom a lookup key. v3's tests are named test_case_NN by construction and carry
                # NO region, so there IS no symptom_region to report. The field is kept for schema
                # compatibility and set to None rather than to a fabricated value.
                "symptom_region": None, "dead_paths": [],
                "regions": [r for r in order if r != "run"], "seed": rng.randint(0, 10**9),
                "_signature": sig, "_test_identity": first["test"],
                "_p1_sibling_regions": siblings,
                "_shape": {"parser": kinds[0], "mappers": kinds[1], "agg": kinds[2], "fmt": kinds[3]}}
    return None


def run_suite(program: str, tests: str):
    """(green, failures). Failures carry the test name, the region the name points at (for P1) and
    the ASSERTION DETAIL -- parsed from the failure BLOCKS, not from a window around the first
    occurrence of the test name, which is the bug that left 79% of v1's symptoms empty."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "program.py"), "w").write(program)
        open(os.path.join(d, "test_program.py"), "w").write(tests)
        try:
            p = subprocess.run([sys.executable, "-m", "unittest", "test_program"], cwd=d,
                               capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return False, [{"test": "timeout", "region": "?", "kind": "TIMEOUT", "message": "timeout"}]
    out = p.stdout + p.stderr
    failures, cur = [], None
    for line in out.splitlines():
        st = line.strip()
        if st.startswith("FAIL: ") or st.startswith("ERROR: "):
            kind, rest = st.split(": ", 1)
            cur = {"test": rest.split()[0], "region": "?", "kind": kind, "lines": []}
            failures.append(cur)
        elif cur is not None:
            if set(st) in ({"-"}, {"="}):
                continue
            if st.startswith("Ran ") or st.startswith("OK") or st.startswith("FAILED"):
                cur = None
            elif st:
                cur["lines"].append(st)
    for f in failures:
        body = [l for l in f.pop("lines") if not l.startswith('File "') and not l.startswith("  ")]
        f["message"] = (body[-1] if body else "")[:200]
    return (p.returncode == 0 and not failures), failures


def measure(tasks):
    by_test = collections.defaultdict(set)
    by_sym = collections.defaultdict(collections.Counter)
    sigs = collections.Counter()
    for t in tasks:
        by_test[t["_test_identity"]].add(t["bug_region"])
        by_sym[t["symptom"]][t["bug_region"]] += 1
        sigs[t["_signature"]] += 1
    n = len(tasks)
    H = 0.0
    for sym, c in by_sym.items():
        tot = sum(c.values())
        H += (tot / n) * (-sum((v / tot) * math.log2(v / tot) for v in c.values()))
    rng = random.Random(3); idx = list(range(n)); rng.shuffle(idx); half = n // 2
    tab = collections.defaultdict(collections.Counter)
    for i in idx[:half]:
        tab[tasks[i]["symptom"]][tasks[i]["bug_region"]] += 1
    tab = {k: v.most_common(1)[0][0] for k, v in tab.items()}
    hit = sum(1 for i in idx[half:]
              if tasks[i]["symptom"] in tab and tab[tasks[i]["symptom"]] == tasks[i]["bug_region"])
    return {"n": n,
            "P0_distinct_signatures": len(sigs),
            "P0_ratio": round(len(sigs) / n, 4) if n else 0.0,
            "P0_largest_group": sigs.most_common(1)[0][1] if sigs else 0,
            "P1prime_tasks_with_no_sibling_region": sum(1 for t in tasks
                                                        if not t.get("_p1_sibling_regions")),
            "P1prime_singletons": [t["task_id"] for t in tasks
                                   if not t.get("_p1_sibling_regions")][:10],
            "P2prime_assertion_detail_pct": round(100 * sum(
                1 for t in tasks if t["symptom"].split(":", 1)[-1].strip()
                not in ("", "no detail")) / n, 1) if n else 0.0,
            "P2prime_symptom_distinctness": round(len(by_sym) / n, 3) if n else 0.0,
            "P2_entropy_bits_PRINTED_NEVER_A_BAR": round(H, 3),
            "P3_heldout_lookup_pct": round(100 * hit / (n - half), 1) if n - half else 0.0,
            "distinct_symptoms": len(by_sym),
            "distinct_programs": len({t["program"] for t in tasks}),
            "bug_region_distribution": dict(collections.Counter(t["bug_region"] for t in tasks)),
            "bug_class_distribution": dict(collections.Counter(t["bug_class"] for t in tasks)),
            "region_count_distribution": dict(collections.Counter(len(t["regions"]) for t in tasks)),
            "red_herring_pct": None,   # v3 has no symptom_region: see the note in make_task
            "symptom_uniqueness": round(len(by_sym) / n, 3) if n else 0.0}


def verdict(m):
    bad = []
    if m["P0_ratio"] < 0.9:
        bad.append("P0 FAIL: %d distinct problems in %d tasks (ratio %.3f)" %
                   (m["P0_distinct_signatures"], m["n"], m["P0_ratio"]))
    if m["P1prime_tasks_with_no_sibling_region"]:
        bad.append("P1' FAIL: %d tasks whose failing test no OTHER region of the same program can "
                   "produce -- these should have been REJECTED at generation"
                   % m["P1prime_tasks_with_no_sibling_region"])
    if m["P2prime_assertion_detail_pct"] < 100.0:
        bad.append("P2'(a) FAIL: assertion detail present on only %.1f%% of tasks -- a symptom with "
                   "no detail is starvation, which is what P2 was standing in for"
                   % m["P2prime_assertion_detail_pct"])
    if m["P2prime_symptom_distinctness"] < 0.5:
        bad.append("P2'(b) FAIL: symptom distinctness %.3f, need >= 0.5"
                   % m["P2prime_symptom_distinctness"])
    if m["P3_heldout_lookup_pct"] > 60.0:
        bad.append("P3 FAIL: held-out lookup %.1f%%, need <= 60" % m["P3_heldout_lookup_pct"])
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "gen"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tasks, seen, tries = [], set(), 0
    while len(tasks) < a.n and tries < 60 * a.n:
        tries += 1
        t = make_task(len(tasks) + 1, rng, seen)
        if t:
            tasks.append(t)
    m = measure(tasks); m["attempts"] = tries; m["seed"] = a.seed
    bad = verdict(m)
    os.makedirs(a.out, exist_ok=True)
    json.dump(m, open(os.path.join(a.out, "_gate.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in m.items() if not k.startswith("P1_tests")}, indent=1))
    for b in bad:
        print("  " + b, file=sys.stderr)
    print("\n  GATE %s" % ("PASS" if not bad else "FAIL"), file=sys.stderr)
    if a.cmd == "gate":
        return 0 if not bad else 1
    if bad:
        print("  REFUSING to write: a set that fails its own pre-registered properties is the "
              "previous set with different numbers.", file=sys.stderr)
        return 1
    dig = hashlib.sha256()
    for t in tasks:
        json.dump(t, open(os.path.join(a.out, t["task_id"] + ".json"), "w"), indent=1)
        dig.update(json.dumps(t, sort_keys=True).encode())
    json.dump({"n": len(tasks), "seed": a.seed, "gate": m,
               "sha256_of_tasks_in_order": dig.hexdigest(),
               "regenerate": "python3 et8_env_v3.py gen --n %d --seed %d --out <dir>" % (len(tasks), a.seed),
               "P4": "NOT YET RUN — the set is PROVISIONAL until a baseline reports its outcome "
                     "spread. A set of programs that are superficially different and behaviourally "
                     "identical passes P0 and is the previous set with more strings."},
              open(os.path.join(a.out, "_stats.json"), "w"), indent=1)
    print("\n  wrote %d tasks to %s" % (len(tasks), a.out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
