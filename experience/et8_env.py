#!/usr/bin/env python3
"""
et8_env.py — Efficient Thinking VIII environment v0: trap-debugging tasks with an external verifier.

One task = a small Python program (a "project" of a few functions), a hidden injected bug from a fixed taxonomy,
a unit-test suite that catches it (the EXTERNAL verifier — never the model judging itself), and the SYMPTOM the
agent sees first (the failing test's output). Families bias which bug classes occur and where the symptom points,
so that a search prior has structure to learn: hypothesis order per (family, symptom), red herrings, and dead paths.

Usage:
  python experience/et8_env.py gen  --n 100 --seed 0 --out experience/tasks/v0            # generate tasks
  python experience/et8_env.py selftest                                                   # G1 gate: verifier catches every bug, passes every clean
  python experience/et8_env.py run  --task experience/tasks/v0/task_0001.json --patch fix.py   # verify a candidate patch

Design: docs/efficient-thinking-8-proposal.md §2. Status: v0, generator + verifier + trajectory schema. No LLM here.
"""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, tempfile, textwrap, hashlib
from dataclasses import dataclass, asdict, field
from typing import Callable

# ----------------------------------------------------------------------------------------------------------------------
# Bug taxonomy v0 (12 classes). Each class: a name, and an injector that rewrites ONE clean region into a buggy one.
# Regions are named so the trajectory logger can score "did the agent inspect the true region".
# ----------------------------------------------------------------------------------------------------------------------

BUG_CLASSES = [
    "off_by_one", "wrong_comparator", "mutable_default", "aliasing", "int_division",
    "width_mismatch", "naive_datetime", "missing_key", "early_return", "swapped_args",
    "unsorted_assumption", "swallowed_exception",
]

# A clean program is assembled from regions. Each region is a function; the "consumer" depends on the "producer",
# so a bug in the producer surfaces as a symptom in the consumer's tests (the red herring).
CLEAN_REGIONS = {
    "producer": textwrap.dedent('''
        def produce(items):
            """Return the list of positive integer scores parsed from items (strings), in input order."""
            out = []
            for i in range(len(items)):
                s = items[i].strip()
                if s.isdigit() and int(s) > 0:
                    out.append(int(s))
            return out
    '''),
    "transform": textwrap.dedent('''
        def normalize(scores, cap=100):
            """Clamp scores to [0, cap] and return as floats scaled to [0, 1]."""
            res = []
            for v in scores:
                if v > cap:
                    v = cap
                res.append(v / cap)
            return res
    '''),
    "aggregate": textwrap.dedent('''
        def summarize(values, bucket=None):
            """Return dict with count, mean, max; optional bucket collects values >= 0.5."""
            if bucket is None:
                bucket = []
            for v in values:
                if v >= 0.5:
                    bucket.append(v)
            n = len(values)
            mean = (sum(values) / n) if n else 0.0
            mx = max(values) if values else 0.0
            return {"count": n, "mean": mean, "max": mx, "bucket": bucket}
    '''),
    "consumer": textwrap.dedent('''
        def report(items, cap=100):
            """End-to-end: parse, normalize, summarize; returns the summary dict with a 'grade' key."""
            scores = produce(items)
            norm = normalize(scores, cap)
            summ = summarize(norm)
            summ["grade"] = "high" if summ["mean"] >= 0.5 else "low"
            return summ
    '''),
}

# Injectors: (region, replacement source). Each returns the buggy region text.
def _inject(cls: str) -> tuple[str, str]:
    if cls == "off_by_one":
        return "producer", CLEAN_REGIONS["producer"].replace("range(len(items))", "range(len(items) - 1)")
    if cls == "wrong_comparator":
        return "transform", CLEAN_REGIONS["transform"].replace("if v > cap:", "if v >= cap:").replace("v = cap", "v = cap - 1")
    if cls == "mutable_default":
        return "aggregate", CLEAN_REGIONS["aggregate"].replace("bucket=None", "bucket=[]").replace(
            "            if bucket is None:\n                bucket = []\n", "")
    if cls == "aliasing":
        return "transform", CLEAN_REGIONS["transform"].replace("res = []", "res = scores").replace(
            "res.append(v / cap)", "res[len(res) and 0] = v / cap" if False else "pass").replace(
            "            for v in scores:\n                if v > cap:\n                    v = cap\n                pass\n",
            "            for i, v in enumerate(scores):\n                if v > cap:\n                    v = cap\n                res[i] = v / cap\n")
    if cls == "int_division":
        return "transform", CLEAN_REGIONS["transform"].replace("res.append(v / cap)", "res.append(v // cap)")
    if cls == "width_mismatch":
        # a non-breaking space in the input is not stripped -> isdigit fails -> score silently dropped
        return "producer", CLEAN_REGIONS["producer"].replace("s = items[i].strip()", "s = items[i].strip(' ')")
    if cls == "naive_datetime":
        # a naive local timestamp leaks into the summary: an extra key the contract does not allow
        return "consumer", CLEAN_REGIONS["consumer"].replace(
            '    summ["grade"] = "high" if summ["mean"] >= 0.5 else "low"\n',
            '    import datetime\n    summ["grade"] = "high" if summ["mean"] >= 0.5 else "low"\n    summ["stamp"] = datetime.datetime.now().isoformat()\n')
    if cls == "missing_key":
        return "consumer", CLEAN_REGIONS["consumer"].replace('summ["mean"] >= 0.5', 'summ["avg"] >= 0.5')
    if cls == "early_return":
        return "producer", CLEAN_REGIONS["producer"].replace(
            "            out.append(int(s))\n", "            out.append(int(s))\n            return out\n")
    if cls == "swapped_args":
        return "consumer", CLEAN_REGIONS["consumer"].replace("normalize(scores, cap)", "normalize(cap, scores)")
    if cls == "unsorted_assumption":
        return "aggregate", CLEAN_REGIONS["aggregate"].replace("mx = max(values) if values else 0.0", "mx = values[-1] if values else 0.0")
    if cls == "swallowed_exception":
        return "producer", CLEAN_REGIONS["producer"].replace(
            "        if s.isdigit() and int(s) > 0:\n            out.append(int(s))\n",
            "        try:\n            if int(s) > 0:\n                out.append(int(s))\n        except Exception:\n            pass\n")
    raise ValueError(cls)

# Tests: the external verifier. Each test names the region it exercises so the SYMPTOM can be attributed.
TESTS = textwrap.dedent('''
    import unittest, datetime
    from program import produce, normalize, summarize, report

    class T(unittest.TestCase):
        # --- producer ---
        def test_produce_all(self):            self.assertEqual(produce(["3","1","2"]), [3,1,2])
        def test_produce_filters(self):        self.assertEqual(produce(["0","-2","x","5"]), [5])
        def test_produce_nbsp(self):           self.assertEqual(produce(["\\u00a07"]), [7])
        def test_produce_bad_token(self):      self.assertEqual(produce(["7a", "8"]), [8])
        def test_produce_underscore(self):     self.assertEqual(produce(["1_0", "2"]), [2])   # isdigit rejects; a bare int() accepts
        # --- transform ---
        def test_normalize_scale(self):        self.assertEqual(normalize([50, 100], 100), [0.5, 1.0])
        def test_normalize_clamp(self):        self.assertEqual(normalize([150], 100), [1.0])
        def test_normalize_no_alias(self):
            s = [10, 20]; normalize(s, 100); self.assertEqual(s, [10, 20])
        def test_normalize_arg_order(self):    self.assertEqual(normalize([25], 50), [0.5])
        # --- aggregate ---
        def test_summarize_basic(self):
            r = summarize([0.2, 0.6, 0.9]); self.assertEqual(r["count"], 3); self.assertAlmostEqual(r["mean"], 0.5666, 3); self.assertEqual(r["max"], 0.9)
        def test_summarize_unsorted_max(self): self.assertEqual(summarize([0.9, 0.1])["max"], 0.9)
        def test_summarize_bucket_fresh(self):
            a = summarize([0.7]); b = summarize([0.8]); self.assertEqual(b["bucket"], [0.8])
        # --- consumer ---
        def test_report_grade_high(self):      self.assertEqual(report(["80","90"], 100)["grade"], "high")
        def test_report_grade_low(self):       self.assertEqual(report(["10","20"], 100)["grade"], "low")
        def test_report_keys(self):
            r = report(["50"], 100); self.assertEqual(set(r.keys()), {"count","mean","max","bucket","grade"})
        def test_report_all_items(self):       self.assertEqual(report(["10","20","30"], 100)["count"], 3)

    if __name__ == "__main__":
        unittest.main(verbosity=0)
''')

REGION_ORDER = ["producer", "transform", "aggregate", "consumer"]

# Families: bias which classes occur and which are dead paths (never the cause, but plausible).
# family -> (class weights, dead_paths)
FAMILIES = {
    "A_boundary":  ({"off_by_one": 4, "wrong_comparator": 3, "early_return": 2, "unsorted_assumption": 1}, ["mutable_default", "aliasing"]),
    "B_state":     ({"mutable_default": 4, "aliasing": 3, "swallowed_exception": 2, "missing_key": 1}, ["off_by_one", "wrong_comparator"]),
    "C_types":     ({"int_division": 4, "width_mismatch": 3, "swapped_args": 2, "naive_datetime": 1}, ["early_return", "unsorted_assumption"]),
    "D_mixed":     ({c: 1 for c in BUG_CLASSES}, []),
}

@dataclass
class Task:
    task_id: str
    family: str
    bug_class: str
    bug_region: str
    program: str            # buggy program source (program.py)
    tests: str              # test source (test_program.py)
    symptom: str            # first failing test name + message, as the agent sees it
    symptom_region: str     # region the failing test EXERCISES (may differ from bug_region: the red herring)
    dead_paths: list[str]
    regions: list[str] = field(default_factory=lambda: list(REGION_ORDER))
    seed: int = 0

def assemble(buggy: dict[str, str] | None = None) -> str:
    parts = []
    for r in REGION_ORDER:
        src = (buggy or {}).get(r, CLEAN_REGIONS[r])
        parts.append(f"# region: {r}\n" + src.strip("\n") + "\n")
    return "\n".join(parts)

def run_tests(program_src: str, tests_src: str = TESTS, timeout: int = 20) -> tuple[bool, list[dict]]:
    """Run the verifier. Returns (all_green, failures[{test, region, message}])."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "program.py"), "w").write(program_src)
        open(os.path.join(d, "test_program.py"), "w").write(tests_src)
        p = subprocess.run([sys.executable, "-m", "unittest", "test_program", "-v"], cwd=d,
                           capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
    failures = []
    for line in out.splitlines():
        line = line.strip()
        for tag in ("FAIL:", "ERROR:"):
            if line.startswith(tag):
                name = line.split()[1]
                region = "producer" if "produce" in name else "transform" if "normalize" in name else \
                         "aggregate" if "summarize" in name else "consumer"
                failures.append({"test": name, "region": region, "kind": tag[:-1]})
    if p.returncode != 0 and not failures:
        # the module did not import (syntax/indentation) — a failure of its own kind, attributed to no region
        failures.append({"test": "import", "region": "module", "kind": "IMPORT_ERROR"})
    green = (p.returncode == 0) and not failures
    # attach the first assertion message per failure, if present
    for f in failures:
        idx = out.find(f["test"])
        seg = out[idx: idx + 600]
        msg = [l for l in seg.splitlines() if "Error" in l or "assert" in l.lower()]
        f["message"] = (msg[-1] if msg else "").strip()[:200]
    return green, failures

def make_task(idx: int, rng: random.Random, family: str | None = None) -> Task:
    fam = family or rng.choice(list(FAMILIES))
    weights, dead = FAMILIES[fam]
    classes, w = zip(*weights.items())
    cls = rng.choices(classes, weights=w, k=1)[0]
    region, buggy_src = _inject(cls)
    program = assemble({region: buggy_src})
    green, failures = run_tests(program)
    if green or not failures:
        raise RuntimeError(f"injector for {cls} produced no failing test — verifier gap")
    # the SYMPTOM is the first failing test in region order — consumer tests fail for producer bugs too (red herring)
    failures.sort(key=lambda f: (-REGION_ORDER.index(f["region"]), f["test"]))  # show the most DOWNSTREAM first
    first = failures[0]
    return Task(task_id=f"task_{idx:04d}", family=fam, bug_class=cls, bug_region=region, program=program, tests=TESTS,
                symptom=f'{first["kind"]}: {first["test"]} — {first["message"]}', symptom_region=first["region"],
                dead_paths=dead, seed=rng.randint(0, 10**9))

def cmd_gen(a):
    rng = random.Random(a.seed); os.makedirs(a.out, exist_ok=True)
    fams = list(FAMILIES)
    stats = {"n": 0, "red_herring": 0, "by_class": {}}
    for i in range(1, a.n + 1):
        t = make_task(i, rng, family=fams[(i - 1) % len(fams)])
        json.dump(asdict(t), open(os.path.join(a.out, f"{t.task_id}.json"), "w"), indent=1)
        stats["n"] += 1; stats["red_herring"] += int(t.symptom_region != t.bug_region)
        stats["by_class"][t.bug_class] = stats["by_class"].get(t.bug_class, 0) + 1
    json.dump(stats, open(os.path.join(a.out, "_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))

def cmd_selftest(a):
    """G1 gate: clean program passes all tests; every bug class is caught; report red-herring rate."""
    green, fails = run_tests(assemble())
    assert green, f"CLEAN PROGRAM FAILS: {fails}"
    caught, herring = 0, 0
    for cls in BUG_CLASSES:
        region, src = _inject(cls)
        g, f = run_tests(assemble({region: src}))
        ok = (not g) and bool(f)
        caught += ok
        f.sort(key=lambda x: -REGION_ORDER.index(x["region"]))
        rh = bool(f) and f[0]["region"] != region
        herring += rh
        print(f"  {cls:22s} region={region:9s} caught={'Y' if ok else 'N'} first_symptom_region={f[0]['region'] if f else '-':9s} red_herring={'Y' if rh else 'N'}")
    print(f"clean=PASS  caught={caught}/{len(BUG_CLASSES)}  red_herring_classes={herring}/{len(BUG_CLASSES)}")
    sys.exit(0 if caught == len(BUG_CLASSES) else 1)

def cmd_run(a):
    t = json.load(open(a.task)); src = open(a.patch).read() if a.patch else t["program"]
    green, fails = run_tests(src, t["tests"])
    print(json.dumps({"task": t["task_id"], "green": green, "failures": fails}, indent=1))

# Trajectory schema (logged by the agent harness, not here): one JSON line per action —
# {"task_id","step","action":"hypothesize|inspect|patch|run","region","hypothesis_class","tokens","verifier":{...},
#  "productive": bool (hindsight: touched bug_region / bug_class)}  → experience/traj/<run>.jsonl

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("--n", type=int, default=100); g.add_argument("--seed", type=int, default=0); g.add_argument("--out", default="experience/tasks/v0"); g.set_defaults(fn=cmd_gen)
    s = sp.add_parser("selftest"); s.set_defaults(fn=cmd_selftest)
    r = sp.add_parser("run"); r.add_argument("--task", required=True); r.add_argument("--patch"); r.set_defaults(fn=cmd_run)
    a = ap.parse_args(); a.fn(a)
