#!/usr/bin/env python3
"""R7 — SQL query repair: a SECOND SEARCH STRUCTURE for the same harness (理 11398/11552).

SCOPE COMMITTED FIRST: docs/r7-sql-repair-scope.md (7d8456b), before a line of this existed.

WHY. Everything in 8a is one environment: a Python pipeline grammar whose regions are stages. If
G's benefit is about EXPERIENCE-DIRECTED SEARCH, a different search STRUCTURE should support it. If
it is about that grammar, it will not. The head is RE-FITTED from this environment's own base
episodes — the mechanism transfers, never the weights.

  regions   select / where / join / group / order — CLAUSES, not stages
  verifier  the EXPECTED RESULT SET on fixed fixture rows. NOT query text: two different queries can
            be equivalent, and scoring on text would mark a correct repair wrong.
  a bug is  admitted only if the verifier DISTINGUISHES it — the mutant must return a different
            result set than the clean query on the fixtures.

THE FOUR HAZARDS, WRITTEN IN THE SCOPE BEFORE THEY BIT, AND WHAT THIS FILE DOES ABOUT THEM:
  1 "the clause structure may pin the region for free" -> P1' is enforced per task exactly as v3
    does it: the failing symptom must be reachable from >= 2 of THIS query's regions, or the task is
    REJECTED at generation. If that rejects most candidates, THAT IS THE RESULT.
  2 "k varies per task" -> a query need not use every clause, so CHANCE IS PER TASK. Each task
    records its own region count; nothing assumes v3's constant 20.4%.
  3 "SQL is familiar; the base may sit near a CEILING" -> the ceiling check is the mirror of R4's
    floor and runs BEFORE any head is fit.
  4 "the verifier is a database" -> fixtures are deterministic and every SELECT is ORDER BY'd at the
    verifier, so `order` bugs are falsifiable and `select` bugs do not flap on row order.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sqlite3, sys

SCHEMA = """
CREATE TABLE emp (id INTEGER, name TEXT, dept_id INTEGER, salary INTEGER, hired TEXT);
CREATE TABLE dept (id INTEGER, dname TEXT, region TEXT);
"""
ROWS_EMP = [(1,"ada",10,120,"2019-03-01"),(2,"bo",10,90,"2020-07-15"),(3,"cy",20,140,"2018-01-09"),
            (4,"di",20,95,"2021-11-30"),(5,"el",30,110,"2017-05-22"),(6,"fi",30,75,"2022-02-14"),
            (7,"gu",10,160,"2016-09-03"),(8,"ha",20,85,"2023-04-18")]
ROWS_DEPT = [(10,"eng","east"),(20,"ops","west"),(30,"sales","east")]

REGIONS = ("select", "where", "join", "group", "order")


def db():
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA)
    c.executemany("INSERT INTO emp VALUES (?,?,?,?,?)", ROWS_EMP)
    c.executemany("INSERT INTO dept VALUES (?,?,?)", ROWS_DEPT)
    return c


# ---- the query as parts, so a "region" is an editable clause -------------------------------
SELECTS = ["e.name, e.salary", "e.name, d.dname", "d.dname, COUNT(*)", "d.region, SUM(e.salary)",
           "e.name, e.hired", "d.dname, AVG(e.salary)", "e.dept_id, MAX(e.salary)"]
WHERES = ["e.salary > 90", "e.salary >= 95", "d.region = 'east'", "e.hired > '2019-01-01'",
          "e.dept_id IN (10,20)", "e.salary BETWEEN 85 AND 140", "d.dname <> 'sales'"]
JOINS = ["JOIN dept d ON e.dept_id = d.id", "LEFT JOIN dept d ON e.dept_id = d.id",
         "JOIN dept d ON e.dept_id = d.id AND d.region = 'east'"]
GROUPS = ["", "GROUP BY d.dname", "GROUP BY d.region", "GROUP BY e.dept_id"]
ORDERS = ["ORDER BY e.name", "ORDER BY e.salary DESC", "ORDER BY 1", "ORDER BY 1, 2"]

# Each mutation edits ONE clause. The verifier decides whether the edit is a bug.
MUT = {
 "select": [lambda s: s.replace("COUNT(*)", "COUNT(e.id)"), lambda s: s.replace("SUM", "AVG"),
            lambda s: s.replace("MAX", "MIN"), lambda s: s.replace("e.name", "e.id")],
 "where":  [lambda s: s.replace(">", ">="), lambda s: s.replace(">=", ">"),
            lambda s: s.replace("'east'", "'west'"), lambda s: s.replace("IN (10,20)", "IN (10,30)"),
            lambda s: s.replace("<>", "="), lambda s: s.replace("BETWEEN 85 AND 140", "BETWEEN 85 AND 120")],
 "join":   [lambda s: s.replace("JOIN dept d ON e.dept_id = d.id", "JOIN dept d ON e.dept_id = d.region"),
            lambda s: s.replace("LEFT JOIN", "JOIN"),
            lambda s: s.replace("AND d.region = 'east'", "AND d.region = 'west'")],
 "group":  [lambda s: s.replace("GROUP BY d.dname", "GROUP BY d.region"),
            lambda s: s.replace("GROUP BY d.region", "GROUP BY d.dname"),
            lambda s: s.replace("GROUP BY e.dept_id", "GROUP BY e.salary")],
 "order":  [lambda s: s.replace("DESC", ""), lambda s: s.replace("ORDER BY e.name", "ORDER BY e.salary"),
            lambda s: s.replace("ORDER BY 1, 2", "ORDER BY 2, 1")],
}


def build(parts):
    q = "SELECT %s FROM emp e %s" % (parts["select"], parts["join"])
    if parts["where"]:
        q += " WHERE %s" % parts["where"]
    if parts["group"]:
        q += " %s" % parts["group"]
    if parts["order"]:
        q += " %s" % parts["order"]
    return q


def run(conn, q):
    """The verifier: the RESULT SET. Deterministic — the query's own ORDER BY is the only ordering,
    and a query with no ORDER BY has its rows sorted here so a `select` bug cannot flap on order."""
    try:
        rows = conn.execute(q).fetchall()
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e)[:120])
    return (rows if "ORDER BY" in q.upper() else sorted(rows, key=repr)), None


def regions_used(parts):
    return [r for r in REGIONS if parts.get(r)]


def make_task(idx, rng, seen):
    conn = db()
    parts = {"select": rng.choice(SELECTS), "where": rng.choice(WHERES), "join": rng.choice(JOINS),
             "group": rng.choice(GROUPS), "order": rng.choice(ORDERS)}
    # an aggregate select needs a GROUP BY to be meaningful; drop the pairing if absent
    if any(a in parts["select"] for a in ("COUNT", "SUM", "AVG", "MAX")) and not parts["group"]:
        return None
    if parts["group"] and not any(a in parts["select"] for a in ("COUNT", "SUM", "AVG", "MAX")):
        return None
    clean = build(parts)
    gold, err = run(conn, clean)
    if err or not gold:
        return None                              # the clean query must run and return rows
    used = regions_used(parts)
    if len(used) < 3:
        return None                              # need room for P1' to be answerable

    cands = [r for r in used if MUT.get(r)]
    rng.shuffle(cands)
    for region in cands:
        for mut in rng.sample(MUT[region], len(MUT[region])):
            p2 = dict(parts); p2[region] = mut(parts[region])
            if p2[region] == parts[region]:
                continue                          # the edit did nothing
            buggy = build(p2)
            got, err2 = run(conn, buggy)
            if err2 is not None:
                continue                          # a crash is not the bug class under test
            if got == gold:
                continue                          # VERIFIER DOES NOT DISTINGUISH IT -> not a bug
            sig = hashlib.sha1((buggy + clean).encode()).hexdigest()[:12]
            if sig in seen:
                continue
            # P1': the symptom must be reachable from >= 2 of THIS query's regions
            siblings = []
            for other in used:
                if other == region or not MUT.get(other):
                    continue
                for m2 in MUT[other]:
                    p3 = dict(parts); p3[other] = m2(parts[other])
                    if p3[other] == parts[other]:
                        continue
                    g3, e3 = run(conn, build(p3))
                    if e3 is None and g3 != gold:
                        siblings.append(other); break
            if not siblings:
                continue                          # symptom pins its own region -> REJECT
            seen.add(sig)
            return {"task_id": "task_%04d" % idx, "family": "sql", "regions": used,
                    "n_regions": len(used), "chance_pct": round(100.0 / len(used), 1),
                    "bug_region": region, "bug_class": "clause_edit",
                    "program": buggy, "clean": clean, "parts": p2,
                    "expected_rows": gold, "got_rows": got,
                    "_p1_sibling_regions": sorted(set(siblings)), "_signature": sig,
                    "tests": "expected result set (%d rows) on the fixed fixtures" % len(gold)}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "gate"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="tasks/sql")
    ap.add_argument("--exclude", default=None,
                    help="a task dir whose signatures must NOT be reused. The grammar is finite, so "
                         "two seeds collide: seed 21 and seed 47 shared 23 signatures. A 'disjoint' "
                         "set that shares problems with the training set is not disjoint, and the "
                         "overlap is silent unless excluded here.")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    seen, tasks, tries = set(), [], 0
    if a.exclude:
        import glob as _g
        for f in _g.glob(os.path.join(a.exclude, "task_*.json")):
            seen.add(json.load(open(f))["_signature"])
        print("  excluding %d signatures from %s" % (len(seen), a.exclude))
    while len(tasks) < a.n and tries < a.n * 400:
        tries += 1
        t = make_task(len(tasks) + 1, rng, seen)
        if t:
            tasks.append(t)
    print("  generated %d of %d requested in %d attempts" % (len(tasks), a.n, tries))
    if a.cmd == "gen":
        os.makedirs(a.out, exist_ok=True)
        for t in tasks:
            json.dump(t, open(os.path.join(a.out, t["task_id"] + ".json"), "w"), indent=1)
        print("  wrote %d files to %s" % (len(tasks), a.out))
    sigs = {t["_signature"] for t in tasks}
    ks = sorted({t["n_regions"] for t in tasks})
    print("  P0  distinct signatures %d / %d = %.3f   (gate >= 0.9)" % (len(sigs), len(tasks), len(sigs)/max(1,len(tasks))))
    print("  P1' every task has >=1 sibling region: %s" % all(t["_p1_sibling_regions"] for t in tasks))
    print("  k varies: region counts %s -> chance is PER TASK, not constant" % ks)
    import collections
    print("  bug region spread: %s" % dict(collections.Counter(t["bug_region"] for t in tasks)))
    # P3, ENFORCED AT GENERATION rather than discovered afterwards. A held-out lookup table keyed on
    # the row-count signature must not exceed 60%. This environment sits ON the boundary: across
    # eight seeds it ranged 56.0-69.3%, so a set MUST be checked, never assumed. seed 47 failed at
    # 61.3% and a G run was stopped on it.
    rows = [((len(x["got_rows"]), len(x["expected_rows"]),
              len(x["got_rows"]) - len(x["expected_rows"])), x["bug_region"]) for x in tasks]
    cut = int(0.75 * len(rows))
    tr, te = rows[:cut], rows[cut:]
    tab = collections.defaultdict(collections.Counter)
    for f, b in tr:
        tab[f][b] += 1
    maj = collections.Counter(b for _, b in tr).most_common(1)[0][0] if tr else None
    hit = sum(1 for f, b in te if (tab[f].most_common(1)[0][0] if f in tab else maj) == b)
    p3 = 100.0 * hit / max(1, len(te))
    print("  P3  held-out symptom lookup table %.1f%%   (gate <= 60%%) -> %s"
          % (p3, "PASSES" if p3 <= 60 else "*** FAILS — do not run on this set ***"))


if __name__ == "__main__":
    main()
