#!/usr/bin/env python3
"""R7 hazard 3: the CEILING CHECK, run BEFORE any head is fit — the mirror of R4's floor.

SQL is far more familiar to a 7B than the synthetic pipeline grammar. If the base can repair these
one-shot at near-ceiling, there is no headroom for a localisation prior and R7 CANNOT answer the
question on this task set. That is a result, and it is reported rather than worked around.

ONE-SHOT and deliberately GENEROUS: the model is handed the buggy query, the schema, the expected
row count and the observed rows, and asked for a corrected query. No inspection, no budget, no
localisation required. This BOUNDS the ceiling from above — the agent loop cannot do better than a
model that is simply told everything.
"""
import glob, json, sqlite3, sys
sys.path.insert(0, "<repo>/experience")
import et8_sql_env as E

def norm(rows):
    """sqlite returns TUPLES; the task JSON round-trips them to LISTS. Comparing the two directly is
    ALWAYS False — the first version of this file did exactly that and scored 0/60, including on
    queries that were correct. Normalise both sides."""
    return [list(r) for r in (rows or [])]


N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
files = sorted(glob.glob("<repo>/experience/tasks/sql/task_*.json"))[:N]
from mlx_lm import load, generate as gen
from mlx_lm.sample_utils import make_sampler
model, tok = load("Qwen/Qwen2.5-7B-Instruct")
SYS = ("You repair a single SQL query. Reply with ONLY the corrected SQL on one line, no prose, "
       "no code fence.\n\nSchema:\n" + E.SCHEMA.strip())
# POSITIVE CONTROL, FIRST: the CLEAN query must score as repaired. If it does not, the scorer is
# broken and every rate below is meaningless. This is the check whose absence produced a 0/60.
_bad = []
for f in files:
    _t = json.load(open(f))
    _g, _e = E.run(E.db(), _t["clean"])
    if _e is not None or norm(_g) != norm(_t["expected_rows"]):
        _bad.append(_t["task_id"])
if _bad:
    print(json.dumps({"VERDICT": "SCORER BROKEN — the clean query does not score as repaired on %d "
                      "of %d tasks. No rate is reported." % (len(_bad), len(files)),
                      "examples": _bad[:5]}, indent=1))
    sys.exit(1)
print("  positive control: the clean query scores as repaired on all %d tasks" % len(files), file=sys.stderr)

ok = 0
for i, f in enumerate(files, 1):
    t = json.load(open(f))
    u = ("This query returns the WRONG rows.\n\nQuery:\n%s\n\nIt returned %d rows; the correct "
         "query returns %d rows.\n\nCorrected query:" % (t["program"], len(t["got_rows"]), len(t["expected_rows"])))
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": u}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    out = gen(model, tok, prompt=p, max_tokens=180, sampler=make_sampler(temp=0.0), verbose=False)
    q = out.strip().split("\n")[0].strip().strip("`").replace("sql", "", 1).strip() if out.strip().startswith("`") else out.strip().split("\n")[0].strip()
    conn = E.db()
    got, err = E.run(conn, q)
    good = (err is None and norm(got) == norm(t["expected_rows"]))
    ok += good
    if i % 20 == 0:
        print("  %d/%d  one-shot repaired %d" % (i, len(files), ok), file=sys.stderr)
n = len(files)
print(json.dumps({"document": "R7 CEILING CHECK — one-shot, generous, before any head is fit",
                  "n": n, "one_shot_repair_pct": round(100.0*ok/n, 1),
                  "reading": ("ABOVE ~80%% -> no headroom; R7 cannot answer on this set and says so"
                              if 100.0*ok/n > 80 else
                              "below the ceiling -> there IS headroom; the agent loop is worth building"),
                  "note": "one-shot is GENEROUS: the model is told the query, the schema, and both "
                          "row counts. The agent loop cannot exceed this bound."}, indent=1))
