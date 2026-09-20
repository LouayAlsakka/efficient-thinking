#!/usr/bin/env python3
"""R7 agent loop — the SAME protocol as et8_agent, over SQL CLAUSES instead of Python stages.

A SEPARATE FILE, not an edit. et8_agent.py produced every number in 8a; teaching it a second
environment would silently re-define the measured thing. The protocol, the inspect-first rule, the
action vocabulary and the head plumbing are mirrored deliberately so the comparison is about the
SEARCH STRUCTURE and not about a rewritten harness.

  regions   the query's own clauses: select / where / join / group / order  (k VARIES per task)
  inspect   reveals ONE clause's text
  patch     replaces ONE clause; the verifier re-runs the query and compares the RESULT SET
  green     the result set equals the expected rows — ANY equivalent query counts, by design

THE HEAD ENTERS EXACTLY WHERE IT DOES IN 8a: at the first post-inspect decision, scoring one
candidate per VISIBLE region, each built by teacher-forcing the opening fields. `head_state` has the
same shape as et8_agent's, so a head fitted here is fitted the same way — the MECHANISM transfers,
never the weights.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et8_sql_env as E

SYSTEM = """You are repairing ONE SQL query that returns the wrong rows. You will be shown the
failing SYMPTOM (expected vs observed row counts), the list of query CLAUSES, and any clauses you
have inspected. Act in small steps. Reply with EXACTLY ONE JSON object and nothing else, one of:
{"action":"hypothesize","region":"<clause>","bug_class":"clause_edit","why":"<one sentence>"}
{"action":"inspect","region":"<clause>"}
{"action":"patch","region":"<clause>","source":"<the full corrected text of that clause>"}
{"action":"run"}
Clauses present in this query: %s.
Schema:
%s
You have %d actions in total. Rules: inspecting a clause you have already seen is wasted; every
patch re-runs the query and shows you the result; hypothesize before you patch. A `patch` replaces
ONLY that clause's text — for `where` give the predicate without the WHERE keyword; for `group` and
`order` include the GROUP BY / ORDER BY keywords; for `join` include the JOIN keyword."""


def parse_action(text):
    import re
    m = re.search(r'\{.*\}', text or "", re.S)
    if not m:
        return {"action": "invalid", "raw": (text or "")[:200]}
    try:
        d = json.loads(m.group(0))
    except Exception:
        try:
            d = json.loads(m.group(0).replace("\n", "\\n"))
        except Exception:
            return {"action": "invalid", "raw": (text or "")[:200]}
    return d if isinstance(d, dict) else {"action": "invalid", "raw": (text or "")[:200]}


def run_episode(model, tok, task, budget, run_id, log, model_id, head_state=None, inspect_first=True):
    import et8_agent as A
    parts = dict(task["parts"])
    regions = task["regions"]
    conn = E.db()
    expected = [list(r) for r in task["expected_rows"]]
    inspected, history = {}, []
    total_in = total_out = 0
    msgs = [{"role": "system", "content": SYSTEM % (", ".join(regions), E.SCHEMA.strip(), budget)}]

    def state():
        got, err = E.run(conn, E.build(parts))
        s = ["SYMPTOM: the query returns %s rows; the correct query returns %d."
             % (("an error: " + err) if err else len(got or []), len(expected))]
        s.append("Clauses: " + ", ".join(regions))
        for r, v in inspected.items():
            s.append("  %s: %s" % (r, v))
        if history:
            s.append("So far: " + " | ".join(history[-6:]))
        return "\n".join(s)

    green = False
    first_correct = None
    for step in range(1, budget + 1):
        cur = msgs + [{"role": "user", "content": state()}]
        if head_state is not None and inspected and not any(h.startswith("hypothesize ") for h in history):
            import numpy as _np
            scores, pres = {}, {}
            for r in regions:
                pre = '{"action": "hypothesize", "region": "%s", "bug_class": "' % r
                hs = head_state["hidden"](model, tok, cur, [head_state["layer"]], prefix=pre)
                z = _np.array(hs[head_state["layer"]].astype(head_state["mx"].float32), copy=False)
                z = (z.astype(_np.float64) - head_state["mu"]) / head_state["sd"]
                scores[r] = float(z @ head_state["w"] + head_state["b"]); pres[r] = pre
            pick = max(scores, key=scores.get)
            text, ni, no = A.generate(model, tok, cur, temp=0.0)
            total_in += ni; total_out += no
            gp = parse_action(text)
            if not (gp.get("action") == "hypothesize" and gp.get("region") == pick):
                text, ni, no = A.generate(model, tok, cur, temp=0.0, prefix=pres[pick])
                total_in += ni; total_out += no
            meta = {"head": True, "head_pick": pick, "agent_pick": gp.get("region")}
        elif inspect_first and step == 1:
            r = regions[0]
            text = json.dumps({"action": "inspect", "region": r}); meta = {"inspect_first": True}
        else:
            text, ni, no = A.generate(model, tok, cur, temp=0.0 if step == 1 else 0.7)
            total_in += ni; total_out += no
            meta = {}

        act = parse_action(text)
        kind, region = act.get("action"), act.get("region")
        rec = {"run_id": run_id, "model": model_id, "task_id": task["task_id"], "family": "sql",
               "step": step, "action": kind, "region": region,
               "region_hit": (region == task["bug_region"]), "candidates": meta,
               "tokens_in": total_in, "tokens_out": total_out}
        if kind == "inspect" and region in regions:
            if region in inspected:
                rec["action"] = "repeat_inspect"
            inspected[region] = parts[region]
            history.append("inspect %s" % region)
        elif kind == "hypothesize" and region in regions:
            if first_correct is None and region == task["bug_region"]:
                first_correct = step
            history.append("hypothesize %s" % region)
        elif kind == "patch" and region in regions:
            new = act.get("source") or ""
            if new.strip() == str(parts[region]).strip():
                rec["action"] = "noop_patch"
            else:
                parts[region] = new
                got, err = E.run(conn, E.build(parts))
                ok = (err is None and [list(r) for r in (got or [])] == expected)
                rec["patch_ok"] = bool(ok)
                history.append("patch %s -> %s" % (region, "GREEN" if ok else ("error" if err else "still wrong")))
                if ok:
                    green = True
        elif kind == "run":
            history.append("run")
        else:
            rec["action"] = "invalid"
        log.write(json.dumps(rec) + "\n"); log.flush()
        if green:
            break
    return {"run_id": run_id, "task_id": task["task_id"], "family": "sql", "green": green,
            "actions": step, "tokens_in": total_in, "tokens_out": total_out,
            "first_correct_hypothesis_step": first_correct, "n_regions": task["n_regions"],
            "chance_pct": task["chance_pct"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--head"); ap.add_argument("--head-layer", type=int, default=18)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import et8_agent as A
    head = None
    if a.head:
        import numpy as _np, mlx.core as _mx, et8_head_v3 as _H
        z = _np.load(a.head)
        head = {"w": z["w"], "b": float(z["b"][0]), "mu": z["mu"], "sd": z["sd"],
                "layer": a.head_layer, "hidden": _H.hidden_at, "mx": _mx}
        print("  head loaded: layer %d" % a.head_layer, file=sys.stderr)
    files = sorted(glob.glob(os.path.join(a.tasks, "task_*.json")))
    if a.limit: files = files[:a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    model, tok = A.load_model(a.model)
    ep = open(a.out + ".episodes.jsonl", "w"); st = open(a.out + ".steps.jsonl", "w")
    for i, f in enumerate(files, 1):
        t = json.load(open(f))
        r = run_episode(model, tok, t, a.budget, run_id, st, a.model, head_state=head)
        ep.write(json.dumps(r) + "\n"); ep.flush()
        print("[%d/%d] %s green=%s actions=%d" % (i, len(files), t["task_id"], r["green"], r["actions"]),
              file=sys.stderr)
    ep.close(); st.close()


if __name__ == "__main__":
    main()
