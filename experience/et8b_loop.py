#!/usr/bin/env python3
"""ET-8b — the ACCUMULATION loop. New files; et8_agent.py is untouched (R7's discipline).

Pre-registration: docs/et8b-loop-gates.md. 8a §9 showed accumulation could not be tested on a single
fixed decision — generation 1's training states were BYTE-IDENTICAL to generation 0's, because the
prior never changed what it later saw. This loop is the fix: the state at decision k depends on the
choices made at decisions < k, so a second generation trains on states its parent produced.

  episode = inspect -> DECISION 1 -> patch -> run -> (red?) inspect -> DECISION 2 -> patch -> run
            -> (red?) DECISION 3 -> patch -> run
  up to THREE head decisions; budget 12 MODEL ACTIONS total across them (fixed now, no ladder).

WHY A DECISION IS RECORDED EVEN WHEN THE HEAD IS ABSENT. G3 compares the base loop's decision-2
state against the head loop's decision-2 state on the SAME problem. That comparison only exists if
both arms log the state they were in, so `decision_state` is written by both.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et8_agent as A
import et8_env_v3 as V3
# BUG_CLASSES and the suite runner come from THE MODULE THE MEASURED AGENT USES (et8_agent.E),
# not from a module I picked. A loop that ran a different verifier than 8a would not be comparable.
E = A.E

DECISIONS = 3
BUDGET = 12


def decision_state_key(inspected, history, patched):
    """What distinguishes one decision from another, for G2 (distinctness) and G3 (dependence).

    It is the AGENT'S OWN SITUATION: what it has seen, what it has tried, and what the verifier said
    — not the task id. Two episodes of the SAME task that diverged at decision 1 have different
    decision-2 keys, and that difference is exactly what G3 measures.
    """
    return hashlib.sha1(json.dumps({
        "inspected": sorted(inspected), "history": history, "patched": patched
    }, sort_keys=True).encode()).hexdigest()


def run_episode(model, tok, task, run_id, log, model_id, head_state=None, budget=BUDGET):
    program = task["program"]
    regions = [r for r in task["regions"] if r != "run"]
    tests = task["tests"]
    inspected, history, patched = {}, [], []
    total_in = total_out = 0
    actions = 0
    green = False
    decisions_made = 0
    msgs = [{"role": "system", "content": A.SYSTEM % (", ".join(E.BUG_CLASSES), budget)}]
    cur_program = program
    first_correct = None

    def state_text():
        s = ["SYMPTOM: %s" % task["symptom"]]
        for r, v in inspected.items():
            s.append("# region: %s\n%s" % (r, v))
        if history:
            s.append("So far: " + " | ".join(history[-8:]))
        return "\n".join(s)

    for d in range(1, DECISIONS + 1):
        if actions >= budget or green:
            break
        # ---- inspect before every decision (the loop's own rule; decision 1 mirrors 8a's inspect-first)
        target = regions[0] if d == 1 else next((r for r in regions if r not in inspected), regions[0])
        if target not in inspected:
            inspected[target] = A.region_source(cur_program, target) or ""
            history.append("inspect %s" % target)
            actions += 1
            log.write(json.dumps({"run_id": run_id, "task_id": task["task_id"], "model": model_id,
                                  "decision": d, "action": "inspect", "region": target,
                                  "step": actions}) + "\n")
        if actions >= budget:
            break

        dkey = decision_state_key(inspected, history, patched)
        cur = msgs + [{"role": "user", "content": state_text()}]
        meta = {"decision": d, "state_key": dkey}

        if head_state is not None:
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
            total_in += ni; total_out += no; actions += 1
            gp = A.parse_action(text)
            if not (gp.get("action") == "hypothesize" and gp.get("region") == pick):
                text, ni, no = A.generate(model, tok, cur, temp=0.0, prefix=pres[pick])
                total_in += ni; total_out += no
            meta.update({"head": True, "head_pick": pick, "agent_pick": gp.get("region")})
        else:
            text, ni, no = A.generate(model, tok, cur, temp=0.0 if d == 1 else 0.7)
            total_in += ni; total_out += no; actions += 1

        act = A.parse_action(text)
        region = act.get("region") if act.get("action") == "hypothesize" else None
        hit = (region == task["bug_region"])
        if hit and first_correct is None:
            first_correct = d
        decisions_made += 1
        history.append("hypothesize %s" % (region or "?"))
        log.write(json.dumps({"run_id": run_id, "task_id": task["task_id"], "model": model_id,
                              "decision": d, "action": "hypothesize", "region": region,
                              "region_hit": bool(hit), "step": actions,
                              "decision_state": dkey, "candidates": meta,
                              "tokens_in": total_in, "tokens_out": total_out}) + "\n")
        if actions >= budget:
            break

        # ---- patch the region it named, then run the verifier
        pr = region if region in regions else regions[0]
        if pr not in inspected:
            inspected[pr] = A.region_source(cur_program, pr) or ""
        cur2 = msgs + [{"role": "user", "content": state_text() +
                        "\nNow patch region %s. Reply with the patch action only." % pr}]
        ptext, ni, no = A.generate(model, tok, cur2, temp=0.0 if d == 1 else 0.7,
                                   prefix='{"action": "patch", "region": "%s", "source": "' % pr)
        total_in += ni; total_out += no; actions += 1
        pa = A.parse_action(ptext)
        src = pa.get("source") or ""
        applied = False
        if src.strip() and src.strip() != (inspected.get(pr) or "").strip():
            newprog = A.replace_region(cur_program, pr, src)
            if newprog:
                cur_program = newprog; applied = True
        patched.append({"decision": d, "region": pr, "applied": bool(applied)})
        g, fails = E.run_tests(cur_program, tests)   # the agent's OWN verifier call (et8_agent:332)
        history.append("patch %s -> %s" % (pr, "GREEN" if g else "still red"))
        log.write(json.dumps({"run_id": run_id, "task_id": task["task_id"], "model": model_id,
                              "decision": d, "action": "patch" if applied else "noop_patch",
                              "region": pr, "patch_ok": bool(g), "step": actions,
                              "tokens_in": total_in, "tokens_out": total_out}) + "\n")
        if g:
            green = True
            break
        if pr in inspected:
            inspected[pr] = A.region_source(cur_program, pr) or inspected[pr]

    return {"run_id": run_id, "task_id": task["task_id"], "family": task.get("family", "v3"),
            "green": green, "actions": actions, "decisions": decisions_made,
            "tokens_in": total_in, "tokens_out": total_out,
            "first_correct_decision": first_correct,
            "agent_claims_pass": bool(green)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--head"); ap.add_argument("--head-layer", type=int, default=18)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    head = None
    if a.head:
        import numpy as _np, mlx.core as _mx, et8_head_v3 as _H
        z = _np.load(a.head)
        head = {"w": z["w"], "b": float(z["b"][0]), "mu": z["mu"], "sd": z["sd"],
                "layer": a.head_layer, "hidden": _H.hidden_at, "mx": _mx}
        print("  head: layer %d" % a.head_layer, file=sys.stderr)
    files = sorted(glob.glob(os.path.join(a.tasks, "task_*.json")))
    if a.limit: files = files[:a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    model, tok = A.load_model(a.model)
    ep = open(a.out + ".episodes.jsonl", "w"); st = open(a.out + ".steps.jsonl", "w")
    for i, f in enumerate(files, 1):
        t = json.load(open(f))
        r = run_episode(model, tok, t, run_id, st, a.model, head_state=head, budget=a.budget)
        ep.write(json.dumps(r) + "\n"); ep.flush(); st.flush()
        print("[%d/%d] %s green=%s decisions=%d actions=%d"
              % (i, len(files), t["task_id"], r["green"], r["decisions"], r["actions"]), file=sys.stderr)
    ep.close(); st.close()


if __name__ == "__main__":
    main()
