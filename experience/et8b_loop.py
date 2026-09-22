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
    """TRAJECTORY key — what the agent has done, independent of which task it is doing.

    ⚠️ THIS IS NOT G2's INSTRUMENT AND THE FIRST VERSION OF THIS FILE USED IT AS ONE. G2 asks for
    distinct decision PROMPTS per episode; this hashes region NAMES and history STRINGS and carries
    no task content, so it read 8 distinct values over 300 episodes at decision 1 and "failed" a
    gate that was never being measured. Kept because it is the right instrument for a different
    question — how much the agent's own trajectory varies — and logged beside the prompt hash.
    """
    return hashlib.sha1(json.dumps({
        "inspected": sorted(inspected), "history": history, "patched": patched
    }, sort_keys=True).encode()).hexdigest()


def prompt_key(state_text):
    """G2's and G3's instrument: a hash of the ACTUAL decision prompt the agent is about to read.

    It carries the task's symptom, the inspected sources, and the history — everything that makes
    one decision different from another. G3 compares this across arms on the SAME task, where the
    task content is held constant and only the trajectory can move it.
    """
    return hashlib.sha1(state_text.encode()).hexdigest()


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

        st_text = state_text()
        dkey = decision_state_key(inspected, history, patched)
        pkey = prompt_key(st_text)
        cur = msgs + [{"role": "user", "content": st_text}]
        meta = {"decision": d, "trajectory_key": dkey, "prompt_key": pkey}

        # a dict keyed by decision number is the PER-DECISION form; anything else is the shared head.
        hs_d = head_state.get(d) if isinstance(head_state, dict) and d in head_state else head_state
        if hs_d is not None and isinstance(hs_d, dict) and "w" in hs_d:
            head_state_local = hs_d
        elif isinstance(head_state, dict) and "w" in head_state:
            head_state_local = head_state
        else:
            head_state_local = None
        if head_state_local is not None:
            head_state = head_state if isinstance(head_state, dict) and "w" not in head_state else head_state
            hstate = head_state_local
            import numpy as _np
            scores, pres = {}, {}
            for r in regions:
                pre = '{"action": "hypothesize", "region": "%s", "bug_class": "' % r
                hs = hstate["hidden"](model, tok, cur, [hstate["layer"]], prefix=pre)
                z = _np.array(hs[hstate["layer"]].astype(hstate["mx"].float32), copy=False)
                z = (z.astype(_np.float64) - hstate["mu"]) / hstate["sd"]
                # §10's nonlinear arm: if the npz carries MLP weights, score through them.
                # The linear path is untouched — an npz without mlp_W1 behaves exactly as before,
                # so the two arms differ in the HEAD and in nothing else.
                if hstate.get("mlp_W1") is not None:
                    _h = z @ hstate["mlp_W1"] + hstate["mlp_b1"]
                    scores[r] = float(_np.maximum(_h, 0) @ hstate["mlp_W2"] + hstate["mlp_b2"])
                else:
                    scores[r] = float(z @ hstate["w"] + hstate["b"])
                pres[r] = pre
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
                              "decision_state": pkey, "trajectory_key": dkey, "candidates": meta,
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
            # `agent_claims_pass` USED TO BE bool(green) -- the VERIFIER's verdict under another
            # name. docs/et8b-loop-gates.md §3 labels the VII bridge arm by "the agent's OWN
            # judgment of success ... not the verifier", so an arm reading that field would have
            # agreed with gen1 BY CONSTRUCTION, and §4's table would have read the tautology as
            # "the verifier's bits were not what carried the gain". Nothing ever read it, so no
            # published number was affected; it was a trap, not a live error.
            #
            # The loop has no agent-side success signal at all: the harness runs the tests and
            # feeds the result straight back into the history, so the agent never forms an
            # independent belief here to record. The claim is collected OFFLINE instead, by
            # experience/et8b_selfassess.py, which replays each patch state with the verdict
            # withheld. (Measured there: the agent said "tests_pass": false on 568 of 568 points,
            # including all 83 where the patch worked, so its 85.4% agreement with the verifier IS
            # the 85.4% base rate. The arm is closed by that precondition.)
            "agent_claims_pass": None,
            "agent_claims_pass_NOTE": "not measured here; see experience/et8b_selfassess.py"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--head", help="ONE shared head for all decisions")
    ap.add_argument("--heads", nargs=3, metavar=("D1","D2","D3"),
                    help="THREE per-decision heads, in decision order. gen0's rule (spec §3) chose "
                         "per-decision on held-out accuracy, so this is the path the loop uses; "
                         "--head remains for the shared alternative the rule did not select.")
    ap.add_argument("--head-layer", type=int, default=18)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    head = None
    if a.head or a.heads:
        import numpy as _np, mlx.core as _mx, et8_head_v3 as _H
        def _load(p):
            z = _np.load(p)
            _k = z.files
            return {"w": z["w"], "b": float(z["b"][0]), "mu": z["mu"], "sd": z["sd"],
                    "mlp_W1": z["mlp_W1"] if "mlp_W1" in _k else None,
                    "mlp_b1": z["mlp_b1"] if "mlp_b1" in _k else None,
                    "mlp_W2": z["mlp_W2"] if "mlp_W2" in _k else None,
                    "mlp_b2": float(z["mlp_b2"][0]) if "mlp_b2" in _k else 0.0,
                    "layer": a.head_layer, "hidden": _H.hidden_at, "mx": _mx}
        if a.heads:
            head = {d + 1: _load(p) for d, p in enumerate(a.heads)}
            print("  heads: per-decision, layer %d, %d files" % (a.head_layer, len(a.heads)),
                  file=sys.stderr)
        else:
            head = _load(a.head)
            print("  head: shared, layer %d" % a.head_layer, file=sys.stderr)
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
