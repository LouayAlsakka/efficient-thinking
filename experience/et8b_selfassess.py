#!/usr/bin/env python3
"""ET-8b bridge arm (VII) — the agent's OWN judgment of success, collected offline.

WHY THIS FILE EXISTS. docs/et8b-loop-gates.md §3 labels the bridge arm by "the agent's OWN judgment
of success (its final 'tests pass' claim, not the verifier)". The loop has no such signal:
et8b_loop.py writes `agent_claims_pass = bool(green)`, which is the VERIFIER'S verdict under a
different name. An arm labelled by that agrees with gen1 by construction, and §4's reading table
would have turned a tautology into "the verifier's bits were not what carried the gain".

So the claim is collected here, offline, over the trajectories that already exist — which keeps
§3's "same states, everything else identical": the rollouts are not re-run and not altered.

THE PRECONDITION, MEASURED BEFORE ANY HEAD IS FITTED. If the agent's self-assessment agrees with the
verifier almost always, the bridge arm is near-tautological no matter how it is labelled and it
cannot answer VII's question. This script reports that agreement rate FIRST. A useful arm needs the
agent to be wrong about itself a substantial fraction of the time; how much is 理's call, but the
number has to be on the table before the arm is built on top of it.

WHAT THE MODEL IS SHOWN. Exactly the state after its own patch and BEFORE the test result: the
symptom, the regions it inspected, the history with the verdict suffix STRIPPED ("patch X" and not
"patch X -> GREEN"). Leaking the suffix would be asking the verifier again.
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ASK = ("You just applied the patch above. Before the tests are run, say whether you believe they "
       "now pass.\nAnswer with exactly one JSON object and nothing else:\n"
       '{"tests_pass": true} or {"tests_pass": false}')


def patch_points(steps, tasks_dir, agent):
    """Rebuild, for every patch step, the state the agent held BEFORE seeing the verdict."""
    by = collections.defaultdict(list)
    for s in steps:
        by[(s["run_id"], s["task_id"])].append(s)
    out = []
    for (rid, tid), st in by.items():
        st.sort(key=lambda s: s.get("step", 0))
        task = json.load(open(os.path.join(tasks_dir, tid + ".json")))
        program = task["program"]
        inspected, history = {}, []
        for s in st:
            act = s.get("action")
            if act == "inspect" and s.get("region"):
                inspected[s["region"]] = agent.region_source(program, s["region"]) or ""
                history.append("inspect %s" % s["region"])
            elif act == "hypothesize":
                history.append("hypothesize %s" % (s.get("region") or "?"))
            elif act in ("patch", "noop_patch"):
                lines = ["SYMPTOM: %s" % task["symptom"]]
                for r, v in inspected.items():
                    lines.append("# region: %s\n%s" % (r, v))
                # the patch itself is in the history; the VERDICT is not, and must not be
                h = history + ["patch %s" % s.get("region")]
                lines.append("So far: " + " | ".join(h[-8:]))
                lines.append("")
                lines.append(ASK)
                out.append({"task_id": tid, "run_id": rid, "decision": s.get("decision"),
                            "region": s.get("region"), "action": act,
                            "verifier_pass": bool(s.get("patch_ok")),
                            "bug_region": task["bug_region"],
                            "user": "\n".join(lines)})
                history.append("patch %s -> %s"
                               % (s.get("region"), "GREEN" if s.get("patch_ok") else "still red"))
    return out


def parse_claim(text):
    """Must BE a verdict, not contain one. Unparseable is None and is reported, never guessed."""
    t = (text or "").strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            v = json.loads(t[i:j + 1]).get("tests_pass")
            if isinstance(v, bool):
                return v
        except Exception:
            pass
    low = t.lower()
    if low.startswith("true") or low.startswith("yes"):
        return True
    if low.startswith("false") or low.startswith("no"):
        return False
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N patch points, for a smoke run")
    a = ap.parse_args()
    import et8_agent as A
    E = A.E
    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    pts = patch_points(steps, a.tasks, A)
    if a.limit:
        pts = pts[:a.limit]
    print("  patch points: %d" % len(pts), file=sys.stderr)
    model, tok = A.load_model(a.model)
    sysmsg = A.SYSTEM % (", ".join(E.BUG_CLASSES), 12)
    t0 = time.time()
    with open(a.out, "w") as f:
        n_un = 0
        for i, p in enumerate(pts):
            txt = A.generate(model, tok, [{"role": "system", "content": sysmsg},
                                          {"role": "user", "content": p["user"]}],
                             max_tokens=24, temp=0.0)
            claim = parse_claim(txt if isinstance(txt, str) else txt[0])
            n_un += claim is None
            rec = {k: p[k] for k in ("task_id", "run_id", "decision", "region", "action",
                                     "verifier_pass", "bug_region")}
            rec["agent_claims_pass"] = claim
            rec["raw"] = (txt if isinstance(txt, str) else txt[0])[:120]
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 100 == 0:
                r = (time.time() - t0) / (i + 1)
                print("  %d/%d  %.2fs/pt  eta %.0fm  unparsed %d"
                      % (i + 1, len(pts), r, r * (len(pts) - i - 1) / 60, n_un), file=sys.stderr)

    rows = [json.loads(l) for l in open(a.out)]
    ok = [r for r in rows if r["agent_claims_pass"] is not None]
    agree = sum(1 for r in ok if r["agent_claims_pass"] == r["verifier_pass"])
    tp = sum(1 for r in ok if r["agent_claims_pass"] and r["verifier_pass"])
    fp = sum(1 for r in ok if r["agent_claims_pass"] and not r["verifier_pass"])
    fn = sum(1 for r in ok if not r["agent_claims_pass"] and r["verifier_pass"])
    print("\n  THE PRECONDITION — does the agent's own judgment differ from the verifier's?",
          file=sys.stderr)
    print("    parsed %d of %d (unparsed %d, dropped not guessed)" % (len(ok), len(rows),
          len(rows) - len(ok)), file=sys.stderr)
    if ok:
        print("    agreement with the verifier   %d/%d = %.1f%%" % (agree, len(ok),
              100 * agree / len(ok)), file=sys.stderr)
        print("    claims pass, actually red     %d   (%.1f%%)" % (fp, 100 * fp / len(ok)),
              file=sys.stderr)
        print("    claims fail, actually green   %d   (%.1f%%)" % (fn, 100 * fn / len(ok)),
              file=sys.stderr)
        print("    claims pass and IS green      %d" % tp, file=sys.stderr)
        print("\n    READING: at ~100%% agreement the bridge arm is a tautology however it is "
              "labelled\n    and cannot answer VII's question. The disagreement rate is the arm's "
              "headroom.", file=sys.stderr)
    json.dump({"document": "ET-8b bridge arm precondition — agent self-assessment vs the verifier",
               "why": "et8b_loop.py's agent_claims_pass is bool(green), the verifier renamed; the "
                      "claim had to be collected separately or the arm labels itself with the "
                      "thing it is supposed to be independent of",
               "patch_points": len(rows), "parsed": len(ok), "unparsed": len(rows) - len(ok),
               "agreement_pct": round(100 * agree / len(ok), 1) if ok else None,
               "claims_pass_actually_red": fp, "claims_fail_actually_green": fn,
               "reading": "at ~100% agreement the arm is a tautology; the disagreement rate is its headroom",
               "signed": "Sautee (sha-ta)"},
              open(a.out.replace(".jsonl", "") + "_precondition.json", "w"), indent=1)


if __name__ == "__main__":
    main()
