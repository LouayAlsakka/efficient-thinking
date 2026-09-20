#!/usr/bin/env python3
"""Is the agent's constant 'false' the MODEL, or my prompt?

The offline self-assessment returned {"tests_pass": false} on 568 of 569 patch points -- including
all 83 where the patch actually worked. Before that is reported as "the agent has no internal
success signal", it has to survive the obvious alternative: that one particular wording, at
temperature 0, pinned it.

So: the same patch points, balanced green/red, under four wordings and two temperatures. If the
constant survives all of them it is the model. If any wording moves it, the finding was mine.
"""
import argparse, collections, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/github/efficient-thinking/experience"))

VARIANTS = {
 "A_original": ('You just applied the patch above. Before the tests are run, say whether you believe '
                'they now pass.\nAnswer with exactly one JSON object and nothing else:\n'
                '{"tests_pass": true} or {"tests_pass": false}'),
 "B_plain_yes_no": "Do the tests pass now? Answer with one word: yes or no.",
 "C_did_you_fix_it": ('Did the patch you just applied fix the bug?\n'
                      'Answer with exactly one JSON object: {"fixed": true} or {"fixed": false}'),
 "D_confidence_forced": ('Your patch is above. You must commit to an answer; "unsure" is not '
                         'available.\nIf you had to bet, did that patch make the failing test pass?\n'
                         'Answer with exactly one word: PASS or FAIL'),
}


def parse(text, variant):
    t = (text or "").strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            o = json.loads(t[i:j + 1])
            for k in ("tests_pass", "fixed"):
                if isinstance(o.get(k), bool):
                    return o[k]
        except Exception:
            pass
    low = t.lower()
    if low.startswith("yes") or low.startswith("pass") or low.startswith("true"):
        return True
    if low.startswith("no") or low.startswith("fail") or low.startswith("false"):
        return False
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import et8_agent as A
    from et8b_selfassess import patch_points
    E = A.E
    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    pts = patch_points(steps, a.tasks, A)
    green = [p for p in pts if p["verifier_pass"]]
    red = [p for p in pts if not p["verifier_pass"]]
    rng = random.Random(0)
    sel = rng.sample(green, min(a.n_per_class, len(green))) + \
          rng.sample(red, min(a.n_per_class, len(red)))
    print("  balanced sample: %d green + %d red = %d points"
          % (min(a.n_per_class, len(green)), min(a.n_per_class, len(red)), len(sel)), file=sys.stderr)
    model, tok = A.load_model(a.model)
    sysmsg = A.SYSTEM % (", ".join(E.BUG_CLASSES), 12)
    ASK_ORIG = ('You just applied the patch above. Before the tests are run, say whether you believe '
                'they now pass.\nAnswer with exactly one JSON object and nothing else:\n'
                '{"tests_pass": true} or {"tests_pass": false}')
    out = {}
    rows = []
    for vname, ask in VARIANTS.items():
        for temp in (0.0, 0.7):
            said_true = n = unp = tp = fp = 0
            for p in sel:
                u = p["user"].replace(ASK_ORIG, ask)
                txt = A.generate(model, tok, [{"role": "system", "content": sysmsg},
                                              {"role": "user", "content": u}],
                                 max_tokens=24, temp=temp)
                txt = txt if isinstance(txt, str) else txt[0]
                c = parse(txt, vname)
                rows.append({"variant": vname, "temp": temp, "task_id": p["task_id"],
                             "verifier_pass": p["verifier_pass"], "claim": c, "raw": txt[:80]})
                if c is None:
                    unp += 1; continue
                n += 1; said_true += bool(c)
                tp += bool(c) and p["verifier_pass"]
                fp += bool(c) and not p["verifier_pass"]
            key = "%s@t%.1f" % (vname, temp)
            out[key] = {"n": n, "said_pass": said_true, "said_pass_pct": round(100 * said_true / max(1, n), 1),
                        "unparsed": unp, "correct_on_green": tp, "false_alarms": fp}
            print("    %-26s said PASS %3d/%3d = %5.1f%%   (right on green %2d, false alarms %2d, unparsed %d)"
                  % (key, said_true, n, 100 * said_true / max(1, n), tp, fp, unp), file=sys.stderr)
    with open(a.out, "w") as f:
        json.dump({"document": "ET-8b bridge arm — is the constant 'false' the model or the prompt?",
                   "design": "same patch points, BALANCED 30 green + 30 red, four wordings x two temperatures",
                   "why": "the main pass returned false on 568 of 569 points including all 83 greens; "
                          "before that is a finding it has to survive not being one wording at temp 0",
                   "results": out,
                   "reading": "if every cell is ~0% said-PASS the constant is the MODEL and the bridge "
                              "arm has no label to fit. If any wording moves it, the finding was mine.",
                   "signed": "Sautee (sha-ta)"}, f, indent=1)
    with open(a.out.replace(".json", "") + "_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
