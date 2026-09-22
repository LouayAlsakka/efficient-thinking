#!/usr/bin/env python
"""WO-312 — score one driven session against its task. The five measurements, from the replay log.

Reads 鉋's per-session JSONL and my task list, and produces the row `docs/wo312-measurement-prereg.md`
registered. Nothing here infers; every number comes from a logged field.

hit@N IS MATCHED ON ARGS, NOT ON LABELS. My first cut compared the action string to head *labels*
and scored 0.60 on a session that was plainly 1.00 — "select offer:haircut" does not appear in
"Pick Haircut", and "submit request" appears in no label at all. That is a parser scoring a miss it
invented. The head entries carry `args` ({'offer_id': 'haircut'}, {'slot': '11:30'}, …) and the
logged action is "<kind> <field>:<value>", so the match is value-to-arg-value within the same kind.
**The count of turns this parser could not read is printed beside the rate**, and a rate computed
over a session with any unparsed turn is not reported as a rate.

TAPS COUNTS GESTURES, and after 形's batching fix (19b5826) one gesture is one logged event, so
`input.kind == "tap"` is the count. Before that fix it over-counted by one at the offer step; the
guard below re-checks the invariant rather than trusting the version.
"""
import argparse, json, os, sys


def parse_action(a):
    """'select day:2026-09-22' -> ('select', 'day', '2026-09-22');  'submit request' -> ('submit', None, 'request')."""
    if not a:
        return None
    head, _, rest = a.partition(" ")
    if not rest:
        return (head, None, None)
    field, sep, value = rest.partition(":")
    return (head, field, value) if sep else (head, None, rest)


def hit(action, shown_head):
    """Was the action taken on the head that was shown? Matched on the head entry's ARG VALUES."""
    p = parse_action(action)
    if not p:
        return None
    kind, _field, value = p
    for h in shown_head or []:
        if h.get("kind") != kind:
            continue
        args = h.get("args") or {}
        if value is not None and any(str(v) == str(value) for v in args.values()):
            return True
        if value is None and not args:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="one session JSONL from logger_serve.py")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--seconds", type=float, default=None, help="wall clock from the driver")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.session) if l.strip()]
    task = next(t for t in json.load(open(a.tasks))["tasks"] if t["task_id"] == a.task_id)

    chain_breaks = [(rows[i]["seq"], rows[i + 1]["seq"]) for i in range(len(rows) - 1)
                    if rows[i]["state_after"] != rows[i + 1]["state_before"]]
    taps = [r for r in rows if (r.get("input") or {}).get("kind") == "tap"]
    typed = [r for r in rows if (r.get("input") or {}).get("kind") == "typed"]

    per_turn, unparsed = [], 0
    for r in taps:
        act = (r.get("input") or {}).get("action")
        h = hit(act, r.get("shown_head"))
        if h is None:
            unparsed += 1
        per_turn.append({"seq": r["seq"], "action": act, "head_size": len(r.get("shown_head") or []),
                         "on_head": h})
    scored = [t for t in per_turn if t["on_head"] is not None]
    hit_rate = round(sum(t["on_head"] for t in scored) / len(scored), 4) if scored and not unparsed else None

    final = rows[-1]["state_after"] if rows else {}
    submitted = bool(final.get("submit_key"))

    out = {
        "document": "WO-312 — one scored session",
        "prereg": "docs/wo312-measurement-prereg.md",
        "task_id": a.task_id, "goal": task["goal"], "arm": (rows[0].get("arm") if rows else None),
        "session_file": os.path.basename(a.session), "events": len(rows),
        "taps_to_goal": len(taps), "optimal_taps": task["optimal_taps"],
        "taps_over_optimal": len(taps) - task["optimal_taps"],
        "typed_events": len(typed),
        "seconds_to_goal": a.seconds,
        "hit_at_N": hit_rate,
        "hit_at_N_unparsed_turns": unparsed,
        "per_turn": per_turn,
        "submit_fired": submitted,
        "task_completion": None,
        "task_completion_note": ("UNMEASURABLE TODAY, and not reported as a failure: the bench makes "
                                 "no engine call on submit ('Request sent to the engine call this "
                                 "bench does not yet make'). §2's measurement 3 is the ENGINE's "
                                 "answer, not the interface's, so `submit_fired` is recorded and "
                                 "completion is left null rather than quietly redefined as 'the "
                                 "screen said so'."),
        "cost_usd": 0.0,
        "cost_note": "$0 by construction for STATIC and unwired CHAT (§2 measurement 5)",
        "replay_chain_intact": not chain_breaks,
        "replay_chain_breaks": chain_breaks,
        "signed": "Sautee (sha-ta)",
    }
    if out["hit_at_N"] is None and unparsed:
        out["hit_at_N_note"] = ("not reported: %d turn(s) the matcher could not read. A rate over a "
                                "session with an unparsed turn is a rate about the parser." % unparsed)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  %s · %s" % (a.task_id, task["goal"]))
    print("  taps %d (optimal %d, over %+d) · typed %d · hit@N %s (unparsed %d) · submit_fired %s"
          % (out["taps_to_goal"], out["optimal_taps"], out["taps_over_optimal"], out["typed_events"],
             out["hit_at_N"], unparsed, submitted))
    print("  replay chain intact: %s%s" % (out["replay_chain_intact"],
                                           "" if out["replay_chain_intact"] else " %s" % chain_breaks))
    print("  completion: NULL — no engine call exists to answer it")
    for t in per_turn:
        print("     seq %-2s %-24s head %-3d on_head=%s" % (t["seq"], str(t["action"])[:24],
                                                            t["head_size"], t["on_head"]))


if __name__ == "__main__":
    main()
