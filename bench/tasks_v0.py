#!/usr/bin/env python
"""WO-312 — the scripted task list, with a known optimal path so taps-to-goal has a floor.

WO-312: "First a scripted user (a task list with a known optimal path, so taps-to-goal has a floor)."
This builds that list against 庭's compiled fixture — the same file the bench renders, verified by
md5 rather than by eye — and computes each task's optimal action sequence in WO-264's vocabulary.

TWO THINGS THIS FILE EXISTS TO SAY OUT LOUD, BOTH BEFORE ANY RUN:

1. **The optimal tap count is CONSTANT on this world.** quick-cuts.chelsea sells ONE offer, so the
   path is offer -> staff -> day -> slot -> identity -> submit for every target that exists. Taps to
   goal therefore has ZERO variance across tasks for a correct arm: it can only go UP (a wrong
   guess) and, for PREDICTED, down by collapsing a step. The WO's headline measure cannot separate
   two arms that both navigate correctly here. **hit@N and seconds-to-goal carry the signal on this
   venue; taps-to-goal discriminates only once a SECOND venue with more than one offer exists** —
   which is also the WO's own programmability test, so it is not extra work, it is the next step.

2. **A task list keyed to "today" is not a fixture.** The bench's day strip is built from the run
   date, and Mondays are closed. A task naming 2026-09-26 is unreachable on a run three weeks later.
   Targets are therefore expressed as (staff, Nth OPEN day, Nth slot) and resolved to dates only for
   a stated `--from`, which is stamped into the artefact.

The DEPTH features (how far the target sits from the default surface) are recorded per task because
they are the only thing that varies here, and they are what a predicted head could exploit.
"""
import argparse, hashlib, json, os, random, sys

DAY_NAMES = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]


def open_days(hours, start, n_open):
    """The next n_open OPEN days from `start` (a date), by the fixture's declared hours."""
    import datetime
    out, d = [], start
    while len(out) < n_open:
        wd = DAY_NAMES[(d.weekday() + 1) % 7]          # python Monday=0 -> DAY_NAMES Sunday=0
        if wd in hours:
            out.append({"date": d.isoformat(), "weekday": wd})
        d += datetime.timedelta(days=1)
    return out


def slots_for(hours, weekday, grid):
    lo, hi = hours[weekday]
    return ["%02d:%02d" % (m // 60, m % 60) for m in range(lo * 60, hi * 60, grid)]


def main():
    import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="the compiled venue the bench renders")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--open-days", type=int, default=7, help="how many OPEN days a task may target")
    ap.add_argument("--from", dest="frm", default="2026-09-23", help="reference date; stamped, not implicit")
    ap.add_argument("--seed", type=int, default=312)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = open(a.space, "rb").read()
    space = json.loads(raw)
    tx = space["transaction"]
    hours = tx["schedule"]["declared"]["hours"]
    grid = tx["schedule"]["declared"]["slot_grid_min"]
    roster = tx["resource"]["roster"]
    offers = space["offer_sheet"]
    start = datetime.date.fromisoformat(a.frm)
    days = open_days(hours, start, a.open_days)

    targets = []
    for si, staff in enumerate(roster):
        for di, day in enumerate(days):
            sl = slots_for(hours, day["weekday"], grid)
            for ti, t in enumerate(sl):
                targets.append({"staff": staff, "staff_index": si,
                                "open_day_index": di, "weekday": day["weekday"], "date": day["date"],
                                "slot": t, "slot_index": ti, "slots_that_day": len(sl)})
    rng = random.Random(a.seed)

    # STRATIFY on the only things that vary: which staff, how far down the day strip, and where in
    # the day. An unstratified sample of 100 from 378 would leave the late-day tail thin, and the
    # late tail is exactly where a predicted head would earn its keep.
    def bucket(t):
        third = 0 if t["slot_index"] < t["slots_that_day"] / 3 else (
            1 if t["slot_index"] < 2 * t["slots_that_day"] / 3 else 2)
        return (t["staff_index"], min(t["open_day_index"], 3), third)
    by = {}
    for t in targets:
        by.setdefault(bucket(t), []).append(t)
    keys = sorted(by)
    for k in keys:
        rng.shuffle(by[k])
    picked, i = [], 0
    while len(picked) < a.n:
        k = keys[i % len(keys)]
        if by[k]:
            picked.append(by[k].pop())
        i += 1
        if i > len(keys) * 200:
            break

    offer = offers[0]
    tasks = []
    for n, t in enumerate(picked):
        path = [
            # ONE TAP, TWO DISPATCHES: the offer press fires SELECT offer_ids AND OPEN_SHEET book.
            # taps counts the press; the logger will see two reducer actions for it.
            {"kind": "select", "label": "Pick %s" % offer["name"], "args": {"offer_id": offer["offer_id"]},
             "dispatches": ["SELECT offer_ids", "OPEN_SHEET book"]},
            {"kind": "select", "label": "Book with %s" % t["staff"], "args": {"staff": t["staff"]}},
            {"kind": "select", "label": t["date"], "args": {"day": t["date"]}},
            # THE SLOT ARG IS "HH:MM", NOT AN ISO DATETIME. Read out of the bench's own dispatch:
            # onSelect: (t) => dispatch({verb:'SELECT', field:'slot', value:t}) where t comes
            # straight from slotsFor(weekday), i.e. "09:00". My first version wrote
            # "2026-09-23T09:00" and would have driven nothing — a task list expressed in terms the
            # thing it measures cannot consume. Checked against the running UI, not the fixture.
            {"kind": "select", "label": t["slot"], "args": {"slot": t["slot"]}},
            {"kind": "form-fill", "label": "name and phone", "args": {"form": "book"}},
            {"kind": "submit", "label": "Request this booking", "args": {"action": "request"}},
        ]
        tasks.append({
            "task_id": "qc-%03d" % (n + 1),
            "goal": "Book %s with %s on %s (%s) at %s" % (offer["name"], t["staff"], t["date"],
                                                          t["weekday"], t["slot"]),
            "target": {k: t[k] for k in ("staff", "date", "weekday", "slot")},
            "optimal_path": path,
            "optimal_taps": sum(1 for s in path if s["kind"] != "form-fill"),
            "depth": {"staff_index": t["staff_index"], "open_day_index": t["open_day_index"],
                      "slot_index": t["slot_index"], "slots_that_day": t["slots_that_day"],
                      "slot_fraction_into_day": round(t["slot_index"] / max(1, t["slots_that_day"] - 1), 3)},
        })

    taps = sorted({t["optimal_taps"] for t in tasks})
    out = {
        "document": "WO-312 — scripted task list v0, quick-cuts.chelsea",
        "prereg": "docs/wo312-measurement-prereg.md",
        "space_file": os.path.basename(a.space), "space_md5": hashlib.md5(raw).hexdigest(),
        "reference_date": a.frm, "open_days_offered": a.open_days, "seed": a.seed,
        "n_tasks": len(tasks), "n_reachable_targets": len(targets),
        "offer_count": len(offers), "roster": roster,
        "OPTIMAL_TAPS_IS_CONSTANT": {
            "values_present": taps,
            "why": ("one offer means one path: offer -> staff -> day -> slot -> identity -> submit. "
                    "Taps-to-goal has no variance across tasks for an arm that navigates correctly, "
                    "so it cannot separate two correct arms on this venue. Registered before the run: "
                    "hit@N and seconds-to-goal carry the signal here; taps-to-goal becomes a "
                    "discriminating measure only on a second venue with more than one offer."),
        },
        "what_varies": ("only DEPTH — which of 3 staff, how far down the open-day strip, and how far "
                        "into the day's %d slots. Those are recorded per task because they are the "
                        "only thing a predicted head could exploit." % len(slots_for(hours, days[0]["weekday"], grid))),
        "stratified_on": "(staff_index, min(open_day_index,3), slot third)",
        "tasks": tasks,
        "verified_against_running_ui": ("the arg shapes are read from the bench's own dispatch calls in "
                                       "src/main.tsx, not inferred from the fixture: day takes a date "
                                       "string, slot takes HH:MM, and the offer press fires two reducer "
                                       "actions for one tap"),
        "signed": "Sautee (sha-ta)",
    }
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  %d tasks over %d reachable targets · optimal taps %s (constant) · md5 %s"
          % (len(tasks), len(targets), taps, out["space_md5"][:8]))
    import collections
    print("  staff:", dict(collections.Counter(t["target"]["staff"] for t in tasks)))
    print("  day index:", dict(sorted(collections.Counter(t["depth"]["open_day_index"] for t in tasks).items())))
    print("  slot third:", dict(sorted(collections.Counter(
        0 if t["depth"]["slot_fraction_into_day"] < 1/3 else (1 if t["depth"]["slot_fraction_into_day"] < 2/3 else 2)
        for t in tasks).items())))
    print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
