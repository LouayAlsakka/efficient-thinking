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
    # DEFAULT IS TODAY, because the bench's day strip is nextDays(TODAY, 14) — verified against the
    # running UI, whose first chip was the run date itself. A list generated from a different
    # reference date still names reachable days, but its open_day_index is then off by one against
    # the strip the user actually sees, and open_day_index is one of only three things that vary here.
    ap.add_argument("--from", dest="frm", default=str(__import__("datetime").date.today()),
                    help="reference date; stamped, not implicit. Defaults to today.")
    ap.add_argument("--seed", type=int, default=312)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = open(a.space, "rb").read()
    space = json.loads(raw)
    tx = space["transaction"]
    hours = tx["schedule"]["declared"]["hours"]
    grid = tx["schedule"]["declared"]["slot_grid_min"]
    # .get, not [] — a venue with resource.kind "none" HAS NO ROSTER KEY. The synthetic
    # no-staff probe crashed here, which is what the probe was for.
    roster = (tx.get("resource") or {}).get("roster") or []
    offers = space["offer_sheet"]
    start = datetime.date.fromisoformat(a.frm)
    days = open_days(hours, start, a.open_days)

    # THE PATH IS READ FROM THE TRANSACTION BLOCK, NOT ASSUMED. Every offer in a venue shares that
    # block, which is why three offers do not change the depth (理 12184, on the schema at a2a
    # ed97c406): `resource.kind: "none"` or `select: "auto"` removes the staff step, and
    # granularity decides whether a day and a time are chosen at all. A generator that hardcodes
    # offer -> staff -> day -> slot -> submit measures v0 and nothing else, and the WO's own
    # acceptance test is that a NEW compiled definition works with zero code either side. This is
    # that test applied to the harness.
    res = tx.get("resource") or {}
    picks_staff = res.get("kind") not in (None, "none") and res.get("select") != "auto"
    gran = (tx.get("schedule") or {}).get("granularity", "slot")
    picks_day = gran in ("slot", "date_range", "occurrence")
    picks_time = gran == "slot"
    if not picks_staff:
        roster = []

    targets = []
    for oi, offer in enumerate(offers):
        staff_opts = roster if picks_staff else [None]
        for si, staff in enumerate(staff_opts):
            for di, day in enumerate(days if picks_day else days[:1]):
                # RESPECT duration_min: a 60-minute offer on a 30-minute grid cannot start in the
                # last slot before closing. v0 could never show this — one offer, 30 min, 30-min
                # grid — and v1's dry-clean exposed it: 6 of 396 slot-offer pairs a week would run
                # past closing, and the engine agrees (庭 measured 242 / 231 / 242 bookable, the
                # 231 being exactly one lost slot a day). A task list that names a slot the engine
                # refuses is a task the scripted user cannot complete, and §4 would read that as
                # the WORLD being broken.
                sl = slots_for(hours, day["weekday"], grid) if picks_time else [None]
                if picks_time:
                    dm = offer.get("duration_min") or grid
                    close_min = hours[day["weekday"]][1] * 60
                    sl = [t_ for t_ in sl
                          if int(t_[:2]) * 60 + int(t_[3:]) + dm <= close_min]
                for ti, t in enumerate(sl):
                    targets.append({"offer": offer, "offer_index": oi,
                                    "staff": staff, "staff_index": si,
                                    "open_day_index": di, "weekday": day["weekday"], "date": day["date"],
                                    "slot": t, "slot_index": ti, "slots_that_day": len(sl)})
    rng = random.Random(a.seed)

    # STRATIFY on the only things that vary: which staff, how far down the day strip, and where in
    # the day. An unstratified sample of 100 from 378 would leave the late-day tail thin, and the
    # late tail is exactly where a predicted head would earn its keep.
    def bucket(t):
        third = 0 if t["slot_index"] < t["slots_that_day"] / 3 else (
            1 if t["slot_index"] < 2 * t["slots_that_day"] / 3 else 2)
        return (t["offer_index"], t["staff_index"], min(t["open_day_index"], 3), third)
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

    tasks = []
    for n, t in enumerate(picked):
        offer = t["offer"]
        path = [{"kind": "select", "label": "Pick %s" % offer["name"],
                 "args": {"offer_id": offer.get("offer_id") or offer["name"]},
                 "dispatches": ["SELECT offer_ids", "OPEN_SHEET book"]}]
        if picks_staff:
            path.append({"kind": "select", "label": "Book with %s" % t["staff"],
                         "args": {"staff": t["staff"]}})
        if picks_day:
            path.append({"kind": "select", "label": t["date"], "args": {"day": t["date"]}})
        if picks_time:
            path.append({"kind": "select", "label": t["slot"], "args": {"slot": t["slot"]}})
        path.append({"kind": "form-fill", "label": "name and phone", "args": {"form": "book"}})
        path.append({"kind": "submit", "label": "Request this booking", "args": {"action": "request"}})
        goal = "Book %s%s on %s (%s)%s" % (
            offer["name"], " with %s" % t["staff"] if t["staff"] else "", t["date"], t["weekday"],
            " at %s" % t["slot"] if t["slot"] else "")
        tasks.append({
            "task_id": "qc-%03d" % (n + 1), "goal": goal,
            "target": {"offer_id": offer.get("offer_id") or offer["name"], "staff": t["staff"],
                       "date": t["date"], "weekday": t["weekday"], "slot": t["slot"]},
            "optimal_path": path,
            "optimal_taps": sum(1 for s_ in path if s_["kind"] != "form-fill"),
            "depth": {"offer_index": t["offer_index"], "staff_index": t["staff_index"],
                      "open_day_index": t["open_day_index"],
                      "slot_index": t["slot_index"], "slots_that_day": t["slots_that_day"],
                      "slot_fraction_into_day": round(t["slot_index"] / max(1, t["slots_that_day"] - 1), 3)},
        })

    taps = sorted({t["optimal_taps"] for t in tasks})
    out = {
        "document": "WO-312 — scripted task list v0, quick-cuts.chelsea",
        "prereg": "docs/wo312-measurement-prereg.md",
        "space_file": os.path.basename(a.space), "space_md5": hashlib.md5(raw).hexdigest(),
        "reference_date": a.frm, "reference_date_is_today": a.frm == str(datetime.date.today()), "open_days_offered": a.open_days, "seed": a.seed,
        "n_tasks": len(tasks), "n_reachable_targets": len(targets),
        "offer_count": len(offers), "roster": roster,
        "path_shape_from_transaction": {"picks_staff": picks_staff, "picks_day": picks_day,
                                        "picks_time": picks_time, "granularity": gran,
                                        "resource_kind": res.get("kind"), "resource_select": res.get("select")},
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
        "stratified_on": "(offer_index, staff_index, min(open_day_index,3), slot third)",
        "duration_respected": ("slots whose start + the offer's duration_min runs past closing are "
                               "not offered as targets; v1's 60-minute dry-clean on a 30-minute grid "
                               "loses the last slot of each day, which is what the engine reports too"),
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
    print("  offers:", dict(collections.Counter(t["target"]["offer_id"] for t in tasks)))
    print("  day index:", dict(sorted(collections.Counter(t["depth"]["open_day_index"] for t in tasks).items())))
    print("  slot third:", dict(sorted(collections.Counter(
        0 if t["depth"]["slot_fraction_into_day"] < 1/3 else (1 if t["depth"]["slot_fraction_into_day"] < 2/3 else 2)
        for t in tasks).items())))
    print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
