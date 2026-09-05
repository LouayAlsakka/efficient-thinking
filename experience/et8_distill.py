#!/usr/bin/env python3
"""
et8_distill.py — Efficient Thinking VIII DISTILL stage v0 (proposal §3.2): trajectories → candidate lessons.

Reads one or more harness runs (<run>.steps.jsonl + <run>.episodes.jsonl), labels episodes τ+ / τ−, clusters decisions by
CONDITION = (family, symptom_region), and emits CANDIDATE lessons in the record schema — the two lists, "what worked" and
"what didn't", with the fields that keep them honest:

  condition            where the lesson applies (family, symptom_region)
  action_prior         "what worked": the first region/class that led to green, ranked by (green rate, actions)
  action_avoid         "what didn't": regions/classes that consumed budget without green (fixation targets, noop patches)
  alternatives_tried   the regions actually tried in this condition — a "works" with one alternative is weak (proposal §5, review §6.1)
  evidence             n_episodes, n_green, runs, models — counts, not adjectives
  confound_check       filled by et8_verify.py (held-out replay); "UNVERIFIED" here by construction
  scope                the condition + the models it was measured on
  confidence           n_green / n_episodes shrunk toward 0.5 by (n / (n + k)), k = 5  — one episode is an anecdote
  expiry               rounds without fresh evidence before confidence decays (γ = 0.9 / round)
  reopen_condition     for dead paths: what would reopen them ("green from this region in ≥ 2 held-out episodes")
  status               "candidate" until verified; never "admitted" from this script

Nothing here is admitted into a prior. This script produces the auditable intermediate object; the gate is et8_verify.py.

Usage:
  python experience/et8_distill.py --runs experience/traj/v03_3b experience/traj/v03_7b --out experience/lessons/v0
    -> lessons.json (records), lessons.txt (text-memory rendering for mechanism A), report.json (cluster stats)
"""
from __future__ import annotations
import argparse, json, os, collections, statistics

K_SHRINK = 5.0

def load_run(prefix: str):
    steps = [json.loads(l) for l in open(prefix + ".steps.jsonl")]
    eps = [json.loads(l) for l in open(prefix + ".episodes.jsonl")]
    return steps, eps

def label_episodes(eps, steps):
    """τ+ = green and actions ≤ median of green episodes; τ− = not green, or green at > 1.5 × median."""
    green_actions = [e["actions"] for e in eps if e["green"]]
    med = statistics.median(green_actions) if green_actions else None
    by_task = collections.defaultdict(list)
    for s in steps: by_task[(s["run_id"], s["task_id"])].append(s)
    out = []
    for e in eps:
        st = sorted(by_task[(e["run_id"], e["task_id"])], key=lambda s: s["step"])
        if e["green"] and med is not None and e["actions"] <= med: lab = "tau+"
        elif e["green"] and med is not None and e["actions"] > 1.5 * med: lab = "tau-"
        elif e["green"]: lab = "ok"
        else: lab = "tau-"
        regions = [s["region"] for s in st if s["region"]]
        first_hyp = next((s for s in st if s["action"] == "hypothesize"), None)
        out.append({**e, "label": lab, "symptom_region": st[0]["symptom_region"] if st else None,
                    "bug_region": st[0]["bug_region"] if st else None,
                    "regions_touched": list(dict.fromkeys(regions)),
                    "first_region": regions[0] if regions else None,
                    "first_hypothesis_class": first_hyp["hypothesis_class"] if first_hyp else None,
                    "wasted": sum(1 for s in st if s["action"] in ("repeat_inspect", "repeat_hypothesis", "repeat_patch", "noop_patch", "invalid")),
                    "models": sorted({s["model"] for s in st})})
    return out, med

def shrink(green, n):
    return round((green / n) * (n / (n + K_SHRINK)) + 0.5 * (K_SHRINK / (n + K_SHRINK)), 3) if n else 0.5

def distill(eps_labeled):
    clusters = collections.defaultdict(list)
    for e in eps_labeled:
        clusters[(e["family"], e["symptom_region"])].append(e)
    lessons, report = [], {}
    for (fam, sym), es in sorted(clusters.items()):
        n = len(es); n_green = sum(1 for e in es if e["green"])
        # what worked: first region of τ+/ok episodes, ranked by green rate then mean actions
        first_reg = collections.defaultdict(lambda: [0, 0, []])  # region -> [n, green, actions]
        for e in es:
            if e["first_region"]:
                r = first_reg[e["first_region"]]; r[0] += 1; r[1] += int(e["green"]); r[2].append(e["actions"])
        ranked = sorted(first_reg.items(), key=lambda kv: (-(kv[1][1] / kv[1][0]), statistics.mean(kv[1][2])))
        works = [{"first_region": r, "n": v[0], "green": v[1], "mean_actions": round(statistics.mean(v[2]), 2)} for r, v in ranked if v[1] > 0]
        # what didn't: regions that were the ONLY region touched in failed episodes (fixation) — dead paths for this condition
        fix = collections.Counter(e["regions_touched"][0] for e in es if not e["green"] and len(e["regions_touched"]) == 1 and e["regions_touched"])
        true_regions = collections.Counter(e["bug_region"] for e in es)
        avoid = [{"region": r, "failed_fixations": c, "was_the_bug_region_in": true_regions.get(r, 0)} for r, c in fix.most_common()]
        alternatives = sorted({e["first_region"] for e in es if e["first_region"]})
        models = sorted({m for e in es for m in e["models"]})
        rec = {
            "condition": {"family": fam, "symptom_region": sym},
            "action_prior": works[:2],
            "action_avoid": avoid[:3],
            "alternatives_tried": alternatives,
            "evidence": {"n_episodes": n, "n_green": n_green, "runs": sorted({e["run_id"] for e in es}), "models": models,
                         "true_bug_regions": dict(true_regions)},
            "confound_check": "UNVERIFIED — et8_verify.py replays this lesson on held-out tasks in scope and on a shuffled-family control",
            "scope": {"family": fam, "symptom_region": sym, "models": models},
            "confidence": shrink(n_green, n),
            "expiry": {"decay_per_round": 0.9, "rounds_without_evidence": 0},
            "reopen_condition": "a dead-path region yields green in ≥ 2 held-out episodes of this condition",
            "status": "candidate",
        }
        lessons.append(rec)
        report[f"{fam}|{sym}"] = {"n": n, "green": n_green, "fixation_failures": sum(fix.values()),
                                  "mean_wasted": round(statistics.mean(e["wasted"] for e in es), 2)}
    return lessons, report

def render_text(lessons) -> str:
    """Mechanism A: the text-memory rendering. Short, so its token cost is measured honestly against the parametric priors."""
    lines = []
    for L in lessons:
        c = L["condition"]; ev = L["evidence"]
        if not L["action_prior"] and not L["action_avoid"]:
            continue
        head = f"[{c['family']}, symptom in {c['symptom_region']}] (n={ev['n_episodes']}, conf={L['confidence']})"
        w = "; ".join(f"start at {p['first_region']} ({p['green']}/{p['n']} green, ~{p['mean_actions']} actions)" for p in L["action_prior"]) or "no winning start recorded yet"
        a = "; ".join(f"do not stay on {x['region']} ({x['failed_fixations']} failed fixations)" for x in L["action_avoid"]) or "no dead path recorded yet"
        lines.append(f"{head}\n  worked: {w}\n  did not: {a}")
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run prefixes (without .steps.jsonl)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    steps, eps = [], []
    for r in a.runs:
        s, e = load_run(r); steps += s; eps += e
    labeled, med = label_episodes(eps, steps)
    lessons, report = distill(labeled)
    os.makedirs(a.out, exist_ok=True)
    json.dump(lessons, open(os.path.join(a.out, "lessons.json"), "w"), indent=1)
    json.dump({"episodes": len(eps), "green": sum(e["green"] for e in eps), "median_green_actions": med,
               "labels": dict(collections.Counter(e["label"] for e in labeled)), "clusters": report,
               "runs": a.runs}, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    txt = render_text(lessons); open(os.path.join(a.out, "lessons.txt"), "w").write(txt)
    print(json.dumps({"episodes": len(eps), "labels": dict(collections.Counter(e["label"] for e in labeled)),
                      "clusters": len(lessons), "text_memory_chars": len(txt)}, indent=1))
    print(txt)

if __name__ == "__main__":
    main()
