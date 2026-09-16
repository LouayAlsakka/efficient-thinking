#!/usr/bin/env python3
"""Mechanism F — an additive logit bias on the action-choice token.

    python3 et8_logit_bias.py build --runs experience/traj/base_v1_7b --out experience/bias/v1_7b
    python3 et8_logit_bias.py show  --bias experience/bias/v1_7b

理's definition (docs/efficient-thinking-8-experience-priors.md:727): "an additive logit bias on
the action-choice token at the decision position, built from the same contrastive statistics as C
— per region class, the log-ratio of action frequencies in productive versus wasted episodes,
clipped to ±2 logits, zero elsewhere."

WHY IT IS THE CHEAPEST CARRIER: C rewrites hidden states and needs a forward hook; A spends ~8,000
extra input tokens per episode. F is a table of at most a few dozen numbers added to the logits of
a handful of action tokens. It costs one addition at one position.

AND WHY ITS EFFECT IS DIRECTLY READABLE: `noop_patch` is the action that dominates wasted episodes
(231/525 steps at baseline, 532/819 under A). F either lowers its share or it does not, and no
other quantity has to be inferred to see that.
"""
import argparse, collections, glob, json, math, os

CLIP = 2.0


def episodes(runs):
    for r in runs:
        p = r if r.endswith(".jsonl") else r + ".episodes.jsonl"
        if os.path.exists(p):
            for l in open(p):
                yield json.loads(l)


def steps(runs):
    for r in runs:
        p = r.replace(".episodes.jsonl", "") + ".steps.jsonl"
        if os.path.exists(p):
            for l in open(p):
                yield json.loads(l)


def cmd_build(a):
    # productive vs wasted, by the same tau split C uses: an episode is PRODUCTIVE if it went
    # green, WASTED otherwise. The bias is per REGION CLASS (the task family), per action.
    ep = {e["task_id"]: e for e in episodes(a.runs)}
    pos = collections.defaultdict(collections.Counter)   # family -> action -> count, productive
    neg = collections.defaultdict(collections.Counter)   # family -> action -> count, wasted
    n_steps = 0
    for s in steps(a.runs):
        e = ep.get(s.get("task_id"))
        if not e:
            continue
        act = s.get("action") or s.get("kind")
        if not act:
            continue
        n_steps += 1
        (pos if e.get("green") else neg)[e.get("family", "?")][act] += 1

    bias = {}
    for fam in sorted(set(pos) | set(neg)):
        p_tot = sum(pos[fam].values()) or 1
        n_tot = sum(neg[fam].values()) or 1
        row = {}
        for act in sorted(set(pos[fam]) | set(neg[fam])):
            # Laplace: an action seen ONLY in wasted episodes must not produce -inf, and an action
            # seen 0 times either side must contribute nothing rather than a division error.
            pp = (pos[fam][act] + 1) / (p_tot + 1)
            pn = (neg[fam][act] + 1) / (n_tot + 1)
            b = max(-CLIP, min(CLIP, math.log(pp / pn)))
            row[act] = round(b, 4)
        bias[fam] = row

    os.makedirs(a.out, exist_ok=True)
    meta = {"definition": "additive logit bias on the action-choice token, per region class; "
                          "log-ratio of action frequencies in productive vs wasted episodes, "
                          "clipped to +/-%.1f" % CLIP,
            "source_runs": a.runs, "steps_counted": n_steps,
            "productive_episodes": sum(1 for e in ep.values() if e.get("green")),
            "wasted_episodes": sum(1 for e in ep.values() if not e.get("green")),
            "clip": CLIP, "bias": bias,
            "smoothing": "Laplace +1 on both sides — an action seen only in wasted episodes must "
                         "not give -inf, and an unseen action must contribute 0 rather than raise"}
    json.dump(meta, open(os.path.join(a.out, "bias.json"), "w"), indent=1, ensure_ascii=False)
    print(f"  steps counted: {n_steps}   productive eps {meta['productive_episodes']} · "
          f"wasted {meta['wasted_episodes']}")
    for fam, row in bias.items():
        hot = sorted(row.items(), key=lambda kv: kv[1])
        print(f"  {fam:12s} most DISCOURAGED {hot[0][0]}={hot[0][1]:+.2f}   "
              f"most ENCOURAGED {hot[-1][0]}={hot[-1][1]:+.2f}")
    print(f"  -> {a.out}/bias.json")


def cmd_show(a):
    d = json.load(open(os.path.join(a.bias, "bias.json")))
    print(json.dumps(d, indent=1, ensure_ascii=False)[:2000])


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--runs", nargs="+", required=True)
    b.add_argument("--out", required=True); b.set_defaults(fn=cmd_build)
    s = sp.add_parser("show"); s.add_argument("--bias", required=True); s.set_defaults(fn=cmd_show)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
