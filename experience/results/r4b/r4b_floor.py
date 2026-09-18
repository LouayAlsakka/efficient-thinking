#!/usr/bin/env python3
"""R4b floor check — pre-registered, run BEFORE any probe is fit (r4_prereg.json, 理 11300).

REFUSES to report on fewer than 300 episodes. The first R4 run reported 262 and 262 is not 300:
a rate over a partial set is a rate over an unknown population.
"""
import collections, json, sys
SP = sys.argv[1]; OUT = sys.argv[2]
FLOOR = 0.05
eps, steps = [], []
for s in (1, 2, 3, 4):
    eps += [json.loads(l) for l in open("%s/r4b_base/r4b8_s%d.episodes.jsonl" % (SP, s))]
    steps += [json.loads(l) for l in open("%s/r4b_base/r4b8_s%d.steps.jsonl" % (SP, s))]
n = len(eps)
if n != 300:
    json.dump({"VERDICT": "VOID -- %d of 300 episodes. No rate is reported over a partial set." % n},
              open(OUT, "w"), indent=1)
    print("VOID: %d of 300 episodes." % n); sys.exit(1)

green = sum(1 for e in eps if e.get("green"))
acts = sum(e.get("actions", 0) for e in eps) / n
ad = collections.Counter(s.get("action") for s in steps)
tot = sum(ad.values())
res = {
 "measurement": "R4b FLOOR CHECK — Meta-Llama-3.1-8B-Instruct-4bit on v3 seed 21's 300, inspect-first, budget 12.",
 "prereg": "results/r4_prereg.json (the floor) + 理 11300 (this model, one more and then stop)",
 "episodes": n,
 "base": "%d/%d = %.1f%%, %.2f mean actions" % (green, n, 100.0 * green / n, acts),
 "comparators_on_the_SAME_300": {
   "Qwen2.5-7B-Instruct (the base of the whole package)": "18.7%, 10.63 actions",
   "Llama-3.2-3B-Instruct-4bit (R4's first model)": "1.5%, 11.95 actions — FLOORED"},
 "pre-registered floor": "5%",
 "VERDICT": ("ABOVE THE FLOOR — R4 is answerable on this model and the probe is fit next."
             if green / n >= FLOOR else
             "BELOW THE FLOOR — v3 is out of this model's range. R4 = (c) per 理 11300: 8a says v3 "
             "is calibrated to the 7B and out of range for a 4-bit 3B and for Llama-3.1-8B-4bit, "
             "and the generality question is handed to 8c/8d by name. NO THIRD MODEL."),
 "action_distribution_because_a_floor_must_be_shown_not_to_be_a_FORMAT_failure": {
   k: "%.1f%%" % (100.0 * v / tot) for k, v in ad.most_common()},
 "how_to_read_the_distribution": "a high `invalid` share would mean the model cannot emit the "
   "protocol and the floor is a harness artefact. A high `noop_patch` share means it emits the "
   "protocol fine and cannot repair — a capability floor. The 3B's was noop_patch 30.9% / invalid "
   "5.9%, which is why its 1.5% was reported as a floor and not as a bug.",
 "bounds": ["One seed's 300 (v3 seed 21), the same set every other arm in the package used.",
            "4-bit quantisation. A bf16 Llama-3.1-8B could differ; not measured, not claimed.",
            "The floor is a pre-registered threshold on the BASE, fixed before this model ran."],
 "signed": "Sautee (sha-ta)"}
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
