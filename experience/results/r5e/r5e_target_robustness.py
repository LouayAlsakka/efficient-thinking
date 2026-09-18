#!/usr/bin/env python3
"""Would the R5-E budget selection change if the target were my number instead of 理's?

理 fixed the target at 9,170 tokens/episode. My own arithmetic from the measured 300s gives:
    G model tokens 6,706.2  +  controller 2,487.9  =  9,194.1     (ratio to base 1.341 = R2's +34%)
a difference of 24.1 tokens, 0.26%. The selection rule is NEAREST, which is robust to that unless
two budgets happen to sit nearly equidistant. This says whether they do.

WRITTEN AS A SEPARATE FILE ON PURPOSE. The ladder script was RUNNING when this was needed, and bash
reads a script incrementally -- editing r5e_ladder.sh mid-run would have changed the selection step
it had not reached yet.
"""
import json, sys
SP = sys.argv[1]
rows = json.load(open(SP + "/r5e_ladder.json"))["ladder"]
out = {"rule": "nearest by absolute distance", "targets": {}}
for name, tgt in (("理's fixed 9170", 9170.0), ("my arithmetic 9194.1", 9194.1)):
    pick = min(rows, key=lambda r: abs(r["tokens_per_episode"] - tgt))
    out["targets"][name] = {
        "selected_budget": pick["budget"],
        "distances": {r["budget"]: round(abs(r["tokens_per_episode"] - tgt), 1) for r in rows}}
a = out["targets"]["理's fixed 9170"]["selected_budget"]
b = out["targets"]["my arithmetic 9194.1"]["selected_budget"]
out["SAME_SELECTION"] = (a == b)
out["reading"] = ("the 0.26%% target difference does not change the selected budget (%d) -- the "
                  "choice is robust to it" % a) if a == b else (
                  "⚠️ THE TARGET DIFFERENCE CHANGES THE SELECTION: %d vs %d. 理 must rule which "
                  "target stands before the 300 is run." % (a, b))
json.dump(out, open(SP + "/r5e_target_robustness.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
