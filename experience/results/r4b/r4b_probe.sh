#!/usr/bin/env bash
# R4 continuation — ONLY runs if the floor check cleared. Per r4_prereg.json: base on 300 -> probe
# on the model's OWN post-inspect states with all three controls -> G.
#
# THE FLOOR IS A GATE, NOT A NOTE. This script reads r4b_floor.json and refuses to fit anything if
# the verdict is BELOW. A probe fitted into a floor returns "no room" wearing the clothes of
# "no signal", which is the one reading R4 exists to avoid.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/github/efficient-thinking"; PY="$HOME/mlx-serve-venv/bin/python"
M="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }
cd "$R" || exit 1

V=$("$PY" -c "import json;print(json.load(open('$SP/r4b_floor.json'))['VERDICT'][:5])" 2>/dev/null)
if [ "$V" != "ABOVE" ]; then
  say "STOP: floor verdict is '$V', not ABOVE. No probe is fit. R4 = (c) per 理 11300."
  exit 1
fi
say "floor cleared -> states at layers 18 and 27, on the model's OWN post-inspect decisions"
"$PY" experience/et8_head_v3.py states \
   --runs "$SP/r4b_base/r4b8_s1" "$SP/r4b_base/r4b8_s2" "$SP/r4b_base/r4b8_s3" "$SP/r4b_base/r4b8_s4" \
   --tasks experience/tasks/v3 --model "$M" --layers 18 27 --out "$SP/r4b_states.npz" \
   > "$SP/r4b_states.log" 2>&1 || { say "states FAILED"; tail -20 "$SP/r4b_states.log"; exit 1; }
say "states done -> fit, with all three controls (permutation, PCA-8, five-draw n=100)"
"$PY" experience/et8_head_v3.py fit --states "$SP/r4b_states.npz" --out "$SP/r4b_fit.json" \
   > "$SP/r4b_fit.log" 2>&1 || { say "fit FAILED"; tail -20 "$SP/r4b_fit.log"; exit 1; }
say "R4 PROBE COMPLETE"
"$PY" -c "import json;print(json.dumps(json.load(open('$SP/r4b_fit.json')),indent=1)[:2500])"
