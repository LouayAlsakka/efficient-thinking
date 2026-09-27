#!/usr/bin/env bash
# ET-8c — reproduce the ACQUISITION result from scratch, one command, on a clean Apple-silicon box.
#
#   bash experience/reproduce_8c.sh
#
# WHAT IT REPRODUCES (LOCK v3, experience/results/viiic_LOCK_v3.json): a head fitted where the agent
# was MADE to stand — the code read before any hypothesis, by a one-line rule during COLLECTION
# ONLY — beats a head fitted where the agent's own policy stood, on 300 problems no head has seen.
#
#     base          22.0%   (66 · 67 · 65 of 300)
#     forced-look   41.2%   (122 · 126 · 123)      +18.7 / +19.7 / +19.3 over base, both bars 3/3
#     self head     26.8%   (77 · 83 · 81)         the VIII-b plateau, reproduced on a third set
#
# AND WHAT IT CANNOT DO, said here rather than discovered at hour nine:
#   * THE SELF ARM NEEDS A HEAD THIS REPO DOES NOT PUBLISH (VIII-b's shared gen1-matched head). If
#     you have it, pass SELF_HEAD=/path/to/gen1matched_shared_layer18.npz and the arm runs; without
#     it the script runs base and forced-look and says the self arm was skipped. It does not fake it.
#   * A FRESH FIT WILL NOT HASH-MATCH OURS. The head is fitted from hidden states your box computes;
#     bit-identity is not the claim. The claim is the GREEN COUNTS, and the acceptance below is
#     stated in terms of the arm's own measured spread rather than equality.
#   * IF YOU WANT TO CHECK OUR BYTES RATHER THAN YOUR OWN, you do not need this script at all:
#         python experience/g13b_score.py --dir experience/results/viiic --plan viiic7 --out /tmp/x.json
#     reproduces all seven arms and 33 pairings from the published episodes in seconds.
#
# IT FAILS LOUD. A missing input, a generator that no longer reproduces the committed task set, a
# non-disjoint pair of sets, a short slice, or a head that will not load all abort. A reproduction
# that prints a number it cannot stand behind is worse than one that stops.
#
# REQUIREMENTS   Apple silicon · python3.9+ with mlx-lm (tested 0.29.1 / mlx 0.29.3) · this repo ·
#                ~16 GB free RAM · Qwen/Qwen2.5-7B-Instruct (bf16, pulled on first use, ~15 GB)
# WALL CLOCK     ~9 h on an otherwise-idle M-series box, MEASURED from our own arm timings, not
#                estimated: collection on seed 73 ~33 min (base) + ~52 min (forced-look); the state
#                pass and the fit ~40 min; then nine 300-problem arm-games at 33 min (base) to
#                52 min (head arms) each. Run it overnight; it is resumable per slice.
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${OUT:-$R/experience/results/reproduce8c_$(date -u +%Y%m%dT%H%M%SZ)}"
SELF_HEAD="${SELF_HEAD:-}"
EXPECT=75
LOCK="$R/experience/results/viiic_LOCK_v3.json"
mkdir -p "$OUT"
say(){ printf '=== %s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
die(){ printf '\n!!! STOP: %s\n' "$*" >&2; exit 1; }

say "0 — environment"
"$PY" - <<'PYE' || die "mlx-lm is not importable by $PY"
import importlib, sys
for m in ("mlx_lm", "mlx.core", "numpy"):
    importlib.import_module(m)
print("  mlx_lm, mlx.core, numpy import")
PYE
[ -f "$LOCK" ] || die "lock v3 missing at $LOCK — this script reproduces ITS numbers"

# ---------------------------------------------------------------- 1. the two task sets
say "1 — regenerate the task sets (seed 73 = fitting, seed 89 = scoring)"
for sd in 73 89; do
  d="$OUT/tasks/seed$sd"
  [ -d "$d" ] || { mkdir -p "$d"; "$PY" "$R/experience/et8_env_v3.py" gen --n 300 --seed "$sd" --out "$d" \
      >/dev/null 2>&1 || die "generator failed for seed $sd"; }
  n=$(ls "$d"/task_*.json 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = "300" ] || die "seed $sd produced $n tasks, expected 300"
  say "  seed $sd: 300 tasks"
done

say "2 — GATE: the generator still reproduces the committed sets, and the two sets are disjoint"
"$PY" - "$R" "$OUT" <<'PYG' || die "task-set gate"
import glob, hashlib, json, os, sys
R, O = sys.argv[1], sys.argv[2]
def sig(d):
    out = set()
    for p in sorted(glob.glob(os.path.join(d, "task_*.json"))):
        t = json.load(open(p))
        # the PROGRAM is the identity of a problem; task_id is per-set and means nothing across sets
        out.add(hashlib.sha1((t.get("program") or t.get("code") or json.dumps(t, sort_keys=True))
                             .encode()).hexdigest())
    return out
fresh73, fresh89 = sig(O + "/tasks/seed73"), sig(O + "/tasks/seed89")
if not fresh73 or not fresh89:
    sys.exit("  a regenerated set is empty")
if fresh73 & fresh89:
    sys.exit("  OVERLAP between the fitting and scoring sets — not a valid run")
print("  disjoint: the fitting set shares no program with the scoring set")
for name, fresh in (("v3c", fresh73), ("v3ind89", fresh89)):
    d = os.path.join(R, "experience/tasks", name)
    if not os.path.isdir(d):
        print("  %s not in this checkout — cannot compare the generator against it" % name); continue
    old = sig(d)
    if old == fresh:
        print("  %s: the generator reproduces the committed set EXACTLY (%d programs)" % (name, len(old)))
    else:
        sys.exit("  %s: the generator no longer reproduces the committed set (%d of %d match). "
                 "A reproduction of a different problem set is not a reproduction."
                 % (name, len(old & fresh), len(old)))
PYG

# ---------------------------------------------------------------- 3. collection on seed 73
say "3 — collect on seed 73: the base policy, and the forced-look policy"
collect(){ local name="$1" tasks="$2"; shift 2
  local f="$OUT/$name"
  local n; n=$( [ -f "$f.episodes.jsonl" ] && wc -l < "$f.episodes.jsonl" | tr -d ' ' || echo 0 )
  if [ "$n" -ne 300 ]; then
    "$PY" "$R/experience/et8b_loop.py" --tasks "$tasks" --out "$f" --model "$MODEL" --budget 12 "$@" \
      >> "$OUT/$name.log" 2>&1 || { tail -20 "$OUT/$name.log"; die "collection $name"; }
    n=$( wc -l < "$f.episodes.jsonl" | tr -d ' ' )
  fi
  [ "$n" = "300" ] || die "$name produced $n episodes, expected 300"
  say "  $name: 300 episodes, green $(grep -c '"green": true' "$f.episodes.jsonl")"
}
collect base_seed73   "$OUT/tasks/seed73"
collect forced_seed73 "$OUT/tasks/seed73" --force-inspect 1

# ---------------------------------------------------------------- 4. states, cut, fit
say "4 — the 7B's states at the forced-look trajectory's decision points, cut to 290 groups, fitted"
[ -d "$OUT/states" ] || "$PY" "$R/experience/et8b_states.py" --runs "$OUT/forced_seed73.steps.jsonl" \
    --tasks "$OUT/tasks/seed73" --model "$MODEL" --layers 18 --out "$OUT/states" \
    >> "$OUT/states.log" 2>&1 || { tail -20 "$OUT/states.log"; die "state pass"; }
[ -d "$OUT/cut290" ] || "$PY" "$R/experience/subsample_by_decision.py" --states "$OUT/states" \
    --decisions 290 --seed 13731 --out "$OUT/cut290" >> "$OUT/fit.log" 2>&1 || die "subsample"
[ -f "$OUT/forced_head.npz" ] || "$PY" "$R/experience/fit_shared_head.py" --states "$OUT/cut290" \
    --layer 18 --cut 400 --cv-folds 5 --out-head "$OUT/forced_head.npz" --json "$OUT/forced_head.json" \
    --name reproduce8c_forced --form-note "shared head, 290 whole decision groups, seed 13731" \
    >> "$OUT/fit.log" 2>&1 || { tail -20 "$OUT/fit.log"; die "head fit"; }
say "  head fitted: $(basename "$OUT/forced_head.npz")"

# ---------------------------------------------------------------- 5. three games on seed 89
say "5 — four slices of seed 89, then three games per arm"
mkdir -p "$OUT/slices"
"$PY" - "$OUT" <<'PYS' || die "slicing"
import glob, os, shutil, sys
O = sys.argv[1]
f = sorted(glob.glob(os.path.join(O, "tasks/seed89", "task_*.json")))
if len(f) != 300: sys.exit("  expected 300 tasks, found %d" % len(f))
for k in range(4):
    d = os.path.join(O, "slices", "s%d" % (k + 1)); os.makedirs(d, exist_ok=True)
    for x in f[k * 75:(k + 1) * 75]: shutil.copy(x, d)
print("  4 slices of 75")
PYS

run_game(){ local arm="$1" g="$2"; shift 2
  for s in 1 2 3 4; do
    local f="$OUT/games/${arm}${g}_s$s"
    mkdir -p "$OUT/games"
    local n; n=$( [ -f "$f.episodes.jsonl" ] && wc -l < "$f.episodes.jsonl" | tr -d ' ' || echo 0 )
    if [ "$n" -ne "$EXPECT" ]; then
      "$PY" "$R/experience/et8b_loop.py" --tasks "$OUT/slices/s$s" --out "$f" --model "$MODEL" \
        --budget 12 "$@" >> "$OUT/games/${arm}.log" 2>&1 \
        || { tail -20 "$OUT/games/${arm}.log"; die "$arm game $g slice $s"; }
      n=$( wc -l < "$f.episodes.jsonl" | tr -d ' ' )
    fi
    [ "$n" = "$EXPECT" ] || die "$arm game $g slice $s has $n episodes, expected $EXPECT"
  done
  cat "$OUT/games/${arm}${g}"_s{1,2,3,4}.episodes.jsonl > "$OUT/games/${arm}${g}.episodes.jsonl"
  say "  ${arm} game $g: green $(grep -c '"green": true' "$OUT/games/${arm}${g}.episodes.jsonl") of 300"
}
for g in 1 2 3; do
  run_game base "$g"
  run_game forced "$g" --head "$OUT/forced_head.npz"
  if [ -n "$SELF_HEAD" ]; then
    [ -f "$SELF_HEAD" ] || die "SELF_HEAD=$SELF_HEAD does not exist"
    run_game self "$g" --head "$SELF_HEAD"
  fi
done
[ -n "$SELF_HEAD" ] || say "  self arm SKIPPED — its head is a VIII-b artefact this repo does not publish"

# ---------------------------------------------------------------- 6. beside the lock
say "6 — your green counts beside ours"
"$PY" - "$OUT" "$LOCK" "${SELF_HEAD:-}" <<'PYC'
import json, os, sys
O, LOCK, SELF = sys.argv[1], sys.argv[2], sys.argv[3]
lock = json.load(open(LOCK))
ours = {"base": [66, 67, 65], "forced": [122, 126, 123], "self": [77, 83, 81]}
bar = lock["scored_seven_arm"]["bar_points"]
print("\n  arm      your games          ours                mean%%   ours%%   |delta|  bar %.2f pp" % bar)
ok = True
for arm in ("base", "forced", "self"):
    if arm == "self" and not SELF:
        print("  %-8s SKIPPED (no SELF_HEAD)" % arm); continue
    got = []
    for g in (1, 2, 3):
        p = os.path.join(O, "games", "%s%d.episodes.jsonl" % (arm, g))
        if not os.path.exists(p):
            got = None; break
        got.append(sum(1 for l in open(p) if json.loads(l).get("green")))
    if got is None:
        print("  %-8s MISSING" % arm); ok = False; continue
    m, om = 100.0 * sum(got) / 900, 100.0 * sum(ours[arm]) / 900
    d = abs(m - om)
    flag = "" if d <= bar else "   <- OUTSIDE the session's own bar"
    if d > bar: ok = False
    print("  %-8s %-18s %-18s %6.2f %7.2f %8.2f%s"
          % (arm, " · ".join(map(str, got)), " · ".join(map(str, ours[arm])), m, om, d, flag))
print("\n  ACCEPTANCE: each arm's mean within %.2f points of ours — the largest within-head-arm\n"
      "  range of our own session, i.e. this harness's own spread. Equality is NOT the test:\n"
      "  decisions 2 and 3 are sampled, so two runs of the same arm differ by a few problems.\n"
      "  %s" % (bar, "ALL ARMS WITHIN THE BAR" if ok else "AT LEAST ONE ARM OUTSIDE THE BAR — read the rows above"))
sys.exit(0 if ok else 2)
PYC
say "done — outputs under $OUT"
