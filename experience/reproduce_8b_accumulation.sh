#!/usr/bin/env bash
# ET-8b — reproduce the ACCUMULATION result, one command, on a clean Apple-silicon box.
#
#   bash experience/reproduce_8b_accumulation.sh            # seed 73 (v3c)
#   SET=v3rep bash experience/reproduce_8b_accumulation.sh  # seed 47, the second replication
#
# WHAT IT REPRODUCES (results/et8b_ACCUMULATION.json): a head fitted on the trajectories of an
# agent that was ITSELF head-steered beats the head that steered it, at no extra cost.
#     seed 73   gen0 34.3% -> gen1 40.0%    +5.7 [+0.7, +10.7]   McNemar p = 0.0396
#     seed 47   gen0 34.0% -> gen1 43.7%    +9.7 [+4.7, +15.0]   McNemar p = 0.00042
#     model tokens/episode: gen1 +0.1% (seed 73) and -3.0% (seed 47) against gen0
#
# AND THAT IT STOPS THERE. A third generation does not improve on the second:
#     seed 73   gen1 40.0% -> gen2 38.0%    -2.0 [-6.3, +2.3]    p = 0.4408
#     seed 47   gen1 43.7% -> gen2 36.7%    -7.0 [-11.0, -3.0]   p = 0.00145
# Both are checked. A reproduction that confirms the rise and not the stop has reproduced half a
# claim, and the half it leaves out is the one a reader is most likely to doubt.
#
# THE THING A REPRODUCER MUST NOT SKIP: both heads were fitted on v3 (seed 21) and must never have
# seen the set they run on. That is checked here by SIGNATURE DISJOINTNESS, not by trusting the seed
# — two generator runs with different seeds are not guaranteed disjoint, and task IDs are per-set so
# comparing them across sets is meaningless (an ID gate here would flag a false overlap, or pass a
# real one by accident).
#
# IT FAILS LOUD. A missing input, a head md5 mismatch, a non-disjoint task set, a short slice, or a
# paired interval that does not contain the published point all abort. A reproduction that prints a
# number it cannot stand behind is worse than one that stops.
#
# REQUIREMENTS   Apple silicon · python3.9+ with mlx-lm (tested 0.29.1 / mlx 0.29.3) · this repo ·
#                ~16 GB free RAM · Qwen/Qwen2.5-7B-Instruct (bf16, pulled on first use, ~15 GB)
# WALL CLOCK     ~50 min per arm x 3 arms = ~2.5 h on an otherwise-idle M-series box.
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
SET="${SET:-v3c}"
OUT="${OUT:-$R/experience/results/reproduce8b_${SET}_$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
H="${H:-$R/experience/results/et8b_heads}"
TRAIN_SET="${TRAIN_SET:-v3}"          # the set BOTH heads were fitted on; must be disjoint from SET
EXPECT=75
die(){ echo; echo "FAILED: $*" >&2; echo "Nothing is reported. Fix the cause and re-run." >&2; exit 1; }
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }

# published points, per set, for the final containment check
case "$SET" in
  v3c)   PUB_G1G0=5.7;  PUB_G2G1=-2.0;  PUB_SEED=73 ;;
  v3rep) PUB_G1G0=9.7;  PUB_G2G1=-7.0;  PUB_SEED=47 ;;
  *)     PUB_G1G0="";   PUB_G2G1="";    PUB_SEED="?" ;;
esac

say "ET-8b accumulation reproduction — set $SET (seed $PUB_SEED)"
command -v "$PY" >/dev/null || die "python3 not on PATH"
"$PY" -c "import mlx_lm, numpy" 2>/dev/null || die "mlx-lm and numpy required"
[ -d "$R/experience/tasks/$SET" ] || die "no task set at experience/tasks/$SET"

# ------------------------------------------------- (a) the heads, by md5
declare -a WANT=(
 "gen0_head_d1_layer18.npz 461d5b1ebfe1cd204c5918e3026ab2e2"
 "gen0_head_d2_layer18.npz 778a949f35ad7a7bb1a879d621726016"
 "gen0_head_d3_layer18.npz be2d1b4177c20b02982ca4810110e63b"
 "gen1_head_d1_layer18.npz f814f6c428d4c7a5d4f32dd25bd8cfc3"
 "gen1_head_d2_layer18.npz 28a5c401e43f7981628a08452dbb79f3"
 "gen1_head_d3_layer18.npz 24ad0b9cadc1bc0dc969f0e36678a0d3"
 "gen2_head_d1_layer18.npz f814f6c428d4c7a5d4f32dd25bd8cfc3"
 "gen2_head_d2_layer18.npz 28a5c401e43f7981628a08452dbb79f3"
 "gen2_head_d3_layer18.npz ac96bc32e8cfb0067efb5481434313cb")
for w in "${WANT[@]}"; do
  f="${w%% *}"; want="${w##* }"; p="$H/$f"
  [ -f "$p" ] || die "missing head $p"
  got=$("$PY" -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$p")
  [ "$got" = "$want" ] || die "head $f md5 $got, expected $want"
done
say "gate 1: all nine heads match their published md5"
# gen2's d1 and d2 are byte-identical to gen1's BY CONSTRUCTION — decision 1's state depends only on
# the task, so every generation fits the same d1 head, and d2 follows from d1. Only d3 differs. If a
# future change breaks that identity, these md5s catch it.

# ------------------------------------------------- (b) DISJOINTNESS, measured
"$PY" - "$R" "$TRAIN_SET" "$SET" <<'PYG' || die "disjointness gate"
import glob,hashlib,json,os,sys
R,train,run=sys.argv[1],sys.argv[2],sys.argv[3]
def sigs(d):
    out=set()
    for f in glob.glob(os.path.join(R,"experience/tasks",d,"task_*.json")):
        t=json.load(open(f))
        out.add(hashlib.sha1((t["program"]+"|"+t["symptom"]+"|"+t["bug_region"]).encode()).hexdigest())
    return out
a,b=sigs(train),sigs(run)
print("  %s %d signatures · %s %d · overlap %d"%(train,len(a),run,len(b),len(a&b)))
if not a or not b: sys.exit("  a task set is empty")
if a&b: sys.exit("  OVERLAP — the heads have seen problems in the run set. This is not a valid run.")
print("  disjoint: no head trained on %s has seen any problem in %s"%(train,run))
PYG
say "gate 2: task sets are disjoint, measured not assumed"

# ------------------------------------------------- (c) slices
mkdir -p "$OUT/slices"
"$PY" - "$R" "$SET" "$OUT" <<'PYS' || die "slicing"
import glob,os,shutil,sys
R,S,O=sys.argv[1],sys.argv[2],sys.argv[3]
f=sorted(glob.glob(os.path.join(R,"experience/tasks",S,"task_*.json")))
if len(f)!=300: sys.exit("  expected 300 tasks, found %d"%len(f))
for k in range(4):
    d=os.path.join(O,"slices","s%d"%(k+1)); os.makedirs(d,exist_ok=True)
    for x in f[k*75:(k+1)*75]: shutil.copy(x,d)
print("  4 slices of 75")
PYS

run_arm(){
  local name="$1"; shift
  mkdir -p "$OUT/$name"
  for s in 1 2 3 4; do
    # RESUMABLE. Four arms x four slices is hours; a kill in arm 4 used to re-run arms 1-3 from
    # zero, and a re-run is a different sample. A slice that already holds EXPECT complete
    # episodes is kept and not touched: the same rows, not new ones.
    local f="$OUT/$name/a_s$s"
    local n; n=$( [ -f "$f.episodes.jsonl" ] && wc -l < "$f.episodes.jsonl" | tr -d ' ' || echo 0 )
    if [ "$n" -ne "$EXPECT" ]; then
      "$PY" "$R/experience/et8b_loop.py" --tasks "$OUT/slices/s$s" --out "$f" \
         --model "$MODEL" --budget 12 "$@" >> "$OUT/$name/s$s.log" 2>&1
      n=$( [ -f "$f.episodes.jsonl" ] && wc -l < "$f.episodes.jsonl" | tr -d ' ' || echo 0 )
      say "  $name slice $s: $n episodes"
    else
      say "  $name slice $s: $n episodes (kept, already complete)"
    fi
    [ "$n" -eq "$EXPECT" ] || { tail -20 "$OUT/$name/s$s.log"; die "$name slice $s produced $n of $EXPECT"; }
  done
  cat "$OUT/$name"/a_s{1,2,3,4}.episodes.jsonl > "$OUT/$name.episodes.jsonl"
}
say "arm 1/4: base (no head)"
run_arm base
say "arm 2/4: gen0"
run_arm gen0 --heads "$H/gen0_head_d1_layer18.npz" "$H/gen0_head_d2_layer18.npz" "$H/gen0_head_d3_layer18.npz"
say "arm 3/4: gen1"
run_arm gen1 --heads "$H/gen1_head_d1_layer18.npz" "$H/gen1_head_d2_layer18.npz" "$H/gen1_head_d3_layer18.npz"
say "arm 4/4: gen2"
run_arm gen2 --heads "$H/gen2_head_d1_layer18.npz" "$H/gen2_head_d2_layer18.npz" "$H/gen2_head_d3_layer18.npz"

for p in "base gen0" "base gen1" "gen0 gen1" "gen1 gen2" "gen0 gen2" "base gen2"; do
  set -- $p
  "$PY" "$R/experience/paired_stats.py" --base "$OUT/$1.episodes.jsonl" --g "$OUT/$2.episodes.jsonl" \
    --expect 300 --name "reproduction ($SET): $2 vs $1" --out "$OUT/${2}_vs_${1}.json" \
    2>&1 | grep -v 'NotOpenSSL\|warnings.warn'
done

# ------------------------------------------------- (d) does it agree with what was published?
[ -n "$PUB_G1G0" ] && "$PY" - "$OUT" "$PUB_G1G0" "$PUB_G2G1" <<'PYC'
import json,sys
o,rise,stop=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
bad=[]
for name,f,pub in (("gen1 vs gen0  (the rise)","gen1_vs_gen0.json",rise),
                   ("gen2 vs gen1  (the stop)","gen2_vs_gen1.json",stop)):
    d=json.load(open(o+"/"+f)); lo,hi=d["success_95CI"]; pt=d["success_delta_points"]
    ok = lo <= pub <= hi
    print("\n  %s"%name)
    print("    PUBLISHED   %+.1f"%pub)
    print("    REPRODUCED  %+.1f  95%% CI [%+.1f, %+.1f]  p=%.4g"%(pt,lo,hi,d["McNemar"]["exact_two_sided_p"]))
    print("    %s"%("REPRODUCED: the published point lies inside this run's interval." if ok else
          "🔴 DISAGREES: the published point is OUTSIDE this run's interval."))
    if not ok: bad.append(name)
if bad:
    print("\n  🔴 %s disagree(s). Report THAT, not a number."%" and ".join(bad)); sys.exit(3)
print("\n  BOTH LIMBS REPRODUCED — the rise and the stop.")
PYC
rc=$?
say "artifacts in $OUT"
[ "${rc:-0}" -eq 0 ] || die "the reproduction disagrees with the published interval"
say "DONE"
