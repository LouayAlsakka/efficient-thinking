#!/usr/bin/env bash
# ET-8 — reproduce the matched-compute result, ONE command, on a clean Apple-silicon box.
#
#   bash experience/reproduce_matched_compute.sh
#
# WHAT IT REPRODUCES (8a §7.6, R5-E): the base handed G's ENTIRE compute budget still falls short.
#     base @ budget 16   22.3%   9,234.2 tokens/episode
#     G    @ budget 12   31.7%   9,194.1 tokens/episode ALL-IN  (6,706 model + 2,488 controller)
#     paired on the same 300:  +9.3 points  95% CI [+3.7, +15.0]  McNemar p = 0.002031
#
# IT FAILS LOUD. Every gate below aborts the run rather than printing a number beside a warning:
# a missing input, a head whose md5 does not match, a short slice, a task or prompt gate under 0.9,
# or a paired table whose interval does not contain the published one. A reproduction that prints
# a number it cannot stand behind is worse than one that stops.
#
# REQUIREMENTS   Apple silicon · python3.9+ with mlx-lm (tested 0.29.1 / mlx 0.29.3) · this repo ·
#                ~16 GB free RAM · the model below, pulled automatically on first use (~15 GB)
# MODEL          Qwen/Qwen2.5-7B-Instruct   (bf16; NOT a 4-bit quant — the numbers are bf16's)
# WALL CLOCK     measured on box A (M-series, nothing else on the GPU):
#                  base@16 300 episodes  ~3.5 h      G@12 300 episodes  ~3.0 h
#                  states + fit           not re-run here; the fitted head ships with the repo
#                TOTAL ~6.5-7 h. Run it overnight; it prints progress per slice.
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
OUT="${OUT:-$R/experience/results/reproduce_$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HEAD="${HEAD:-$R/experience/results/head_layer18_r5e.npz}"   # overridable ONLY so the md5 control can be exercised
HEAD_MD5="ce4e9aa6b75e5ff4b03b5ad1545354d3"
TASKS="${TASKS:-$R/experience/tasks/v3}"
EXPECT_PER_SLICE=75
die(){ echo; echo "FAILED: $*" >&2; echo "Nothing is reported. Fix the cause and re-run." >&2; exit 1; }
say(){ echo "=== $(date -u +%H:%M:%SZ) $*"; }

say "ET-8 matched-compute reproduction"
say "repo $R"
say "model $MODEL"

# ---------------------------------------------------------------- (a) inputs
command -v "$PY" >/dev/null || die "python3 not on PATH"
"$PY" -c "import mlx_lm, mlx.core" 2>/dev/null || die "mlx-lm is not importable by $PY"
"$PY" -c "import mlx_lm;print('  mlx_lm',mlx_lm.__version__)"
[ -d "$TASKS" ] || die "task set missing: $TASKS"
NT=$(ls "$TASKS"/task_*.json 2>/dev/null | wc -l | tr -d ' ')
[ "$NT" -ge 300 ] || die "task set has $NT tasks, need >= 300 ($TASKS)"
[ -f "$HEAD" ] || die "head file missing: $HEAD"
GOT=$("$PY" - "$HEAD" <<'PY'
import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())
PY
)
[ "$GOT" = "$HEAD_MD5" ] || die "head md5 mismatch: expected $HEAD_MD5, got $GOT. This is not the head the paper used."
say "inputs ok: $NT tasks, head md5 $GOT"

# ---------------------------------------------------------------- gates on the task set
"$PY" "$R/experience/et8_task_gate.py" --tasks "$TASKS" 2>/dev/null | tee "$OUT.gates.txt" || true
mkdir -p "$OUT"

slice_dir(){ printf "%s/s%s" "$OUT/slices" "$1"; }
mkdir -p "$OUT/slices"
"$PY" - "$TASKS" "$OUT/slices" <<'PY'
import glob,os,shutil,sys
t,o=sys.argv[1],sys.argv[2]
f=sorted(glob.glob(os.path.join(t,"task_*.json")))[:300]
for i,p in enumerate(f):
    d=os.path.join(o,"s%d"%(i//75+1)); os.makedirs(d,exist_ok=True); shutil.copy(p,d)
print("  sliced 300 tasks into 4 x 75")
PY

# ---------------------------------------------------------------- (b) the two arms
run_arm(){  # name outdir budget [head args...]
  local name="$1" dir="$2" budget="$3"; shift 3
  mkdir -p "$dir"
  for s in 1 2 3 4; do
    say "$name slice $s/4 (budget $budget)"
    "$PY" "$R/experience/et8_agent.py" --tasks "$(slice_dir $s)" --out "$dir/a_s$s" \
        --model "$MODEL" --budget "$budget" --inspect-first "$@" >> "$dir/s$s.log" 2>&1
    local n; n=$(wc -l < "$dir/a_s$s.episodes.jsonl" 2>/dev/null | tr -d ' '); n=${n:-0}
    [ "$n" -eq "$EXPECT_PER_SLICE" ] || die "$name slice $s produced $n of $EXPECT_PER_SLICE episodes (see $dir/s$s.log)"
  done
}
run_arm "BASE@16" "$OUT/base16" 16
run_arm "G@12"    "$OUT/g12"    12 --head "$HEAD" --head-layer 18

# ---------------------------------------------------------------- (c) the paired table, (d) the check
"$PY" "$R/experience/paired_stats.py" \
  --base "$OUT"/base16/a_s1.episodes.jsonl "$OUT"/base16/a_s2.episodes.jsonl \
         "$OUT"/base16/a_s3.episodes.jsonl "$OUT"/base16/a_s4.episodes.jsonl \
  --g    "$OUT"/g12/a_s1.episodes.jsonl "$OUT"/g12/a_s2.episodes.jsonl \
         "$OUT"/g12/a_s3.episodes.jsonl "$OUT"/g12/a_s4.episodes.jsonl \
  --name "REPRODUCTION: base@16 (compute-matched) vs G@12" --expect 300 \
  --out "$OUT/paired.json" || die "paired_stats refused the arms (unequal sizes or short set)"

"$PY" - "$OUT/paired.json" <<'PY' || exit 1
import json,sys
d=json.load(open(sys.argv[1]))
lo,hi=d["success_95CI"]; delta=d["success_delta_points"]
PUB=(9.3,3.7,15.0)
print()
print("  PUBLISHED   +%.1f points  95%% CI [+%.1f, +%.1f]"%PUB)
print("  REPRODUCED  %+.1f points  95%% CI [%+.1f, %+.1f]   base %.1f%% -> G %.1f%%"
      %(delta,lo,hi,d["base_success_pct"],d["G_success_pct"]))
ok = lo <= PUB[0] <= hi and PUB[1] <= delta <= PUB[2]
print()
if ok:
    print("  REPRODUCED: the published point estimate lies inside this run's interval and this run's")
    print("  estimate lies inside the published interval.")
else:
    print("  *** NOT REPRODUCED *** the two intervals do not contain each other's point estimate.")
    print("  This is a real disagreement, not a rounding difference. Report it with both intervals.")
sys.exit(0 if ok else 1)
PY
say "artifacts in $OUT"

# ---------------------------------------------------------------- CONTROLS (run 2026-09-20)
# A gate nobody has seen fire is not a gate. Each was exercised before this script shipped:
#   tampered head (b+1.0)     -> "head md5 mismatch: expected ce4e9aa6..., got 96e578c7..."  ABORTS
#   TASKS=/nonexistent        -> "task set missing"                                          ABORTS
#   a table reading +0.5 [-2.0,+3.0] against the published +9.3 [+3.7,+15.0]
#                             -> "*** NOT REPRODUCED ***", exit 1                            REFUSES
# Reproduce the controls with:
#   HEAD=/path/to/tampered.npz bash experience/reproduce_matched_compute.sh
#   TASKS=/nonexistent        bash experience/reproduce_matched_compute.sh
