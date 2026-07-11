#!/bin/bash
# On llm2: 1x CONTROL for the param-doubling test. Same recipe as llm1's 2x run (10 shards, same
# watchdog/hang-fix, same TARGET) but width 96 (= the baseline arch) instead of 136. Comparing
# llm1's d8w136 vs this d8w96 on IDENTICAL data isolates the pure effect of doubling parameters.
set -u
cd "$HOME/chess-scaling" || exit 1
L=runs/llm2_1x.log
: > "$L"
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
RUN=runs/conv_1x_w96_ctrl
mkdir -p "$RUN"
LADDER=2500,2800,3050; GPR=28; MT=0.04
TARGET=90000

say "TRAIN START 1x control depth8 width96 on 10 shards (llm2)"
extra=""; cum=0; restart=0
while true; do
  set -f
  caffeinate -i env PYTHONPATH=. ./.venv/bin/python -u scripts/train.py \
    --data "data/eval.000*.npz" --run-dir "$RUN" --arch conv --width 96 --depth 8 \
    --objective soft --value-head --grad-clip 1.0 --lr 5e-4 --batch-size 1024 --epochs 1 \
    --ckpt-every 2000 --seed "$restart" $extra >> "$RUN/train.log" 2>&1 &
  P=$!; set +f
  say "launched pid $P seed=$restart extra='$extra'"
  last=""; ts=$(date +%s); launch_ts=$ts; stalled=0; started=0
  while kill -0 "$P" 2>/dev/null; do
    sleep 30
    cur=$(grep -oE "step [0-9]+" "$RUN/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    now=$(date +%s)
    if [ -n "$cur" ]; then
      started=1
      if [ "$cur" != "$last" ]; then last="$cur"; ts="$now"; fi
    fi
    if [ "$started" -eq 0 ] && [ $((now - launch_ts)) -ge 1500 ]; then
      say "no first step 25min after launch (load/init hang) -> restart"; stalled=1
      kill -9 "$P" 2>/dev/null; sleep 5; break
    fi
    if [ "$started" -eq 1 ] && [ $((now - ts)) -ge 300 ]; then
      say "STALL at step ${cur:-?} -> kill+restart warm from ckpt"; stalled=1
      kill -9 "$P" 2>/dev/null; sleep 5; break
    fi
  done
  st=$(grep -oE "step [0-9]+" "$RUN/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+"); st=${st:-0}
  cum=$((cum + st))
  if [ "$stalled" -eq 1 ]; then
    say "segment stalled at step $st (cum ~$cum / $TARGET)"
  elif grep -qE "Traceback|RuntimeError|MemoryError|Killed" "$RUN/train.log" 2>/dev/null && [ "$st" -lt 40000 ]; then
    say "segment crashed at step $st (cum ~$cum / $TARGET)"
  else
    say "segment finished epoch at step $st (cum ~$cum / $TARGET)"
  fi
  [ "$cum" -ge "$TARGET" ] && { say "reached target ~$cum"; break; }
  [ ! -f "$RUN/model.npz" ] && { say "no ckpt yet (early load hang) -> cold retry"; restart=$((restart+1)); extra=""; continue; }
  restart=$((restart + 1)); extra="--init $RUN/model.npz"
done
say "TRAIN COMPLETE (cum ~$cum steps)"

say "EVAL START 1x control (MCTS-800) on ladder $LADDER"
PYTHONPATH=. ./.venv/bin/python scripts/eval_search.py \
  --run-dir "$RUN" --method mcts --sims 800 \
  --ladder "$LADDER" --games-per-rung "$GPR" --movetime "$MT" --seed 0 >> "$L" 2>&1
say "EVAL COMPLETE"
say "ALL DONE (1x control vs 2x on same 10 shards)"
