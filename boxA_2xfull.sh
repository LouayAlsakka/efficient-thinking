#!/bin/bash
# DECISIVE data-vs-capacity cell: 2x net (depth8 width136, ~6.9M) on FULL data (~380M positions),
# so it's matched to the 1x full-data baseline (2734). If it beats 2734 -> capacity helps once
# well-fed (10-shard null was data starvation). If ~2734 -> data/signal is the true bottleneck.
# Long run (~24-30h): 2x net is ~2x slower/step and a full epoch is ~380k steps.
set -u
cd "$HOME/chess-scaling" || exit 1
L=runs/2xfull.log
: > "$L"
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
RUN=runs/conv_2x_w136_full
mkdir -p "$RUN"
LADDER=2500,2800,3050; GPR=28; MT=0.04
TARGET=380000            # ~1 full epoch over ~380M positions (matched to the 1x baseline)
LOAD_GRACE=3600          # 64GB load is slow -> 60min grace before calling it a load hang

say "TRAIN START 2x net depth8 width136 on FULL data (~380M) — decisive capacity-vs-data cell"
extra=""; cum=0; restart=0
while true; do
  set -f
  caffeinate -i env PYTHONPATH=. ./.venv/bin/python -u scripts/train.py \
    --data "data/eval.*.npz" --run-dir "$RUN" --arch conv --width 136 --depth 8 \
    --objective soft --value-head --grad-clip 1.0 --lr 5e-4 --batch-size 1024 --epochs 1 \
    --ckpt-every 4000 --seed "$restart" $extra >> "$RUN/train.log" 2>&1 &
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
    if [ "$started" -eq 0 ] && [ $((now - launch_ts)) -ge "$LOAD_GRACE" ]; then
      say "no first step ${LOAD_GRACE}s after launch (load/init hang) -> restart"; stalled=1
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
  elif grep -qE "Traceback|RuntimeError|MemoryError|Killed" "$RUN/train.log" 2>/dev/null && [ "$st" -lt 360000 ]; then
    say "segment crashed at step $st (cum ~$cum / $TARGET)"
  else
    say "segment finished epoch at step $st (cum ~$cum / $TARGET)"
  fi
  [ "$cum" -ge "$TARGET" ] && { say "reached target ~$cum"; break; }
  [ ! -f "$RUN/model.npz" ] && { say "no ckpt yet (early load hang) -> cold retry"; restart=$((restart+1)); extra=""; continue; }
  restart=$((restart + 1)); extra="--init $RUN/model.npz"
done
say "TRAIN COMPLETE (cum ~$cum steps)"

say "EVAL START 2x-full net (MCTS-800) on ladder $LADDER — compare vs 1x full-data 2734"
PYTHONPATH=. ./.venv/bin/python scripts/eval_search.py \
  --run-dir "$RUN" --method mcts --sims 800 \
  --ladder "$LADDER" --games-per-rung "$GPR" --movetime "$MT" --seed 0 >> "$L" 2>&1
say "EVAL COMPLETE"
say "ALL DONE (2x FULL-data vs 1x full-data 2734)"
