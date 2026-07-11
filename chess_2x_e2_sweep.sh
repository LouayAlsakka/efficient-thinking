#!/bin/bash
# Fires when the running 2x-full epoch-1 finishes. Then: (1) DOUBLE THE TRAINING — a 2nd full-data
# epoch warm-started from the epoch-1 net; (2) test whether this better evaluator RAISES THE SEARCH
# CEILING — absolute Elo at MCTS-800 & 3200, plus a sims-sweep incl 6400 (does deeper search now pay
# off, unlike the current net where 6400 = 3200 = 2839?).
set -u
cd "$HOME/chess-scaling" || exit 1
WAIT_PID="${1:-16890}"
L=runs/2x_e2_sweep.log; : > "$L"
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
SRC=runs/conv_2x_w136_full          # epoch-1 net (warm-start source)
RUN=runs/conv_2x_w136_e2            # epoch-2 net
LADDER=2500,2800,3050

say "waiting for epoch-1 run (pid $WAIT_PID) to finish..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
say "epoch-1 done. epoch-1 model: $(ls -la $SRC/model.npz 2>/dev/null | awk '{print $5}') bytes"
[ ! -f "$SRC/model.npz" ] && { say "ERROR: no epoch-1 model; abort"; exit 1; }
mkdir -p "$RUN"

# ---- (1) DOUBLE THE TRAINING: 2nd full-data epoch, warm-started, with the stall watchdog ----
say "TRAIN epoch-2 START (warm-start from epoch-1, full data)"
extra="--init $SRC/model.npz"; cum=0; restart=0; TARGET=360000
while true; do
  set -f
  caffeinate -i env PYTHONPATH=. ./.venv/bin/python -u scripts/train.py \
    --data "data/eval.*.npz" --run-dir "$RUN" --arch conv --width 136 --depth 8 \
    --objective soft --value-head --grad-clip 1.0 --lr 3e-4 --batch-size 1024 --epochs 1 \
    --ckpt-every 4000 --seed "$restart" $extra >> "$RUN/train.log" 2>&1 &
  P=$!; set +f
  say "epoch-2 launched pid $P seed=$restart"
  last=""; ts=$(date +%s); lts=$ts; stalled=0; started=0
  while kill -0 "$P" 2>/dev/null; do
    sleep 30
    cur=$(grep -oE "step [0-9]+" "$RUN/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    now=$(date +%s)
    if [ -n "$cur" ]; then started=1; [ "$cur" != "$last" ] && { last="$cur"; ts="$now"; }; fi
    if [ "$started" -eq 0 ] && [ $((now-lts)) -ge 3600 ]; then say "load hang -> restart"; stalled=1; kill -9 "$P" 2>/dev/null; sleep 5; break; fi
    if [ "$started" -eq 1 ] && [ $((now-ts)) -ge 300 ]; then say "STALL at step ${cur:-?} -> restart"; stalled=1; kill -9 "$P" 2>/dev/null; sleep 5; break; fi
  done
  st=$(grep -oE "step [0-9]+" "$RUN/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+"); st=${st:-0}; cum=$((cum+st))
  say "epoch-2 segment ended at step $st (cum ~$cum / $TARGET, stalled=$stalled)"
  [ "$cum" -ge "$TARGET" ] && { say "epoch-2 reached target"; break; }
  [ "$stalled" -eq 0 ] && { say "epoch-2 finished epoch normally"; break; }
  restart=$((restart+1)); extra="--init $RUN/model.npz"
done
say "TRAIN epoch-2 COMPLETE (2x net, double training)"

# ---- (2) does the better evaluator raise the search ceiling? ----
for S in 800 3200; do
  say "ABS-ELO eval MCTS-$S on 2-epoch 2x net (vs 2734/2839 baselines)"
  PYTHONPATH=. ./.venv/bin/python scripts/eval_search.py --run-dir "$RUN" --method mcts --sims "$S" \
    --ladder "$LADDER" --games-per-rung 24 --movetime 0.04 --seed 0 >> "$L" 2>&1
  say "ABS-ELO MCTS-$S done"
done
say "SIMS-SWEEP on 2x net: does 6400 beat 3200 now? (baseline 800)"
PYTHONPATH=. ./.venv/bin/python scripts/sims_sweep.py --run-dir "$RUN" \
  --baseline 800 --sweep 1600,3200,6400 --games 24 >> "$L" 2>&1
say "SIMS-SWEEP done"
say "ALL DONE (2x net, double training, 6400 ceiling test)"
