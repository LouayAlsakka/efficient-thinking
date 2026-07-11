#!/bin/bash
# Two-machine chess close-out (llm1 = train, llm2 = eval; neither idles).
#   llm2 is ALREADY busy: absolute MCTS-6400 on the current net (definitive current-net ceiling).
#   This script (on llm1): wait for epoch-1 -> epoch-2 DOUBLE TRAINING (batch 4096, warm-start) ->
#   ship the double-trained net to llm2 (free by then) for the full ceiling test (abs MCTS-800/3200
#   + sims-sweep incl 6400): does a BETTER evaluator raise the 2839 search ceiling?
set -u
cd "$HOME/chess-scaling" || exit 1
L=runs/finish_chess.log; : > "$L"
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
E1=runs/conv_2x_w136_full        # epoch-1 net (warm-start source)
RUN=runs/conv_2x_w136_e2         # epoch-2 (double-trained) net

# wait for the epoch-1 run to FULLY finish (train + its own MCTS-800 eval) to avoid GPU contention
say "waiting for epoch-1 run (train+eval) 'ALL DONE'..."
while ! grep -q "ALL DONE" runs/2xfull.log 2>/dev/null; do sleep 120; done
say "epoch-1 fully done. model: $(ls -la $E1/model.npz 2>/dev/null | awk '{print $5}') bytes"
[ ! -f "$E1/model.npz" ] && { say "ERROR: no epoch-1 model; abort"; exit 1; }

# ---- llm1: epoch-2 DOUBLE TRAINING (batch 4096, LR scaled, stall watchdog) ----
mkdir -p "$RUN"
say "epoch-2 START (double training, batch 4096, warm-start)"
extra="--init $E1/model.npz"; cum=0; restart=0; TARGET=90000   # ~1 epoch at batch 4096 (~394M/4096)
while true; do
  set -f
  caffeinate -i env PYTHONPATH=. ./.venv/bin/python -u scripts/train.py \
    --data "data/eval.*.npz" --run-dir "$RUN" --arch conv --width 136 --depth 8 \
    --objective soft --value-head --grad-clip 1.0 --lr 1e-3 --batch-size 4096 --epochs 1 \
    --ckpt-every 2000 --seed "$restart" $extra >> "$RUN/train.log" 2>&1 &
  P=$!; set +f
  say "epoch-2 launched pid $P seed=$restart"
  last=""; ts=$(date +%s); lts=$ts; stalled=0; started=0
  while kill -0 "$P" 2>/dev/null; do
    sleep 30
    cur=$(grep -oE "step [0-9]+" "$RUN/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    now=$(date +%s)
    if [ -n "$cur" ]; then started=1; [ "$cur" != "$last" ] && { last="$cur"; ts="$now"; }; fi
    if [ "$started" -eq 0 ] && [ $((now-lts)) -ge 3600 ]; then say "load hang -> restart"; stalled=1; kill -9 "$P" 2>/dev/null; sleep 5; break; fi
    if [ "$started" -eq 1 ] && [ $((now-ts)) -ge 300 ]; then say "STALL step ${cur:-?} -> restart"; stalled=1; kill -9 "$P" 2>/dev/null; sleep 5; break; fi
  done
  st=$(grep -oE "step [0-9]+" "$RUN/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+"); st=${st:-0}; cum=$((cum+st))
  say "epoch-2 segment ended step $st (cum ~$cum/$TARGET stalled=$stalled)"
  [ "$cum" -ge "$TARGET" ] && { say "epoch-2 target reached"; break; }
  [ "$stalled" -eq 0 ] && { say "epoch-2 finished normally"; break; }
  restart=$((restart+1)); extra="--init $RUN/model.npz"
done
say "epoch-2 COMPLETE (double-trained 2x net)"

# ---- ship to llm2 (free by now) for the full ceiling test ----
say "waiting for llm2 to be free (its 6400-current eval to finish)..."
while ssh -o ConnectTimeout=10 llm2 'pgrep -f "eval_search.py|sims_sweep.py" >/dev/null' 2>/dev/null; do sleep 120; done
say "llm2 free. rsync epoch-2 net -> llm2, launch ceiling test"
rsync -az "$RUN/model.npz" "$RUN/config.json" llm2:chess-scaling/runs/conv_2x_e2/ >>"$L" 2>&1
ssh llm2 'cd chess-scaling
  nohup caffeinate -i bash -c '"'"'
    L=runs/2x_e2_ceiling.log; : > "$L"
    for S in 800 3200; do
      echo "=== abs-Elo MCTS-$S on double-trained 2x net ===" >> "$L"
      PYTHONPATH=. ./.venv/bin/python scripts/eval_search.py --run-dir runs/conv_2x_e2 --method mcts \
        --sims $S --ladder 2500,2800,3050 --games-per-rung 24 --movetime 0.04 --seed 0 >> "$L" 2>&1
    done
    echo "=== sims-sweep: does 6400 beat 3200 on the better net? ===" >> "$L"
    PYTHONPATH=. ./.venv/bin/python scripts/sims_sweep.py --run-dir runs/conv_2x_e2 --baseline 800 \
      --sweep 1600,3200,6400 --games 24 >> "$L" 2>&1
    echo "ALL DONE" >> "$L"
  '"'"' > /dev/null 2>&1 &
  echo launched' >>"$L" 2>&1
say "ALL DONE on llm1 side; llm2 running epoch-2 ceiling test (log: runs/2x_e2_ceiling.log)"
