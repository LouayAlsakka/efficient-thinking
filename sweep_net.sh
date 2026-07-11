#!/bin/bash
# Capacity-sweep worker: train one width on FULL data (batch 4096) after a prerequisite job's log
# shows "ALL DONE", then eval MCTS-800 on the high ladder. Args: WIDTH LABEL WAITLOG
set -u
cd "$HOME/chess-scaling" || exit 1
W="$1"; LAB="$2"; WAITLOG="$3"
RUN="runs/cap_${LAB}_w${W}"
L="runs/sweep_${LAB}.log"; : > "$L"
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$L"; }
mkdir -p "$RUN"

say "waiting for prerequisite ($WAITLOG) 'ALL DONE'..."
while ! grep -q "ALL DONE" "$WAITLOG" 2>/dev/null; do sleep 120; done
say "prerequisite done. training capacity net $LAB (w$W) on full data (batch 4096)"

extra=""; cum=0; restart=0; TARGET=90000       # ~1 epoch at batch 4096 (~394M/4096)
while true; do
  set -f
  caffeinate -i env PYTHONPATH=. ./.venv/bin/python -u scripts/train.py \
    --data "data/eval.*.npz" --run-dir "$RUN" --arch conv --width "$W" --depth 8 \
    --objective soft --value-head --grad-clip 1.0 --lr 1e-3 --batch-size 4096 --epochs 1 \
    --ckpt-every 3000 --seed "$restart" $extra >> "$RUN/train.log" 2>&1 &
  P=$!; set +f
  say "$LAB launched pid $P seed=$restart"
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
  say "$LAB segment ended step $st (cum ~$cum/$TARGET stalled=$stalled)"
  [ "$cum" -ge "$TARGET" ] && { say "$LAB target reached"; break; }
  [ "$stalled" -eq 0 ] && { say "$LAB finished normally"; break; }
  restart=$((restart+1)); extra="--init $RUN/model.npz"
done
say "$LAB training COMPLETE"
say "eval MCTS-800 absolute on $LAB net"
PYTHONPATH=. ./.venv/bin/python scripts/eval_search.py --run-dir "$RUN" --method mcts --sims 800 \
  --ladder 2500,2800,3050 --games-per-rung 24 --movetime 0.04 --seed 0 >> "$L" 2>&1
say "ALL DONE ($LAB w$W capacity net)"
