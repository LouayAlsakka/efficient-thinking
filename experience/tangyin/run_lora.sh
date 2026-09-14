#!/usr/bin/env bash
# Mechanism E: LoRA on the 63-example train split. 理's ruling — rank 8, 3 epochs.
# 3 epochs x 63 examples at batch size 1 = 189 iterations.
# Committed before running, per the series rule; the adapter path carries the rank and the epochs
# so a restore-by-filename cannot pick up the wrong one (窯's T3.4v, six for six).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HOME/claude/agents/Sautee/venvs/vlm312/bin/python}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${OUT:-$HERE/adapters/tangyin_lora_r8_e3}"
mkdir -p "$OUT"
"$PY" -m mlx_lm lora \
  --model "$MODEL" --train \
  --data "$HERE/lora_data" \
  --fine-tune-type lora --num-layers 16 \
  --batch-size 1 --iters 189 \
  --learning-rate 1e-5 \
  --steps-per-report 20 --steps-per-eval 60 --val-batches 7 \
  --adapter-path "$OUT" --save-every 60 \
  --max-seq-length 1024 --seed 7 2>&1 | tee "$OUT/train.log"
echo "adapter: $OUT"
