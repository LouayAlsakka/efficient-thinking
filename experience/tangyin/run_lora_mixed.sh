#!/usr/bin/env bash
# Item 1: the form-preserving arm. ONE EPOCH over the mixed set (119 examples = 119 iters at
# batch 1), rank 8, lr 1e-5, seed 7 — every hyperparameter identical to the plain arm so the
# ONLY variable is the anchor data.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HOME/claude/agents/Sautee/venvs/vlm312/bin/python}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${OUT:-$HERE/adapters/tangyin_lora_r8_mixed_e1}"
mkdir -p "$OUT"
"$PY" -m mlx_lm lora \
  --model "$MODEL" --train \
  --data "$HERE/lora_data_mixed" \
  --fine-tune-type lora --num-layers 16 \
  --batch-size 1 --iters 119 \
  --learning-rate 1e-5 \
  --steps-per-report 20 --steps-per-eval 40 --val-batches 14 \
  --adapter-path "$OUT" --save-every 40 \
  --max-seq-length 1024 --seed 7 2>&1 | tee "$OUT/train.log"
echo "adapter: $OUT"
