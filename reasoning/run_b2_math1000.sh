#!/bin/zsh
# B2 — MATH n=1000 rerun to tighten §4's crossover p-value beyond n=500.
# Registered flip: 7B+sc@16 vs 14B true-greedy on the seed-0 n=1000 MATH set (math1000.jsonl).
# Order: the two greedy caches + 7B samples first (the flip ingredients), 14B samples last.
set -e
cd /Users/lab/chess-scaling
PY=./.venv/bin/python
DATA=reasoning/data/math1000.jsonl
echo "[B2] start $(date)"
$PY reasoning/reason_cache.py generate --math --model mlx-community/Qwen2.5-14B-Instruct-4bit \
    --data $DATA --problems 1000 --nmax 1 --temp 0 --out reasoning/cache/math1000_14B_greedy.jsonl
$PY reasoning/reason_cache.py generate --math --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --data $DATA --problems 1000 --nmax 1 --temp 0 --out reasoning/cache/math1000_7B_greedy.jsonl
$PY reasoning/reason_cache.py generate --math --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --data $DATA --problems 1000 --nmax 16 --temp 0.8 --out reasoning/cache/math1000_7B.jsonl
$PY reasoning/reason_cache.py generate --math --model mlx-community/Qwen2.5-14B-Instruct-4bit \
    --data $DATA --problems 1000 --nmax 16 --temp 0.8 --out reasoning/cache/math1000_14B.jsonl
echo "[B2] DONE $(date)"
