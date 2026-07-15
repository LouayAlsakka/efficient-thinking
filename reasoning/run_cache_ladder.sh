#!/bin/bash
# Run a list of "tag:model_id" specs sequentially through the batched cache generator.
# Resumable (reason_cache appends). Usage: bash reasoning/run_cache_ladder.sh "3B:mlx-community/..." ...
cd "$(dirname "$0")/.." || exit 1
LOG=${LOG:-reasoning/results/cache_$(hostname -s).log}
PROB=${PROB:-500}; NMAX=${NMAX:-32}
mkdir -p reasoning/cache reasoning/results
echo "=== ladder START $(date +%Y-%m-%d_%H:%M:%S) on $(hostname -s): $* ===" >> "$LOG"
for spec in "$@"; do
  t=${spec%%:*}; m=${spec#*:}
  echo "=== $t START $(date +%H:%M:%S) ===" >> "$LOG"
  caffeinate -i ./.venv/bin/python reasoning/reason_cache.py generate \
    --model "$m" --problems "$PROB" --nmax "$NMAX" \
    --out "reasoning/cache/gsm8k_${t}.jsonl" >> "$LOG" 2>&1
  echo "=== $t END $(date +%H:%M:%S) ===" >> "$LOG"
done
echo "ALL_DONE $(date +%H:%M:%S)" >> "$LOG"
