#!/usr/bin/env python
"""Phase-2 reward-weighted fine-tuning (REINFORCE) of a supervised policy.

Example:
  PYTHONPATH=. python scripts/reinforce.py \
      --warm-start runs/big/d15_w1024_onehot_t1.0 --run-dir runs/phase2/d15w1024 \
      --steps 800 --batch-size 128 --horizon 12 --n-engines 12
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessnet.reinforce import ReinforceConfig, train_reinforce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", required=True, help="supervised run dir")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--openings-pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--horizon", type=int, default=12,
                    help="greedy rollout plies before engine bootstrap (0=one-ply)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--judge-depth", type=int, default=10)
    ap.add_argument("--n-engines", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = ReinforceConfig(
        warm_start=args.warm_start, run_dir=args.run_dir,
        openings_pgn=args.openings_pgn, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, ent_coef=args.ent_coef,
        horizon=args.horizon, temperature=args.temperature,
        judge_depth=args.judge_depth, n_engines=args.n_engines, seed=args.seed)
    train_reinforce(cfg)


if __name__ == "__main__":
    main()
