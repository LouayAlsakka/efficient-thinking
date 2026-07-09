#!/usr/bin/env python
"""Train one scaling-curve point.

Example:
  PYTHONPATH=. python scripts/train.py --depth 10 --width 64 \
      --data 'data/smoke.*.npz' --run-dir runs/d10w64 --epochs 8
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessnet.dataset import Dataset
from chessnet.train import RunConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob for shard .npz files")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--init", default=None, help="warm-start weights (.npz) to fine-tune from")
    ap.add_argument("--encoding", choices=["onehot", "packed"], default="onehot")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--widths", default=None,
                    help="explicit body-width schedule, e.g. 1024,512,256,128,64 "
                         "(funnel) or 64,128,256,512 (pyramid); overrides depth/width")
    ap.add_argument("--activation", choices=["gelu", "relu"], default="gelu")
    ap.add_argument("--head", choices=["dense", "factored"], default="dense",
                    help="factored = separate from/to heads (~30x fewer head params)")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--arch", choices=["mlp", "dualpath", "conv"], default="mlp",
                    help="conv = AlphaZero-style residual conv tower over 8x8 planes "
                         "(width=channels, depth=#blocks; onehot only)")
    ap.add_argument("--conv-head-channels", type=int, default=8,
                    help="conv: 1x1 reduce channels before the dense move head")
    ap.add_argument("--objective", choices=["soft", "hard"], default="hard",
                    help="soft = advantage-weighted distribution (needs --soft "
                         "labels); hard = single-best-move imitation")
    ap.add_argument("--tau", type=float, default=0.08,
                    help="soft-target softmax temperature (winprob space)")
    ap.add_argument("--soft-target", choices=["softmax", "ratio"], default="softmax",
                    help="soft target shape: softmax(Boltzmann) or ratio(linear advantage)")
    ap.add_argument("--soft-baseline", choices=["min", "mean", "half"], default="min",
                    help="ratio-mode baseline b in (v_i-b)/(v*-b)")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help=">0: atomically save model.npz every N steps (crash safety)")
    ap.add_argument("--value-head", action="store_true",
                    help="also train a scalar Eval(N) value head (closed-loop Stage 0; "
                         "forces soft batches for the win-prob target)")
    ap.add_argument("--value-weight", type=float, default=0.5,
                    help="weight of the value MSE in the combined loss")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help=">0: clip gradients to this global L2 norm (stability)")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.data))
    if not shards:
        sys.exit(f"no shards matched {args.data!r}")

    cfg = RunConfig(
        encoding=args.encoding, depth=args.depth, width=args.width,
        widths=tuple(int(x) for x in args.widths.split(",")) if args.widths else None,
        activation=args.activation, residual=not args.no_residual, head=args.head,
        arch=args.arch, conv_head_channels=args.conv_head_channels,
        value_head=args.value_head, value_weight=args.value_weight,
        grad_clip=args.grad_clip,
        objective=args.objective, tau=args.tau,
        soft_target=args.soft_target, soft_baseline=args.soft_baseline,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, val_frac=args.val_frac, seed=args.seed,
        data_glob=args.data, run_dir=args.run_dir, ckpt_every=args.ckpt_every,
        init=args.init,
    )
    ds = Dataset(shards, encoding=args.encoding)
    print(f"loaded {len(ds)} positions from {len(shards)} shard(s)")
    train(cfg, ds)


if __name__ == "__main__":
    main()
