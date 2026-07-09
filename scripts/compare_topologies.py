#!/usr/bin/env python
"""Compare network TOPOLOGIES at a fixed (winning) objective, on identical data.

Isolates architecture from objective/size: every contender trains with the same
soft objective on the same positions; we report absolute Elo AND Elo-per-Mparam
(efficiency) since shapes have different natural param counts.

Contenders (override with --topologies): constant · funnel · pyramid · hourglass ·
bottleneck · factored-head · dualpath (wide+deep, "circle iterations").

Example:
  PYTHONPATH=. python scripts/compare_topologies.py --data 'data/eval.000*.npz' \
      --out runs/topology --min-candidates 3 --epochs 3 --games-per-rung 40
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessnet.dataset import Dataset
from chessnet.train import RunConfig, train, load_run
from chessnet.player import ModelPlayer
from chessnet.evaluate import regret_eval, run_match, estimate_elo, load_openings

# name -> ModelConfig-ish kwargs (constant uses depth/width; others use widths).
TOPOLOGIES = {
    "constant-1024x6": dict(width=1024, depth=6),
    "funnel-1024to64": dict(widths=(1024, 512, 256, 128, 64)),
    "pyramid-64to1024": dict(widths=(64, 128, 256, 512, 1024)),
    "hourglass": dict(widths=(1024, 256, 128, 256, 1024)),
    "bottleneck-512to8": dict(widths=(512, 128, 32, 8)),
    "factored-1024x6": dict(width=1024, depth=6, head="factored"),
    "dualpath": dict(arch="dualpath", width=256, wide_width=2048,
                     deep_width=256, deep_layers=6, iters=2, merge="gate"),
    # conv = AlphaZero-style residual conv tower (width=channels, depth=#blocks).
    "conv-96x8": dict(arch="conv", width=96, depth=8, conv_head_channels=8),
    "conv-64x10": dict(arch="conv", width=64, depth=10, conv_head_channels=8),
}


def eval_model(run_dir, encoding, ladder, games, openings, regret_boards,
               movetime, seed):
    model, cfg = load_run(run_dir)
    player = ModelPlayer(model, encoding=encoding, mode="masked", seed=seed)
    rr = regret_eval(player, [b.copy() for b in regret_boards], depth=10)
    specs = [{"kind": "random"}] + [{"kind": "sf_elo", "elo": e} for e in ladder]
    results = [run_match(player, s, games, movetime=movetime, openings=openings,
                         seed=seed + i) for i, s in enumerate(specs)]
    elo, margin = estimate_elo(results)
    # legality health in the deployed reject mode
    rej = ModelPlayer(model, encoding=encoding, mode="reject", seed=seed)
    leg = regret_eval(rej, [b.copy() for b in regret_boards[:80]], depth=6)
    return {"mean_regret": rr.mean_regret, "blunder_rate": rr.blunder_rate,
            "elo": elo, "elo_margin": margin, "illegal_rate": leg.illegal_rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topologies", default=",".join(TOPOLOGIES),
                    help="comma list of names from the TOPOLOGIES table")
    ap.add_argument("--objective", default="soft")
    ap.add_argument("--soft-target", default="softmax")
    ap.add_argument("--soft-baseline", default="min")
    ap.add_argument("--soft-sharpen", type=float, default=1.0)
    ap.add_argument("--min-candidates", type=int, default=3)
    ap.add_argument("--depth-default", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ladder", default="1320,1500")
    ap.add_argument("--games-per-rung", type=int, default=40)
    ap.add_argument("--regret-positions", type=int, default=150)
    ap.add_argument("--openings-pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--movetime", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="label for this machine's shard of work")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.data))
    if not shards:
        sys.exit(f"no shards matched {args.data!r}")
    ds = Dataset(shards, encoding="onehot")
    if args.min_candidates > 1 and ds.has_soft:
        ds = ds.filter_multipv(args.min_candidates)
    print(f"{len(ds):,} training positions | topologies: {args.topologies}")

    openings = load_openings(args.openings_pgn, 200, seed=args.seed + 1)
    regret_boards = load_openings(args.openings_pgn, args.regret_positions,
                                  min_ply=10, max_ply=60, seed=args.seed + 5)
    ladder = [int(x) for x in args.ladder.split(",")]

    os.makedirs(args.out, exist_ok=True)
    summary = {"data": args.data, "n_positions": len(ds), "tag": args.tag,
               "objective": args.objective, "soft_target": args.soft_target,
               "soft_sharpen": args.soft_sharpen, "ladder": ladder, "points": []}

    for name in args.topologies.split(","):
        name = name.strip()
        spec = dict(TOPOLOGIES[name])
        run_dir = os.path.join(args.out, name)
        cfg = RunConfig(encoding="onehot", objective=args.objective,
                        soft_target=args.soft_target, soft_baseline=args.soft_baseline,
                        soft_sharpen=args.soft_sharpen, batch_size=args.batch_size,
                        epochs=args.epochs, lr=args.lr, data_glob=args.data,
                        run_dir=run_dir, seed=args.seed,
                        depth=spec.pop("depth", args.depth_default), **spec)
        print(f"\n=== {name} ({cfg.model_config().param_estimate()/1e6:.2f}M params) ===")
        _, hist = train(cfg, ds)
        ev = eval_model(run_dir, "onehot", ladder, args.games_per_rung, openings,
                        regret_boards, args.movetime, args.seed)
        params = cfg.model_config().param_estimate()
        point = {"name": name, "params": params,
                 "val_top1": hist[-1]["val_top1"],
                 "elo_per_mparam": ev["elo"] / (params / 1e6), **ev}
        summary["points"].append(point)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> {name}: {params/1e6:.2f}M top1={point['val_top1']:.3f} "
              f"elo={ev['elo']:.0f} elo/Mp={point['elo_per_mparam']:.0f}")

    print(f"\nTOPOLOGY SWEEP DONE -> {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
