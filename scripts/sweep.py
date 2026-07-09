#!/usr/bin/env python
"""Efficiency-oriented scaling sweep (proposal 1, 3.3; extended per user).

The study's core question is *rating per coefficient*, so this sweeps several
efficiency levers around a shared reference config and records, for every point,
both raw quality (top-1, regret, Elo) and efficiency (quality per million
params, quality per training example, train wall-clock cost):

  depth      N    : number of layers            (--depths)
  width      W    : layer size                  (--widths)
  encoding        : input size, onehot vs packed(--encodings)
  train-size      : fraction of training data   (--train-sizes)

The label-depth lever (§9 confound) is swept separately by pointing --data at
sets labeled at different Stockfish depths and merging reports; positions are
identical across depths because label.py sampling is seeded.

Example:
  PYTHONPATH=. python scripts/sweep.py --data 'data/real.*.npz' --sweep runs/eff1 \
      --depths 5,15,30 --widths 64,256,1024 --encodings onehot,packed \
      --train-sizes 0.25,0.5,1.0 --games-per-rung 60 --epochs 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

from chessnet.dataset import Dataset, PHASE_NAMES
from chessnet.train import RunConfig, train, load_run
from chessnet.player import ModelPlayer
from chessnet.evaluate import regret_eval, run_match, estimate_elo


def _random_boards(n, seed, skip_plies=8):
    rng = random.Random(seed)
    boards, tries = [], 0
    while len(boards) < n and tries < n * 50:
        tries += 1
        b = chess.Board()
        for _ in range(rng.randint(skip_plies, skip_plies + 30)):
            mv = list(b.legal_moves)
            if not mv:
                break
            b.push(rng.choice(mv))
        if not b.is_game_over():
            boards.append(b.copy())
    return boards


def evaluate_run(run_dir, encoding, ladder, games_per_rung, regret_boards,
                 movetime, mode, seed, openings=None):
    model, cfg = load_run(run_dir)
    player = ModelPlayer(model, encoding=encoding, mode=mode, seed=seed)
    rr = regret_eval(player, [b.copy() for b in regret_boards], depth=10)
    # random-mover anchor (~300 Elo) brackets sub-1320 models; SF rungs above.
    specs = [{"kind": "random"}] + [{"kind": "sf_elo", "elo": e} for e in ladder]
    results = [run_match(player, spec, games_per_rung, movetime=movetime,
                         openings=openings, seed=seed + i)
               for i, spec in enumerate(specs)]
    elo, margin = estimate_elo(results)
    # RAW illegal-selection counter (training-health metric), measured in the
    # deployed 'reject' mode: how often the model's own top pick is illegal, and
    # how many retries the reject wrapper needs. Independent of the always-legal
    # played move, so it shows whether legality improves across trainings.
    reject_player = ModelPlayer(model, encoding=encoding, mode="reject", seed=seed)
    leg = regret_eval(reject_player, [b.copy() for b in regret_boards[:100]],
                      depth=6)
    return {"mean_regret": rr.mean_regret, "blunder_rate": rr.blunder_rate,
            "elo": elo, "elo_margin": margin, "illegal_rate": leg.illegal_rate,
            "illegal_attempts": leg.mean_illegal_attempts}


def build_configs(args):
    """Return list of (axes:set, cfg) specs, deduped, sharing the ref point.

    Default: one-lever axes around a reference config. With --grid, instead build
    the FULL width x train-size cross-product (at ref depth/encoding) so the
    report can show the data x parameters interaction — i.e. where extra
    parameters plateau because there is too little training data.
    """
    ref = dict(depth=args.ref_depth, width=args.ref_width,
               encoding=args.ref_encoding, train_fraction=1.0)
    specs = {}  # key -> (axes:set, cfg)

    def add(axis, **over):
        cfg = {**ref, **over}
        key = (cfg["depth"], cfg["width"], cfg["encoding"], cfg["train_fraction"])
        if key in specs:
            specs[key][0].add(axis)
        else:
            specs[key] = ({axis}, cfg)

    if args.grid:
        widths = _ints(args.widths)
        fracs = _floats(args.train_sizes) or [1.0]
        for w in widths:
            for f in fracs:
                add("grid", width=w, train_fraction=f)
        return list(specs.values())

    for d in _ints(args.depths):
        add("depth", depth=d)
    for w in _ints(args.widths):
        add("width", width=w)
    for e in args.encodings.split(",") if args.encodings else []:
        add("encoding", encoding=e.strip())
    for f in _floats(args.train_sizes):
        add("train_size", train_fraction=f)
    return list(specs.values())


def _ints(s):
    return [int(x) for x in s.split(",")] if s else []


def _floats(s):
    return [float(x) for x in s.split(",")] if s else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--depths", default="5,10,15,20,30")
    ap.add_argument("--widths", default="64,256,1024")
    ap.add_argument("--encodings", default="", help="e.g. onehot,packed")
    ap.add_argument("--train-sizes", default="", help="fractions e.g. 0.25,0.5,1.0")
    ap.add_argument("--grid", action="store_true",
                    help="sweep the full width x train-size grid (data x params "
                         "interaction) instead of one-lever axes")
    ap.add_argument("--ref-depth", type=int, default=15)
    ap.add_argument("--ref-width", type=int, default=64)
    ap.add_argument("--ref-encoding", default="onehot")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ladder", default="1100,1500,1900")
    ap.add_argument("--games-per-rung", type=int, default=40)
    ap.add_argument("--regret-positions", type=int, default=200)
    ap.add_argument("--openings-pgn", default="data/lichess/2013-01.pgn",
                    help="real-game PGN for in-distribution eval openings")
    ap.add_argument("--movetime", type=float, default=0.05)
    ap.add_argument("--mode", choices=["masked", "raw", "reject"], default="masked")
    ap.add_argument("--label-depth", type=int, default=None,
                    help="tag: Stockfish depth this --data was labeled at")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shards = sorted(glob.glob(args.data))
    if not shards:
        sys.exit(f"no shards matched {args.data!r}")

    # datasets are cached per encoding (reused across depth/width/train-size)
    ds_cache = {}
    def get_ds(enc):
        if enc not in ds_cache:
            ds_cache[enc] = Dataset(shards, encoding=enc)
        return ds_cache[enc]

    ladder = _ints(args.ladder)
    # in-distribution eval: real-game openings + real-game regret positions
    if args.openings_pgn and os.path.exists(args.openings_pgn):
        from chessnet.evaluate import load_openings
        openings = load_openings(args.openings_pgn, 200, seed=args.seed + 1)
        regret_boards = load_openings(args.openings_pgn, args.regret_positions,
                                      min_ply=10, max_ply=60, seed=args.seed + 5)
        print(f"eval openings: {len(openings)} real-game positions")
    else:
        openings = None
        regret_boards = _random_boards(args.regret_positions, args.seed + 777)
    configs = build_configs(args)

    os.makedirs(args.sweep, exist_ok=True)
    n_total = len(get_ds(args.ref_encoding))
    phase_dist = {PHASE_NAMES.get(k, str(k)): v
                  for k, v in get_ds(args.ref_encoding).phase_counts().items()}
    summary = {"data": args.data, "n_positions": n_total, "mode": args.mode,
               "ladder": ladder, "label_depth": args.label_depth,
               "ref": {"depth": args.ref_depth, "width": args.ref_width,
                       "encoding": args.ref_encoding},
               "phase_distribution": phase_dist, "points": []}
    print(f"loaded {n_total} positions | phases {phase_dist} | "
          f"{len(configs)} configs")

    for axes, c in configs:
        tag = f"d{c['depth']}_w{c['width']}_{c['encoding']}_t{c['train_fraction']}"
        run_dir = os.path.join(args.sweep, tag)
        cfg = RunConfig(encoding=c["encoding"], depth=c["depth"], width=c["width"],
                        train_fraction=c["train_fraction"],
                        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
                        data_glob=args.data, run_dir=run_dir, seed=args.seed)
        print(f"\n=== {tag} (axes={sorted(axes)}) ===")
        _, history = train(cfg, get_ds(c["encoding"]))
        with open(os.path.join(run_dir, "metrics.json")) as f:
            m = json.load(f)
        ev = evaluate_run(run_dir, c["encoding"], ladder, args.games_per_rung,
                          regret_boards, args.movetime, args.mode, args.seed,
                          openings=openings)
        params = cfg.model_config().param_estimate()
        mparams = params / 1e6
        top1 = history[-1]["val_top1"]
        point = {"axes": sorted(axes), "depth": c["depth"], "width": c["width"],
                 "encoding": c["encoding"], "input_dim": m["input_dim"],
                 "train_fraction": c["train_fraction"], "n_train": m["n_train"],
                 "params": params, "wall_sec": m["wall_sec"],
                 "label_depth": args.label_depth,
                 "val_top1": top1, "val_top5": history[-1]["val_top5"], **ev,
                 # --- efficiency: quality per coefficient / per cost ---
                 "top1_per_mparam": top1 / mparams,
                 "elo_per_mparam": ev["elo"] / mparams,
                 "top1_per_train_k": top1 / (m["n_train"] / 1000.0),
                 "sec_per_top1": m["wall_sec"] / max(top1, 1e-6)}
        summary["points"].append(point)
        with open(os.path.join(args.sweep, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> {mparams:.2f}M params  top1={top1:.3f}  elo={ev['elo']:.0f}  "
              f"illegal={ev['illegal_rate']:.3f}  "
              f"eff: top1/Mp={point['top1_per_mparam']:.3f} "
              f"elo/Mp={point['elo_per_mparam']:.0f}")

    print(f"\nSWEEP DONE -> {os.path.join(args.sweep, 'summary.json')}")


if __name__ == "__main__":
    main()
