#!/usr/bin/env python
"""A/B the training objectives on IDENTICAL data (don't assume — measure).

A multi-PV soft-labeled shard carries BOTH targets:
  * `target`            -> the single best move (the OLD objective: imitation)
  * `soft_idx/soft_wp`  -> the per-move advantage map (the NEW objective)

So we can train one model per objective on the exact same positions/architecture
and compare true strength (regret on real positions + bracketed-ladder Elo). The
only thing that differs is the loss. Results are written to a summary the report
generator renders as a head-to-head.

Example:
  PYTHONPATH=. python scripts/compare_objectives.py --data 'data/soft.*.npz' \
      --out runs/objective_ab --widths 256,1024 --depth 10 --epochs 8 \
      --games-per-rung 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob

from chessnet.dataset import Dataset
from chessnet.train import RunConfig, train, load_run
from chessnet.player import ModelPlayer
from chessnet.evaluate import (regret_eval, run_match, estimate_elo,
                               load_openings)


def eval_model(run_dir, encoding, ladder, games, openings, regret_boards,
               movetime, seed):
    model, cfg = load_run(run_dir)
    player = ModelPlayer(model, encoding=encoding, mode="masked", seed=seed)
    rr = regret_eval(player, [b.copy() for b in regret_boards], depth=10)
    specs = [{"kind": "random"}] + [{"kind": "sf_elo", "elo": e} for e in ladder]
    results = [run_match(player, s, games, movetime=movetime, openings=openings,
                         seed=seed + i) for i, s in enumerate(specs)]
    elo, margin = estimate_elo(results)
    return {"mean_regret": rr.mean_regret, "blunder_rate": rr.blunder_rate,
            "elo": elo, "elo_margin": margin,
            "matches": [{"opponent": r.opponent_name, "score": r.score,
                         "games": r.games} for r in results]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="soft-labeled shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="256,1024")
    ap.add_argument("--variants", default="hard,soft:softmax:min,soft:ratio:min",
                    help="comma list; soft variants as soft:<softmax|ratio>:<min|mean|half>")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--min-candidates", type=int, default=3,
                    help="only train on positions with >= this many recorded moves "
                         "(so the soft target has a real distribution)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--ladder", default="1320,1500")
    ap.add_argument("--games-per-rung", type=int, default=60)
    ap.add_argument("--regret-positions", type=int, default=150)
    ap.add_argument("--openings-pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--movetime", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shards = sorted(glob.glob(args.data))
    if not shards:
        sys.exit(f"no shards matched {args.data!r}")
    ds = Dataset(shards, encoding="onehot")
    if not ds.has_soft:
        sys.exit("dataset has no soft labels; re-label with --soft")
    print(f"loaded {len(ds):,} soft-labeled positions")
    if args.min_candidates > 1:
        ds = ds.filter_multipv(args.min_candidates)
        print(f"filtered to {len(ds):,} positions with >= {args.min_candidates} "
              f"candidate moves (fair soft-target subset)")

    openings = load_openings(args.openings_pgn, 200, seed=args.seed + 1)
    regret_boards = load_openings(args.openings_pgn, args.regret_positions,
                                  min_ply=10, max_ply=60, seed=args.seed + 5)
    ladder = [int(x) for x in args.ladder.split(",")]
    widths = [int(x) for x in args.widths.split(",")]

    # parse variants: "hard" | "soft:<softmax|ratio>:<min|mean|half>[:<sharpen γ>]"
    variants = []
    for v in args.variants.split(","):
        parts = v.split(":")
        if parts[0] == "hard":
            variants.append({"name": "hard", "objective": "hard"})
        else:
            st = parts[1] if len(parts) > 1 else "softmax"
            bl = parts[2] if len(parts) > 2 else "min"
            sh = float(parts[3]) if len(parts) > 3 else 1.0
            name = f"soft-{st}" + (f"-{bl}" if st == "ratio" else "")
            name += f"-g{sh:g}" if (st == "ratio" and sh != 1.0) else ""
            variants.append({"name": name, "objective": "soft", "soft_target": st,
                             "soft_baseline": bl, "soft_sharpen": sh})

    os.makedirs(args.out, exist_ok=True)
    summary = {"data": args.data, "n_positions": len(ds), "depth": args.depth,
               "ladder": ladder, "tau": args.tau,
               "variants": [v["name"] for v in variants], "points": []}

    for width in widths:
        for v in variants:
            tag = f"w{width}_{v['name']}"
            run_dir = os.path.join(args.out, tag)
            cfg = RunConfig(encoding="onehot", depth=args.depth, width=width,
                            objective=v["objective"], tau=args.tau,
                            soft_target=v.get("soft_target", "softmax"),
                            soft_baseline=v.get("soft_baseline", "min"),
                            soft_sharpen=v.get("soft_sharpen", 1.0),
                            batch_size=args.batch_size, epochs=args.epochs,
                            lr=args.lr, data_glob=args.data, run_dir=run_dir,
                            seed=args.seed)
            print(f"\n=== TRAIN {tag} ===")
            _, hist = train(cfg, ds)
            print(f"=== EVAL {tag} ===")
            ev = eval_model(run_dir, "onehot", ladder, args.games_per_rung,
                            openings, regret_boards, args.movetime, args.seed)
            point = {"width": width, "objective": v["name"],
                     "params": cfg.model_config().param_estimate(),
                     "val_top1": hist[-1]["val_top1"], **ev}
            summary["points"].append(point)
            with open(os.path.join(args.out, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  -> {v['name']}: top1={point['val_top1']:.3f} "
                  f"regret={ev['mean_regret']:.3f} elo={ev['elo']:.0f}")

    # head-to-head deltas vs the hard baseline, per width
    print("\n===== OBJECTIVE A/B (variant − hard) =====")
    by = {(p["width"], p["objective"]): p for p in summary["points"]}
    for w in widths:
        h = by.get((w, "hard"))
        if not h:
            continue
        print(f"  W={w} (hard: regret {h['mean_regret']:.3f}, elo {h['elo']:.0f}):")
        for v in variants:
            if v["name"] == "hard":
                continue
            s = by.get((w, v["name"]))
            if s:
                print(f"    {v['name']:<18} Δregret={s['mean_regret']-h['mean_regret']:+.3f} "
                      f"Δblunder={s['blunder_rate']-h['blunder_rate']:+.3f} "
                      f"Δelo={s['elo']-h['elo']:+.0f} Δtop1={s['val_top1']-h['val_top1']:+.3f}")
    print(f"\nwrote {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
