#!/usr/bin/env python
"""ET-VI chess stage trajectory — label-bias vs Elo WITHIN one net family (the matched-family run
that resolves the confounded cross-net comparison). For each stage checkpoint model_it{N}.npz from a
from-scratch chess self-play run, measure:
  Elo         : open-loop (raw policy) Elo vs the calibrated Stockfish ladder
  label_bias  : mean |V^pi - V*|, V^pi = win-rate under the checkpoint's own play, V* = Stockfish
F1-chess reads off the crossover: does the net plateau (Elo flat) as label bias floors?

  PYTHONPATH=. ./.venv/bin/python chessnet/et6_chess_traj_analyze.py --run runs/et6_chess_traj
"""
import argparse, glob, json, os, random, re
import numpy as np
import chess, chess.engine
import mlx.core as mx
from chessnet.model import ModelConfig, PolicyNet
from chessnet.player import ModelPlayer
from chessnet.labeler import score_to_winprob, DEFAULT_STOCKFISH
from chessnet.evaluate import load_openings
from scripts.chess_evalfirst import eval_open_loop_elo
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from et6_chess_labelbias import rollout_winrate


def load_ckpt(run, npz):
    cfg_d = json.load(open(os.path.join(run, "config.json")))
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    net = PolicyNet(cfg); net.load_weights(npz); mx.eval(net.parameters())
    return net, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/et6_chess_traj")
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--ladder", type=int, nargs="+", default=[1320, 1700, 2100])
    ap.add_argument("--pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/et6_chess_traj_analysis.json")
    a = ap.parse_args()
    ckpts = sorted(glob.glob(os.path.join(a.run, "model_it*.npz")),
                   key=lambda p: int(re.search(r"model_it(\d+)", p).group(1)))
    try:
        openings = load_openings(a.pgn, 200)
    except Exception as e:
        print(f"  openings load failed ({e}); standard start", flush=True); openings = None

    # one fixed position set (seed-fixed) so label bias is comparable across stages
    rng = random.Random(a.seed)
    boards = []
    while len(boards) < a.positions:
        b = chess.Board()
        for _ in range(rng.randint(10, 30)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over() and b.legal_moves:
            boards.append(b)

    eng = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH); eng.configure({"Threads": 2, "Hash": 256})
    lim = chess.engine.Limit(depth=a.depth)
    traj = []
    for npz in ckpts:
        it = int(re.search(r"model_it(\d+)", npz).group(1))
        net, cfg = load_ckpt(a.run, npz)
        elo, margin = eval_open_loop_elo(net, cfg, a.eval_games, openings, a.ladder, a.seed + it)
        player = ModelPlayer(net, encoding=cfg.encoding, mode="masked", temperature=0.4, seed=a.seed)
        bias = float(np.mean([abs(rollout_winrate(player, b, a.k, 80, eng, lim) -
                                  score_to_winprob(eng.analyse(b, lim)["score"], b.turn)) for b in boards]))
        traj.append({"iter": it, "elo": round(float(elo), 0), "elo_margin": round(float(margin), 0),
                     "label_bias": round(bias, 3)})
        print(f"  it{it:>3}  Elo={elo:+.0f}+/-{margin:.0f}  label_bias={bias:.3f}", flush=True)
    eng.quit()

    elos = [r["elo"] for r in traj]; biases = [r["label_bias"] for r in traj]
    corr = float(np.corrcoef(elos, biases)[0, 1]) if len(traj) > 2 and np.std(elos) > 0 else float("nan")
    print(f"\n[traj] {len(traj)} stages | corr(Elo, label_bias) = {corr:+.3f}  "
          f"({'bias FALLS as Elo rises — label ceiling recedes with strength' if corr < -0.1 else 'no clear relationship'})")
    json.dump({"run": a.run, "corr_elo_bias": round(corr, 3), "trajectory": traj},
              open(a.out, "w"), indent=2)
    print(f"[traj] wrote {a.out}")


if __name__ == "__main__":
    main()
