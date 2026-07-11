#!/usr/bin/env python
"""Search-distillation on a FIXED, diverse position set (no new games).

Jump-start from a strong net (e.g. conv_value_llm1, whose MCTS-800 plays ~2800). Each iteration:
run MCTS on the *current* net over a fixed set of diverse positions, and distill the resulting
visit-distribution (improved policy) + backed-up root value into the net. Repeat. This is
AlphaZero-style policy/value iteration but on a fixed diverse dataset (avoids self-play's
diversity collapse) with a strong search teacher. Question: does the RAW policy climb toward the
search's level (~2800) -- and where does it flatten (the value-net ceiling)?

  PYTHONPATH=. python scripts/distill.py --run-dir runs/distill --init runs/conv_value_llm1/model.npz \
      --arch-from runs/conv_value_llm1 --sims 800 --positions 4000 --workers 16 --iters 20
"""
from __future__ import annotations
import argparse, json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
import chess, chess.pgn

from chessnet.model import ModelConfig, PolicyNet
from chessnet.encoding import board_to_codes, move_to_index
from chessnet.search import MCTSPlayer
from chessnet.train import _clip_grads, load_run, RunConfig
from chessnet.player import ModelPlayer
from chessnet.evaluate import run_match, estimate_elo, load_openings
from selfplay import train_steps                      # reuse the training loop


def sample_positions(pgn, n, seed, lo=8, hi=90):
    rng = random.Random(seed); fens = []
    with open(pgn) as f:
        while len(fens) < n:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            mv = list(g.mainline_moves())
            if len(mv) < lo + 2:
                continue
            b = g.board()
            for x in mv[:rng.randint(lo, min(hi, len(mv) - 1))]:
                b.push(x)
            if not b.is_game_over() and any(b.legal_moves):
                fens.append(b.fen())
    return fens


def _worker(task):
    run_dir, sims, fens, seed = task
    model, cfg = load_run(run_dir)
    player = MCTSPlayer(model, sims=sims, seed=seed)
    out = []
    for fen in fens:
        b = chess.Board(fen)
        root = player._run(b.copy())
        moves = list(root.children.keys())
        if not moves:
            continue
        visits = np.array([root.children[m].N for m in moves], dtype=np.float32)
        probs = visits / max(visits.sum(), 1.0)
        mirrored = b.turn == chess.BLACK
        idxs = np.array([move_to_index(m, mirrored) for m in moves], dtype=np.int32)
        val = root.q() if root.N else 0.5                # search-backed value for side to move
        codes, meta = board_to_codes(b)
        out.append((codes, meta, idxs, probs, float(val)))
    return out


def raw_elo(model, cfg, ladder, games, openings, movetime, seed):
    raw = ModelPlayer(model, encoding=cfg.encoding, mode="masked", seed=seed)
    specs = [{"kind": "random"}] + [{"kind": "sf_elo", "elo": e} for e in ladder]
    res = [run_match(raw, s, games, movetime=movetime, openings=openings, seed=seed + i)
           for i, s in enumerate(specs)]
    return estimate_elo(res)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--arch-from", required=True)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--positions", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--ladder", default="2000,2400,2700")
    ap.add_argument("--check-every", type=int, default=2)
    ap.add_argument("--check-games", type=int, default=20)
    ap.add_argument("--movetime", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    _, base = load_run(args.arch_from)
    cfg = ModelConfig(encoding="onehot", arch=base.arch, width=base.width, depth=base.depth,
                      value_head=True)
    json.dump(__import__("dataclasses").asdict(RunConfig(
        encoding="onehot", arch=base.arch, width=base.width, depth=base.depth,
        value_head=True, run_dir=args.run_dir)),
        open(os.path.join(args.run_dir, "config.json"), "w"), indent=2)
    mx.random.seed(args.seed)
    model = PolicyNet(cfg); model.load_weights(args.init); mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    ladder = [int(x) for x in args.ladder.split(",")]
    openings = load_openings(args.pgn, 200, seed=args.seed + 5)

    fens = sample_positions(args.pgn, args.positions, args.seed + 1)   # FIXED diverse set
    model.save_weights(os.path.join(args.run_dir, "model.npz"))        # workers load this
    print(f"[distill] {cfg.arch} w{cfg.width} d{cfg.depth} | sims={args.sims} "
          f"positions={len(fens)} | init={args.init}", flush=True)
    e0 = raw_elo(model, cfg, ladder, args.check_games, openings, args.movetime, args.seed)
    print(f"[distill] iter -1 (start)  raw Elo = {e0:.0f}", flush=True)

    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(args.workers)
    per = max(1, len(fens) // args.workers)
    chunks = [fens[i:i + per] for i in range(0, len(fens), per)]
    log = []
    for it in range(args.iters):
        t0 = time.time()
        tasks = [(args.run_dir, args.sims, ch, args.seed + it * 100 + w)   # fresh MCTS targets
                 for w, ch in enumerate(chunks)]
        buffer = [s for part in pool.map(_worker, tasks) for s in part]
        loss = train_steps(model, opt, buffer, args.steps, args.batch, args.clip, rng)
        model.save_weights(os.path.join(args.run_dir, "model.npz"))
        rec = {"iter": it, "n": len(buffer), "loss": round(loss, 3), "sec": round(time.time()-t0, 1)}
        if (it + 1) % args.check_every == 0:
            rec["raw_elo"] = round(raw_elo(model, cfg, ladder, args.check_games, openings,
                                           args.movetime, args.seed + it))
        log.append(rec)
        json.dump(log, open(os.path.join(args.run_dir, "distill_log.json"), "w"))
        msg = f"  iter {it}: n={len(buffer)} loss {loss:.3f}"
        if "raw_elo" in rec:
            msg += f"   RAW Elo {rec['raw_elo']}  (start {e0:.0f})"
        print(msg + f"  {rec['sec']:.0f}s", flush=True)
    print("[distill] DONE", flush=True)


if __name__ == "__main__":
    main()
