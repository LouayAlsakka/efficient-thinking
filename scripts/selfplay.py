#!/usr/bin/env python
"""Stage 3 — self-play expert iteration (AlphaZero-style, NO Stockfish).

Loop: the net (with MCTS) plays games against itself -> record each position, the MCTS
visit distribution (improved policy target), and the eventual game outcome (value target)
-> train the net toward them -> repeat. The only external input is the rules of chess
(legal moves + who won). Start from a random net to learn purely from self-play.

  PYTHONPATH=. python scripts/selfplay.py --run-dir runs/sp1 --width 64 --depth 6 \
      --sims 48 --games 60 --iters 100
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import chess

from chessnet.model import ModelConfig, PolicyNet
from chessnet.encoding import (board_to_codes, codes_to_onehot_batch, MOVE_DIM)
from chessnet.search import MCTSPlayer
from chessnet.train import _clip_grads


def play_game(player, max_moves, temp_moves, rng):
    """One self-play game. Returns list of (codes, meta, idxs, probs, mover)."""
    board = chess.Board()
    hist = []
    while not board.is_game_over(claim_draw=True) and len(hist) < max_moves:
        moves, idxs, probs = player.search_visits(board)
        if len(moves) == 0:
            break
        codes, meta = board_to_codes(board)
        hist.append((codes, meta, idxs, probs, board.turn))
        if len(hist) <= temp_moves:                 # explore early: sample by visits
            mv = moves[rng.choice(len(moves), p=probs)]
        else:                                        # then greedy (most-visited)
            mv = moves[int(np.argmax(probs))]
        board.push(mv)
    res = board.result(claim_draw=True)
    winner = chess.WHITE if res == "1-0" else chess.BLACK if res == "0-1" else None
    out = []
    for codes, meta, idxs, probs, mover in hist:
        v = 0.5 if winner is None else (1.0 if mover == winner else 0.0)
        out.append((codes, meta, idxs, probs, v))
    return out, res


def _worker(task):
    """Parallel self-play worker (spawn-safe): load latest model, play n games, return
    samples + result counts. Runs in its own process so many games proceed concurrently."""
    run_dir, sims, dir_alpha, n_games, max_moves, temp_moves, seed = task
    from chessnet.train import load_run
    model, cfg = load_run(run_dir)
    player = MCTSPlayer(model, sims=sims, dirichlet_alpha=dir_alpha, seed=seed)
    rng = np.random.default_rng(seed)
    samples = []
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    for _ in range(n_games):
        s, r = play_game(player, max_moves, temp_moves, rng)
        samples += s
        results[r if r in results else "*"] += 1
    return samples, results


def _loss_fn(m, x, ix, pb, vt):
    logits, v = m(x, return_value=True)
    logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    chosen = mx.take_along_axis(logp, ix, axis=1)      # [B,K]
    pol = -mx.mean(mx.sum(pb * chosen, axis=1))
    return pol + 0.5 * mx.mean((v - vt) ** 2)


def train_steps(model, opt, buffer, steps, batch, clip, rng):
    """Sample `steps` minibatches from the REPLAY BUFFER (not just latest games) —
    prevents catastrophic forgetting. Encodes only each minibatch (cheap)."""
    lg = nn.value_and_grad(model, _loss_fn)
    n = len(buffer)
    last = 0.0
    for _ in range(steps):
        sel = rng.integers(0, n, size=min(batch, n))
        bs = [buffer[i] for i in sel]
        codes = np.stack([s[0] for s in bs]); meta = np.stack([s[1] for s in bs])
        X = codes_to_onehot_batch(codes, meta)
        K = max(len(s[2]) for s in bs)
        idxs = np.zeros((len(bs), K), np.int32); prb = np.zeros((len(bs), K), np.float32)
        for i, s in enumerate(bs):
            k = len(s[2]); idxs[i, :k] = s[2]; prb[i, :k] = s[3]
        val = np.array([s[4] for s in bs], np.float32)
        l, g = lg(model, mx.array(X), mx.array(idxs), mx.array(prb), mx.array(val))
        if clip > 0:
            g = _clip_grads(g, clip)
        opt.update(model, g); mx.eval(model.parameters(), opt.state)
        last = l.item()
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1,
                    help=">1: generate self-play games across N parallel processes")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--temp-moves", type=int, default=16)
    ap.add_argument("--dirichlet", type=float, default=0.3,
                    help="root Dirichlet-noise alpha for self-play exploration (0=off)")
    ap.add_argument("--buffer", type=int, default=120000, help="replay buffer max positions")
    ap.add_argument("--steps", type=int, default=300, help="grad steps per iteration")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--init", default=None, help="warm-start model.npz (else random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=0, help="save a distinct stage checkpoint model_it{N}.npz every N iters")
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    cfg = ModelConfig(encoding="onehot", arch="conv", width=args.width,
                      depth=args.depth, value_head=True)
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        from dataclasses import asdict
        # store a RunConfig-compatible dict so load_run works
        from chessnet.train import RunConfig
        rc = RunConfig(encoding="onehot", arch="conv", width=args.width, depth=args.depth,
                       value_head=True, run_dir=args.run_dir)
        json.dump(asdict(rc), f, indent=2)
    mx.random.seed(args.seed)
    model = PolicyNet(cfg)
    if args.init:
        model.load_weights(args.init)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    print(f"[selfplay] conv w{args.width} d{args.depth} | sims={args.sims} "
          f"games/iter={args.games} | init={'warm' if args.init else 'RANDOM'}", flush=True)

    from collections import deque
    buffer = deque(maxlen=args.buffer)
    model.save_weights(os.path.join(args.run_dir, "model.npz"))   # so workers load it
    pool = None
    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(args.workers)
    log = []
    for it in range(args.iters):
        t0 = time.time()
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}; new = 0
        if pool is not None:                                       # PARALLEL self-play
            per = max(1, args.games // args.workers)
            tasks = [(args.run_dir, args.sims, args.dirichlet, per, args.max_moves,
                      args.temp_moves, args.seed + it * 1000 + w) for w in range(args.workers)]
            for samples, res in pool.map(_worker, tasks):
                buffer.extend(samples); new += len(samples)
                for k in results: results[k] += res[k]
        else:                                                      # serial (1 worker)
            player = MCTSPlayer(model, sims=args.sims,
                                dirichlet_alpha=args.dirichlet, seed=args.seed + it)
            for _ in range(args.games):
                s, r = play_game(player, args.max_moves, args.temp_moves, rng)
                buffer.extend(s); new += len(s); results[r if r in results else "*"] += 1
        loss = (train_steps(model, opt, buffer, args.steps, args.batch, args.clip, rng)
                if len(buffer) >= args.batch else 0.0)
        model.save_weights(os.path.join(args.run_dir, "model.npz"))
        if args.ckpt_every and (it + 1) % args.ckpt_every == 0:      # stage checkpoint for the label-bias-vs-Elo trajectory
            model.save_weights(os.path.join(args.run_dir, f"model_it{it + 1}.npz"))
        dt = time.time() - t0
        rec = {"iter": it, "new": new, "buffer": len(buffer), "loss": round(loss, 3),
               "results": results, "sec": round(dt, 1)}
        log.append(rec)
        with open(os.path.join(args.run_dir, "selfplay_log.json"), "w") as f:
            json.dump(log, f)
        print(f"  iter {it}: +{new} buf={len(buffer)}  loss {loss:.3f}  "
              f"W/B/D {results['1-0']}/{results['0-1']}/{results['1/2-1/2']}  {dt:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
