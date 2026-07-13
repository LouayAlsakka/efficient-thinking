#!/usr/bin/env python
"""Chess eval-first self-training (AlphaZero expert iteration) — does distilling HIGH-FIDELITY MCTS
targets break the from-scratch ~2000 self-play plateau?

The bet: the prior from-scratch flywheel stalled at ~2000 because its self-play search was shallow ->
weak targets. Here we use strong-search MCTS self-play (visit-policy + game-outcome value = the
game-rules oracle), distilled value-first into the evaluator, and watch the OPEN-LOOP Elo climb.

Reuses the existing infra: MCTSPlayer.search_visits (policy target), the value head (value target),
ModelPlayer + run_match + estimate_elo (the calibrated Elo ladder). Robust for unattended runs:
checkpoints every iter, Elo eval guarded so it never kills self-play.

  ./.venv/bin/python scripts/chess_evalfirst.py --arch-from runs/conv_value_full --sims 100 \
      --iters 200 --games 32 --run-dir runs/evalfirst_scratch
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import chess
from chessnet.model import PolicyNet
from chessnet.train import load_run
from chessnet.encoding import ENCODERS, MOVE_DIM
from chessnet.search import BatchedMCTSPlayer
from chessnet.player import ModelPlayer
from chessnet.evaluate import run_match, estimate_elo, load_openings


def self_play_game(player, py_rng, np_rng, max_moves=200, temp_moves=20, open_plies=0):
    board = chess.Board()
    for _ in range(open_plies):
        if board.is_game_over(claim_draw=True):
            break
        board.push(py_rng.choice(list(board.legal_moves)))
    hist = []
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        moves, idxs, probs = player.search_visits(board)
        if len(moves) == 0:
            break
        hist.append((player.encode(board), idxs.copy(), probs.copy(), board.turn))
        if ply < temp_moves and probs.sum() > 0:
            mv = moves[int(np_rng.choice(len(moves), p=probs))]
        else:
            mv = moves[int(np.argmax(probs))]
        board.push(mv); ply += 1
    res = board.result(claim_draw=True)
    z = {chess.WHITE: 0.5, chess.BLACK: 0.5} if res == "1/2-1/2" else \
        ({chess.WHITE: 1.0, chess.BLACK: 0.0} if res == "1-0" else {chess.WHITE: 0.0, chess.BLACK: 1.0})
    return [(enc, idxs, probs, z[mover]) for (enc, idxs, probs, mover) in hist], res


def train_on_buffer(net, opt, lg, buf, epochs, batch, py_rng):
    data = list(buf)
    for _ in range(epochs):
        py_rng.shuffle(data)
        for i in range(0, len(data), batch):
            chunk = data[i:i + batch]
            X = mx.array(np.stack([c[0] for c in chunk]))
            P = np.zeros((len(chunk), MOVE_DIM), dtype=np.float32)
            for r, c in enumerate(chunk):
                P[r, c[1]] = c[2]
            V = mx.array(np.array([c[3] for c in chunk], dtype=np.float32))
            _, g = lg(net, X, mx.array(P), V)
            opt.update(net, g); mx.eval(net.parameters(), opt.state)


def eval_open_loop_elo(net, cfg, games, openings, ladder, seed):
    """Open-loop (raw policy) Elo vs the calibrated ladder. Guarded by caller."""
    player = ModelPlayer(net, encoding=cfg.encoding, mode="masked", seed=seed)
    specs = [{"kind": "random"}] + [{"kind": "sf_elo", "elo": e} for e in ladder]
    res = [run_match(player, s, games, openings=openings, seed=seed + i) for i, s in enumerate(specs)]
    return estimate_elo(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch-from", default="runs/conv_value_full",
                    help="run whose ARCH/config to copy (weights are re-initialized = from scratch)")
    ap.add_argument("--init", default=None, help="optional: warm-start weights from this run")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--batch-mcts", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=100000)
    ap.add_argument("--temp-moves", type=int, default=20)
    ap.add_argument("--open-plies", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--ladder", type=int, nargs="+", default=[1320, 1500, 1700, 1900, 2100])
    ap.add_argument("--pgn", default="data/lichess/2013-12.pgn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default="runs/evalfirst_scratch")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    import random
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    _, cfg = load_run(args.arch_from)                       # copy arch (incl. value_head)
    net = PolicyNet(cfg)
    if args.init:
        net2, _ = load_run(args.init); net.update(net2.parameters()); print(f"[evalfirst] warm-start {args.init}", flush=True)
    else:
        print(f"[evalfirst] FROM SCRATCH, arch from {args.arch_from} (d{cfg.depth} w{cfg.width} {cfg.encoding} value_head={cfg.value_head})", flush=True)
    mx.eval(net.parameters())
    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(net, X, P, V):
        logits, value = net(X, return_value=True)
        logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        pol = -(P * logp).sum(axis=1).mean()
        val = ((value.reshape(-1) - V) ** 2).mean()
        return pol + val
    lg = nn.value_and_grad(net, loss_fn)

    try:
        openings = load_openings(args.pgn, 200)
    except Exception as e:
        print(f"[evalfirst] openings load failed ({e}); using standard start", flush=True); openings = None

    from collections import deque
    buf = deque(maxlen=args.buffer)
    hist = []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        player = BatchedMCTSPlayer(net, batch=args.batch_mcts, encoding=cfg.encoding, sims=args.sims,
                                   dirichlet_alpha=0.3, dirichlet_eps=0.25, seed=args.seed + it)
        ng = nmoves = 0
        for g in range(args.games):
            try:
                samples, res = self_play_game(player, py_rng, np_rng,
                                              temp_moves=args.temp_moves, open_plies=args.open_plies)
                buf.extend(samples); ng += 1; nmoves += len(samples)
            except Exception as e:
                print(f"  [warn] game {g} failed: {e}", flush=True)
        train_on_buffer(net, opt, lg, buf, args.epochs, args.batch, py_rng)
        net.save_weights(os.path.join(args.run_dir, "model.npz"))

        line = f"  it{it:>3} games={ng} buf={len(buf):>6} moves/game={nmoves/max(ng,1):.0f} ({time.time()-t0:.0f}s)"
        if it % args.eval_every == 0 or it == 1:
            try:
                elo, margin = eval_open_loop_elo(net, cfg, args.eval_games, openings, args.ladder, args.seed + it)
                hist.append({"iter": it, "buf": len(buf), "open_loop_elo": round(elo), "margin": round(margin)})
                line += f"   OPEN-LOOP Elo={elo:.0f}±{margin:.0f}"
            except Exception as e:
                line += f"   [elo eval failed: {e}]"
            json.dump({"config": vars(args), "history": hist}, open(os.path.join(args.run_dir, "evalfirst.json"), "w"), indent=2)
        print(line, flush=True)

    print(f"[evalfirst] done ({time.time()-t0:.0f}s); curve in {args.run_dir}/evalfirst.json", flush=True)


if __name__ == "__main__":
    main()
