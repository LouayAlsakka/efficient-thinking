"""Phase-2: reward-weighted policy-gradient training (proposal 4.2 option 2).

Supervised imitation teaches "what move looks engine-like here" but not "how to
convert a won game" — the measured failure (the policy draws even against a
random mover). Phase 2 attacks that directly with REINFORCE + an engine critic:

  * warm-start from a supervised PolicyNet (REINFORCE from scratch over 4096
    actions is hopelessly high-variance);
  * from a real-game position s, sample a legal move a ~ π(·|s);
  * roll the game forward greedily for a short horizon (our side = the policy,
    opponent = a frozen copy of the policy — "self"), then read the resulting
    position's win-probability from Stockfish. That reward credits moves that
    make *progress* toward winning, not just moves that hold a static eval;
  * baseline = the engine's win-prob for s itself (an actor-critic baseline), so
    advantage = reward - value(s) is positive only when the move beats the
    position's intrinsic value;
  * maximize E[advantage · log π(a|s)] with an entropy bonus to avoid collapse.

horizon=0 recovers the literal one-ply eval-delta variant from the proposal.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import chess
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .encoding import ENCODERS, legal_move_mask, decode_move, MOVE_DIM
from .enginepool import EnginePool
from .model import PolicyNet
from .train import load_run, RunConfig


@dataclass
class ReinforceConfig:
    warm_start: str                  # run dir of the supervised model to start from
    run_dir: str = "runs/phase2"
    openings_pgn: str = "data/lichess/2013-01.pgn"
    steps: int = 500
    batch_size: int = 128
    lr: float = 1e-5                 # small — we are fine-tuning a trained policy
    ent_coef: float = 0.01
    horizon: int = 12                # greedy rollout plies before engine bootstrap
    temperature: float = 1.0         # sampling temperature for the explored move
    judge_depth: int = 10
    n_engines: int = 6
    opening_min_ply: int = 8
    opening_max_ply: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 10
    save_every: int = 100

    def encoding_from_model(self):  # filled at load time
        return None


def _masked_logits_np(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.where(mask, logits, -1e30)
    return out


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def pg_loss(model, x, actions, advantages, masks, ent_coef):
    """REINFORCE loss with entropy bonus. actions/advantages/masks are constants."""
    logits = model(x)
    neg = mx.where(masks, logits, mx.full(logits.shape, -1e30))
    logp = neg - mx.logsumexp(neg, axis=1, keepdims=True)      # log-softmax
    chosen = mx.take_along_axis(logp, actions[:, None], axis=1).squeeze(1)
    pg = -mx.mean(advantages * chosen)
    p = mx.exp(logp)
    # entropy over legal moves (masked -inf logp contribute exp(-inf)=0)
    entropy = -mx.mean(mx.sum(mx.where(masks, p * logp, mx.zeros(p.shape)), axis=1))
    return pg - ent_coef * entropy, entropy


def _greedy_move(model, encode, board):
    x = mx.array(encode(board)[None, :])
    logits = np.array(model(x)[0])
    mask = legal_move_mask(board)
    idx = int(np.argmax(np.where(mask, logits, -1e30)))
    mv = decode_move(idx, board)
    return mv


def _terminal_value(board: chess.Board, pov: chess.Color) -> float:
    res = board.result(claim_draw=True)
    if res == "1/2-1/2":
        return 0.5
    win_white = res == "1-0"
    return 1.0 if win_white == (pov == chess.WHITE) else 0.0


class Rollout:
    """Computes the reward for a sampled move via short greedy self-play rollout."""

    def __init__(self, model, frozen, encode, horizon):
        self.model = model
        self.frozen = frozen       # opponent = frozen copy of the policy
        self.encode = encode
        self.horizon = horizon

    def leaf(self, board: chess.Board, mover: chess.Color):
        """Play greedily for `horizon` plies; return (leaf_board, terminal?)."""
        b = board.copy()
        for _ in range(self.horizon):
            if b.is_game_over(claim_draw=True):
                return b, True
            net = self.model if b.turn == mover else self.frozen
            mv = _greedy_move(net, self.encode, b)
            if mv is None:
                return b, True
            b.push(mv)
        return b, b.is_game_over(claim_draw=True)


def load_openings_for_rl(pgn_path, n, min_ply, max_ply, seed):
    import random
    import chess.pgn
    rng = random.Random(seed)
    boards = []
    with open(pgn_path) as fh:
        while len(boards) < n:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            moves = list(game.mainline_moves())
            if len(moves) < max_ply + 4:
                continue
            b = game.board()
            ply = rng.randint(min_ply, max_ply)
            for mv in moves[:ply]:
                b.push(mv)
            if not b.is_game_over():
                boards.append(b.copy())
    return boards


def train_reinforce(cfg: ReinforceConfig):
    os.makedirs(cfg.run_dir, exist_ok=True)
    mx.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model, mcfg = load_run(cfg.warm_start)
    frozen, _ = load_run(cfg.warm_start)       # fixed opponent for rollouts
    encoding = mcfg.encoding
    encode = ENCODERS[encoding]
    # RunConfig-shaped config so the standard eval harness (load_run) works on
    # the phase-2 checkpoint; RL hyperparameters go in a separate metadata file.
    run_cfg = RunConfig(encoding=encoding, depth=mcfg.depth, width=mcfg.width,
                        activation=mcfg.activation, residual=mcfg.residual,
                        run_dir=cfg.run_dir)
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump(asdict(run_cfg), f, indent=2)
    with open(os.path.join(cfg.run_dir, "phase2_meta.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    opt = optim.Adam(learning_rate=cfg.lr)
    loss_and_grad = nn.value_and_grad(model, pg_loss)
    rollout = Rollout(model, frozen, encode, cfg.horizon)

    print(f"[phase2] warm-start {cfg.warm_start} (d{mcfg.depth} w{mcfg.width} "
          f"{encoding}) | horizon={cfg.horizon} engines={cfg.n_engines}")
    boards = load_openings_for_rl(cfg.openings_pgn, max(2000, cfg.batch_size * 4),
                                  cfg.opening_min_ply, cfg.opening_max_ply, cfg.seed)
    print(f"[phase2] {len(boards)} real-game start positions")

    history = []
    t0 = time.time()
    with EnginePool(cfg.n_engines, depth=cfg.judge_depth) as pool:
        for step in range(cfg.steps):
            idx = rng.integers(0, len(boards), size=cfg.batch_size)
            batch = [boards[i] for i in idx]
            povs = [b.turn for b in batch]

            # 1. forward (no grad) -> sample an exploratory legal move per board
            X = np.stack([encode(b) for b in batch]).astype(np.float32)
            logits = np.array(model(mx.array(X)))
            masks = np.stack([legal_move_mask(b) for b in batch])
            probs = _softmax_np(_masked_logits_np(logits / cfg.temperature, masks))
            actions = np.array([rng.choice(MOVE_DIM, p=probs[i])
                                for i in range(len(batch))])

            # 2. reward = winprob at the rollout leaf; baseline = winprob(s)
            leaves, leaf_povs = [], []
            base_items = list(zip(batch, povs))
            for i, b in enumerate(batch):
                mv = decode_move(int(actions[i]), b)
                s1 = b.copy()
                s1.push(mv) if mv else None
                if mv is None or s1.is_game_over(claim_draw=True):
                    leaves.append(("terminal",
                                   _terminal_value(s1, povs[i]) if mv else 0.0))
                    continue
                leaf, term = rollout.leaf(s1, povs[i])
                if term:
                    leaves.append(("terminal", _terminal_value(leaf, povs[i])))
                else:
                    leaves.append(("eval", leaf))
                    leaf_povs.append((leaf, povs[i]))
            # engine calls: baselines for all, bootstrap for non-terminal leaves
            baselines = pool.winprobs(base_items)
            leaf_wps = pool.winprobs(leaf_povs) if leaf_povs else []
            rewards, k = np.zeros(len(batch)), 0
            for i, (kind, val) in enumerate(leaves):
                if kind == "terminal":
                    rewards[i] = val
                else:
                    rewards[i] = leaf_wps[k]; k += 1
            advantages = rewards - np.array(baselines)

            # 3. policy-gradient update
            (loss, entropy), grads = loss_and_grad(
                model, mx.array(X), mx.array(actions), mx.array(advantages.astype(np.float32)),
                mx.array(masks), cfg.ent_coef)
            grads = optim.clip_grad_norm(grads, cfg.grad_clip)[0]
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)

            rec = {"step": step, "reward": float(rewards.mean()),
                   "advantage": float(advantages.mean()),
                   "entropy": float(entropy.item()), "loss": float(loss.item())}
            history.append(rec)
            if step % cfg.log_every == 0:
                rate = (step + 1) * cfg.batch_size / (time.time() - t0)
                print(f"  step {step} reward {rec['reward']:.3f} "
                      f"adv {rec['advantage']:+.3f} ent {rec['entropy']:.2f} "
                      f"({rate:.0f} samp/s)")
            if step and step % cfg.save_every == 0:
                model.save_weights(os.path.join(cfg.run_dir, "model.npz"))

    model.save_weights(os.path.join(cfg.run_dir, "model.npz"))
    with open(os.path.join(cfg.run_dir, "metrics.json"), "w") as f:
        json.dump({"history": history, "wall_sec": time.time() - t0}, f, indent=2)
    print(f"[phase2] done in {time.time()-t0:.1f}s -> {cfg.run_dir}")
    return model, history
