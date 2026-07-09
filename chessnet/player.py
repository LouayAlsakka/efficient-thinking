"""Turn a trained PolicyNet into a move-picking player (proposal 4.3).

Two inference modes, reported separately:
  * raw    : argmax/sample over all 4096 logits. If the picked index is not a
             legal move the move is illegal -> counted, and we fall back to a
             random legal move so the game can continue (fallback stated
             explicitly, per proposal). Illegal-move rate is a tracked metric.
  * masked : restrict the softmax to legal (from,to) indices before choosing.
             Gives the "pure policy strength" number.

Temperature 0 -> argmax; temperature > 0 -> sample from the (optionally masked)
distribution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess
import mlx.core as mx
import numpy as np

from .encoding import ENCODERS, legal_move_mask, decode_move, index_to_from_to
from .model import PolicyNet


@dataclass
class MoveChoice:
    move: chess.Move          # a legal move to play
    raw_index: int            # the index the net actually picked
    was_illegal: bool         # True if the net's FIRST pick was illegal
    illegal_attempts: int = 0  # reject mode: how many illegal picks before a legal one


class ModelPlayer:
    """Three inference modes (proposal 4.3 + the deployed 'reject' design):

      masked : zero out illegal logits BEFORE choosing. Needs the full legal set
               (move generation in the hot path). Kept for the "pure policy
               strength" measurement only.
      reject : DEPLOYED path. Keep the model a pure single-pass predictor — no
               masking. Take its ranked output and play the top move; if it is
               illegal, fall through to the next-ranked move, and so on. Cost is
               ~zero when the top move is legal (the common case as legality is
               learned). `illegal_attempts` tracks the retry cost.
      raw    : play the single top pick; if illegal it's a loss/fallback. Used to
               report the raw illegal-move rate as a scaling metric.
    """

    def __init__(self, model: PolicyNet, encoding: str = "onehot",
                 mode: str = "masked", temperature: float = 0.0,
                 seed: int = 0):
        assert mode in ("raw", "masked", "reject")
        self.model = model
        self.encode = ENCODERS[encoding]
        self.mode = mode
        self.temperature = temperature
        self.rng = random.Random(seed)
        self._np = np.random.default_rng(seed)

    def _logits(self, board: chess.Board) -> np.ndarray:
        x = mx.array(self.encode(board)[None, :])
        out = self.model(x)
        mx.eval(out)
        return np.array(out[0])

    def _pick_index(self, logits: np.ndarray) -> int:
        if self.temperature <= 0.0:
            return int(np.argmax(logits))
        z = logits / self.temperature
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(self.rng.choices(range(len(p)), weights=p, k=1)[0])

    def _ranked_order(self, logits: np.ndarray) -> np.ndarray:
        """Descending order of move indices. Temperature>0 uses Gumbel noise to
        turn the ranking into a sample-without-replacement order (so rejection
        walks a properly sampled sequence, not just next-best-logit)."""
        if self.temperature <= 0.0:
            return np.argsort(-logits)
        g = -np.log(-np.log(self._np.random.random(logits.shape) + 1e-12) + 1e-12)
        return np.argsort(-(logits / self.temperature + g))

    def choose(self, board: chess.Board) -> MoveChoice:
        logits = self._logits(board)

        if self.mode == "masked":
            mask = legal_move_mask(board)
            logits = np.where(mask, logits, -1e30)
            idx = self._pick_index(logits)
            move = decode_move(idx, board)
            if move is None:  # only if board had no legal moves
                move = self.rng.choice(list(board.legal_moves))
            return MoveChoice(move, idx, was_illegal=False)

        if self.mode == "reject":
            # Walk the model's own ranking; play the first legal move. No mask is
            # ever built — we only validate the moves we actually try (usually 1).
            attempts = 0
            for idx in self._ranked_order(logits):
                move = decode_move(int(idx), board)
                if move is not None:
                    return MoveChoice(move, int(idx),
                                      was_illegal=(attempts > 0),
                                      illegal_attempts=attempts)
                attempts += 1
            move = self.rng.choice(list(board.legal_moves))  # unreachable if legal moves exist
            return MoveChoice(move, -1, was_illegal=True, illegal_attempts=attempts)

        # raw mode
        idx = self._pick_index(logits)
        move = decode_move(idx, board)
        if move is None:
            fallback = self.rng.choice(list(board.legal_moves))
            return MoveChoice(fallback, idx, was_illegal=True, illegal_attempts=1)
        return MoveChoice(move, idx, was_illegal=False)
