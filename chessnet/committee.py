"""Committee / ensemble inference (see docs/committee-findings.md).

Given K policy(+value) nets with different views, combine them per position:
  - soft vote: average the members' legal-move probability vectors, pick the argmax
  - hard vote: each member picks its top move; take the plurality
plus an `agreement` signal (how many members back the chosen move) — the free,
self-contained confidence meter validated in the committee test (unanimity ⇒ ~65-94%
correct). High agreement -> trust; low agreement -> explore.

EnsemblePlayer exposes .choose(board)->MoveChoice, so it drops into the same match/
ladder harness as ModelPlayer and SearchPlayer.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import chess

from .encoding import ENCODERS, move_to_index
from .player import MoveChoice


class EnsemblePlayer:
    def __init__(self, models, encoding: str = "onehot", combine: str = "mean_prob",
                 seed: int = 0):
        assert combine in ("mean_prob", "vote")
        self.models = list(models)
        self.encode = ENCODERS[encoding]
        self.combine = combine
        self._rng = np.random.default_rng(seed)

    def _member_probs(self, board):
        """Return (legal_moves, [prob_vector per member]) — each vector over legal moves."""
        x = mx.array(self.encode(board)[None, :])
        mirrored = board.turn == chess.BLACK
        legal = list(board.legal_moves)
        idx = [move_to_index(mv, mirrored) for mv in legal]
        out = []
        for m in self.models:
            logits = np.array(m(x)[0])
            z = np.array([logits[i] for i in idx], dtype=np.float64)
            z -= z.max()
            p = np.exp(z); p /= p.sum()
            out.append(p)
        return legal, out

    def _combined(self, board):
        """Return (legal, chosen_index, agreement_fraction)."""
        legal, probs = self._member_probs(board)
        if not legal:
            return legal, None, 0.0
        if self.combine == "mean_prob":
            mean = np.mean(probs, axis=0)
            choice = int(np.argmax(mean))
        else:  # plurality of per-member argmax
            votes = [int(np.argmax(p)) for p in probs]
            counts = np.bincount(votes, minlength=len(legal))
            choice = int(np.argmax(counts))
        # agreement = fraction of members whose OWN top move is the chosen move
        tops = [int(np.argmax(p)) for p in probs]
        agree = float(np.mean([t == choice for t in tops]))
        return legal, choice, agree

    def choose(self, board: chess.Board) -> MoveChoice:
        if board.is_game_over() or not any(board.legal_moves):
            return MoveChoice(None, 0, was_illegal=False)
        legal, choice, _ = self._combined(board)
        if choice is None:
            return MoveChoice(None, 0, was_illegal=False)
        mv = legal[choice]
        return MoveChoice(mv, move_to_index(mv, board.turn == chess.BLACK),
                          was_illegal=False)

    def agreement(self, board: chess.Board) -> float:
        """Confidence meter in [0,1]: fraction of members backing the consensus move."""
        if board.is_game_over() or not any(board.legal_moves):
            return 0.0
        return self._combined(board)[2]
