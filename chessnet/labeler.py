"""Stockfish labeling harness (proposal 4.1).

For each position we ask Stockfish, at a *fixed* search budget (depth or nodes,
held constant across the whole study — proposal 9 "label depth confound"), for:
  * the best move (the supervised cross-entropy target), and
  * optionally the eval of every legal move, so we can build a softened target
    distribution and, later, compute the regret metric (proposal 5.3).

Evals are stored in win-probability space via the standard logistic mapping of
centipawns; mate scores map to ~0/1. This is the space the proposal wants for
the regret metric (centipawns are nonlinear near decided positions).

A single engine process is reused across many positions (UCI is stateful but
`analyse` sets the position each call). Parallelism is achieved by running
several `StockfishLabeler` instances in separate processes (see scripts/label.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import chess.engine

DEFAULT_STOCKFISH = "/opt/homebrew/bin/stockfish"


def cp_to_winprob(cp: float) -> float:
    """Centipawns (from side-to-move POV) -> win probability in [0,1].

    Uses the logistic used by Lichess/Stockfish WDL rescaling (~1/400 slope).
    """
    return 1.0 / (1.0 + math.pow(10.0, -cp / 400.0))


def score_to_winprob(score: chess.engine.PovScore, pov: chess.Color) -> float:
    s = score.pov(pov)
    if s.is_mate():
        return 1.0 if s.mate() > 0 else 0.0
    return cp_to_winprob(s.score())


@dataclass
class LabelBudget:
    """Fixed search budget. Set exactly one of depth/nodes; depth is default."""
    depth: int | None = 12
    nodes: int | None = None
    multipv: int = 1   # >1 to get per-move evals for softened / regret targets

    def limit(self) -> chess.engine.Limit:
        if self.nodes is not None:
            return chess.engine.Limit(nodes=self.nodes)
        return chess.engine.Limit(depth=self.depth)


@dataclass
class PositionLabel:
    fen: str
    best_uci: str
    best_winprob: float                 # win prob for side-to-move after best move
    move_winprobs: dict[str, float]     # uci -> win prob (only if multipv>1)


class StockfishLabeler:
    def __init__(self, path: str = DEFAULT_STOCKFISH, threads: int = 1,
                 hash_mb: int = 256, budget: LabelBudget | None = None):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        self.budget = budget or LabelBudget()

    def label(self, board: chess.Board) -> PositionLabel:
        pov = board.turn
        limit = self.budget.limit()
        if self.budget.multipv > 1:
            infos = self.engine.analyse(board, limit, multipv=self.budget.multipv)
            move_wp: dict[str, float] = {}
            best_uci, best_wp = None, -1.0
            for info in infos:
                pv = info.get("pv")
                if not pv:
                    continue
                uci = pv[0].uci()
                wp = score_to_winprob(info["score"], pov)
                move_wp[uci] = wp
                if best_uci is None:  # multipv results are ordered best-first
                    best_uci, best_wp = uci, wp
            return PositionLabel(board.fen(), best_uci, best_wp, move_wp)
        info = self.engine.analyse(board, limit)
        best = info["pv"][0]
        wp = score_to_winprob(info["score"], pov)
        return PositionLabel(board.fen(), best.uci(), wp, {best.uci(): wp})

    def close(self):
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
