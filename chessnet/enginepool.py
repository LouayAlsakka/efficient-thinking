"""A small pool of persistent Stockfish engines for parallel position judging.

Phase-2 training needs a win-probability estimate for many positions per step
(the critic baseline and the rollout bootstrap). Each engine is a separate
process, so we borrow one per thread from a queue and fan the batch out with a
thread pool — the GIL is released during the blocking UCI round-trip.
"""

from __future__ import annotations

import queue
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

from .labeler import score_to_winprob, DEFAULT_STOCKFISH


class EnginePool:
    def __init__(self, n_engines: int = 4, depth: int = 10,
                 path: str = DEFAULT_STOCKFISH, hash_mb: int = 128):
        self.limit = chess.engine.Limit(depth=depth)
        self._engines = queue.Queue()
        self._all = []
        for _ in range(n_engines):
            eng = chess.engine.SimpleEngine.popen_uci(path)
            eng.configure({"Threads": 1, "Hash": hash_mb})
            self._all.append(eng)
            self._engines.put(eng)
        self._pool = ThreadPoolExecutor(max_workers=n_engines)

    def _winprob(self, board: chess.Board, pov: chess.Color) -> float:
        eng = self._engines.get()
        try:
            info = eng.analyse(board, self.limit)
            return score_to_winprob(info["score"], pov)
        finally:
            self._engines.put(eng)

    def winprobs(self, items) -> list[float]:
        """items: iterable of (board, pov). Returns win probs in order."""
        items = list(items)
        return list(self._pool.map(lambda it: self._winprob(it[0], it[1]), items))

    def close(self):
        self._pool.shutdown(wait=True)
        for eng in self._all:
            eng.quit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
