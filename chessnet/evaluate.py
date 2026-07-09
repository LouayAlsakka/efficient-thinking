"""Evaluation harness (proposal 5): regret metric, ladder matches, Elo.

Three things the scaling report needs, all here:

  1. regret_eval  — held-out move quality in win-probability space (5.3):
        regret = winprob(judge best move) - winprob(model move)
     averaged over positions never seen in training, plus blunder rate
     (moves losing > 10% win probability). Lower is better; a move that loses
     nothing scores perfectly even if it differs from Stockfish's choice.

  2. run_match    — N games vs a fixed-strength Stockfish (UCI_Elo-limited),
     alternating colors from varied openings, returning W/D/L and (in raw mode)
     the illegal-move rate.

  3. estimate_elo — MLE performance rating of the model given results against a
     ladder of known-Elo opponents (self-contained; no external BayesElo/Ordo).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import chess
import chess.engine
import chess.pgn

from .labeler import score_to_winprob, DEFAULT_STOCKFISH
from .player import ModelPlayer


# --- 1. regret / held-out move quality --------------------------------------

@dataclass
class RegretResult:
    n: int
    mean_regret: float          # avg winprob lost vs best move
    blunder_rate: float         # frac of moves losing > 0.10 winprob
    illegal_rate: float         # frac of moves where the model's TOP pick was illegal
    mean_illegal_attempts: float = 0.0  # avg illegal picks walked before a legal one


def regret_eval(player: ModelPlayer, boards, engine_path=DEFAULT_STOCKFISH,
                depth=12, blunder_thresh=0.10) -> RegretResult:
    """Move quality + the RAW illegal-selection counter.

    `illegal_rate` and `mean_illegal_attempts` measure how often the MODEL
    proposes illegal moves BEFORE the reject wrapper fixes it — a pure
    training-health signal that must be tracked separately from the always-legal
    played move (the reject fallback is unrelated to training progress). Only
    meaningful when the player is in 'reject' or 'raw' mode; 'masked' never
    proposes illegal moves so both read 0.
    """
    limit = chess.engine.Limit(depth=depth)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    total_regret = 0.0
    blunders = illegals = total_attempts = n = 0
    try:
        for board in boards:
            if board.is_game_over():
                continue
            pov = board.turn
            best_wp = score_to_winprob(
                engine.analyse(board, limit)["score"], pov)
            choice = player.choose(board)
            illegals += int(choice.was_illegal)
            total_attempts += choice.illegal_attempts
            after = board.copy()
            after.push(choice.move)
            if after.is_game_over():
                # terminal: winprob from mover's POV of the resulting position
                res = after.result()
                move_wp = 1.0 if res == ("1-0" if pov == chess.WHITE else "0-1") \
                    else (0.5 if res == "1/2-1/2" else 0.0)
            else:
                move_wp = score_to_winprob(
                    engine.analyse(after, limit)["score"], pov)
            regret = max(0.0, best_wp - move_wp)
            total_regret += regret
            blunders += int(regret > blunder_thresh)
            n += 1
    finally:
        engine.quit()
    if n == 0:
        return RegretResult(0, 0.0, 0.0, 0.0, 0.0)
    return RegretResult(n, total_regret / n, blunders / n, illegals / n,
                        total_attempts / n)


# --- 2. matches vs a calibrated opponent ladder -----------------------------
#
# Two hard-won lessons from the first sweep:
#   * Stockfish's UCI_Elo has a FLOOR of 1320 (SF silently clamps below it), so a
#     ladder of "1100" rungs was really all 1320 — a model weaker than 1320 then
#     scores ~0 against every rung and its performance rating is unmeasurable
#     noise. Fix: expose a `random`-mover anchor (~300 Elo) so sub-1320 models
#     are bracketed, and clamp/label SF rungs at their true 1320+ strength.
#   * Matches started from RANDOM-legal openings, which are out-of-distribution
#     for a model trained on real games. Fix: start from real Lichess openings.

SF_ELO_FLOOR = 1320                 # Stockfish UCI_Elo minimum (verified)
RANDOM_MOVER_ELO = 300              # rough Lichess-equivalent of random legal play


@dataclass
class MatchResult:
    opponent_elo: int               # the opponent's assumed rating (for the MLE)
    opponent_name: str = ""
    clamped: bool = False           # True if a sub-floor SF Elo was raised to 1320
    wins: int = 0
    draws: int = 0
    losses: int = 0
    illegal_moves: int = 0
    total_moves: int = 0

    @property
    def games(self):
        return self.wins + self.draws + self.losses

    @property
    def score(self):
        return self.wins + 0.5 * self.draws

    @property
    def illegal_rate(self):
        return self.illegal_moves / max(1, self.total_moves)


def load_openings(pgn_path: str, n: int, min_ply=6, max_ply=16,
                  seed=0) -> list[chess.Board]:
    """Sample `n` realistic opening positions from real games (in-distribution)."""
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    with open(pgn_path) as fh:
        while len(boards) < n:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            moves = list(game.mainline_moves())
            if len(moves) < max_ply + 4:
                continue
            board = game.board()
            ply = rng.randint(min_ply, max_ply)
            for mv in moves[:ply]:
                board.push(mv)
            if not board.is_game_over():
                boards.append(board.copy())
    return boards


def _random_opening(rng: random.Random, plies: int) -> chess.Board:
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        if board.is_game_over():
            return chess.Board()
    return board


def play_game(player: ModelPlayer, opponent_move, model_is_white: bool,
              start: chess.Board, max_moves=300):
    """opponent_move: callable(board) -> chess.Move for the non-model side."""
    board = start.copy()
    illegal = moves_played = 0
    while not board.is_game_over(claim_draw=True) and moves_played < max_moves:
        if board.turn == (chess.WHITE if model_is_white else chess.BLACK):
            choice = player.choose(board)
            illegal += int(choice.was_illegal)
            board.push(choice.move)
        else:
            board.push(opponent_move(board))
        moves_played += 1
    return board.result(claim_draw=True), illegal, moves_played


def run_match(player: ModelPlayer, opponent, n_games: int,
              engine_path=DEFAULT_STOCKFISH, movetime=0.05, opening_plies=4,
              openings: list[chess.Board] | None = None, seed=0) -> MatchResult:
    """Play a match vs one opponent.

    `opponent` is a spec dict:
      {"kind": "random"}                       -> random legal mover (~300 Elo)
      {"kind": "sf_elo", "elo": 1500}          -> Stockfish UCI_Elo-limited
    Matches start from `openings` (real positions) when provided, else from a
    short random opening.
    """
    rng = random.Random(seed)
    engine = None
    if opponent["kind"] == "random":
        opp_rng = random.Random(seed + 12345)
        opponent_move = lambda b: opp_rng.choice(list(b.legal_moves))
        res = MatchResult(opponent_elo=RANDOM_MOVER_ELO, opponent_name="random")
    else:  # sf_elo
        want = int(opponent["elo"])
        real = max(SF_ELO_FLOOR, want)
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": real})
        limit = chess.engine.Limit(time=movetime)
        opponent_move = lambda b: engine.play(b, limit).move
        res = MatchResult(opponent_elo=real, opponent_name=f"SF{real}",
                          clamped=(real != want))
    try:
        for g in range(n_games):
            model_white = (g % 2 == 0)
            if openings:
                start = openings[g % len(openings)]
            else:
                start = _random_opening(rng, opening_plies)
            outcome, illegal, moves = play_game(
                player, opponent_move, model_white, start)
            res.illegal_moves += illegal
            res.total_moves += moves
            if outcome == "1/2-1/2":
                res.draws += 1
            elif (outcome == "1-0") == model_white:
                res.wins += 1
            else:
                res.losses += 1
    finally:
        if engine is not None:
            engine.quit()
    return res


# --- 3. Elo estimation (MLE performance rating) -----------------------------

def _expected_score(r_model: float, r_opp: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (r_opp - r_model) / 400.0))


def estimate_elo(results: list[MatchResult]) -> tuple[float, float]:
    """MLE model Elo from results vs known-Elo opponents + a rough ±95% margin.

    Solves for R such that sum of expected scores == actual total score
    (equivalent to the MLE under the logistic/Bradley-Terry model with fixed
    opponents), via bisection. Returns (elo, margin).
    """
    total_games = sum(r.games for r in results)
    total_score = sum(r.score for r in results)
    if total_games == 0:
        return 0.0, 0.0
    # Perfect/zero scores have unbounded MLE; clamp for a finite estimate.
    frac = total_score / total_games
    eps = 0.5 / total_games
    frac = min(max(frac, eps), 1 - eps)
    target = frac * total_games

    lo, hi = -1000.0, 4000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        exp = sum(r.games * _expected_score(mid, r.opponent_elo) for r in results)
        if exp < target:
            lo = mid
        else:
            hi = mid
    elo = (lo + hi) / 2
    # crude margin: ±400/sqrt(n) scaled by score spread (proposal 9: <400 games
    # => ±50+ Elo). Good enough to draw error bars on the scaling curve.
    margin = 400.0 / math.sqrt(max(1, total_games)) * 2.0
    return elo, margin
