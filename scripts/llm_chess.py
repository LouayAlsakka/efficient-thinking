#!/usr/bin/env python
"""How good is a FRONTIER LLM at raw chess? — Kimi-K2.5 (Bedrock) vs a Stockfish ladder.

The efficiency thesis, pushed to its cheeky extreme: our 3.45M-parameter specialist plays ~2150 open-
loop (~2800 with search). Here we measure a *frontier general model's* raw chess Elo the same way (MLE
performance rating over a random-mover + Stockfish-UCI_Elo ladder). Expectation: the giant generalist
plays far below the tiny specialist — capability-per-parameter, made vivid. Kimi picks moves from the
legal list (UCI); illegal/garbled replies fall back to a random legal move (counted, reported).

  ./.venv/bin/python scripts/llm_chess.py --games 8 --ladder 1320 1700 --movetime 0.05
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
import chess, chess.engine
import boto3

SF = "/opt/homebrew/bin/stockfish"
MODEL_ID = "moonshotai.kimi-k2.5"
RANDOM_ELO = 300


def kimi_move(rt, board, illegal_counter):
    legal = [m.uci() for m in board.legal_moves]
    sysp = ("You are a world-class chess engine. Given a position you output the single strongest move "
            "in UCI notation (e.g. e2e4, g8f6, e7e8q). Output ONLY the move token, nothing else.")
    user = f"FEN: {board.fen()}\nLegal moves (UCI): {' '.join(legal)}\nBest move (UCI):"
    body = {"messages": [{"role": "system", "content": sysp}, {"role": "user", "content": user}],
            "max_tokens": 8, "temperature": 0.3}
    try:
        r = json.loads(rt.invoke_model(modelId=MODEL_ID, body=json.dumps(body))["body"].read())
        txt = r["choices"][0]["message"]["content"].strip().split()[0].lower().strip(".,")
        if txt in legal:
            return chess.Move.from_uci(txt)
    except Exception:
        pass
    illegal_counter[0] += 1
    return chess.Move.from_uci(random.choice(legal))


def play_game(rt, kimi_white, sf, sf_elo, movetime, illegal):
    board = chess.Board()
    limit = chess.engine.Limit(time=movetime)
    while not board.is_game_over(claim_draw=True) and board.fullmove_number < 120:
        if (board.turn == chess.WHITE) == kimi_white:
            board.push(kimi_move(rt, board, illegal))
        elif sf is None:                                    # random-mover anchor
            board.push(random.choice(list(board.legal_moves)))
        else:
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": max(1320, sf_elo)})
            board.push(sf.play(board, limit).move)
    res = board.result(claim_draw=True)
    if res == "1/2-1/2":
        return 0.5
    return 1.0 if (res == "1-0") == kimi_white else 0.0


def expected(elo, opp):
    return 1.0 / (1.0 + 10 ** ((opp - elo) / 400.0))


def estimate_elo(results):                                  # results: list of (opp_elo, score_frac, games)
    lo, hi = -200.0, 3200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        exp = sum(g * expected(mid, o) for o, s, g in results)
        act = sum(g * s for o, s, g in results)
        if exp < act: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8, help="games per rung (alternating colors)")
    ap.add_argument("--ladder", type=int, nargs="+", default=[1320, 1700])
    ap.add_argument("--movetime", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/llm_chess.json")
    args = ap.parse_args()
    random.seed(args.seed)
    rt = boto3.client("bedrock-runtime", region_name="us-east-1")
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    illegal = [0]; nmoves = [0]
    print(f"[llm-chess] Kimi-K2.5 vs random + SF{args.ladder} | {args.games} games/rung", flush=True)

    results = []
    # random-mover anchor
    for name, opp_elo, engine in [("random", RANDOM_ELO, None)] + [(f"SF{e}", e, sf) for e in args.ladder]:
        sc = 0.0; t0 = time.time()
        for g in range(args.games):
            sc += play_game(rt, g % 2 == 0, engine, opp_elo, args.movetime, illegal)
        frac = sc / args.games
        results.append((opp_elo, frac, args.games))
        print(f"  vs {name:<7} (Elo {opp_elo}): score {frac*100:4.0f}%   ({time.time()-t0:.0f}s)", flush=True)

    elo = estimate_elo(results)
    sf.quit()
    print(f"\n>>> Kimi-K2.5 raw chess Elo ≈ {elo:.0f}   (illegal/garbled moves: {illegal[0]}, fell back to random)", flush=True)
    print(f">>> for contrast, our 3.45M specialist: ~2150 open-loop, ~2800 with MCTS.", flush=True)
    json.dump({"model": MODEL_ID, "elo": round(elo), "illegal_fallbacks": illegal[0],
               "results": [{"opp": o, "score": s, "games": g} for o, s, g in results]},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
