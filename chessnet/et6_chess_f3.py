#!/usr/bin/env python
"""ET-VI chess F3 (C4-matched) — conversion narrowness on WINNING positions.

F3's real claim, mirrored from Connect-4: on positions the side-to-move is winning (Stockfish V* high),
'narrow' = few value-preserving moves (moves that KEEP the win, by Stockfish), and label bias
|V^pi - V*| should be higher on narrow (only-move) wins where converting demands skill the policy lacks.
This replaces the earlier cp-gap-on-random-positions proxy with the matched metric.

  PYTHONPATH=. ./.venv/bin/python chessnet/et6_chess_f3.py --run runs/conv_value_full
"""
import argparse, json, os, random, sys
import numpy as np
import chess, chess.engine
from chessnet.labeler import score_to_winprob, DEFAULT_STOCKFISH
from chessnet.player import ModelPlayer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from et6_chess_labelbias import load_policy, rollout_winrate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/conv_value_full")
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--win-thresh", type=float, default=0.70)
    ap.add_argument("--keep-thresh", type=float, default=0.40, help="opp winprob below this after our move = win kept")
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/et6_chess_f3.json")
    a = ap.parse_args()
    net, enc = load_policy(a.run)
    player = ModelPlayer(net, encoding=enc, mode="masked", temperature=a.temp, seed=a.seed)
    rng = random.Random(a.seed)
    eng = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH); eng.configure({"Threads": 2, "Hash": 256})
    lim = chess.engine.Limit(depth=a.depth)

    rows = []
    tries = 0
    while len(rows) < a.positions and tries < a.positions * 60:
        tries += 1
        b = chess.Board()
        for _ in range(rng.randint(8, 30)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if b.is_game_over() or not b.legal_moves:
            continue
        vstar = score_to_winprob(eng.analyse(b, lim)["score"], b.turn)
        if vstar < a.win_thresh:
            continue
        # conversion narrowness: how many legal moves keep the win (opponent winprob low after our move)
        nkeep = 0
        for mv in b.legal_moves:
            c = b.copy(); c.push(mv)
            if c.is_game_over():
                nkeep += (c.result() == ("1-0" if b.turn == chess.WHITE else "0-1"))
                continue
            if score_to_winprob(eng.analyse(c, lim)["score"], c.turn) < a.keep_thresh:
                nkeep += 1
        vpi = rollout_winrate(player, b, a.k, a.max_plies, eng, lim)
        rows.append({"v_star": round(vstar, 3), "v_pi": round(vpi, 3), "bias": round(abs(vpi - vstar), 3),
                     "n_winning_moves": nkeep, "n_legal": b.legal_moves.count()})
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}/{a.positions}", flush=True)
    eng.quit()

    nw = np.array([r["n_winning_moves"] for r in rows]); bias = np.array([r["bias"] for r in rows])
    corr = float(np.corrcoef(nw, bias)[0, 1]) if len(rows) > 2 and nw.std() > 0 else float("nan")
    def mb(lo, hi):
        m = (nw >= lo) & (nw <= hi)
        return (round(float(bias[m].mean()), 3), int(m.sum())) if m.sum() else (None, 0)
    summ = {"n": len(rows), "mean_bias": round(float(bias.mean()), 3),
            "only_move(1)": mb(1, 1), "narrow(2-3)": mb(2, 3), "wide(>=4)": mb(4, 99),
            "corr_nwin_bias": round(corr, 3)}
    print(f"\n[chess-F3] winning positions n={len(rows)}, mean bias={summ['mean_bias']}")
    print(f"[chess-F3] bias by conversion narrowness: only-move {summ['only_move(1)']}  "
          f"narrow {summ['narrow(2-3)']}  wide {summ['wide(>=4)']}   corr(n_winning_moves,bias)={summ['corr_nwin_bias']:+.3f}")
    print(f"[chess-F3] {'bias concentrates on narrow wins (F3 SUPPORTED cross-domain)' if summ['corr_nwin_bias']<-0.1 else 'not clearly concentrated'}")
    json.dump({"run": a.run, "summary": summ, "rows": rows}, open(a.out, "w"), indent=2)
    print(f"[chess-F3] wrote {a.out}")


if __name__ == "__main__":
    main()
