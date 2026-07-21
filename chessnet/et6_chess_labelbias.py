#!/usr/bin/env python
"""ET-VI chess arm — label-bias map with a Stockfish oracle (extends E-A to the flagship domain).

The C4 value head is exact; the chess net's value head did not train usefully, so we measure the term
that matters most and needs no value head: the LABEL BIAS itself. V^pi = win-rate under the chess
policy's own self-play from a position; V* = Stockfish win-probability (proxy oracle at fixed depth).
label_bias = |V^pi - V*|. F3-chess: does that bias concentrate on tactically NARROW positions, where
'narrow' = Stockfish's centipawn gap from the best move to the second-best (only-move sharpness)?
Registered (from E-A): self-play labels are biased vs the oracle, and the bias concentrates on the
skill-demanding (narrow) positions.

  PYTHONPATH=. ./.venv/bin/python chessnet/et6_chess_labelbias.py --run runs/conv_value_full
"""
import argparse, json, os, random
import numpy as np
import chess, chess.engine
import mlx.core as mx
from chessnet.model import ModelConfig, PolicyNet
from chessnet.player import ModelPlayer
from chessnet.labeler import score_to_winprob, DEFAULT_STOCKFISH


def load_policy(run):
    cfg_d = json.load(open(os.path.join(run, "config.json")))
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    net = PolicyNet(cfg); net.load_weights(os.path.join(run, "model.npz")); mx.eval(net.parameters())
    return net, cfg_d.get("encoding", "onehot")


def rollout_winrate(player, board0, k, max_plies, eng, limit):
    """Win-rate from board0's side to move over k policy self-play games (Stockfish-adjudicated if capped)."""
    pov = board0.turn; wins = draws = 0
    for _ in range(k):
        b = board0.copy(); plies = 0
        while not b.is_game_over() and plies < max_plies:
            b.push(player.choose(b).move); plies += 1
        if b.is_game_over():
            r = b.result()  # '1-0','0-1','1/2-1/2'
            s = 1.0 if (r == "1-0" and pov == chess.WHITE) or (r == "0-1" and pov == chess.BLACK) else \
                (0.5 if r == "1/2-1/2" else 0.0)
        else:
            wp = score_to_winprob(eng.analyse(b, limit)["score"], pov)  # adjudicate the unfinished game
            s = 1.0 if wp > 0.6 else (0.0 if wp < 0.4 else 0.5)
        wins += s == 1.0; draws += s == 0.5
    return (wins + 0.5 * draws) / k


def narrowness(board, eng, limit):
    """Centipawn gap best - second-best (only-move sharpness). Large = narrow. None if <2 moves."""
    info = eng.analyse(board, limit, multipv=3)
    if len(info) < 2:
        return None
    def cp(i):
        return i["score"].pov(board.turn).score(mate_score=100000)
    return abs(cp(info[0]) - cp(info[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/conv_value_full")
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/et6_chess_labelbias.json")
    a = ap.parse_args()
    net, enc = load_policy(a.run)
    player = ModelPlayer(net, encoding=enc, mode="masked", temperature=a.temp, seed=a.seed)
    rng = random.Random(a.seed)
    eng = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH); eng.configure({"Threads": 2, "Hash": 256})
    limit = chess.engine.Limit(depth=a.depth)

    # sample non-terminal midgame positions by random play
    boards = []
    while len(boards) < a.positions:
        b = chess.Board()
        for _ in range(rng.randint(10, 30)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over() and b.legal_moves:
            boards.append(b)

    rows = []
    for i, b in enumerate(boards):
        vstar = score_to_winprob(eng.analyse(b, limit)["score"], b.turn)
        vpi = rollout_winrate(player, b, a.k, a.max_plies, eng, limit)
        nar = narrowness(b, eng, limit)
        rows.append({"v_star": round(vstar, 3), "v_pi": round(vpi, 3),
                     "bias": round(abs(vpi - vstar), 3), "narrow_cp": nar})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(boards)}  mean_bias={np.mean([r['bias'] for r in rows]):.3f}", flush=True)
    eng.quit()

    bias = np.array([r["bias"] for r in rows])
    nar = np.array([r["narrow_cp"] for r in rows if r["narrow_cp"] is not None], dtype=float)
    bn = np.array([r["bias"] for r in rows if r["narrow_cp"] is not None])
    corr = float(np.corrcoef(nar, bn)[0, 1]) if len(nar) > 2 else float("nan")
    # bins by narrowness: wide (<50cp), mid (50-200), narrow (>200)
    def mbias(lo, hi):
        m = (nar >= lo) & (nar < hi)
        return (round(float(bn[m].mean()), 3), int(m.sum())) if m.sum() else (None, 0)
    summary = {"mean_label_bias": round(float(bias.mean()), 3),
               "corr_narrowness_bias": round(corr, 3),
               "wide_<50cp": mbias(0, 50), "mid_50-200": mbias(50, 200), "narrow_>200cp": mbias(200, 1e9)}
    print(f"\n[chess] mean label bias |V^pi-V*| = {summary['mean_label_bias']}")
    print(f"[chess] bias by narrowness: wide {summary['wide_<50cp']}  mid {summary['mid_50-200']}  "
          f"narrow {summary['narrow_>200cp']}   corr(narrow_cp,bias)={summary['corr_narrowness_bias']:+.3f}")
    print(f"[chess] F3-chess: {'bias concentrates on NARROW positions (supported)' if summary['corr_narrowness_bias']>0.1 else 'not clearly concentrated'}")
    json.dump({"run": a.run, "summary": summary, "rows": rows}, open(a.out, "w"), indent=2)
    print(f"[chess] wrote {a.out}")


if __name__ == "__main__":
    main()
