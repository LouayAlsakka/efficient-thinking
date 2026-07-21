#!/usr/bin/env python
"""ET-VI F4 (chess) — 'what's special about ~2000': does the supervised net's ceiling back-predict
from its label-source fidelity, and is it label-bound or capacity-bound?

Chess has an Elo scale that solved Connect-4 lacks, so F4's quantitative form lives here. On midgame
positions, with one Stockfish at two depths (shallow = the training-label depth, deep = the oracle):
  net_regret    = winprob lost by the NET's move vs the deep-SF best move  (the net's play-fidelity)
  label_regret  = winprob lost by SHALLOW-SF's best move vs the deep-SF best move  (the label ceiling
                  the net was trained toward — the fidelity of its label SOURCE)
  narrow_shift  = |shallow-SF eval - deep-SF eval|  (tactical depth-sensitivity = F3's metric (b))
Reading: if net_regret >> label_regret, the ~2000 ceiling is CAPACITY/fit-bound (the 3.45M net cannot
reach even its label's fidelity — consistent with E-A's fit-bound supervised net); if net_regret ~=
label_regret, it is LABEL-bound (it reached its label source and only a better source raises it). Either
way, 'what's special about ~2000' is named: the regret floor the net actually sits at, and which term
caps it. Nothing self-graded; the oracle is Stockfish.

  PYTHONPATH=. ./.venv/bin/python chessnet/et6_f4.py --run runs/conv_value_full
"""
import argparse, json, os, random
import numpy as np
import chess, chess.engine
import mlx.core as mx
from chessnet.model import ModelConfig, PolicyNet
from chessnet.player import ModelPlayer
from chessnet.labeler import score_to_winprob, DEFAULT_STOCKFISH


def load_player(run):
    cfg_d = json.load(open(os.path.join(run, "config.json")))
    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})
    net = PolicyNet(cfg); net.load_weights(os.path.join(run, "model.npz")); mx.eval(net.parameters())
    return ModelPlayer(net, encoding=cfg_d.get("encoding", "onehot"), mode="masked", temperature=0.0)


def move_wp(board, mv, eng, lim, pov):
    a = board.copy(); a.push(mv)
    if a.is_game_over():
        r = a.result()
        return 1.0 if r == ("1-0" if pov == chess.WHITE else "0-1") else (0.5 if r == "1/2-1/2" else 0.0)
    return score_to_winprob(eng.analyse(a, lim)["score"], pov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/conv_value_full")
    ap.add_argument("--positions", type=int, default=120)
    ap.add_argument("--shallow", type=int, default=6, help="training-label Stockfish depth")
    ap.add_argument("--deep", type=int, default=20, help="oracle Stockfish depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/et6_f4.json")
    a = ap.parse_args()
    player = load_player(a.run)
    rng = random.Random(a.seed)
    eng = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH); eng.configure({"Threads": 2, "Hash": 256})
    ls, ld = chess.engine.Limit(depth=a.shallow), chess.engine.Limit(depth=a.deep)

    boards = []
    while len(boards) < a.positions:
        b = chess.Board()
        for _ in range(rng.randint(8, 30)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over() and b.legal_moves:
            boards.append(b)

    rows = []
    for i, b in enumerate(boards):
        pov = b.turn
        deep = eng.analyse(b, ld, multipv=2)
        best_deep_wp = score_to_winprob(deep[0]["score"], pov)
        best_deep_mv = deep[0]["pv"][0]
        # net move regret vs deep oracle
        net_mv = player.choose(b).move
        net_reg = max(0.0, best_deep_wp - move_wp(b, net_mv, eng, ld, pov))
        # label-source (shallow SF) best move, its regret under the deep oracle
        shallow = eng.analyse(b, ls)
        shal_mv = shallow["pv"][0]
        lab_reg = max(0.0, best_deep_wp - move_wp(b, shal_mv, eng, ld, pov))
        # two-depth narrowness / tactical depth-sensitivity
        shal_wp = score_to_winprob(shallow["score"], pov)
        narrow_shift = abs(shal_wp - best_deep_wp)
        rows.append({"net_regret": round(net_reg, 3), "label_regret": round(lab_reg, 3),
                     "narrow_shift": round(narrow_shift, 3)})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(boards)}", flush=True)
    eng.quit()

    nr = np.mean([r["net_regret"] for r in rows]); lr = np.mean([r["label_regret"] for r in rows])
    ns = np.mean([r["narrow_shift"] for r in rows])
    bound = "CAPACITY/fit-bound" if nr > lr + 0.02 else ("LABEL-bound" if abs(nr - lr) <= 0.02 else "below-label(?)")
    summ = {"n": len(rows), "net_regret": round(float(nr), 3), "label_regret_shallow_source": round(float(lr), 3),
            "mean_narrow_shift": round(float(ns), 3), "verdict": bound,
            "shallow_depth": a.shallow, "deep_depth": a.deep}
    print(f"\n[F4-chess] net regret vs deep-SF = {summ['net_regret']}  |  label-source (d{a.shallow}) regret = "
          f"{summ['label_regret_shallow_source']}")
    print(f"[F4-chess] the ~2000 ceiling is {bound}: "
          f"{'the net cannot reach even its label source fidelity (fit/capacity caps it)' if 'CAPACITY' in bound else 'the net reached its label source; only a better source raises it'}")
    print(f"[F4-chess] mean two-depth narrow shift (tactical depth-sensitivity) = {summ['mean_narrow_shift']}")
    json.dump({"run": a.run, "summary": summ, "rows": rows}, open(a.out, "w"), indent=2)
    print(f"[F4-chess] wrote {a.out}")


if __name__ == "__main__":
    main()
