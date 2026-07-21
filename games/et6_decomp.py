#!/usr/bin/env python
"""ET-VI E-A — label-fidelity decomposition (F1 mechanism + F2 irreducibility).

For a trained Connect-4 net, on solved positions (V* = exact negamax), decompose total error into:
    total |V_net - V*|  =  fit |V_net - V^pi|  (+)  label-bias |V^pi - V*|
where V^pi = win-rate over k games rolled out from the position under the net's OWN policy (sampled
from the policy head — its current level of play). F1 predicts the plateau floor is label bias, not fit
error: the net has learned its own labels, and the residual lives in the labels. F2 predicts label bias
is flat in rollout count k while its variance shrinks ~1/sqrt(k) — more self-play sharpens the wrong
number.

  PYTHONPATH=games ./.venv/bin/python games/et6_decomp.py --net games/results/et6_ea_scratch.safetensors
"""
import argparse, json, random
import numpy as np
import mlx.core as mx
from connect4 import C4, value01 as true_value01
from c4_net import C4Net, masked_policy
from c4_mcts import search, visit_probs


def build_positions(n, rng):
    """n decisive solved positions (kept as C4 objects so we can roll out), balanced win/loss."""
    out, want = [], n // 2
    nw = nl = 0
    while len(out) < n:
        s = C4(); depth = rng.randint(14, 30); ok = True
        for _ in range(depth):
            if s.terminal() is not None or not s.legal():
                ok = False; break
            s = s.play(rng.choice(s.legal()))
        if not ok or s.terminal() is not None:
            continue
        v = true_value01(s)
        if v == 1.0 and nw < want:
            out.append((s, v)); nw += 1
        elif v == 0.0 and nl < want:
            out.append((s, v)); nl += 1
    return out


def net_value(net, s):
    _, value = net(mx.array([s.encode()]))
    return float(value[0])


def net_probs(net, s):
    logits, _ = net(mx.array([s.encode()]))
    return np.array(masked_policy(logits[0], s.legal()).tolist())


def rollout_value(net, s0, k, rng, mcts_sims=0, npr=None):
    """Win-rate from s0's side-to-move over k games. mcts_sims=0 -> sample the net's policy head;
    mcts_sims>0 -> play by net+MCTS (matching the policy that generated the training labels)."""
    wins = draws = 0
    for _ in range(k):
        s = s0; parity = 0
        while s.terminal() is None and s.legal():
            if mcts_sims:
                root = search(net, s, sims=mcts_sims, dirichlet=0.0, rng=npr)
                vp = visit_probs(root)
                move = int(rng.choice(7, p=vp)) if vp.sum() > 0 else s.legal()[0]
            else:
                p = net_probs(net, s)
                move = int(rng.choice(7, p=p / p.sum())) if p.sum() > 0 else s.legal()[0]
            if not s.can_play(move):
                move = s.legal()[0]
            s = s.play(move); parity ^= 1
        t = s.terminal()
        if t == "win":
            if parity == 1:                 # s0's side made the last, winning move
                wins += 1
        elif t is not None:
            draws += 1
    return (wins + 0.5 * draws) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="games/results/et6_ea_scratch.safetensors")
    ap.add_argument("--positions", type=int, default=150)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--mcts-sims", type=int, default=0, help=">0: roll out V^pi under net+MCTS (label-generating policy)")
    ap.add_argument("--channels", type=int, default=64); ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--skip-f2", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="games/results/et6_decomp.json")
    a = ap.parse_args()
    net = C4Net(a.channels, a.blocks); net.load_weights(a.net); net.eval()
    mx.eval(net.parameters())
    rng = random.Random(a.seed); nprng = np.random.default_rng(a.seed)
    pos = build_positions(a.positions, rng)
    print(f"[et6] {len(pos)} solved positions; V^pi over k={a.k} rollouts under the net's policy")
    fit, bias, total, rows = [], [], [], []
    for i, (s, vstar) in enumerate(pos):
        vnet = net_value(net, s)
        vpi = rollout_value(net, s, a.k, nprng, mcts_sims=a.mcts_sims, npr=nprng)
        fit.append(abs(vnet - vpi)); bias.append(abs(vpi - vstar)); total.append(abs(vnet - vstar))
        rows.append({"v_net": round(vnet, 3), "v_pi": round(vpi, 3), "v_star": vstar})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pos)}  fit={np.mean(fit):.3f} bias={np.mean(bias):.3f} total={np.mean(total):.3f}", flush=True)
    F = dict(fit=round(float(np.mean(fit)), 3), label_bias=round(float(np.mean(bias)), 3),
             total=round(float(np.mean(total)), 3))
    print(f"\n[F1] fit |V_net-V^pi| = {F['fit']}   label-bias |V^pi-V*| = {F['label_bias']}   "
          f"total |V_net-V*| = {F['total']}")
    print(f"[F1] plateau floor is {'LABEL BIAS' if F['label_bias'] > F['fit'] else 'FIT ERROR'} "
          f"(bias {'>' if F['label_bias'] > F['fit'] else '<='} fit) — "
          f"{'supports F1: net learned its labels, residual is in the labels' if F['label_bias'] > F['fit'] else 'against F1'}")

    f2 = {}
    if a.skip_f2:
        json.dump({"F1": F, "F2_bias_vs_k": f2, "n_positions": len(pos), "k": a.k,
                   "mcts_sims": a.mcts_sims, "rows": rows}, open(a.out, "w"), indent=2)
        print(f"\n[et6] wrote {a.out} (F1 only, mcts_sims={a.mcts_sims})"); return
    # F2: label bias vs k on a subset
    print("\n[F2] label-bias vs rollout count k (mean flat, variance ~1/sqrt(k)):")
    sub = pos[:50]; f2 = {}
    for k in (10, 100, 1000):
        biases = [abs(rollout_value(net, s, k, nprng, mcts_sims=a.mcts_sims, npr=nprng) - vstar) for s, vstar in sub]
        f2[k] = {"mean": round(float(np.mean(biases)), 3), "std": round(float(np.std(biases)), 3)}
        print(f"  k={k:>4}  mean|V^pi-V*|={f2[k]['mean']:.3f}  std={f2[k]['std']:.3f}")

    json.dump({"F1": F, "F2_bias_vs_k": f2, "n_positions": len(pos), "k": a.k, "rows": rows},
              open(a.out, "w"), indent=2)
    print(f"\n[et6] wrote {a.out}")


if __name__ == "__main__":
    main()
