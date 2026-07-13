#!/usr/bin/env python
"""Train the Connect-4 net on the depth-8 oracle labels (games/c4_data.npz).

Policy = cross-entropy vs the oracle's best column; value = MSE vs the squashed oracle score. Saves
weights to games/c4_net.safetensors. This is the Stage-1 supervised evaluator; strength (open- vs
closed-loop) is then measured by c4_eval.py.

  PYTHONPATH=games ./.venv/bin/python games/c4_train.py --epochs 40 --channels 64 --blocks 4
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from c4_net import C4Net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="games/c4_data.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="games/c4_net.safetensors")
    args = ap.parse_args()

    d = np.load(args.data)
    X, M, V = d["X"].astype(np.float32), d["M"].astype(np.int32), d["V"].astype(np.float32)
    n = len(X); nval = int(n * args.val_frac)
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    X, M, V = X[perm], M[perm], V[perm]
    Xtr, Mtr, Vtr = X[nval:], M[nval:], V[nval:]
    Xva, Mva, Vva = X[:nval], M[:nval], V[:nval]
    print(f"[train] {len(Xtr)} train / {len(Xva)} val | C={args.channels} blocks={args.blocks}", flush=True)

    net = C4Net(args.channels, args.blocks)
    mx.eval(net.parameters())
    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(net, x, m, v):
        logits, value = net(x)
        pol = nn.losses.cross_entropy(logits, m, reduction="mean")
        val = nn.losses.mse_loss(value, v, reduction="mean")
        return pol + val, (pol, val)

    lg = nn.value_and_grad(net, loss_fn)

    def batches(Xa, Ma, Va, bs, shuffle=True):
        idx = np.random.permutation(len(Xa)) if shuffle else np.arange(len(Xa))
        for i in range(0, len(Xa), bs):
            j = idx[i:i + bs]
            yield mx.array(Xa[j]), mx.array(Ma[j]), mx.array(Va[j])

    def evaluate(Xa, Ma, Va):
        net.eval()
        correct = 0; verr = 0.0; tot = 0
        for x, m, v in batches(Xa, Ma, Va, 512, shuffle=False):
            logits, value = net(x)
            correct += int((mx.argmax(logits, axis=1) == m).sum().item())
            verr += float((mx.abs(value - v)).sum().item()); tot += x.shape[0]
        return 100.0 * correct / tot, verr / tot

    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        tl = 0.0; nb = 0
        for x, m, v in batches(Xtr, Mtr, Vtr, args.batch):
            (l, _), grads = lg(net, x, m, v)
            opt.update(net, grads); mx.eval(net.parameters(), opt.state)
            tl += l.item(); nb += 1
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            pacc, vmae = evaluate(Xva, Mva, Vva)
            print(f"  ep{ep+1:>3} loss={tl/nb:.4f}  val: policy_acc={pacc:.1f}% value_mae={vmae:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    net.save_weights(args.out)
    print(f"[train] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
