#!/usr/bin/env python
"""The framework in a third modality — sequential CONTROL (a gridworld MDP).

A gridworld is the "Connect-4 of control": known stochastic dynamics, so value iteration gives the exact
optimal value V* (a perfect oracle). We then hand the controller a *degraded* evaluator — V* corrupted by
Gaussian noise of scale σ (a mediocre value function) — and ask the two framework questions:

  * evaluator × search: does h-step lookahead (MPC / closed loop) lift a noisy evaluator's policy over the
    greedy 1-step (open loop) policy? by how much, vs the evaluator's quality?
  * evaluator is the bottleneck: as σ grows (worse evaluator), how far can search compensate before both
    collapse — i.e., is search or evaluator quality the binding constraint?

Return is reported as % of optimal (V*(start)). Pure NumPy; the oracle is exact.

  python control/gridworld.py
"""
import numpy as np

N = 8                     # grid size
GAMMA = 0.95
SLIP = 0.1                # prob the intended move slips to a random neighbour
STEP_COST = -0.02
GOAL_R = 1.0
PIT_R = -1.0
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def build_mdp(seed=0):
    rng = np.random.default_rng(seed)
    goal = (N - 1, N - 1)
    pits = set()
    while len(pits) < N:                                  # scatter a few pits
        c = (int(rng.integers(N)), int(rng.integers(N)))
        if c not in ((0, 0), goal):
            pits.add(c)
    S = [(r, c) for r in range(N) for c in range(N)]
    idx = {s: i for i, s in enumerate(S)}
    nS, nA = len(S), len(ACTIONS)
    P = np.zeros((nS, nA, nS)); R = np.zeros((nS, nA))
    terminal = np.zeros(nS, bool)
    for s in S:
        i = idx[s]
        if s == goal or s in pits:
            terminal[i] = True; P[i, :, i] = 1.0; continue
        for a, (dr, dc) in enumerate(ACTIONS):
            outcomes = {}
            for (pr, (ar, ac)) in [(1 - SLIP, (dr, dc))] + [(SLIP / 4, d) for d in ACTIONS]:
                nr, nc = min(max(s[0] + ar, 0), N - 1), min(max(s[1] + ac, 0), N - 1)
                j = idx[(nr, nc)]
                outcomes[j] = outcomes.get(j, 0.0) + pr
            for j, pr in outcomes.items():
                P[i, a, j] += pr
                sj = S[j]
                R[i, a] += pr * (GOAL_R if sj == goal else PIT_R if sj in pits else STEP_COST)
    return S, idx, P, R, terminal, idx[(0, 0)]


def value_iteration(P, R, terminal, iters=2000):
    nS, nA, _ = P.shape
    V = np.zeros(nS)
    for _ in range(iters):
        Q = R + GAMMA * (P @ V)
        Vn = np.where(terminal, 0.0, Q.max(1))
        if np.max(np.abs(Vn - V)) < 1e-9:
            V = Vn; break
        V = Vn
    return V


def lookahead_policy(V_eval, P, R, terminal, h):
    """h-step lookahead (MPC): apply the Bellman optimality backup h-1 times from V_eval, then act
    greedily. h=1 = greedy 1-step (open loop); larger h = deeper closed-loop search. Efficient (matrix
    ops); returns an action per state."""
    V = V_eval.copy()
    for _ in range(h - 1):
        V = np.where(terminal, 0.0, (R + GAMMA * (P @ V)).max(1))
    return (R + GAMMA * (P @ V)).argmax(1)


def rollout_return(policy_action, P, R, terminal, start, episodes=400, max_steps=80, seed=1):
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(episodes):
        i = start; disc = 1.0; g = 0.0
        for _ in range(max_steps):
            if terminal[i]:
                break
            a = policy_action(i)
            g += disc * R[i, a]; disc *= GAMMA
            i = int(rng.choice(len(terminal), p=P[i, a]))
        tot += g
    return tot / episodes


def main():
    S, idx, P, R, terminal, start = build_mdp()
    Vstar = value_iteration(P, R, terminal)
    opt_pol = lookahead_policy(Vstar, P, R, terminal, 1)
    opt = rollout_return(lambda i: opt_pol[i], P, R, terminal, start)
    rng_r = np.random.default_rng(7)
    rand = rollout_return(lambda i: int(rng_r.integers(len(ACTIONS))), P, R, terminal, start)
    vstd = float(Vstar[~terminal].std())                  # value spread; noise is scaled to it
    def norm(r):                                          # 0% = random policy, 100% = optimal
        return 100.0 * (r - rand) / (opt - rand) if opt > rand else 0.0
    print(f"[gridworld] {N}x{N}, slip={SLIP}; optimal return {opt:.3f}, random {rand:.3f} "
          f"(scores: 0%=random, 100%=optimal; evaluator noise σ in units of value-spread {vstd:.3f})\n")
    print(f"{'evaluator noise σ':>18} | {'open-loop h=1':>14} {'h=2':>8} {'h=3':>8}")
    rng = np.random.default_rng(3)
    rows = []
    for sr in [0.0, 0.25, 0.5, 1.0, 2.0]:
        Vhat = Vstar + sr * vstd * rng.standard_normal(len(Vstar))
        Vhat[terminal] = 0.0
        res = {}
        for h in (1, 2, 3):
            pol = lookahead_policy(Vhat, P, R, terminal, h)
            res[h] = norm(rollout_return(lambda i: pol[i], P, R, terminal, start))
        rows.append({"sigma_rel": sr, **{f"h{h}": round(res[h], 1) for h in (1, 2, 3)}})
        print(f"{sr:>18} | {res[1]:>13.1f}% {res[2]:>7.1f}% {res[3]:>7.1f}%")

    import json
    json.dump({"optimal_return": round(float(opt), 3), "rows": rows}, open("control/gridworld_results.json", "w"), indent=2)
    print("\n[gridworld] wrote control/gridworld_results.json")
    print("Reading: σ=0 (perfect evaluator) → open-loop already optimal, search adds nothing; as σ grows,")
    print("search (h>1) recovers a noisy evaluator — until σ is large enough that even lookahead collapses")
    print("(the evaluator becomes the binding constraint). The evaluator×search decomposition in control.")


if __name__ == "__main__":
    main()
