# Connect-4 calibrated rating scale

Reference ladder, Bradley-Terry fit, anchored random := 0 Elo. Logistic goodness-of-fit (mean |pred-obs| win-rate) = **0.058** (lower = the Elo/logistic model fits this game well).

| rung | Elo | meaning |
|---|---:|---|
| random | +0 | floor (no lookahead) |
| depth-1 | +804 | 1-ply tactical lookahead |
| depth-2 | +954 | 2-ply tactical lookahead |
| depth-3 | +963 | 3-ply tactical lookahead |
| depth-4 | +1017 | 4-ply tactical lookahead |
| depth-6 | +1050 | 6-ply tactical lookahead |
| depth-5 | +1070 | 5-ply tactical lookahead |

## Our agents, placed on the calibrated scale
| agent | Elo | vs-ladder scores |
|---|---:|---|
| open-loop (raw net) | +644 | random:98% depth-1:28% depth-2:12% depth-3:6% depth-4:25% depth-5:5% depth-6:8% |
| closed-loop (MCTS 200) | +880 | random:98% depth-1:55% depth-2:34% depth-3:41% depth-4:30% depth-5:31% depth-6:32% |
