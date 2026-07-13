# Connect-4 evaluator×search decomposition (Arm A, game #2)

Net: small conv (C64, 4 residual blocks, BatchNorm), policy+value, trained on 12k depth-8-oracle
labels (54% top-1 vs oracle, value MAE 0.136). Closed loop = PUCT MCTS, 200 sims. Opponents = ab_best
depth ladder. 30 games/opponent, alternating colors, 2 random opening plies.

## Strength vs depth ladder (score %, Elo gap)
| opponent | open-loop (raw net) | closed-loop (MCTS 200) |
|----------|--------------------:|-----------------------:|
| depth-1  | 17%  (-280)         | 47%  (-23)             |
| depth-2  | 10%  (-382)         | 23%  (-207)            |
| depth-3  | 23%  (-207)         | 38%  (-83)             |
| depth-4  | 22%  (-223)         | 38%  (-83)             |
| depth-5  | 12%  (-352)         | 25%  (-191)            |

## Search lift (head-to-head, same evaluator)
closed-loop beats open-loop 83% (W23 D4 L3) -> +280 Elo from search.

## Reading
- Search adds +280 Elo head-to-head — strikingly close to chess's +286.
- But the LEVER BALANCE differs: the raw net loses even to depth-1 (1-ply tactical) play, so search
  carries proportionally MORE of the load here than in chess. Much of the lift is tactical (immediate
  wins/blocks the weak net misses) — a clean demonstration of "search extracts what the evaluator
  failed to represent."
- CAVEAT: this evaluator is undertrained (12k labels, 54% top-1) vs the chess one (millions). So this is
  "modest evaluator + search." Data-scaling run (50k+) launched to trace the open-loop ceiling and
  separate "simpler game -> search dominates" from "smaller evaluator -> search dominates."
