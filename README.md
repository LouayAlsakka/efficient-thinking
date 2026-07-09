# Efficient Thinking: Tradeoffs in Game-Playing AI

A three-stage study of **where chess strength comes from** — network capacity, search, and
self-learning — using a deliberately tiny model on modest hardware (two Apple-Silicon machines,
[MLX](https://github.com/ml-explore/mlx)).

**Headline:** a **3.45M-parameter (14 MB) convolutional network** plays at **~2150 Elo** as a
single forward pass, and adding **MCTS search** lifts the *same weights* to **~2800-class strength
with zero extra parameters** — strength bought with *thinking*, not *growing*.

📄 **Full white paper (with graphs): [`docs/whitepaper.pdf`](docs/whitepaper.pdf)** ·
[`docs/whitepaper.html`](docs/whitepaper.html) · [`docs/whitepaper.md`](docs/whitepaper.md)

> **On the numbers:** absolute Elo is measured against a Stockfish ladder and carries systematic
> uncertainty (±~100) near the top rung. The **relative** results below — MCTS-vs-fixed-depth, the
> cascade speed/score curve, and the Stage-3 negatives — are the robust claims; treat the absolute
> "~2800" as an efficiency indicator, not an engine-matching claim.

## The three stages

| Stage | Question | Result |
|---|---|---|
| **1 · Open loop** | how strong is one forward pass? | conv **≫** MLP at equal data; ceiling **~2150**, saturates with capacity/data |
| **2 · Closed loop** | how much does search add? | MCTS **beats & out-scales** fixed-depth → **~2800** at 0 extra params; a wide→narrow **cascade** matches flat MCTS at up to **4.8× less compute** |
| **3 · Self-learning** | can it improve with no teacher/labels? | self-play, a self-referential ladder, **evolution**, and plurality-voting committees **all fail to cross ~2000**; model **agreement is a robust confidence signal**. The wall is the quality of *self-generated signal*, not capacity |

**Unifying finding:** *the learned evaluator is the bottleneck.* Search redistributes the net's
knowledge and self-generated signal cannot exceed its own quality — only a better evaluator (better
labels, more scale, or a better aggregator than plurality voting) raises the ceiling.

## Quickstart

```bash
pip install -r requirements.txt          # mlx, numpy, python-chess
# also install Stockfish for rating/labels:  brew install stockfish

# Play the closed loop (search on the included 3.45M value net) vs a Stockfish ladder:
PYTHONPATH=. python scripts/eval_search.py --run-dir runs/conv_value_llm1 \
    --method mcts --sims 800 --ladder 2400,2700,3000 --games-per-rung 20

# The wide→narrow MCTS cascade (same strength, ~1.6–4.8x faster):
PYTHONPATH=. python scripts/eval_search.py --run-dir runs/conv_value_llm1 \
    --method mstage --mstages 8:150:3.0,3:250:1.5,1:400:0.5 --ladder 2400,2700,3000
```

The best model (`runs/conv_value_llm1/`, conv-96x8 + value head, 3.45M params) is included so search
runs out of the box — no training required.

## Repository layout

```
chessnet/
  model.py       PolicyNet — conv / MLP / dual-path towers + scalar value head
  search.py      closed-loop search: alpha-beta beam-minimax, MCTS/PUCT, quiescence,
                 the wide->narrow MCTS cascade (MultiStageMCTSPlayer)
  committee.py   ensemble inference — soft/hard vote + the agreement (confidence) signal
  train.py       supervised trainer (hard + soft/advantage-weighted objectives, value head)
  encoding.py    board -> 773-float onehot / 8x8x17 conv planes; side-to-move normalization
  evaluate.py    Stockfish-ladder Elo (MLE fit), match harness
  player.py      model-as-player (masked / raw inference)
scripts/
  eval_search.py     raw-policy vs search Elo (methods: ab | mcts | cascade | mstage)
  cascade_sweep.py   N=1..10 cascade level sweep (score & speed vs #levels)
  selfplay.py        Stage-3 self-play expert iteration (AlphaZero-style, no engine)
  selfplay_league.py self-referential Elo ladder (frozen-champion gating)
  evolve.py          neuroevolution: mutate / play / score / select (plateau-escape test)
  committee_test.py  agreement->correctness + consensus-vs-single (Stockfish-scored)
  build_balanced.py  curate balanced opening/midgame/endgame start positions
  train.py           train one config;  make_paper.py renders the white paper
tests/               pytest pipeline test
docs/whitepaper.*    the white paper (md / html / pdf)
```

## Reproducing from scratch (data)

Training uses the public **Lichess cloud-eval database** (~394M deep multi-PV positions).
`scripts/ingest_evals.py` converts it into the compact shard format the trainer reads
(`data/eval.*.npz`), and `scripts/train.py` trains a config. Rating and value/label generation
require a local **Stockfish** binary. The raw dataset is large (~100 GB) and is not included.

## Citation

```bibtex
@misc{alsakka2026efficientthinking,
  title  = {Efficient Thinking: Tradeoffs in Game-Playing AI},
  author = {Louay Alsakka},
  year   = {2026},
  note   = {https://github.com/louayalsakka/efficient-thinking}
}
```

## License
MIT — see [LICENSE](LICENSE).
