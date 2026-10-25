# Connect Four

## Interface

The model plays standard 7-column by 6-row Connect Four through a fixed
`2 x 7 x 7` binary interface. Plane zero contains stones belonging to the
player to move, plane one contains opponent stones, and tensor row zero is
always zero padding. The model emits seven column scores. Full columns are
masked mechanically.

Horizontal reflection is exact: evaluating a reflected board reverses the
seven scores. The network stores four canonical output heads and reuses the
same computation for the reflected view.

## Architecture

```text
98 binary inputs
  -> 16 shared length-four line filters over 69 winning windows
  -> 64 global streaming accumulators
  -> 32 mixing units
  -> 4 canonical heads
  -> 7 reflection-equivariant scores
```

All physical hidden activations are binary. Learned weights are integers in
`[-15, 15]`; learned biases are integers in `[-127, 127]`; outputs are bounded
to `[0, 15]`. Immediate wins and mandatory blocks are handled by a fixed local
tactical branch using the same streamed length-four patterns.

Rows, columns, and the two diagonal directions can run in parallel. Assuming
one window every three **redstone** ticks, the longest 24-window direction
alone takes 72 redstone ticks, or 7.2 seconds, to drain. This is not the
network's end-to-end latency: the custom Connect Four model does not yet have a
schematic backend, and its global, mixing, output, tactical, and routing paths
must also be counted. The retained checkpoint's naive sparse serial mapping
has 1,705 comparator stages before routing, so a fast implementation requires
parallel reduction rather than the ordinary serial MCNN accumulator layout.

## Data

The generator labels all seven legal columns with the exact Pascal Pons solver.
Trajectory policies deliberately include optimal, noisy, random, and
anti-optimal games. Suboptimal trajectories change position coverage but never
change labels. Final test generation excludes every training board and its
reflection.

The retained corpus contains 150,000 unique positions from 8,280 games. It
includes 119,968 suboptimal trajectory moves. The independent benchmark has
10,000 additional positions with no exact or reflected overlap.

Install the pinned external solver and opening book:

```bash
uv run python -m scripts.connect4.setup_solver
```

Generate a scalable corpus:

```bash
uv run python -m scripts.connect4.generate_data \
  --output artifacts/connect4/data/positions_150k.npz \
  --positions 150000
```

## Commands

Train a continuous teacher:

```bash
uv run python -m scripts.connect4.train_teacher \
  --dataset artifacts/connect4/data/positions_150k.npz \
  --output-dir artifacts/connect4/teacher
```

Translate and refine the physical model:

```bash
uv run python -m scripts.connect4.train_physical \
  --teacher artifacts/connect4/teacher/model.pt \
  --dataset artifacts/connect4/data/positions_150k.npz \
  --output-dir artifacts/connect4/physical
```

Run the retained model:

```bash
uv run python -m scripts.connect4.play
```

## Retained result

The final physical model is `artifacts/connect4/final/model.pt`; its continuous
teacher is retained beside it for comparison. On the disjoint 10,000-position
benchmark:

| Metric | Teacher | Physical |
| --- | ---: | ---: |
| Outcome preserved | 95.41% | 92.16% |
| Exact optimal move | 82.91% | 78.26% |
| Mean solver regret | 0.65 | 1.04 |
| Opening outcome preserved | 92.61% | 88.77% |
| Middle outcome preserved | 94.62% | 90.86% |
| Late outcome preserved | 98.29% | 96.31% |
| Immediate wins | 100% | 100% |
| Mandatory blocks | 100% | 100% |

The physical model won 198 of 200 games against random play. Against perfect
replies from balanced sampled positions it converted 8 of 20 existing wins,
but held none of 20 draws. This is a playable physical approximation, not a
perfect Connect Four engine. See `artifacts/connect4/final/performance.png` and
`evaluation.json` for the retained benchmark.
