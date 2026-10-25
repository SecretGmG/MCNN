# 7x7 Go

The Go pipeline deliberately has one model and one representation. It evaluates
legal successor boards using 98 binary inputs: one `7x7` plane for the player
considering the move and one for the opponent. The controller adds captures,
filters illegal moves, and treats pass as an ordinary candidate that wins exact
ties.

```text
98 board bits, padded to 104
  -> 128 -> 128 -> 128 -> 128 -> 8
  -> sum(outputs 0:4) - sum(outputs 4:8)
```

Every layer is the physical streamed Minecraft layer described in
[`minecraft_network.md`](minecraft_network.md): eight-neuron cells, cyclic
one-directional diagonal strips, ternary weights, and exactly 15 threshold-count
steps. The retained model uses shift depths `12, 4, 4, 4, 4`; its first layer
uses all 104 input diagonals and later layers use the first 40 diagonals in the
physical shifter traversal. The unshifted diagonal is last in that traversal.

The dataset target excludes accumulated prisoner totals. Immediate captures are
added mechanically at inference, so the network learns positional value rather
than reconstructing game history. Training splits complete source games, applies
random D4 augmentation, and anneals from continuous weights and activations to
the exact ternary and integer-threshold circuit.

The complete retained run is under `artifacts/go7x7/`:

- `data.npz`: normalized training data
- `model.pt`: exact projected model
- `config.json`: physical architecture
- `training.json` and `history.json`: metrics and training history
- `training.png`: concise optimization plot

The short cleanup run used 28,087 examples and selected epoch 18. On its
game-disjoint validation split it reached 6.97 points mean move regret, 6.21
points centered RMSE, and 79.2% ordering accuracy for move pairs separated by
at least three points. Train and validation mean regret were 6.84 and 6.97
points respectively, providing no sign of conventional train-set overfitting
in this run. This is a validated physical baseline, not yet a strong Go player.

For the retained shift depths `12, 4, 4, 4, 4`, the diagonal widths are
`104, 40, 40, 40, 40`. The timing formula in `minecraft_network.md` gives a
first-result latency of 2,744 RT (`274.4 s`) with preloaded inputs and direct
final-output reads. Evaluating all 49 placements plus pass as one 50-position
stream adds only 98 RT (`9.8 s`), so the final result arrives after `284.2 s`.
The corresponding steady-state scoring throughput is five positions per
second; running 50 independent end-to-end passes would be much slower.

## Commands

```bash
# Generate more KataGo data.
uv run python -m scripts.go.generate_data \
  --output artifacts/go7x7/data.npz --games 192 --workers 6

# Train the physical model.
uv run python -m scripts.go.train \
  --dataset artifacts/go7x7/data.npz --output-dir artifacts/go7x7

# Play in the browser at http://127.0.0.1:8765/.
uv run python -m scripts.go.play

# Evaluate against a retained noisy-KataGo match suite.
uv run python -m scripts.go.evaluate \
  --model artifacts/go7x7/model.pt \
  --suite artifacts/go7x7/match_suite.json \
  --output artifacts/go7x7/match.json
```

KataGo defaults to the executable, configuration, and network under `tools/`;
all paths can be overridden through command-line flags.
