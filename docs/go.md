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
one-sided block bands, ternary weights, and exactly 15 threshold-count steps.
The retained model uses shift depths `12, 4, 4, 4, 4`; its first layer is dense
over the padded input and later layers use five adjacent wrapped cells.

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

The short cleanup run used 28,087 examples and selected epoch 22. On its
game-disjoint validation split it reached 6.82 points mean move regret, 6.21
points centered RMSE, and 78.9% ordering accuracy for move pairs separated by
at least three points. Train and validation mean regret were 7.09 and 6.82
points respectively, providing no sign of conventional train-set overfitting
in this run. This is a validated physical baseline, not yet a strong Go player.

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
