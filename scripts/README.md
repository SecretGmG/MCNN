# Commands

Commands are grouped by the problem they operate on:

- `scripts.mnist`: digit training and evaluation
- `scripts.go`: 7x7 Go data generation, training, evaluation, and play
- `scripts.connect4`: Connect Four data generation, training, evaluation, and play
- `scripts.common`: model-agnostic export and shared command infrastructure

Run commands as modules from the repository root:

```bash
uv run python -m scripts.mnist.train --help
uv run python -m scripts.go.play --help
uv run python -m scripts.connect4.play --help
```

The general dense-model schematic exporter is independent of any dataset:

```bash
uv run python -m scripts.common.export_schematic \
  artifacts/shift_digits_64_64_10/model.pt
```
