# MCNN

MCNN trains neural networks that can be executed by compact redstone circuits.
The core hardware uses ternary weights, integer activations from 0 to 15, and
eight-neuron streamed cells connected by a wrapping barrel shifter.

The repository contains three focused workloads:

- `scripts/mnist/`: digit training and evaluation, including schematic export
- `scripts/go/`: 7x7 Go data generation, physical training, evaluation, and play
- `scripts/connect4/`: exact-data generation, teacher distillation, and play

Shared command infrastructure lives in `scripts/common/`; reusable models and
rules live in `src/mcnn/`. See [`docs/minecraft_network.md`](docs/minecraft_network.md)
for the circuit definition and the workload documents under `docs/` for the
retained models and commands.

## Setup

```bash
uv sync
uv run pytest -q
```

Run commands as modules from the repository root:

```bash
uv run python -m scripts.mnist.train --help
uv run python -m scripts.go.play --help
uv run python -m scripts.connect4.play --help
```

The trained artifacts required by the browser demos are retained under
`artifacts/`. Local KataGo binaries/models and the external Connect Four solver
are ignored; their small setup/configuration files remain under `tools/`.
