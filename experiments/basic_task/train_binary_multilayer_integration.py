"""Export a deterministic 5 -> 4 -> 5 binary multilayer MCNN fixture.

The model is specified directly as an exact discrete MCNN. That keeps the
fixture stable: any Minecraft mismatch points at export, placement, or
inter-layer wiring rather than training variance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]

from mcnn import (
    DiscreteLayer,
    DiscreteMCNN,
    SchematicExportConfig,
    save_schematic,
)


def binary_inputs() -> Tensor:
    return torch.tensor(
        [[(value >> bit) & 1 for bit in range(5)] for value in range(32)],
        dtype=torch.float32,
    )


def exact_model() -> DiscreteMCNN:
    return DiscreteMCNN(
        [
            DiscreteLayer(
                weight=torch.tensor(
                    [
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 1],
                    ],
                    dtype=torch.int16,
                ),
                bias=torch.tensor([0, 0, 0, 0], dtype=torch.int16),
                output_min=0,
                output_max=1,
            ),
            DiscreteLayer(
                weight=torch.tensor(
                    [
                        [14, 0, 0, 0],
                        [0, -14, 0, 0],
                        [6, 0, -4, 0],
                        [0, 0, 0, 12],
                        [5, -4, 3, -2],
                    ],
                    dtype=torch.int16,
                ),
                bias=torch.tensor([1, 15, 2, 3, 5], dtype=torch.int16),
                output_min=0,
                output_max=15,
            ),
        ]
    )


def hidden_targets(x: Tensor) -> Tensor:
    h0 = x[:, 0]
    h1 = x[:, 1]
    h2 = x[:, 2]
    h3 = x[:, 3] + x[:, 4]
    return torch.stack((h0, h1, h2, h3), dim=1).round().clamp(0.0, 1.0)


def output_targets(hidden: Tensor) -> Tensor:
    h0, h1, h2, h3 = hidden.unbind(dim=1)
    return (
        torch.stack(
            (
                1 + 14 * h0,
                15 - 14 * h1,
                2 + 6 * h0 - 4 * h2,
                3 + 12 * h3,
                5 + 5 * h0 - 4 * h1 + 3 * h2 - 2 * h3,
            ),
            dim=1,
        )
        .round()
        .clamp(0.0, 15.0)
    )


@torch.no_grad()
def metrics(model: DiscreteMCNN, x: Tensor, y: Tensor) -> dict[str, float]:
    outputs = model.forward(x)
    matches = outputs.eq(y)
    return {
        "mse": F.mse_loss(outputs, y).item(),
        "mae": (outputs - y).abs().mean().item(),
        "signal_accuracy": matches.to(torch.float32).mean().item(),
        "row_accuracy": matches.all(dim=1).to(torch.float32).mean().item(),
    }


def save_artifact(model: DiscreteMCNN, args: argparse.Namespace) -> None:
    x = binary_inputs()
    hidden = hidden_targets(x)
    targets = output_targets(hidden)
    final_metrics = metrics(model, x, targets)

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "metrics": {"final": final_metrics},
            "config": {
                "artifact": str(args.artifact),
                "output": str(args.output),
                "io_wiring": args.io_wiring,
            },
            "task": {
                "description": "5 binary inputs through a 4-unit hidden layer to 5 redstone signal outputs.",
                "inputs": x.to(torch.int16),
                "hidden_targets": hidden.to(torch.int16),
                "targets": targets.to(torch.int16),
            },
        },
        args.artifact,
    )
    print(
        f"final_exact_mse={final_metrics['mse']:.4f} "
        f"signal_acc={final_metrics['signal_accuracy']:.3f} "
        f"row_acc={final_metrics['row_accuracy']:.3f} "
        f"nonzero_weights={model.nonzero_weights()} "
        f"max_physical_bias={model.max_physical_bias()} "
        f"layer_sizes={model.layer_sizes()}"
    )
    print(f"physical_biases={model.physical_biases()}")
    print(f"saved_model={args.artifact}")


def export(model: DiscreteMCNN, args: argparse.Namespace) -> None:
    config = SchematicExportConfig(
        emit_input_wiring=args.io_wiring,
        emit_output_wiring=args.io_wiring,
        emit_layer_io_connectors=args.io_wiring,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved_path = save_schematic(model, args.output, config=config)
    print(f"saved_schematic={saved_path}")
    print(f"layer_sizes={model.layer_sizes()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / "mcnn_binary_5x4x5_integration.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "mcnn_binary_5x4x5_integration.schem",
    )
    parser.add_argument(
        "--io-wiring", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = exact_model()
    save_artifact(model, args)
    export(model, args)


if __name__ == "__main__":
    main()
