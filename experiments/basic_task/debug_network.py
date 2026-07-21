"""Export a deterministic 8 -> 64 -> 8 MCNN fixture named large_example.

Each layer starts with a weight row linearly spanning the integer range
[-32, 32]. Each subsequent unit receives the same row circularly shifted
one position to the right.

The model is specified directly as an exact discrete MCNN. This keeps the
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
    save_schematic,
)


def binary_inputs() -> Tensor:
    """Return all 256 possible 8-bit binary inputs."""
    return torch.tensor(
        [[(value >> bit) & 1 for bit in range(8)] for value in range(256)],
        dtype=torch.float32,
    )


def barrel_shifted_weights(
    out_features: int,
    in_features: int,
    *,
    minimum: int = -32,
    maximum: int = 32,
) -> Tensor:
    """Create linearly spaced integer rows, barrel-shifted one over.

    Row zero consists of ``in_features`` values distributed approximately
    uniformly across [minimum, maximum]. Each following row is circularly
    shifted one position to the right.
    """
    base_row = torch.linspace(
        minimum,
        maximum,
        steps=in_features,
        dtype=torch.float64,
    ).round().to(torch.int16)

    return torch.stack(
        [
            torch.roll(base_row, shifts=row_index)
            for row_index in range(out_features)
        ],
        dim=0,
    )


def large_example() -> DiscreteMCNN:
    """Construct the deterministic 8 -> 64 -> 8 MCNN."""
    return DiscreteMCNN(
        [
            DiscreteLayer(
                weight=barrel_shifted_weights(
                    out_features=64,
                    in_features=8,
                ),
                bias=torch.zeros(64, dtype=torch.int16),
                output_min=0,
                output_max=1,
            ),
            DiscreteLayer(
                weight=barrel_shifted_weights(
                    out_features=8,
                    in_features=64,
                ),
                bias=torch.zeros(8, dtype=torch.int16),
                output_min=0,
                output_max=15,
            ),
        ]
    )


@torch.no_grad()
def layer_targets(
    model: DiscreteMCNN,
    inputs: Tensor,
) -> tuple[Tensor, Tensor]:
    """Calculate exact hidden and output targets from the discrete model."""
    hidden_layer = model.layers[0]
    output_layer = model.layers[1]

    hidden_pre = (
        inputs.to(torch.int32)
        @ hidden_layer.weight.to(torch.int32).T
        + hidden_layer.bias.to(torch.int32)
    )

    hidden = hidden_pre.clamp(
        int(hidden_layer.output_min),
        int(hidden_layer.output_max),
    )

    output_pre = (
        hidden
        @ output_layer.weight.to(torch.int32).T
        + output_layer.bias.to(torch.int32)
    )

    output = output_pre.clamp(
        int(output_layer.output_min),
        int(output_layer.output_max),
    )

    return hidden.to(torch.float32), output.to(torch.float32)


@torch.no_grad()
def metrics(
    model: DiscreteMCNN,
    inputs: Tensor,
    targets: Tensor,
) -> dict[str, float]:
    outputs = model.forward(inputs)
    matches = outputs.eq(targets)

    return {
        "mse": F.mse_loss(outputs, targets).item(),
        "mae": (outputs - targets).abs().mean().item(),
        "signal_accuracy": matches.to(torch.float32).mean().item(),
        "row_accuracy": matches.all(dim=1).to(torch.float32).mean().item(),
    }


def save_artifact(
    model: DiscreteMCNN,
    args: argparse.Namespace,
) -> None:
    inputs = binary_inputs()
    hidden_targets, targets = layer_targets(model, inputs)
    final_metrics = metrics(model, inputs, targets)

    args.artifact.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "name": "large_example",
            "model": model.state_dict(),
            "metrics": {
                "final": final_metrics,
            },
            "config": {
                "name": "large_example",
                "artifact": str(args.artifact),
                "output": str(args.output),
                "io_wiring": args.io_wiring,
                "weight_min": -32,
                "weight_max": 32,
                "barrel_shift": 1,
                "architecture": [8, 64, 8],
            },
            "task": {
                "description": (
                    "8 binary inputs through a 64-unit hidden layer "
                    "to 8 redstone signal outputs. Weight rows are "
                    "linearly spaced from -32 to 32 and circularly "
                    "shifted one position per unit."
                ),
                "inputs": inputs.to(torch.int16),
                "hidden_targets": hidden_targets.to(torch.int16),
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


def export(
    model: DiscreteMCNN,
    args: argparse.Namespace,
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    saved_path = save_schematic(model, args.output)

    print(f"saved_schematic={saved_path}")
    print(f"layer_sizes={model.layer_sizes()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "integration"
            / "large_example.pt"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "integration"
            / "large_example.schem"
        ),
    )

    parser.add_argument(
        "--io-wiring",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = large_example()

    save_artifact(model, args)
    export(model, args)


if __name__ == "__main__":
    main()
