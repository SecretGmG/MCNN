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



def weights(
    out_features: int,
    in_features: int,
) -> Tensor:
    """Create linearly spaced integer rows, barrel-shifted one over.

    Row zero consists of ``in_features`` values distributed approximately
    uniformly across [minimum, maximum]. Each following row is circularly
    shifted one position to the right.
    """
    return torch.randint(-3, 4, (out_features, in_features), dtype=torch.int16)

def large_example() -> DiscreteMCNN:
    """Construct the deterministic 8 -> 64 -> 8 MCNN."""
    return DiscreteMCNN(
        [
            DiscreteLayer(
                weight=weights(
                    out_features=8,
                    in_features=8,
                ),
                bias=torch.zeros(8, dtype=torch.int16),
                output_min=0,
                output_max=1,
            ),
            DiscreteLayer(
                weight=weights(
                    out_features=8,
                    in_features=8,
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



def save_artifact(
    model: DiscreteMCNN,
    args: argparse.Namespace,
) -> None:
    args.artifact.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "name": "large_example",
            "model": model.state_dict(),
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
            },
        },
        args.artifact,
    )

    print(
        f"nonzero_weights={model.nonzero_weights()} "
        f"layer_sizes={model.layer_sizes()}"
    )
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



    hidden, out = layer_targets(model, torch.zeros(8))
    print(f"zeros: hidden:{hidden} out: {out}")
    hidden, out = layer_targets(model, torch.ones(8))
    print(f"ones : hidden:{hidden} out: {out}")

    for i in range(8):
        one_hot = torch.zeros(8)
        one_hot[i] = 1
        hidden, out = layer_targets(model, one_hot)
        print(f"one hot {i}: hidden:{hidden} out: {out}")



if __name__ == "__main__":
    main()
