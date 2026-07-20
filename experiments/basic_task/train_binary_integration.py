"""Train and export a small 5-input/5-output binary MCNN integration task.

The default target is intentionally linear after the binary inputs. That keeps
the exported Minecraft model compact while still exercising positive weights,
negative weights, nonzero biases, and output saturation.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]

from mcnn import (
    DiscreteMCNN,
    SchematicExportConfig,
    TrainableMCNN,
    save_schematic,
)


def binary_inputs() -> Tensor:
    return torch.tensor(
        [[(value >> bit) & 1 for bit in range(5)] for value in range(32)],
        dtype=torch.float32,
    )


def binary_integration_targets(x: Tensor) -> Tensor:
    weight = torch.tensor(
        [
            [2, 4, 8, 0, 0],
            [-3, -3, -3, -3, -2],
            [3, 3, 3, 3, 3],
            [5, -4, 3, -2, 1],
            [0, 0, 0, 0, 14],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor([1, 15, 2, 5, 1], dtype=torch.float32)
    return F.linear(x, weight, bias).round().clamp(0.0, 15.0)


@torch.no_grad()
def exact_metrics(
    model: TrainableMCNN | DiscreteMCNN, x: Tensor, y: Tensor
) -> dict[str, float]:
    if isinstance(model, DiscreteMCNN):
        outputs = model.forward(x)
    else:
        model.eval()
        outputs = model(x, exact=True)
    matches = outputs.eq(y)
    return {
        "mse": F.mse_loss(outputs, y).item(),
        "mae": (outputs - y).abs().mean().item(),
        "signal_accuracy": matches.to(torch.float32).mean().item(),
        "row_accuracy": matches.all(dim=1).to(torch.float32).mean().item(),
    }


def serializable_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def train(args: argparse.Namespace) -> DiscreteMCNN:
    torch.manual_seed(args.seed)
    x = binary_inputs()
    y = binary_integration_targets(x)
    hidden_sizes = tuple(int(size) for size in args.hidden.split(",") if size)
    model = TrainableMCNN(
        input_features=5,
        hidden_sizes=hidden_sizes,
        output_features=5,
        output_min=0.0,
        output_max=15.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = exact_metrics(model, x, y)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss = F.mse_loss(outputs, y)
        loss.backward()
        optimizer.step()

        metrics = exact_metrics(model, x, y)
        if metrics["mse"] < best_metrics["mse"]:
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch == args.epochs or epoch % args.report_every == 0:
            print(
                f"epoch={epoch:04d} "
                f"exact_mse={metrics['mse']:.4f} "
                f"signal_acc={metrics['signal_accuracy']:.3f} "
                f"row_acc={metrics['row_accuracy']:.3f} "
                f"best_exact_mse={best_metrics['mse']:.4f}"
            )
        if metrics["row_accuracy"] == 1.0 and args.stop_when_exact:
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            print(f"stopped_exact epoch={epoch:04d}")
            break

    model.load_state_dict(best_state)
    discrete_model = model.discrete()
    discrete_metrics = exact_metrics(discrete_model, x, y)
    print(
        f"final_exact_mse={discrete_metrics['mse']:.4f} "
        f"signal_acc={discrete_metrics['signal_accuracy']:.3f} "
        f"row_acc={discrete_metrics['row_accuracy']:.3f} "
        f"nonzero_weights={discrete_model.nonzero_weights()} "
        f"max_physical_bias={discrete_model.max_physical_bias()} "
        f"layer_sizes={discrete_model.layer_sizes()}"
    )

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": discrete_model.state_dict(),
            "metrics": {"final": discrete_metrics},
            "config": serializable_config(args),
            "task": {
                "description": "5 binary inputs to compact signal-strength diagnostics with positive weights, negative weights, bias, and saturation.",
                "inputs": x.to(torch.int16),
                "targets": y.to(torch.int16),
            },
        },
        args.artifact,
    )
    print(f"saved_model={args.artifact}")
    return discrete_model


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
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument(
        "--stop-when-exact", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / "mcnn_binary_5x5_integration.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "mcnn_binary_5x5_integration.schem",
    )
    parser.add_argument(
        "--io-wiring", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.report_every < 1:
        raise SystemExit("--report-every must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    model = train(args)
    export(model, args)


if __name__ == "__main__":
    main()
