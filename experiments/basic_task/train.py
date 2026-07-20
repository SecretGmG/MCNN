"""Train an MCNN classifier on binarized 8x8 handwritten digits.

This uses scikit-learn's built-in digits dataset: 1,797 grayscale 8x8 images.
Pixels are binarized before they reach the network so the first MCNN layer sees
Minecraft-style binary inputs.
"""

from __future__ import annotations

import argparse
import copy

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mcnn import (
    MAX_BIAS_SIGNAL_STRENGTH,
    DiscreteMCNN,
    RefinementConfig,
    TrainableMCNN,
    evaluate_discrete,
    evolutionary_refinement,
    local_search_refinement,
    repair_physical_bias_cap,
    structural_ablation,
)
from mcnn.refinement import loss_value


def load_binary_digits(
    test_fraction: float,
    validation_fraction: float,
    threshold: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise SystemExit(
            "Install scikit-learn to use this experiment: python -m pip install scikit-learn"
        ) from error

    digits = load_digits()
    x = torch.tensor(digits.images, dtype=torch.float32).reshape(-1, 64)
    x = (x > threshold).to(torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)

    train_val_x, test_x, train_val_y, test_y = train_test_split(
        x,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_val_x,
        train_val_y,
        test_size=validation_fraction,
        random_state=seed,
        stratify=train_val_y,
    )
    return (
        TensorDataset(train_x, train_y),
        TensorDataset(validation_x, validation_y),
        TensorDataset(test_x, test_y),
    )


def training_physical_cost(model: TrainableMCNN, args: argparse.Namespace) -> Tensor:
    penalties = []
    for layer_index in range(len(model.layer_params)):
        weight = model._quantized_weight(layer_index, exact=False)
        bias = model._positive_bias(layer_index)
        inactive = torch.where(
            weight > 0,
            torch.full_like(weight, 15.0),
            torch.where(weight < 0, weight + 15.0, torch.zeros_like(weight)),
        )
        physical_bias = bias + inactive.sum(dim=1)
        penalties.append(
            args.physical_bias_penalty
            * F.relu(physical_bias - args.max_physical_bias).mean()
        )
        penalties.append(
            args.physical_bias_l1_penalty
            * (physical_bias / max(1, args.max_physical_bias)).mean()
        )
        penalties.append(args.nonzero_weight_penalty * weight.abs().mean())
    if not penalties:
        return torch.zeros((), device=next(model.parameters()).device)
    return sum(penalties)


def accuracy(outputs: Tensor, labels: Tensor) -> float:
    predictions = outputs.argmax(dim=1)
    return (predictions == labels).to(torch.float32).mean().item()


@torch.no_grad()
def evaluate(
    model: TrainableMCNN,
    loader: DataLoader,
    device: torch.device,
    *,
    exact: bool,
    loss_name: str,
    target_min: float,
    target_max: float,
    margin: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_examples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x, exact=exact)
        loss = loss_value(
            outputs,
            y,
            loss_name=loss_name,
            target_min=target_min,
            target_max=target_max,
            margin=margin,
        )
        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_accuracy += accuracy(outputs, y) * batch_size
        total_examples += batch_size

    return total_loss / total_examples, total_accuracy / total_examples


def train(args: argparse.Namespace) -> None:
    if args.weight_min > args.weight_max:
        raise SystemExit("--weight-min must be <= --weight-max")
    if args.output_min > args.output_max:
        raise SystemExit("--output-min must be <= --output-max")
    if args.report_every < 1:
        raise SystemExit("--report-every must be >= 1")
    if args.evolution_elites < 1:
        raise SystemExit("--evolution-elites must be >= 1")
    if args.evolution_population < args.evolution_elites:
        raise SystemExit("--evolution-population must be >= --evolution-elites")
    if args.max_physical_bias < 0:
        raise SystemExit("--max-physical-bias must be >= 0")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = RefinementConfig(
        loss=args.loss,
        target_min=args.output_min,
        target_max=args.output_max,
        margin=args.margin,
        output_min=args.output_min,
        output_max=args.output_max,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        max_physical_bias=args.max_physical_bias,
        hidden_unit_penalty=args.hidden_unit_penalty,
        discrete_nonzero_weight_penalty=args.discrete_nonzero_weight_penalty,
        discrete_max_bias_penalty=args.discrete_max_bias_penalty,
        ablation_passes=args.ablation_passes,
        ablation_accuracy_tolerance=args.ablation_accuracy_tolerance,
        evolution_generations=args.evolution_generations,
        evolution_population=args.evolution_population,
        evolution_elites=args.evolution_elites,
        evolution_weight_mutations=args.evolution_weight_mutations,
        evolution_bias_mutations=args.evolution_bias_mutations,
        evolution_zero_probability=args.evolution_zero_probability,
        evolution_remove_unit_probability=args.evolution_remove_unit_probability,
        local_search_steps=args.local_search_steps,
        local_search_accuracy_tolerance=args.local_search_accuracy_tolerance,
        local_search_candidate_window=args.local_search_candidate_window,
        local_search_temperature=args.local_search_temperature,
        seed=args.seed,
        report_every=args.report_every,
    )

    train_data, validation_data, test_data = load_binary_digits(
        args.test_fraction,
        args.validation_fraction,
        args.threshold,
        args.seed,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    hidden_sizes = tuple(int(size) for size in args.hidden.split(",") if size)
    model = TrainableMCNN(
        input_features=64,
        hidden_sizes=hidden_sizes,
        output_features=10,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        output_min=args.output_min,
        output_max=args.output_max,
        hidden_activation=args.hidden_activation,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_exact_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = loss_value(
                outputs,
                y,
                loss_name=args.loss,
                target_min=args.output_min,
                target_max=args.output_max,
                margin=args.margin,
            )
            loss = loss + training_physical_cost(model, args)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(
            model,
            train_loader,
            device,
            exact=False,
            loss_name=args.loss,
            target_min=args.output_min,
            target_max=args.output_max,
            margin=args.margin,
        )
        validation_loss, validation_acc = evaluate(
            model,
            validation_loader,
            device,
            exact=True,
            loss_name=args.loss,
            target_min=args.output_min,
            target_max=args.output_max,
            margin=args.margin,
        )
        if validation_acc > best_exact_acc:
            best_exact_acc = validation_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch == args.epochs or epoch % args.report_every == 0:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"exact_validation_loss={validation_loss:.4f} exact_validation_acc={validation_acc:.3f} "
                f"best_exact_validation_acc={best_exact_acc:.3f}@{best_epoch:03d}"
            )

    model.load_state_dict(best_state)
    discrete_model = model.discrete()
    unrepaired_max_physical_bias = discrete_model.max_physical_bias()
    discrete_model, cap_zeroed_weights = repair_physical_bias_cap(
        discrete_model, config
    )
    validation_loss, validation_acc = evaluate_discrete(
        discrete_model,
        validation_loader,
        device,
        loss_name=args.loss,
        target_min=args.output_min,
        target_max=args.output_max,
        margin=args.margin,
    )
    print(
        f"best_exact_validation_acc={best_exact_acc:.3f} best_epoch={best_epoch:03d} "
        f"discrete_validation_acc={validation_acc:.3f} "
        f"max_physical_bias={discrete_model.max_physical_bias()} "
        f"layer_sizes={discrete_model.layer_sizes()}"
    )
    if cap_zeroed_weights:
        print(
            f"bias_cap_repair zeroed_weights={cap_zeroed_weights} "
            f"unrepaired_max_physical_bias={unrepaired_max_physical_bias} "
            f"max_physical_bias={discrete_model.max_physical_bias()}"
        )

    post_metrics: dict[str, object] = {
        "trained": {
            "validation_loss": validation_loss,
            "validation_accuracy": validation_acc,
            "hidden_units": discrete_model.hidden_units(),
            "nonzero_weights": discrete_model.nonzero_weights(),
            "max_physical_bias": discrete_model.max_physical_bias(),
            "bias_cap_zeroed_weights": cap_zeroed_weights,
            "unrepaired_max_physical_bias": unrepaired_max_physical_bias,
            "layer_sizes": discrete_model.layer_sizes(),
        }
    }

    if args.ablation_passes != 0:
        discrete_model, ablation_metrics = structural_ablation(
            discrete_model, validation_loader, device, config
        )
        post_metrics["ablation"] = ablation_metrics
        print(
            f"ablation accepted={ablation_metrics['accepted']} "
            f"evaluated={ablation_metrics['evaluated']} "
            f"acc={ablation_metrics['accuracy']:.3f} "
            f"layer_sizes={ablation_metrics['layer_sizes']}"
        )

    if args.evolution_generations > 0:
        discrete_model, evolution_metrics = evolutionary_refinement(
            discrete_model, validation_loader, device, config
        )
        post_metrics["evolution"] = evolution_metrics
        print(
            f"evolution acc={evolution_metrics['accuracy']:.3f} "
            f"score={evolution_metrics['score']:.4f} "
            f"layer_sizes={evolution_metrics['layer_sizes']}"
        )

    if args.local_search_steps > 0:
        discrete_model, local_search_metrics = local_search_refinement(
            discrete_model, validation_loader, device, config
        )
        post_metrics["local_search"] = local_search_metrics
        print(
            f"local_search acc={local_search_metrics['accuracy']:.3f} "
            f"score={local_search_metrics['score']:.4f} "
            f"hidden={local_search_metrics['hidden_units']} "
            f"nonzero={local_search_metrics['nonzero_weights']} "
            f"max_physical_bias={local_search_metrics['max_physical_bias']} "
            f"layer_sizes={local_search_metrics['layer_sizes']}"
        )

    final_validation_loss, final_validation_acc = evaluate_discrete(
        discrete_model,
        validation_loader,
        device,
        loss_name=args.loss,
        target_min=args.output_min,
        target_max=args.output_max,
        margin=args.margin,
    )
    final_test_loss, final_test_acc = evaluate_discrete(
        discrete_model,
        test_loader,
        device,
        loss_name=args.loss,
        target_min=args.output_min,
        target_max=args.output_max,
        margin=args.margin,
    )
    post_metrics["final"] = {
        "validation_loss": final_validation_loss,
        "validation_accuracy": final_validation_acc,
        "test_loss": final_test_loss,
        "test_accuracy": final_test_acc,
        "hidden_units": discrete_model.hidden_units(),
        "nonzero_weights": discrete_model.nonzero_weights(),
        "max_physical_bias": discrete_model.max_physical_bias(),
        "layer_sizes": discrete_model.layer_sizes(),
    }
    print(
        f"final_validation_acc={final_validation_acc:.3f} "
        f"final_test_acc={final_test_acc:.3f} "
        f"nonzero_weights={discrete_model.nonzero_weights()} "
        f"max_physical_bias={discrete_model.max_physical_bias()} "
        f"layer_sizes={discrete_model.layer_sizes()}"
    )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": discrete_model.state_dict(),
                "metrics": post_metrics,
                "config": vars(args),
            },
            save_path,
        )
        print(f"saved_model={save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--loss", choices=("mse", "ce", "margin"), default="mse")
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--hidden", default="128,64")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weight-min", type=float, default=-14.0)
    parser.add_argument("--weight-max", type=float, default=14.0)
    parser.add_argument("--output-min", type=float, default=0.0)
    parser.add_argument("--output-max", type=float, default=15.0)
    parser.add_argument(
        "--hidden-activation",
        choices=("bounded_integer", "binary_step"),
        default="bounded_integer",
    )
    parser.add_argument(
        "--max-physical-bias", type=int, default=MAX_BIAS_SIGNAL_STRENGTH
    )
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--save-path", default="")
    parser.add_argument("--hidden-unit-penalty", type=float, default=0.0)
    parser.add_argument("--nonzero-weight-penalty", type=float, default=0.0)
    parser.add_argument("--physical-bias-penalty", type=float, default=0.0)
    parser.add_argument("--physical-bias-l1-penalty", type=float, default=0.0)
    parser.add_argument("--discrete-nonzero-weight-penalty", type=float, default=0.0)
    parser.add_argument("--discrete-max-bias-penalty", type=float, default=0.0)
    parser.add_argument("--ablation-passes", type=int, default=0)
    parser.add_argument("--ablation-accuracy-tolerance", type=float, default=0.0)
    parser.add_argument("--evolution-generations", type=int, default=0)
    parser.add_argument("--evolution-population", type=int, default=24)
    parser.add_argument("--evolution-elites", type=int, default=4)
    parser.add_argument("--evolution-weight-mutations", type=int, default=24)
    parser.add_argument("--evolution-bias-mutations", type=int, default=4)
    parser.add_argument("--evolution-zero-probability", type=float, default=0.15)
    parser.add_argument("--evolution-remove-unit-probability", type=float, default=0.1)
    parser.add_argument("--local-search-steps", type=int, default=0)
    parser.add_argument("--local-search-accuracy-tolerance", type=float, default=0.0)
    parser.add_argument("--local-search-candidate-window", type=int, default=8)
    parser.add_argument("--local-search-temperature", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
