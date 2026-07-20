"""Train an MCNN classifier on binarized 8x8 handwritten digits.

This uses scikit-learn's built-in digits dataset: 1,797 grayscale 8x8 images.
Pixels are binarized before they reach the network so the first MCNN layer sees
Minecraft-style binary inputs.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcnn import (  # noqa: E402
    MAX_BIAS_SIGNAL_STRENGTH,
    DiscreteMCNN,
    MCNNClassifier,
    MCNNLayer,
    discrete_from_model,
    inactive_subtraction,
    is_representable_bias_signal,
    remove_hidden_unit,
)


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


def class_targets(labels: Tensor, classes: int, low: float, high: float) -> Tensor:
    targets = torch.full((labels.shape[0], classes), low, device=labels.device)
    return targets.scatter(1, labels.unsqueeze(1), high)


def accuracy(outputs: Tensor, labels: Tensor) -> float:
    predictions = outputs.argmax(dim=1)
    return (predictions == labels).to(torch.float32).mean().item()


@torch.no_grad()
def evaluate_discrete(
    model: DiscreteMCNN,
    loader: DataLoader,
    device: torch.device,
    *,
    loss_name: str,
    target_min: float,
    target_max: float,
    margin: float,
) -> tuple[float, float]:
    total_loss = 0.0
    total_accuracy = 0.0
    total_examples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model.forward(x)
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


def loss_value(
    outputs: Tensor,
    labels: Tensor,
    *,
    loss_name: str,
    target_min: float,
    target_max: float,
    margin: float = 1.0,
) -> Tensor:
    if loss_name == "ce":
        return F.cross_entropy(outputs, labels)
    if loss_name == "margin":
        correct = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
        wrong = (
            outputs.masked_fill(
                F.one_hot(labels, outputs.shape[1]).to(torch.bool), float("-inf")
            )
            .max(dim=1)
            .values
        )
        return F.relu(margin - (correct - wrong)).mean()
    targets = class_targets(labels, outputs.shape[1], target_min, target_max)
    return F.mse_loss(outputs, targets)


def training_physical_cost(model: MCNNClassifier, args: argparse.Namespace) -> Tensor:
    penalties = []
    for module in model.layers:
        if not isinstance(module, MCNNLayer):
            continue
        weight = module.quantized_weight(exact=False)
        bias = module.positive_bias()
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


@torch.no_grad()
def evaluate(
    model: MCNNClassifier,
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


def discrete_score(
    model: DiscreteMCNN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    if not satisfies_physical_bias_cap(model, args):
        return float("-inf"), float("inf"), 0.0
    loss, acc = evaluate_discrete(
        model,
        loader,
        device,
        loss_name=args.loss,
        target_min=args.output_min,
        target_max=args.output_max,
        margin=args.margin,
    )
    hidden_penalty = args.hidden_unit_penalty * model.hidden_units()
    nonzero_penalty = args.discrete_nonzero_weight_penalty * model.nonzero_weights()
    bias_penalty = args.discrete_max_bias_penalty * (
        model.max_physical_bias() / max(1, args.max_physical_bias)
    )
    return acc - hidden_penalty - nonzero_penalty - bias_penalty, loss, acc


def satisfies_physical_bias_cap(model: DiscreteMCNN, args: argparse.Namespace) -> bool:
    return all(
        physical_bias <= args.max_physical_bias
        and is_representable_bias_signal(physical_bias)
        for physical_bias in model.physical_biases()
    )


def repair_physical_bias_cap(
    model: DiscreteMCNN, args: argparse.Namespace
) -> tuple[DiscreteMCNN, int]:
    repaired = model.clone()
    zeroed_weights = 0
    for layer in repaired.layers:
        for output_index in range(layer.weight.shape[0]):
            physical_bias = int(layer.bias[output_index].item()) + sum(
                inactive_subtraction(int(weight))
                for weight in layer.weight[output_index].tolist()
            )
            if physical_bias <= args.max_physical_bias and is_representable_bias_signal(
                physical_bias
            ):
                continue

            candidates = sorted(
                (
                    (
                        inactive_subtraction(
                            int(layer.weight[output_index, input_index].item())
                        ),
                        abs(int(layer.weight[output_index, input_index].item())),
                        input_index,
                    )
                    for input_index in range(layer.weight.shape[1])
                    if int(layer.weight[output_index, input_index].item()) != 0
                ),
                reverse=True,
            )
            for inactive, _magnitude, input_index in candidates:
                if (
                    physical_bias <= args.max_physical_bias
                    and is_representable_bias_signal(physical_bias)
                ):
                    break
                layer.weight[output_index, input_index] = 0
                physical_bias -= inactive
                zeroed_weights += 1
    return repaired, zeroed_weights


def structural_ablation(
    model: DiscreteMCNN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[DiscreteMCNN, dict[str, float]]:
    pruned = model.clone()
    baseline_score, baseline_loss, baseline_acc = discrete_score(
        pruned, loader, device, args
    )
    min_acc = baseline_acc - args.ablation_accuracy_tolerance

    accepted = 0
    evaluated = 0
    passes = 0
    while args.ablation_passes < 0 or passes < args.ablation_passes:
        passes += 1
        pass_accepted = 0
        candidates: list[tuple[int, int, int]] = []

        for layer_index in range(len(pruned.layers) - 1):
            layer = pruned.layers[layer_index]
            next_layer = pruned.layers[layer_index + 1]
            for unit_index in range(layer.weight.shape[0]):
                incoming = int(layer.weight[unit_index].abs().sum().item())
                outgoing = int(next_layer.weight[:, unit_index].abs().sum().item())
                candidates.append((incoming + outgoing, layer_index, unit_index))
        candidates.sort()

        for _, layer_index, unit_index in candidates:
            if layer_index >= len(pruned.layers) - 1:
                continue
            if unit_index >= pruned.layers[layer_index].weight.shape[0]:
                continue
            candidate = remove_hidden_unit(pruned, layer_index, unit_index)
            if not satisfies_physical_bias_cap(candidate, args):
                continue
            score, loss, acc = discrete_score(candidate, loader, device, args)
            evaluated += 1
            if acc >= min_acc and score >= baseline_score:
                pruned = candidate
                baseline_score = score
                baseline_loss = loss
                baseline_acc = acc
                accepted += 1
                pass_accepted += 1
                print(
                    f"ablation_removed layer={layer_index} unit={unit_index} "
                    f"acc={acc:.3f} layer_sizes={pruned.layer_sizes()}"
                )
        if pass_accepted == 0:
            break

    return pruned, {
        "accepted": accepted,
        "evaluated": evaluated,
        "passes": passes,
        "loss": baseline_loss,
        "accuracy": baseline_acc,
        "hidden_units": pruned.hidden_units(),
        "nonzero_weights": pruned.nonzero_weights(),
        "max_physical_bias": pruned.max_physical_bias(),
        "layer_sizes": pruned.layer_sizes(),
    }


def mutate_discrete(
    model: DiscreteMCNN, rng: random.Random, args: argparse.Namespace
) -> DiscreteMCNN:
    child = model.clone()
    if (
        rng.random() < args.evolution_remove_unit_probability
        and child.hidden_units() > 0
    ):
        hidden_layer_indices = [
            index
            for index in range(len(child.layers) - 1)
            if child.layers[index].weight.shape[0] > 1
        ]
        if hidden_layer_indices:
            layer_index = rng.choice(hidden_layer_indices)
            unit_index = rng.randrange(child.layers[layer_index].weight.shape[0])
            child = remove_hidden_unit(child, layer_index, unit_index)

    for _ in range(args.evolution_weight_mutations):
        layer = rng.choice(child.layers)
        row = rng.randrange(layer.weight.shape[0])
        column = rng.randrange(layer.weight.shape[1])
        if rng.random() < args.evolution_zero_probability:
            layer.weight[row, column] = 0
        else:
            delta = rng.choice((-2, -1, 1, 2))
            old = int(layer.weight[row, column].item())
            layer.weight[row, column] = max(
                args.weight_min, min(args.weight_max, old + delta)
            )

    for _ in range(args.evolution_bias_mutations):
        layer = rng.choice(child.layers)
        index = rng.randrange(layer.bias.shape[0])
        delta = rng.choice((-2, -1, 1, 2))
        old = int(layer.bias[index].item())
        layer.bias[index] = max(1, old + delta)

    return repair_physical_bias_cap(child, args)[0]


def evolutionary_refinement(
    seed_model: DiscreteMCNN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[DiscreteMCNN, dict[str, float]]:
    rng = random.Random(args.seed + 1000)
    seed_model = repair_physical_bias_cap(seed_model, args)[0]
    population = [seed_model.clone()]
    while len(population) < args.evolution_population:
        population.append(mutate_discrete(seed_model, rng, args))

    best_model = seed_model.clone()
    best_score, best_loss, best_acc = discrete_score(best_model, loader, device, args)

    for generation in range(1, args.evolution_generations + 1):
        scored = []
        for candidate in population:
            score, loss, acc = discrete_score(candidate, loader, device, args)
            scored.append((score, loss, acc, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)

        if scored[0][0] > best_score:
            best_score, best_loss, best_acc, best_model = (
                scored[0][0],
                scored[0][1],
                scored[0][2],
                scored[0][3].clone(),
            )

        if (
            generation == 1
            or generation == args.evolution_generations
            or generation % args.report_every == 0
        ):
            print(
                f"evolution_generation={generation:03d} "
                f"best_score={best_score:.4f} best_acc={best_acc:.3f} "
                f"best_loss={best_loss:.4f} layer_sizes={best_model.layer_sizes()}"
            )

        elites = [
            candidate.clone() for _, _, _, candidate in scored[: args.evolution_elites]
        ]
        population = elites.copy()
        while len(population) < args.evolution_population:
            parent = rng.choice(elites)
            population.append(mutate_discrete(parent, rng, args))

    return best_model, {
        "loss": best_loss,
        "accuracy": best_acc,
        "score": best_score,
        "hidden_units": best_model.hidden_units(),
        "max_physical_bias": best_model.max_physical_bias(),
        "nonzero_weights": best_model.nonzero_weights(),
        "layer_sizes": best_model.layer_sizes(),
    }


def mutate_local_search_candidate(
    model: DiscreteMCNN, rng: random.Random, args: argparse.Namespace
) -> DiscreteMCNN:
    child = model.clone()
    moves = ["zero_weight", "nudge_weight", "nudge_bias"]
    if child.hidden_units() > 0:
        moves.append("remove_unit")
    move = rng.choice(moves)

    if move == "remove_unit":
        hidden_layer_indices = [
            index
            for index in range(len(child.layers) - 1)
            if child.layers[index].weight.shape[0] > 1
        ]
        if hidden_layer_indices:
            layer_index = rng.choice(hidden_layer_indices)
            layer = child.layers[layer_index]
            next_layer = child.layers[layer_index + 1]
            costs = [
                (
                    int(layer.weight[unit_index].abs().sum().item())
                    + int(next_layer.weight[:, unit_index].abs().sum().item()),
                    unit_index,
                )
                for unit_index in range(layer.weight.shape[0])
            ]
            costs.sort()
            window = costs[
                : max(1, min(args.local_search_candidate_window, len(costs)))
            ]
            child = remove_hidden_unit(child, layer_index, rng.choice(window)[1])
    elif move == "zero_weight":
        nonzero_positions = [
            (layer_index, row, column)
            for layer_index, layer in enumerate(child.layers)
            for row in range(layer.weight.shape[0])
            for column in range(layer.weight.shape[1])
            if int(layer.weight[row, column].item()) != 0
        ]
        if nonzero_positions:
            layer_index, row, column = rng.choice(nonzero_positions)
            child.layers[layer_index].weight[row, column] = 0
    elif move == "nudge_weight":
        layer = rng.choice(child.layers)
        row = rng.randrange(layer.weight.shape[0])
        column = rng.randrange(layer.weight.shape[1])
        old = int(layer.weight[row, column].item())
        layer.weight[row, column] = max(
            args.weight_min, min(args.weight_max, old + rng.choice((-1, 1)))
        )
    else:
        layer = rng.choice(child.layers)
        index = rng.randrange(layer.bias.shape[0])
        old = int(layer.bias[index].item())
        layer.bias[index] = max(1, old + rng.choice((-2, -1, 1, 2)))

    return repair_physical_bias_cap(child, args)[0]


def local_search_refinement(
    seed_model: DiscreteMCNN,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[DiscreteMCNN, dict[str, float]]:
    rng = random.Random(args.seed + 2000)
    current = repair_physical_bias_cap(seed_model, args)[0]
    current_score, current_loss, current_acc = discrete_score(
        current, loader, device, args
    )
    best = current.clone()
    best_score = current_score
    best_loss = current_loss
    best_acc = current_acc
    min_acc = best_acc - args.local_search_accuracy_tolerance
    accepted = 0

    for step in range(1, args.local_search_steps + 1):
        candidate = mutate_local_search_candidate(current, rng, args)
        if not satisfies_physical_bias_cap(candidate, args):
            continue
        score, loss, acc = discrete_score(candidate, loader, device, args)
        temperature = max(
            0.0,
            args.local_search_temperature
            * (1.0 - step / max(1, args.local_search_steps)),
        )
        accept_worse = temperature > 0.0 and rng.random() < math.exp(
            min(0.0, score - current_score) / temperature
        )
        if acc >= min_acc and (score >= current_score or accept_worse):
            current = candidate
            accepted += 1
            current_score = score
            current_loss = loss
            current_acc = acc
            if score >= best_score:
                best = candidate.clone()
                best_score = score
                best_loss = loss
                best_acc = acc
        if (
            step == 1
            or step == args.local_search_steps
            or step % args.report_every == 0
        ):
            print(
                f"local_search_step={step:04d} accepted={accepted} "
                f"best_score={best_score:.4f} best_acc={best_acc:.3f} "
                f"current_score={current_score:.4f} current_acc={current_acc:.3f} "
                f"hidden={best.hidden_units()} nonzero={best.nonzero_weights()} "
                f"max_physical_bias={best.max_physical_bias()}"
            )

    return best, {
        "accepted": accepted,
        "evaluated": args.local_search_steps,
        "loss": best_loss,
        "accuracy": best_acc,
        "score": best_score,
        "hidden_units": best.hidden_units(),
        "nonzero_weights": best.nonzero_weights(),
        "max_physical_bias": best.max_physical_bias(),
        "layer_sizes": best.layer_sizes(),
    }


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
    model = MCNNClassifier(
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
    discrete_model = discrete_from_model(model)
    unrepaired_max_physical_bias = discrete_model.max_physical_bias()
    discrete_model, cap_zeroed_weights = repair_physical_bias_cap(discrete_model, args)
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
            discrete_model, validation_loader, device, args
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
            discrete_model, validation_loader, device, args
        )
        post_metrics["evolution"] = evolution_metrics
        print(
            f"evolution acc={evolution_metrics['accuracy']:.3f} "
            f"score={evolution_metrics['score']:.4f} "
            f"layer_sizes={evolution_metrics['layer_sizes']}"
        )

    if args.local_search_steps > 0:
        discrete_model, local_search_metrics = local_search_refinement(
            discrete_model, validation_loader, device, args
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
