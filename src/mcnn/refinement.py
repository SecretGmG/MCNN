"""Post-training discrete refinement: repair, ablation, evolution, and local search."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .constants import MAX_BIAS_SIGNAL_STRENGTH
from .discrete import DiscreteMCNN
from .encoding import inactive_subtraction
from .export import is_representable_bias_signal


@dataclass
class RefinementConfig:
    """Parameters that control post-training discrete refinement steps."""

    loss: str = "mse"
    target_min: float = 0.0
    target_max: float = 15.0
    margin: float = 4.0
    output_min: float = 0.0
    output_max: float = 15.0
    weight_min: float = -14.0
    weight_max: float = 14.0
    max_physical_bias: int = MAX_BIAS_SIGNAL_STRENGTH
    hidden_unit_penalty: float = 0.0
    discrete_nonzero_weight_penalty: float = 0.0
    discrete_max_bias_penalty: float = 0.0
    ablation_passes: int = 0
    ablation_accuracy_tolerance: float = 0.0
    evolution_generations: int = 0
    evolution_population: int = 24
    evolution_elites: int = 4
    evolution_weight_mutations: int = 24
    evolution_bias_mutations: int = 4
    evolution_zero_probability: float = 0.15
    evolution_remove_unit_probability: float = 0.1
    local_search_steps: int = 0
    local_search_accuracy_tolerance: float = 0.0
    local_search_candidate_window: int = 8
    local_search_temperature: float = 0.0
    seed: int = 1
    report_every: int = 1


# ---------------------------------------------------------------------------
# Evaluation helpers shared by all refinement stages
# ---------------------------------------------------------------------------


def accuracy(outputs: Tensor, labels: Tensor) -> float:
    predictions = outputs.argmax(dim=1)
    return (predictions == labels).to(torch.float32).mean().item()


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
    targets = torch.full(
        (labels.shape[0], outputs.shape[1]), target_min, device=labels.device
    )
    targets = targets.scatter(1, labels.unsqueeze(1), target_max)
    return F.mse_loss(outputs, targets)


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


# ---------------------------------------------------------------------------
# Discrete scoring and bias cap repair
# ---------------------------------------------------------------------------


def discrete_score(
    model: DiscreteMCNN,
    loader: DataLoader,
    device: torch.device,
    config: RefinementConfig,
) -> tuple[float, float, float]:
    if not satisfies_physical_bias_cap(model, config):
        return float("-inf"), float("inf"), 0.0
    loss, acc = evaluate_discrete(
        model,
        loader,
        device,
        loss_name=config.loss,
        target_min=config.output_min,
        target_max=config.output_max,
        margin=config.margin,
    )
    hidden_penalty = config.hidden_unit_penalty * model.hidden_units()
    nonzero_penalty = config.discrete_nonzero_weight_penalty * model.nonzero_weights()
    bias_penalty = config.discrete_max_bias_penalty * (
        model.max_physical_bias() / max(1, config.max_physical_bias)
    )
    return acc - hidden_penalty - nonzero_penalty - bias_penalty, loss, acc


def satisfies_physical_bias_cap(model: DiscreteMCNN, config: RefinementConfig) -> bool:
    return all(
        physical_bias <= config.max_physical_bias
        and is_representable_bias_signal(physical_bias)
        for physical_bias in model.physical_biases()
    )


def repair_physical_bias_cap(
    model: DiscreteMCNN, config: RefinementConfig
) -> tuple[DiscreteMCNN, int]:
    repaired = model.clone()
    zeroed_weights = 0
    for layer in repaired.layers:
        for output_index in range(layer.weight.shape[0]):
            physical_bias = int(layer.bias[output_index].item()) + sum(
                inactive_subtraction(int(weight))
                for weight in layer.weight[output_index].tolist()
            )
            if (
                physical_bias <= config.max_physical_bias
                and is_representable_bias_signal(physical_bias)
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
                    physical_bias <= config.max_physical_bias
                    and is_representable_bias_signal(physical_bias)
                ):
                    break
                layer.weight[output_index, input_index] = 0
                physical_bias -= inactive
                zeroed_weights += 1
    return repaired, zeroed_weights


# ---------------------------------------------------------------------------
# Structural ablation
# ---------------------------------------------------------------------------


def structural_ablation(
    model: DiscreteMCNN,
    loader: DataLoader,
    device: torch.device,
    config: RefinementConfig,
) -> tuple[DiscreteMCNN, dict[str, float]]:
    from .discrete import remove_hidden_unit

    pruned = model.clone()
    baseline_score, baseline_loss, baseline_acc = discrete_score(
        pruned, loader, device, config
    )
    min_acc = baseline_acc - config.ablation_accuracy_tolerance

    accepted = 0
    evaluated = 0
    passes = 0
    while config.ablation_passes < 0 or passes < config.ablation_passes:
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
            if not satisfies_physical_bias_cap(candidate, config):
                continue
            score, loss, acc = discrete_score(candidate, loader, device, config)
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


# ---------------------------------------------------------------------------
# Evolutionary refinement
# ---------------------------------------------------------------------------


def mutate_discrete(
    model: DiscreteMCNN, rng: random.Random, config: RefinementConfig
) -> DiscreteMCNN:
    from .discrete import remove_hidden_unit

    child = model.clone()
    if (
        rng.random() < config.evolution_remove_unit_probability
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

    for _ in range(config.evolution_weight_mutations):
        layer = rng.choice(child.layers)
        row = rng.randrange(layer.weight.shape[0])
        column = rng.randrange(layer.weight.shape[1])
        if rng.random() < config.evolution_zero_probability:
            layer.weight[row, column] = 0
        else:
            delta = rng.choice((-2, -1, 1, 2))
            old = int(layer.weight[row, column].item())
            layer.weight[row, column] = max(
                config.weight_min, min(config.weight_max, old + delta)
            )

    for _ in range(config.evolution_bias_mutations):
        layer = rng.choice(child.layers)
        index = rng.randrange(layer.bias.shape[0])
        delta = rng.choice((-2, -1, 1, 2))
        old = int(layer.bias[index].item())
        layer.bias[index] = max(1, old + delta)

    return repair_physical_bias_cap(child, config)[0]


def evolutionary_refinement(
    seed_model: DiscreteMCNN,
    loader: DataLoader,
    device: torch.device,
    config: RefinementConfig,
) -> tuple[DiscreteMCNN, dict[str, float]]:
    rng = random.Random(config.seed + 1000)
    seed_model = repair_physical_bias_cap(seed_model, config)[0]
    population = [seed_model.clone()]
    while len(population) < config.evolution_population:
        population.append(mutate_discrete(seed_model, rng, config))

    best_model = seed_model.clone()
    best_score, best_loss, best_acc = discrete_score(best_model, loader, device, config)

    for generation in range(1, config.evolution_generations + 1):
        scored = []
        for candidate in population:
            score, loss, acc = discrete_score(candidate, loader, device, config)
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
            or generation == config.evolution_generations
            or generation % config.report_every == 0
        ):
            print(
                f"evolution_generation={generation:03d} "
                f"best_score={best_score:.4f} best_acc={best_acc:.3f} "
                f"best_loss={best_loss:.4f} layer_sizes={best_model.layer_sizes()}"
            )

        elites = [
            candidate.clone()
            for _, _, _, candidate in scored[: config.evolution_elites]
        ]
        population = elites.copy()
        while len(population) < config.evolution_population:
            parent = rng.choice(elites)
            population.append(mutate_discrete(parent, rng, config))

    return best_model, {
        "loss": best_loss,
        "accuracy": best_acc,
        "score": best_score,
        "hidden_units": best_model.hidden_units(),
        "max_physical_bias": best_model.max_physical_bias(),
        "nonzero_weights": best_model.nonzero_weights(),
        "layer_sizes": best_model.layer_sizes(),
    }


# ---------------------------------------------------------------------------
# Local search refinement
# ---------------------------------------------------------------------------


def mutate_local_search_candidate(
    model: DiscreteMCNN, rng: random.Random, config: RefinementConfig
) -> DiscreteMCNN:
    from .discrete import remove_hidden_unit

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
                : max(1, min(config.local_search_candidate_window, len(costs)))
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
            config.weight_min, min(config.weight_max, old + rng.choice((-1, 1)))
        )
    else:
        layer = rng.choice(child.layers)
        index = rng.randrange(layer.bias.shape[0])
        old = int(layer.bias[index].item())
        layer.bias[index] = max(1, old + rng.choice((-2, -1, 1, 2)))

    return repair_physical_bias_cap(child, config)[0]


def local_search_refinement(
    seed_model: DiscreteMCNN,
    loader: DataLoader,
    device: torch.device,
    config: RefinementConfig,
) -> tuple[DiscreteMCNN, dict[str, float]]:
    rng = random.Random(config.seed + 2000)
    current = repair_physical_bias_cap(seed_model, config)[0]
    current_score, current_loss, current_acc = discrete_score(
        current, loader, device, config
    )
    best = current.clone()
    best_score = current_score
    best_loss = current_loss
    best_acc = current_acc
    min_acc = best_acc - config.local_search_accuracy_tolerance
    accepted = 0

    for step in range(1, config.local_search_steps + 1):
        candidate = mutate_local_search_candidate(current, rng, config)
        if not satisfies_physical_bias_cap(candidate, config):
            continue
        score, loss, acc = discrete_score(candidate, loader, device, config)
        temperature = max(
            0.0,
            config.local_search_temperature
            * (1.0 - step / max(1, config.local_search_steps)),
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
            or step == config.local_search_steps
            or step % config.report_every == 0
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
        "evaluated": config.local_search_steps,
        "loss": best_loss,
        "accuracy": best_acc,
        "score": best_score,
        "hidden_units": best.hidden_units(),
        "nonzero_weights": best.nonzero_weights(),
        "max_physical_bias": best.max_physical_bias(),
        "layer_sizes": best.layer_sizes(),
    }
