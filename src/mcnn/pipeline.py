from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .export import MAX_SIGNAL_STRENGTH, compute_signal_strengths
from .model import MCNNModel


@dataclass
class PipelineConfig:
    loss: str = "mse"
    target_min: float = 0.0
    target_max: float = 15.0
    margin: float = 4.0

    ste_epochs: int = 80
    ste_learning_rate: float = 0.02
    early_stop_patience: int = 10

    evolution_generations: int = 10
    evolution_population: int = 24
    evolution_elites: int = 4
    evolution_weight_mutations: int = 24
    evolution_bias_mutations: int = 4
    evolution_zero_probability: float = 0.15
    evolution_remove_unit_probability: float = 0.1

    hidden_unit_penalty: float = 0.0
    nonzero_weight_penalty: float = 0.0
    bias_signal_penalty: float = 0.0
    signal_penalty_weight: float = 0.05
    signal_penalty_threshold: float = 800.0
    signal_penalty_warmup_epochs: int = 20
    weight_magnitude_penalty: float = 0.0
    selection_shifted_accuracy_weight: float = 0.0
    selection_signal_excess_weight: float = 0.0
    selection_active_weight_weight: float = 0.0
    selection_active_weight_scale: float = 5000.0
    surrogate_temperature_start: float | None = None
    surrogate_temperature_end: float | None = None
    surrogate_temperature_anneal_epochs: int = 0
    integer_attraction_start_weight: float = 0.0
    integer_attraction_end_weight: float = 0.0
    integer_attraction_start_epoch: int = 1
    integer_attraction_anneal_epochs: int = 0
    identity_warmup_epochs: int = 0
    identity_warmup_layer_indices: tuple[int, ...] = ()
    seed: int = 1
    report_every: int = 1


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

    targets = torch.full((labels.shape[0], outputs.shape[1]), target_min, device=labels.device)
    targets = targets.scatter(1, labels.unsqueeze(1), target_max)
    return F.mse_loss(outputs, targets)


@torch.no_grad()
def evaluate_model(
    model: MCNNModel,
    loader: DataLoader,
    device: torch.device,
    *,
    mode: str,
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
        outputs = model(x, mode=mode)
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


def _export_signal_penalty(model: MCNNModel, threshold: float) -> Tensor:
    """Differentiable surrogate penalty for export bias signal overflow.

    Matches export semantics: zero weights contribute no module, so they add no
    bias-signal penalty. Penalizes signals above `threshold` and below zero.
    """
    penalties: list[Tensor] = []
    for layer_index in range(len(model.weights)):
        weight = model._quantized_weight(layer_index, ste=True)
        bias = model._quantized_bias(layer_index, ste=True)

        out_features, _in_features = weight.shape
        row_indices = torch.arange(out_features, device=weight.device)
        even_mask = (row_indices % 2 == 0).to(weight.dtype)
        odd_mask = 1.0 - even_mask

        active = (weight != 0).to(weight.dtype)
        even_signal = bias + ((15.0 + weight) * active).sum(dim=1)
        odd_signal = bias + (15.0 * active).sum(dim=1)
        signal = even_mask * even_signal + odd_mask * odd_signal

        penalties.append(torch.relu(signal - threshold).mean())
        penalties.append(torch.relu(-signal).mean())

    if not penalties:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(penalties).sum()


def _weight_magnitude_penalty(model: MCNNModel) -> Tensor:
    """Encourage exact-zero weights without changing the discrete forward path."""
    penalties = [
        model._quantized_weight(layer_index, ste=True).abs().mean()
        for layer_index in range(len(model.weights))
    ]
    if not penalties:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(penalties).mean()


def _integer_attraction_penalty(model: MCNNModel) -> Tensor:
    """Periodic penalty whose minima are exactly the integer lattice."""
    parameters = [*model.weights, *model.biases]
    penalties = [torch.sin(torch.pi * parameter).square().mean() for parameter in parameters]
    if not penalties:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(penalties).mean()


def _linear_anneal(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(1.0, max(0.0, progress))


def _surrogate_temperature(config: PipelineConfig, epoch: int) -> float | None:
    if config.surrogate_temperature_start is None:
        return None
    end = config.surrogate_temperature_end or config.surrogate_temperature_start
    if config.surrogate_temperature_anneal_epochs <= 1:
        return end
    progress = (epoch - 1) / (config.surrogate_temperature_anneal_epochs - 1)
    return _linear_anneal(config.surrogate_temperature_start, end, progress)


def _integer_attraction_weight(config: PipelineConfig, epoch: int) -> float:
    if epoch < config.integer_attraction_start_epoch:
        return 0.0
    if config.integer_attraction_anneal_epochs <= 1:
        return config.integer_attraction_end_weight
    progress = (epoch - config.integer_attraction_start_epoch) / (
        config.integer_attraction_anneal_epochs - 1
    )
    return _linear_anneal(
        config.integer_attraction_start_weight,
        config.integer_attraction_end_weight,
        progress,
    )


def physicality_metrics(model: MCNNModel, *, signal_target: float) -> dict[str, float | int]:
    """Return exact export-relevant cost measurements for a discrete model."""
    layers = model.discrete_layers()
    total_weight_slots = sum(int(layer.weight.numel()) for layer in layers)
    active_weight_modules = sum(int(layer.weight.count_nonzero().item()) for layer in layers)
    signals = [
        int(signal)
        for bias_signals, _network_signals in compute_signal_strengths(model)
        for signal in bias_signals.tolist()
    ]
    max_signal = max(signals, default=0)
    signal_excess = sum(max(0.0, signal - signal_target) for signal in signals)
    return {
        "total_weight_slots": total_weight_slots,
        "active_weight_modules": active_weight_modules,
        "active_weight_fraction": active_weight_modules / max(1, total_weight_slots),
        "max_signal": max_signal,
        "signal_headroom": MAX_SIGNAL_STRENGTH - max_signal,
        "signal_excess": signal_excess,
    }


def selection_score(
    centered_accuracy: float,
    shifted_accuracy: float | None,
    model: MCNNModel,
    config: PipelineConfig,
) -> tuple[float, dict[str, float | int]]:
    """Score an exportable model by deployment quality and physical cost."""
    shifted_weight = config.selection_shifted_accuracy_weight if shifted_accuracy is not None else 0.0
    quality = (1.0 - shifted_weight) * centered_accuracy + shifted_weight * (shifted_accuracy or 0.0)
    metrics = physicality_metrics(model, signal_target=config.signal_penalty_threshold)
    score = quality
    score -= config.selection_signal_excess_weight * float(metrics["signal_excess"]) / max(
        1, MAX_SIGNAL_STRENGTH
    )
    score -= config.selection_active_weight_weight * float(metrics["active_weight_modules"]) / max(
        1.0, config.selection_active_weight_scale
    )
    return score, metrics


def validate_export_signals(
    model: MCNNModel,
    *,
    max_signal: int = MAX_SIGNAL_STRENGTH,
) -> dict[str, int | bool]:
    signal_pairs = compute_signal_strengths(model)
    violations = 0
    max_seen = 0
    min_seen = 0
    total = 0

    for bias_signals, _network in signal_pairs:
        for value in bias_signals.tolist():
            signal = int(value)
            max_seen = max(max_seen, signal)
            min_seen = min(min_seen, signal)
            total += 1
            if signal < 0 or signal > max_signal:
                violations += 1

    return {
        "ok": violations == 0,
        "violations": violations,
        "max_signal": max_seen,
        "min_signal": min_seen,
        "count": total,
    }


def train_classic(
    model: MCNNModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: PipelineConfig,
    shifted_val_loader: DataLoader | None = None,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ste_learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_selection_score = float("-inf")
    best_exportable_acc = float("-inf")
    best_raw_acc = float("-inf")
    best_epoch = 0
    no_improve = 0

    history_epoch: list[int] = []
    history_train_accuracy: list[float] = []
    history_val_accuracy: list[float] = []
    history_exportable_train_accuracy: list[float] = []
    history_exportable_val_accuracy: list[float] = []
    history_shifted_val_accuracy: list[float] = []
    history_selection_score: list[float] = []
    history_active_weight_modules: list[int] = []
    history_max_signal: list[int] = []
    history_surrogate_temperature: list[float] = []
    history_integer_attraction_weight: list[float] = []

    for epoch in range(1, config.ste_epochs + 1):
        temperature = _surrogate_temperature(config, epoch)
        model.set_binary_surrogate_temperature_(temperature)
        integer_weight = _integer_attraction_weight(config, epoch)
        warmup_active = epoch <= config.identity_warmup_epochs
        for layer_index in config.identity_warmup_layer_indices:
            model.weights[layer_index].requires_grad_(not warmup_active)
            model.biases[layer_index].requires_grad_(not warmup_active)

        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, mode="ste")
            loss = loss_value(
                outputs,
                y,
                loss_name=config.loss,
                target_min=config.target_min,
                target_max=config.target_max,
                margin=config.margin,
            )
            if config.signal_penalty_weight > 0:
                if config.signal_penalty_warmup_epochs > 0:
                    ramp = min(1.0, epoch / float(config.signal_penalty_warmup_epochs))
                else:
                    ramp = 1.0
                penalty_weight = config.signal_penalty_weight * ramp
                signal_penalty = _export_signal_penalty(model, config.signal_penalty_threshold)
                loss = loss + penalty_weight * signal_penalty
            if config.weight_magnitude_penalty > 0:
                loss = loss + config.weight_magnitude_penalty * _weight_magnitude_penalty(model)
            if integer_weight > 0:
                loss = loss + integer_weight * _integer_attraction_penalty(model)

            loss.backward()
            optimizer.step()

        _, train_acc = evaluate_model(
            model,
            train_loader,
            device,
            mode="exact",
            loss_name=config.loss,
            target_min=config.target_min,
            target_max=config.target_max,
            margin=config.margin,
        )
        _, val_acc = evaluate_model(
            model,
            val_loader,
            device,
            mode="exact",
            loss_name=config.loss,
            target_min=config.target_min,
            target_max=config.target_max,
            margin=config.margin,
        )

        exportable_epoch, _ = cap_bias_signals(quantize_model(model))
        _, exportable_train_acc = evaluate_model(
            exportable_epoch,
            train_loader,
            device,
            mode="exact",
            loss_name=config.loss,
            target_min=config.target_min,
            target_max=config.target_max,
            margin=config.margin,
        )
        _, exportable_val_acc = evaluate_model(
            exportable_epoch,
            val_loader,
            device,
            mode="exact",
            loss_name=config.loss,
            target_min=config.target_min,
            target_max=config.target_max,
            margin=config.margin,
        )
        shifted_val_acc: float | None = None
        if shifted_val_loader is not None:
            _, shifted_val_acc = evaluate_model(
                exportable_epoch,
                shifted_val_loader,
                device,
                mode="exact",
                loss_name=config.loss,
                target_min=config.target_min,
                target_max=config.target_max,
                margin=config.margin,
            )
        epoch_score, epoch_physicality = selection_score(
            exportable_val_acc,
            shifted_val_acc,
            exportable_epoch,
            config,
        )

        history_epoch.append(epoch)
        history_train_accuracy.append(train_acc)
        history_val_accuracy.append(val_acc)
        history_exportable_train_accuracy.append(exportable_train_acc)
        history_exportable_val_accuracy.append(exportable_val_acc)
        history_shifted_val_accuracy.append(shifted_val_acc if shifted_val_acc is not None else exportable_val_acc)
        history_selection_score.append(epoch_score)
        history_active_weight_modules.append(int(epoch_physicality["active_weight_modules"]))
        history_max_signal.append(int(epoch_physicality["max_signal"]))
        history_surrogate_temperature.append(temperature if temperature is not None else 0.0)
        history_integer_attraction_weight.append(integer_weight)

        if epoch_score > best_selection_score:
            best_selection_score = epoch_score
            best_exportable_acc = exportable_val_acc
            best_raw_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch == config.ste_epochs or epoch % 10 == 0:
            print(
                f"ste epoch={epoch:03d} train_acc={train_acc:.3f} raw_val={val_acc:.3f} "
                f"exportable_val={exportable_val_acc:.3f} score={epoch_score:.3f} "
                f"best_score={best_selection_score:.3f}"
            )

        if config.early_stop_patience > 0 and no_improve >= config.early_stop_patience:
            print(f"ste early stop at epoch={epoch}")
            break

    model.load_state_dict(best_state)
    model.set_binary_surrogate_temperature_(None)
    for layer_index in config.identity_warmup_layer_indices:
        model.weights[layer_index].requires_grad_(True)
        model.biases[layer_index].requires_grad_(True)
    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_raw_acc,
        "best_exportable_val_accuracy": best_exportable_acc,
        "best_selection_score": best_selection_score,
        "history": {
            "epoch": history_epoch,
            "train_accuracy": history_train_accuracy,
            "val_accuracy": history_val_accuracy,
            "exportable_train_accuracy": history_exportable_train_accuracy,
            "exportable_val_accuracy": history_exportable_val_accuracy,
            "shifted_val_accuracy": history_shifted_val_accuracy,
            "selection_score": history_selection_score,
            "active_weight_modules": history_active_weight_modules,
            "max_signal": history_max_signal,
            "surrogate_temperature": history_surrogate_temperature,
            "integer_attraction_weight": history_integer_attraction_weight,
        },
    }


def quantize_model(model: MCNNModel) -> MCNNModel:
    quantized = model.clone()
    discrete = quantized.discrete_layers()
    quantized.apply_discrete_layers_(discrete)
    return quantized


def cap_bias_signals(model: MCNNModel) -> tuple[MCNNModel, int]:
    """Repair export barrel signals using `compute_signal_strengths` as truth."""
    adjusted = quantize_model(model)
    layers = adjusted.discrete_layers()
    edits = 0

    # Phase 1: bias-only correction from true export signal computation.
    for _ in range(6):
        signal_pairs = compute_signal_strengths(adjusted)
        changed = False

        for layer_index, (bias_signals, _network) in enumerate(signal_pairs):
            layer = layers[layer_index]
            for output_index, signal_value in enumerate(bias_signals.tolist()):
                signal = int(signal_value)
                if signal > MAX_SIGNAL_STRENGTH:
                    layer.bias[output_index] = int(layer.bias[output_index].item()) - (signal - MAX_SIGNAL_STRENGTH)
                    edits += 1
                    changed = True
                elif signal < 0:
                    layer.bias[output_index] = int(layer.bias[output_index].item()) + (-signal)
                    edits += 1
                    changed = True

        if not changed:
            break
        adjusted.apply_discrete_layers_(layers)

    # Phase 2: if even rows still overflow, reduce weights minimally.
    for _ in range(4):
        signal_pairs = compute_signal_strengths(adjusted)
        changed = False

        for layer_index, (bias_signals, _network) in enumerate(signal_pairs):
            layer = layers[layer_index]
            for output_index, signal_value in enumerate(bias_signals.tolist()):
                signal = int(signal_value)
                if signal <= MAX_SIGNAL_STRENGTH or output_index % 2 != 0:
                    continue

                remaining = signal - MAX_SIGNAL_STRENGTH
                row = layer.weight[output_index]
                order = sorted(
                    range(row.shape[0]),
                    key=lambda col: int(row[col].item()),
                    reverse=True,
                )
                for col in order:
                    if remaining <= 0:
                        break
                    old = int(row[col].item())
                    if old <= adjusted.weight_min:
                        continue
                    reducible = old - adjusted.weight_min
                    step = min(reducible, remaining)
                    row[col] = old - step
                    remaining -= step
                    edits += 1
                    changed = True

        if not changed:
            break
        adjusted.apply_discrete_layers_(layers)

    adjusted.apply_discrete_layers_(layers)
    return adjusted, edits


def remove_hidden_unit(model: MCNNModel, layer_index: int, unit_index: int) -> MCNNModel:
    if layer_index < 0 or layer_index >= len(model.weights) - 1:
        raise ValueError("only hidden layers can be pruned")

    pruned = model.clone()

    with torch.no_grad():
        layer_weight = pruned.weights[layer_index].detach().clone()
        layer_bias = pruned.biases[layer_index].detach().clone()
        next_weight = pruned.weights[layer_index + 1].detach().clone()

        if unit_index < 0 or unit_index >= layer_weight.shape[0]:
            raise ValueError("unit_index out of range")

        keep_out = torch.ones(layer_weight.shape[0], dtype=torch.bool)
        keep_out[unit_index] = False
        keep_in = torch.ones(next_weight.shape[1], dtype=torch.bool)
        keep_in[unit_index] = False

        pruned.weights[layer_index] = torch.nn.Parameter(layer_weight[keep_out])
        pruned.biases[layer_index] = torch.nn.Parameter(layer_bias[keep_out])
        pruned.weights[layer_index + 1] = torch.nn.Parameter(next_weight[:, keep_in])

    return pruned


def _score(model: MCNNModel, loader: DataLoader, device: torch.device, config: PipelineConfig) -> tuple[float, float, float]:
    loss, acc = evaluate_model(
        model,
        loader,
        device,
        mode="exact",
        loss_name=config.loss,
        target_min=config.target_min,
        target_max=config.target_max,
        margin=config.margin,
    )
    metrics = physicality_metrics(model, signal_target=config.signal_penalty_threshold)
    overflow = max(0, int(metrics["max_signal"]) - MAX_SIGNAL_STRENGTH)
    score = (
        acc
        - config.hidden_unit_penalty * model.hidden_units()
        - config.nonzero_weight_penalty * model.nonzero_weights()
        - config.bias_signal_penalty * overflow / max(1, MAX_SIGNAL_STRENGTH)
        - config.selection_signal_excess_weight * float(metrics["signal_excess"]) / max(1, MAX_SIGNAL_STRENGTH)
        - config.selection_active_weight_weight
        * float(metrics["active_weight_modules"])
        / max(1.0, config.selection_active_weight_scale)
    )
    return score, loss, acc


def _mutate_discrete(model: MCNNModel, rng: random.Random, config: PipelineConfig) -> MCNNModel:
    child = model.clone()

    if rng.random() < config.evolution_remove_unit_probability and child.hidden_units() > 0:
        candidate_layers = [
            index for index in range(len(child.weights) - 1) if child.weights[index].shape[0] > 1
        ]
        if candidate_layers:
            layer_index = rng.choice(candidate_layers)
            unit_index = rng.randrange(child.weights[layer_index].shape[0])
            child = remove_hidden_unit(child, layer_index, unit_index)

    with torch.no_grad():
        for _ in range(config.evolution_weight_mutations):
            layer_index = rng.randrange(len(child.weights))
            weight = child.weights[layer_index]
            row = rng.randrange(weight.shape[0])
            col = rng.randrange(weight.shape[1])
            old = int(weight[row, col].item())
            if rng.random() < config.evolution_zero_probability:
                new = 0
            else:
                new = max(child.weight_min, min(child.weight_max, old + rng.choice((-2, -1, 1, 2))))
            weight[row, col] = new

        for _ in range(config.evolution_bias_mutations):
            layer_index = rng.randrange(len(child.biases))
            bias = child.biases[layer_index]
            row = rng.randrange(bias.shape[0])
            old = int(bias[row].item())
            bias[row] = max(0, old + rng.choice((-2, -1, 1, 2)))

    child = quantize_model(child)
    child, _ = cap_bias_signals(child)
    return child


def evolutionary_refine(
    seed_model: MCNNModel,
    loader: DataLoader,
    device: torch.device,
    config: PipelineConfig,
) -> tuple[MCNNModel, dict[str, float]]:
    rng = random.Random(config.seed + 1000)

    seed = quantize_model(seed_model)
    seed, _ = cap_bias_signals(seed)

    population = [seed.clone()]
    while len(population) < config.evolution_population:
        population.append(_mutate_discrete(seed, rng, config))

    best = seed.clone()
    best_score, best_loss, best_acc = _score(best, loader, device, config)

    for generation in range(1, config.evolution_generations + 1):
        scored: list[tuple[float, float, float, MCNNModel]] = []
        for candidate in population:
            score, loss, acc = _score(candidate, loader, device, config)
            scored.append((score, loss, acc, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)

        if scored[0][0] > best_score:
            best_score, best_loss, best_acc, best = (
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
                f"evolution generation={generation:03d} best_score={best_score:.4f} "
                f"best_acc={best_acc:.3f} best_loss={best_loss:.4f} "
                f"layer_sizes={best.layer_sizes()}"
            )

        elites = [candidate.clone() for _, _, _, candidate in scored[: config.evolution_elites]]
        population = elites.copy()
        while len(population) < config.evolution_population:
            population.append(_mutate_discrete(rng.choice(elites), rng, config))

    return best, {
        "score": best_score,
        "loss": best_loss,
        "accuracy": best_acc,
        "hidden_units": float(best.hidden_units()),
        "nonzero_weights": float(best.nonzero_weights()),
        "max_bias_signal": float(best.max_bias_signal_strength()),
    }
