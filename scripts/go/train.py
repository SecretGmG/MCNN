"""Train the canonical physical Minecraft network for 7x7 Go."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.common.plotting import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from mcnn.go_model import MinecraftGoConfig, MinecraftGoModel

from .data import GoValueDataset, d4_augment, load_value_dataset, split_parent_groups


@dataclass(frozen=True)
class TrainConfig:
    hidden_sizes: tuple[int, ...] = (128, 128, 128, 128)
    shift_depths: tuple[int, ...] = (12, 4, 4, 4, 4)
    relaxed_epochs: int = 30
    exact_epochs: int = 50
    learning_rate: float = 0.002
    final_learning_rate: float = 0.0002
    weight_decay: float = 1e-5
    groups_per_batch: int = 16
    absolute_weight: float = 0.5
    centered_weight: float = 1.0
    ranking_weight: float = 0.2
    significant_gap: float = 3.0
    quantization_weight: float = 0.02
    score_unit_points: float = 1.25
    split_seed: int = 8201
    seed: int = 26001
    patience: int = 18


def _group_batches(
    groups: list[np.ndarray], groups_per_batch: int, rng: np.random.Generator
) -> list[list[np.ndarray]]:
    order = rng.permutation(len(groups))
    return [
        [groups[int(index)] for index in order[start : start + groups_per_batch]]
        for start in range(0, len(order), groups_per_batch)
    ]


def _ranking_loss(prediction: Tensor, target: Tensor, gap: float) -> Tensor:
    target_delta = target[:, None] - target[None, :]
    prediction_delta = prediction[:, None] - prediction[None, :]
    selected = torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
    selected &= target_delta.abs() >= gap
    if not selected.any():
        return prediction.sum() * 0
    direction = target_delta[selected].sign()
    importance = (target_delta[selected].abs() / gap).clamp(max=5)
    return (importance * torch.nn.functional.softplus(-direction * prediction_delta[selected])).mean()


def _quantization_loss(model: MinecraftGoModel) -> Tensor:
    terms = []
    for layer in model.network.layers:
        weight = layer.weight.clamp(-1, 1)
        terms.append((weight - weight.round()).square().mean())
        threshold = layer.threshold
        terms.append(0.1 * (threshold - threshold.round()).square().mean())
    return torch.stack(terms).mean()


@torch.no_grad()
def predict(model: MinecraftGoModel, dataset: GoValueDataset, mode: str = "exact") -> Tensor:
    return torch.cat(
        [model.score_points(batch, mode=mode) for batch in dataset.boards.split(1024)]
    )


@torch.no_grad()
def metrics(
    model: MinecraftGoModel,
    dataset: GoValueDataset,
    groups: list[np.ndarray],
) -> dict[str, float]:
    prediction = predict(model, dataset, mode="exact")
    indices = np.concatenate(groups)
    error = prediction[indices] - dataset.positional_value[indices]
    centered_errors = []
    regrets = []
    pair_correct = 0
    pair_count = 0
    for group in groups:
        predicted_action = prediction[group] + dataset.captures[group]
        target_action = dataset.action_value[group]
        centered_errors.append(
            (predicted_action - predicted_action.mean())
            - (target_action - target_action.mean())
        )
        selected = int(torch.argmax(predicted_action))
        regrets.append(float(target_action.max() - target_action[selected]))
        target_delta = target_action[:, None] - target_action[None, :]
        prediction_delta = predicted_action[:, None] - predicted_action[None, :]
        pairs = torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
        pairs &= target_delta.abs() >= 3.0
        pair_correct += int(((target_delta[pairs] * prediction_delta[pairs]) > 0).sum())
        pair_count += int(pairs.sum())
    centered = torch.cat(centered_errors)
    regret = np.asarray(regrets)
    return {
        "parents": float(len(groups)),
        "positional_rmse_points": float(error.square().mean().sqrt()),
        "positional_mae_points": float(error.abs().mean()),
        "centered_rmse_points": float(centered.square().mean().sqrt()),
        "mean_regret_points": float(regret.mean()),
        "p90_regret_points": float(np.quantile(regret, 0.9)),
        "over_10_points": float((regret > 10).mean()),
        "significant_pair_accuracy": pair_correct / max(1, pair_count),
    }


def _selection_score(row: dict[str, float]) -> float:
    return row["mean_regret_points"] + 0.1 * row["centered_rmse_points"]


def _plot(history: list[dict[str, object]], output: Path, selected_epoch: int) -> None:
    epochs = [int(row["epoch"]) for row in history]
    exact_epoch = next(
        int(row["epoch"]) for row in history if row["phase"] == "exact"
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].plot(epochs, [row["validation"]["mean_regret_points"] for row in history])
    axes[0].set_ylabel("Mean regret (points)")
    axes[1].plot(epochs, [row["validation"]["centered_rmse_points"] for row in history])
    axes[1].set_ylabel("Centered RMSE")
    axes[2].plot(epochs, [row["temperature"] for row in history])
    axes[2].set_ylabel("Surrogate temperature")
    for axis in axes:
        axis.axvline(exact_epoch, color="#777777", linestyle="--", linewidth=1)
        axis.axvline(selected_epoch, color="#c44e52", linestyle=":", linewidth=1)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    axes[0].legend(
        ["validation", "exact phase", "selected"], frameon=False, fontsize=8
    )
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def train(
    dataset_paths: list[Path], output_dir: Path, config: TrainConfig
) -> tuple[MinecraftGoModel, dict[str, object]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    dataset = load_value_dataset(dataset_paths)
    train_groups, validation_groups = split_parent_groups(dataset, config.split_seed)
    model = MinecraftGoModel(
        MinecraftGoConfig(
            hidden_sizes=config.hidden_sizes,
            shift_depths=config.shift_depths,
            score_unit_points=config.score_unit_points,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_epochs = config.relaxed_epochs + config.exact_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=config.final_learning_rate
    )
    history: list[dict[str, object]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    selected_epoch = 0
    stale = 0

    for epoch in range(total_epochs + 1):
        progress = epoch / max(1, total_epochs)
        temperature = 3.0 * (0.75 / 3.0) ** progress
        if epoch < config.relaxed_epochs:
            quantization_progress = 0.0
        else:
            anneal = (epoch - config.relaxed_epochs) / max(1, config.exact_epochs)
            quantization_progress = 0.5 - 0.5 * np.cos(np.pi * min(1.0, anneal))
        for layer in model.network.layers:
            layer.surrogate_temperature = temperature
            layer.quantization_progress = quantization_progress
        if epoch % 2 == 0 or epoch == total_epochs:
            validation = metrics(model, dataset, validation_groups)
            row = {
                "epoch": epoch,
                "phase": "relaxed" if epoch < config.relaxed_epochs else "exact",
                "temperature": temperature,
                "quantization_progress": quantization_progress,
                "validation": validation,
            }
            history.append(row)
            score = _selection_score(validation)
            if epoch >= config.relaxed_epochs and score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                selected_epoch = epoch
                stale = 0
            elif epoch >= config.relaxed_epochs:
                stale += 2
            if config.patience and stale >= config.patience:
                break
        if epoch == total_epochs:
            break

        mode = "ste"
        model.train()
        for batch_groups in _group_batches(train_groups, config.groups_per_batch, rng):
            indices = np.concatenate(batch_groups)
            inputs = d4_augment(dataset.boards[indices], rng)
            prediction = model.score_points(inputs, mode=mode)
            lookup = {int(index): offset for offset, index in enumerate(indices.tolist())}
            losses = []
            for group in batch_groups:
                local = torch.tensor([lookup[int(index)] for index in group])
                predicted_position = prediction[local]
                target_position = dataset.positional_value[group].clamp(-75, 75)
                predicted_action = predicted_position + dataset.captures[group]
                target_action = dataset.action_value[group].clamp(-75, 75)
                absolute = torch.nn.functional.smooth_l1_loss(
                    predicted_position / 15, target_position / 15
                )
                centered = torch.nn.functional.mse_loss(
                    (predicted_action - predicted_action.mean()) / 15,
                    (target_action - target_action.mean()) / 15,
                )
                ranking = _ranking_loss(predicted_action, target_action, config.significant_gap) / 15
                losses.append(
                    config.absolute_weight * absolute
                    + config.centered_weight * centered
                    + config.ranking_weight * ranking
                )
            loss = torch.stack(losses).mean()
            loss = loss + (
                config.quantization_weight
                * quantization_progress
                * _quantization_loss(model)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                for layer in model.network.layers:
                    layer.weight.clamp_(-1.25, 1.25)
                    layer.threshold.clamp_(-model.network.config.accumulator_limit, model.network.config.accumulator_limit)
        scheduler.step()

    model.load_state_dict(best_state)
    model.project_().eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.pt")
    summary: dict[str, object] = {
        "training": asdict(config),
        "datasets": [str(path) for path in dataset_paths],
        "examples": len(dataset),
        "train_parents": len(train_groups),
        "validation_parents": len(validation_groups),
        "selected_epoch": selected_epoch,
        "train": metrics(model, dataset, train_groups),
        "validation": metrics(model, dataset, validation_groups),
        "physicality": model.physicality(),
    }
    (output_dir / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2) + "\n")
    (output_dir / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    _plot(history, output_dir / "training.png", selected_epoch)
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(TrainConfig.hidden_sizes))
    parser.add_argument("--shift-depths", type=int, nargs="+", default=list(TrainConfig.shift_depths))
    parser.add_argument("--relaxed-epochs", type=int, default=TrainConfig.relaxed_epochs)
    parser.add_argument("--exact-epochs", type=int, default=TrainConfig.exact_epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()
    config = TrainConfig(
        hidden_sizes=tuple(args.hidden_sizes),
        shift_depths=tuple(args.shift_depths),
        relaxed_epochs=args.relaxed_epochs,
        exact_epochs=args.exact_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    _, summary = train(args.dataset, args.output_dir, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
