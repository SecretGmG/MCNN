"""Train the continuous reflection-equivariant Connect Four teacher."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from mcnn.connect4_model import ConnectFourTeacher, ConnectFourTeacherConfig


@dataclass(frozen=True)
class TrainConfig:
    pattern_units: int = 16
    global_units: int = 64
    mixing_units: int = 32
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 0.002
    weight_decay: float = 1e-5
    centered_weight: float = 0.15
    target_temperature: float = 0.75
    patience: int = 10
    validation_fraction: float = 0.2
    critical_sampling_weight: float = 3.0
    opening_sampling_weight: float = 2.0
    seed: int = 45001


def load_data(path: Path, config: TrainConfig):
    with np.load(path) as data:
        x = torch.tensor(data["x"], dtype=torch.float32)
        scores = torch.tensor(data["scores"], dtype=torch.float32)
        legal = torch.tensor(data["legal"], dtype=torch.bool)
        games = data["game"].copy()
        ply = data["ply"].copy()
        source = data["source"].copy()
    unique_games = np.unique(games)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(unique_games)
    validation_games = set(
        unique_games[: max(1, round(config.validation_fraction * len(unique_games)))]
    )
    validation = np.asarray([game in validation_games for game in games])
    return x, scores, legal, games, ply, source, ~validation, validation


def _target_utility(scores: Tensor, legal: Tensor) -> Tensor:
    clipped = scores.clamp(-21, 21)
    utility = 4 * clipped.sign() + clipped / 21
    return utility.masked_fill(~legal, -1e9)


def _loss(
    logits: Tensor,
    scores: Tensor,
    legal: Tensor,
    config: TrainConfig,
) -> Tensor:
    masked_logits = logits.masked_fill(~legal, -1e9)
    utility = _target_utility(scores, legal)
    target = torch.softmax(utility / config.target_temperature, dim=1)
    policy = -(target * torch.log_softmax(masked_logits, dim=1)).sum(dim=1).mean()
    count = legal.sum(dim=1, keepdim=True)
    predicted_center = (masked_logits.masked_fill(~legal, 0).sum(dim=1, keepdim=True) / count)
    target_center = (utility.masked_fill(~legal, 0).sum(dim=1, keepdim=True) / count)
    centered_prediction = (masked_logits - predicted_center).masked_fill(~legal, 0)
    centered_target = (utility - target_center).masked_fill(~legal, 0)
    regression = ((centered_prediction - centered_target).square().sum(dim=1) / count[:, 0]).mean()
    return policy + config.centered_weight * regression


@torch.no_grad()
def evaluate(
    model: ConnectFourTeacher,
    x: Tensor,
    scores: Tensor,
    legal: Tensor,
    indices: np.ndarray,
    ply: np.ndarray,
    source: np.ndarray,
) -> dict[str, object]:
    selected_scores = []
    best_scores = []
    predictions = []
    for batch in np.array_split(indices, max(1, (len(indices) + 1023) // 1024)):
        logits = model(x[batch]).masked_fill(~legal[batch], -1e9)
        selected = logits.argmax(dim=1)
        predictions.append(selected)
        selected_scores.append(scores[batch, selected])
        best_scores.append(scores[batch].masked_fill(~legal[batch], -1e9).max(dim=1).values)
    selected_score = torch.cat(selected_scores)
    best_score = torch.cat(best_scores)
    selected = torch.cat(predictions)
    optimal = selected_score == best_score
    outcome_preserved = torch.where(
        best_score > 0,
        selected_score > 0,
        torch.where(best_score == 0, selected_score >= 0, torch.ones_like(optimal)),
    )
    regret = best_score - selected_score
    result: dict[str, object] = {
        "positions": len(indices),
        "optimal_move_rate": float(optimal.float().mean()),
        "outcome_preservation_rate": float(outcome_preserved.float().mean()),
        "mean_solver_regret": float(regret.mean()),
        "p95_solver_regret": float(torch.quantile(regret, 0.95)),
        "outcome_blunder_rate": float((~outcome_preserved).float().mean()),
    }
    local_ply = ply[indices]
    local_source = source[indices]
    for low, high, name in ((0, 10, "opening"), (10, 24, "middle"), (24, 43, "late")):
        mask = (local_ply >= low) & (local_ply < high)
        if mask.any():
            result[f"{name}_outcome_preservation"] = float(
                outcome_preserved[mask].float().mean()
            )
    for source_id, name in enumerate(("optimal", "noisy", "random", "anti")):
        mask = local_source == source_id
        if mask.any():
            result[f"{name}_outcome_preservation"] = float(
                outcome_preserved[mask].float().mean()
            )
    reflected_logits = model(x[indices[: min(2048, len(indices))]].flip(-1)).flip(1)
    original_logits = model(x[indices[: min(2048, len(indices))]])
    result["reflection_max_error"] = float((original_logits - reflected_logits).abs().max())
    return result


def _selection_score(metrics: dict[str, object]) -> float:
    return (
        100 * float(metrics["outcome_blunder_rate"])
        + float(metrics["mean_solver_regret"])
        + 0.1 * float(metrics["p95_solver_regret"])
    )


def train(
    dataset_path: Path,
    output_dir: Path,
    config: TrainConfig,
    initial_model_path: Path | None = None,
):
    torch.manual_seed(config.seed)
    x, scores, legal, _, ply, source, train_mask, validation_mask = load_data(
        dataset_path, config
    )
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if initial_model_path is not None:
        model = ConnectFourTeacher.load(initial_model_path)
    else:
        model = ConnectFourTeacher(
            ConnectFourTeacherConfig(
                pattern_units=config.pattern_units,
                global_units=config.global_units,
                mixing_units=config.mixing_units,
            )
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.05
    )
    train_scores = scores[train_indices]
    train_legal = legal[train_indices]
    best = train_scores.masked_fill(~train_legal, -1e9).max(dim=1).values
    preserves = torch.where(
        best[:, None] > 0,
        train_scores > 0,
        torch.where(best[:, None] == 0, train_scores >= 0, train_legal),
    ) | ~train_legal
    critical = ~preserves.all(dim=1)
    sampling_weights = torch.ones(len(train_indices))
    sampling_weights += critical * (config.critical_sampling_weight - 1)
    sampling_weights += torch.tensor(ply[train_indices] < 10) * (
        config.opening_sampling_weight - 1
    )
    sampler = WeightedRandomSampler(
        sampling_weights, len(sampling_weights), replacement=True
    )
    loader = DataLoader(
        TensorDataset(x[train_indices], train_scores, train_legal),
        batch_size=config.batch_size,
        sampler=sampler,
    )
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = []
    stale = 0
    for epoch in range(config.epochs + 1):
        validation = evaluate(
            model, x, scores, legal, validation_indices, ply, source
        )
        history.append({"epoch": epoch, **validation})
        score = _selection_score(validation)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == config.epochs or stale >= config.patience:
            break
        model.train()
        for batch_x, batch_scores, batch_legal in loader:
            loss = _loss(model(batch_x), batch_scores, batch_legal, config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.pt")
    summary = {
        "config": asdict(config),
        "initial_model": str(initial_model_path) if initial_model_path else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": best_epoch,
        "train": evaluate(model, x, scores, legal, train_indices, ply, source),
        "validation": evaluate(
            model, x, scores, legal, validation_indices, ply, source
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--pattern-units", type=int, default=TrainConfig.pattern_units)
    parser.add_argument("--global-units", type=int, default=TrainConfig.global_units)
    parser.add_argument("--mixing-units", type=int, default=TrainConfig.mixing_units)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--initial-model", type=Path)
    args = parser.parse_args()
    _, summary = train(
        args.dataset,
        args.output_dir,
        TrainConfig(
            pattern_units=args.pattern_units,
            global_units=args.global_units,
            mixing_units=args.mixing_units,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
        args.initial_model,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
