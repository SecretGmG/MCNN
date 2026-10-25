"""Distill a Connect Four teacher into the exact integer line architecture."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from mcnn.connect4_model import (
    ConnectFourTeacher,
    PhysicalConnectFourModel,
    expand_connect4_teacher,
)

from .train_teacher import TrainConfig, _loss, _selection_score, evaluate, load_data


@dataclass(frozen=True)
class PhysicalTrainConfig:
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.015
    distillation_weight: float = 0.25
    patience: int = 10
    pattern_levels: int = 2
    global_levels: int = 2
    mixing_levels: int = 2
    critical_sampling_weight: float = 3.0
    opening_sampling_weight: float = 2.0
    seed: int = 46001


def train(
    teacher_path: Path,
    dataset_path: Path,
    output_dir: Path,
    config: PhysicalTrainConfig,
    initial_model_path: Path | None = None,
):
    torch.manual_seed(config.seed)
    data_config = TrainConfig(batch_size=config.batch_size, seed=config.seed)
    x, scores, legal, _, ply, source, train_mask, validation_mask = load_data(
        dataset_path, data_config
    )
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    teacher = ConnectFourTeacher.load(teacher_path).eval()
    model = (
        PhysicalConnectFourModel.load(initial_model_path)
        if initial_model_path is not None
        else expand_connect4_teacher(
            teacher,
            pattern_levels=config.pattern_levels,
            global_levels=config.global_levels,
            mixing_levels=config.mixing_levels,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.1
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
    loader = DataLoader(
        TensorDataset(x[train_indices], train_scores, train_legal),
        batch_size=config.batch_size,
        sampler=WeightedRandomSampler(
            sampling_weights, len(sampling_weights), replacement=True
        ),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    history = []
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
            physical = model(batch_x, mode="ste")
            policy_loss = _loss(physical, batch_scores, batch_legal, data_config)
            with torch.no_grad():
                teacher_scores = teacher(batch_x)
                teacher_center = (
                    teacher_scores.masked_fill(~batch_legal, 0).sum(dim=1, keepdim=True)
                    / batch_legal.sum(dim=1, keepdim=True)
                )
            physical_center = (
                physical.masked_fill(~batch_legal, 0).sum(dim=1, keepdim=True)
                / batch_legal.sum(dim=1, keepdim=True)
            )
            distillation = (
                (
                    (physical - physical_center)
                    - (teacher_scores - teacher_center).detach()
                )
                .masked_fill(~batch_legal, 0)
                .square()
                .sum(dim=1)
                .div(batch_legal.sum(dim=1))
                .mean()
            )
            loss = policy_loss + config.distillation_weight * distillation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.load_state_dict(best_state)
    model.quantize_()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.pt")
    summary = {
        "config": asdict(config),
        "initial_model": str(initial_model_path) if initial_model_path else None,
        "selected_epoch": best_epoch,
        "teacher_validation": evaluate(
            teacher, x, scores, legal, validation_indices, ply, source
        ),
        "train": evaluate(model, x, scores, legal, train_indices, ply, source),
        "validation": evaluate(
            model, x, scores, legal, validation_indices, ply, source
        ),
        "physicality": model.physicality_metrics(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=PhysicalTrainConfig.epochs)
    parser.add_argument("--seed", type=int, default=PhysicalTrainConfig.seed)
    parser.add_argument("--learning-rate", type=float, default=PhysicalTrainConfig.learning_rate)
    parser.add_argument(
        "--distillation-weight", type=float, default=PhysicalTrainConfig.distillation_weight
    )
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--pattern-levels", type=int, default=PhysicalTrainConfig.pattern_levels)
    parser.add_argument("--global-levels", type=int, default=PhysicalTrainConfig.global_levels)
    parser.add_argument("--mixing-levels", type=int, default=PhysicalTrainConfig.mixing_levels)
    args = parser.parse_args()
    _, summary = train(
        args.teacher,
        args.dataset,
        args.output_dir,
        PhysicalTrainConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            distillation_weight=args.distillation_weight,
            pattern_levels=args.pattern_levels,
            global_levels=args.global_levels,
            mixing_levels=args.mixing_levels,
            seed=args.seed,
        ),
        args.initial_model,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
