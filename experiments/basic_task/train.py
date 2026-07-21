"""MCNN training pipeline — curated dataset with staged training.

Stages:
  1. STE training (straight-through estimator, soft quantization)
  2. Discrete conversion (exact quantization, bias cap repair)
  3. Structural ablation (remove redundant hidden units)
  4. Evolutionary refinement (discrete search)
  5. Export & visualization

Each stage logs train/val accuracy for plotting.

Usage:
    .venv/bin/python experiments/basic_task/train.py

To customize stages, training hyperparameters, or dataset, edit the config
dict at the bottom of this file.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mcnn import (
    DiscreteMCNN,
    RefinementConfig,
    TrainableMCNN,
    evaluate_discrete,
    evolutionary_refinement,
    repair_physical_bias_cap,
    save_schematic,
    structural_ablation,
)
from mcnn.refinement import loss_value


# ============================================================================
# Data
# ============================================================================

def _valid_shifts(img: np.ndarray, max_shift: int = 2) -> list[tuple[int, int]]:
    """Return all shifts in [-max_shift,max_shift]² that keep active bits in frame."""
    H, W = 8, 8
    active = np.argwhere(img > 0)
    if len(active) == 0:
        return [(0, 0)]
    min_y, min_x = active.min(axis=0)
    max_y, max_x = active.max(axis=0)
    valid = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if 0 <= min_y + dy and max_y + dy < H and 0 <= min_x + dx and max_x + dx < W:
                valid.append((dy, dx))
    if (0, 0) not in valid:
        valid.append((0, 0))
    return valid


def load_curated_dataset(
    validation_fraction: float = 0.2,
    seed: int = 1,
    augment: bool = True,
    bitflip_variants: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load curated dataset, split train/val, optionally augment with shifts+bitflips.

    Returns (train_loader, val_loader, raw_loader) where raw_loader is the
    non-augmented curated set used for discrete evaluation.
    """
    from sklearn.model_selection import train_test_split

    data = np.load("artifacts/curation/curated_dataset.npz")
    x_all = data["x"].astype(np.float32)
    y_all = data["y"].astype(np.int64)

    train_x, val_x, train_y, val_y = train_test_split(
        x_all, y_all, test_size=validation_fraction, random_state=seed, stratify=y_all,
    )

    print(f"curated dataset: train={len(train_x)}, val={len(val_x)}")

    # Non-augmented loader for discrete evaluation
    raw_train = TensorDataset(
        torch.tensor(x_all, dtype=torch.float32),
        torch.tensor(y_all, dtype=torch.long),
    )
    raw_loader = DataLoader(raw_train, batch_size=256)

    if augment:
        rng = np.random.RandomState(seed + 5000)
        aug_x, aug_y = [], []
        for i in range(len(train_x)):
            img = train_x[i].reshape(8, 8)
            label = int(train_y[i])
            for dy, dx in _valid_shifts(img):
                shifted = np.roll(img, (dy, dx), axis=(0, 1)).flatten()
                aug_x.append(shifted)
                aug_y.append(label)
                for _ in range(bitflip_variants):
                    flipped = shifted.copy()
                    flips = rng.choice(64, size=2, replace=False)
                    flipped[flips] = 1.0 - flipped[flips]
                    aug_x.append(flipped)
                    aug_y.append(label)

        train_x_t = torch.tensor(np.stack(aug_x), dtype=torch.float32)
        train_y_t = torch.tensor(aug_y, dtype=torch.long)
        print(f"augmented train: {len(train_x_t)} samples ({len(train_x)} base × ~{len(train_x_t)/len(train_x):.0f}x)")
    else:
        train_x_t = torch.tensor(train_x, dtype=torch.float32)
        train_y_t = torch.tensor(train_y, dtype=torch.long)

    val_xy = TensorDataset(
        torch.tensor(val_x, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.long),
    )

    return (
        DataLoader(TensorDataset(train_x_t, train_y_t), batch_size=128, shuffle=True),
        DataLoader(val_xy, batch_size=128),
        raw_loader,
    )


# ============================================================================
# Metrics
# ============================================================================

def accuracy(outputs: Tensor, labels: Tensor) -> float:
    return (outputs.argmax(dim=1) == labels).to(torch.float32).mean().item()


@torch.no_grad()
def evaluate_ste(model: TrainableMCNN, loader: DataLoader, device: torch.device, exact: bool):
    model.eval()
    tl, ta, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, exact=exact)
        loss = loss_value(out, y, loss_name="mse", target_min=0.0, target_max=15.0, margin=4.0)
        bs = x.shape[0]
        tl += loss.item() * bs
        ta += accuracy(out, y) * bs
        n += bs
    return tl / n, ta / n


@torch.no_grad()
def evaluate_discrete_both(model: DiscreteMCNN, train_loader, val_loader, device):
    train_loss, train_acc = evaluate_discrete(
        model, train_loader, device, loss_name="mse", target_min=0.0, target_max=15.0, margin=4.0)
    val_loss, val_acc = evaluate_discrete(
        model, val_loader, device, loss_name="mse", target_min=0.0, target_max=15.0, margin=4.0)
    return train_acc, val_acc


# ============================================================================
# Logging
# ============================================================================

@dataclass
class TrainingLog:
    records: list[dict] = field(default_factory=list)

    def add(self, stage: str, train_acc: float, val_acc: float, **extra):
        self.records.append({
            "stage": stage,
            "train_acc": round(train_acc, 5),
            "val_acc": round(val_acc, 5),
            **extra,
        })

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)


# ============================================================================
# Stages
# ============================================================================

@dataclass
class StageResult:
    model: DiscreteMCNN
    train_acc: float
    val_acc: float
    info: dict


def stage_ste_training(
    model: TrainableMCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    raw_loader: DataLoader,
    device: torch.device,
    log: TrainingLog,
    epochs: int = 80,
    learning_rate: float = 0.02,
    physicality_weight: float = 0.0,
    max_physical_bias: int = 897,
    early_stop_patience: int = 10,
) -> StageResult:
    """STE training with optional physicality penalty.

    If physicality_weight > 0, a penalty is added that pushes the model to
    have representable physical biases (bias + inactive subtractions <= cap).
    This can be increased gradually (annealed) across epochs to smoothly
    transition from soft to hard quantization.
    """
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_value(out, y, loss_name="mse", target_min=0.0, target_max=15.0, margin=4.0)

            if physicality_weight > 0:
                phys_penalty = _compute_physicality_cost(model, max_physical_bias)
                loss = loss + physicality_weight * phys_penalty

            loss.backward()
            opt.step()

        _train, train_acc = evaluate_ste(model, train_loader, device, exact=False)
        _, val_acc = evaluate_ste(model, val_loader, device, exact=True)
        log.add("ste_train", train_acc, val_acc, epoch=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  ste epoch={epoch:03d} train={train_acc:.3f} val={val_acc:.3f} best={best_acc:.3f}")

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    discrete = model.discrete()

    train_acc, val_acc = evaluate_discrete_both(discrete, raw_loader, val_loader, device)
    log.add("ste_final", train_acc, val_acc)

    info = {
        "best_epoch": best_epoch,
        "nonzero_weights": discrete.nonzero_weights(),
        "max_physical_bias": discrete.max_physical_bias(),
        "layer_sizes": discrete.layer_sizes(),
    }
    print(f"  ste final: val={val_acc:.3f} nonzero={info['nonzero_weights']} max_bias={info['max_physical_bias']}")
    return StageResult(discrete, train_acc, val_acc, info)


def _compute_physicality_cost(model: TrainableMCNN, max_physical_bias: int) -> Tensor:
    """Penalty for weights that push physical bias over the barrel cap."""
    penalties = []
    for layer_idx in range(len(model.layer_params)):
        weight = model._quantized_weight(layer_idx, exact=False)
        bias = model._positive_bias(layer_idx)
        inactive = torch.where(
            weight > 0,
            torch.full_like(weight, 15.0),
            torch.where(weight < 0, weight + 15.0, torch.zeros_like(weight)),
        )
        physical_bias = bias + inactive.sum(dim=1)
        penalties.append(F.relu(physical_bias - max_physical_bias).mean())
    if not penalties:
        return torch.zeros((), device=next(model.parameters()).device)
    return sum(penalties)


def stage_repair(
    discrete: DiscreteMCNN,
    raw_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    log: TrainingLog,
) -> StageResult:
    """Repair physical bias cap violations by zeroing the worst weights."""
    config = RefinementConfig()
    repaired, zeroed = repair_physical_bias_cap(discrete, config)
    train_acc, val_acc = evaluate_discrete_both(repaired, raw_loader, val_loader, device)
    log.add("repair", train_acc, val_acc, zeroed_weights=zeroed)

    info = {
        "zeroed_weights": zeroed,
        "nonzero_weights": repaired.nonzero_weights(),
        "max_physical_bias": repaired.max_physical_bias(),
        "layer_sizes": repaired.layer_sizes(),
    }
    print(f"  repair: zeroed={zeroed} val={val_acc:.3f} max_bias={info['max_physical_bias']}")
    return StageResult(repaired, train_acc, val_acc, info)


def stage_ablation(
    discrete: DiscreteMCNN,
    raw_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    log: TrainingLog,
    passes: int = 1,
    accuracy_tolerance: float = 0.0,
) -> StageResult:
    """Remove redundant hidden units."""
    config = RefinementConfig(ablation_passes=passes, ablation_accuracy_tolerance=accuracy_tolerance)
    pruned, metrics = structural_ablation(discrete, raw_loader, device, config)
    train_acc, val_acc = evaluate_discrete_both(pruned, raw_loader, val_loader, device)
    log.add("ablation", train_acc, val_acc,
            removed=metrics["accepted"],
            layer_sizes=pruned.layer_sizes())

    info = {
        "removed": metrics["accepted"],
        "evaluated": metrics["evaluated"],
        "nonzero_weights": pruned.nonzero_weights(),
        "max_physical_bias": pruned.max_physical_bias(),
        "layer_sizes": pruned.layer_sizes(),
    }
    print(f"  ablation: removed={info['removed']} val={val_acc:.3f} layers={info['layer_sizes']}")
    return StageResult(pruned, train_acc, val_acc, info)


def stage_evolution(
    discrete: DiscreteMCNN,
    raw_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    log: TrainingLog,
    generations: int = 10,
    population: int = 24,
    elites: int = 4,
    weight_mutations: int = 24,
    bias_mutations: int = 4,
    zero_probability: float = 0.15,
    remove_unit_probability: float = 0.1,
    seed: int = 1,
) -> StageResult:
    """Evolutionary refinement of the discrete model."""
    config = RefinementConfig(
        evolution_generations=generations,
        evolution_population=population,
        evolution_elites=elites,
        evolution_weight_mutations=weight_mutations,
        evolution_bias_mutations=bias_mutations,
        evolution_zero_probability=zero_probability,
        evolution_remove_unit_probability=remove_unit_probability,
        seed=seed,
        report_every=1,
    )
    evolved, metrics = evolutionary_refinement(discrete, raw_loader, device, config)
    train_acc, val_acc = evaluate_discrete_both(evolved, raw_loader, val_loader, device)
    log.add("evolution", train_acc, val_acc,
            score=metrics.get("score"),
            layer_sizes=evolved.layer_sizes())

    info = {
        "score": metrics.get("score"),
        "nonzero_weights": evolved.nonzero_weights(),
        "max_physical_bias": evolved.max_physical_bias(),
        "layer_sizes": evolved.layer_sizes(),
    }
    print(f"  evolution: val={val_acc:.3f} score={info['score']:.4f} layers={info['layer_sizes']}")
    return StageResult(evolved, train_acc, val_acc, info)


# ============================================================================
# Pipeline
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the full training pipeline."""
    # Data
    seed: int = 1
    validation_fraction: float = 0.2
    augment: bool = True
    bitflip_variants: int = 2

    # Model
    hidden: list[int] = field(default_factory=lambda: [64])
    hidden_activation: str = "binary_step"

    # STE training
    ste_epochs: int = 80
    ste_learning_rate: float = 0.02
    ste_early_stop: int = 10
    physicality_weight: float = 0.0
    max_physical_bias: int = 897

    # Stages (set to 0/false to skip)
    enable_repair: bool = True
    enable_ablation: bool = True
    ablation_passes: int = 1
    enable_evolution: bool = True
    evolution_generations: int = 10
    evolution_population: int = 24
    evolution_elites: int = 4
    evolution_weight_mutations: int = 24
    evolution_bias_mutations: int = 4
    evolution_zero_probability: float = 0.15
    evolution_remove_unit_probability: float = 0.1

    # Output
    output_dir: str = "artifacts/run"
    export_schematic: bool = False


def run_pipeline(cfg: PipelineConfig):
    device = torch.device("cpu")
    log = TrainingLog()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    # --- Data ---
    train_loader, val_loader, raw_loader = load_curated_dataset(
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
        augment=cfg.augment,
        bitflip_variants=cfg.bitflip_variants,
    )

    # --- Model ---
    model = TrainableMCNN(
        input_features=64,
        hidden_sizes=cfg.hidden,
        output_features=10,
        weight_min=-14, weight_max=14,
        output_min=0, output_max=15,
        hidden_activation=cfg.hidden_activation,
    ).to(device)

    # --- 1. STE Training ---
    print("\n[1/5] STE Training")
    result = stage_ste_training(
        model, train_loader, val_loader, raw_loader, device, log,
        epochs=cfg.ste_epochs,
        learning_rate=cfg.ste_learning_rate,
        physicality_weight=cfg.physicality_weight,
        max_physical_bias=cfg.max_physical_bias,
        early_stop_patience=cfg.ste_early_stop,
    )
    discrete = result.model

    # --- 2. Repair ---
    if cfg.enable_repair:
        print("\n[2/5] Physical Bias Repair")
        result = stage_repair(discrete, raw_loader, val_loader, device, log)
        discrete = result.model

    # --- 3. Ablation ---
    if cfg.enable_ablation:
        print("\n[3/5] Structural Ablation")
        result = stage_ablation(
            discrete, raw_loader, val_loader, device, log,
            passes=cfg.ablation_passes,
        )
        discrete = result.model

    # --- 4. Evolution ---
    if cfg.enable_evolution:
        print("\n[4/5] Evolutionary Refinement")
        result = stage_evolution(
            discrete, raw_loader, val_loader, device, log,
            generations=cfg.evolution_generations,
            population=cfg.evolution_population,
            elites=cfg.evolution_elites,
            weight_mutations=cfg.evolution_weight_mutations,
            bias_mutations=cfg.evolution_bias_mutations,
            zero_probability=cfg.evolution_zero_probability,
            remove_unit_probability=cfg.evolution_remove_unit_probability,
            seed=cfg.seed,
        )
        discrete = result.model

    # --- 5. Final ---
    print("\n[5/5] Final Evaluation")
    train_acc, val_acc = evaluate_discrete_both(discrete, raw_loader, val_loader, device)
    log.add("final", train_acc, val_acc)

    final_info = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "nonzero_weights": discrete.nonzero_weights(),
        "max_physical_bias": discrete.max_physical_bias(),
        "layer_sizes": discrete.layer_sizes(),
    }
    print(f"  train={train_acc:.3f} val={val_acc:.3f}")
    print(f"  nonzero={final_info['nonzero_weights']} max_bias={final_info['max_physical_bias']}")
    print(f"  layers={final_info['layer_sizes']}")

    # --- Save ---
    log.save(out / "log.json")
    torch.save({"model": discrete.state_dict(), "config": cfg.__dict__}, out / "model.pt")

    if cfg.export_schematic:
        print("\n[export] Schematic")
        schem_path = save_schematic(discrete, out / "model")
        print(f"  saved {schem_path}")

    print(f"\nsaved to {out}/")


# ============================================================================
# Default config — edit here to customize
# ============================================================================

if __name__ == "__main__":
    cfg = PipelineConfig(
        seed=1,
        hidden=[64],
        hidden_activation="binary_step",
        ste_epochs=80,
        ste_learning_rate=0.02,
        ste_early_stop=10,
        physicality_weight=0.0,
        ablation_passes=1,
        evolution_generations=10,
        output_dir="artifacts/run",
        export_schematic=True,
    )
    run_pipeline(cfg)
