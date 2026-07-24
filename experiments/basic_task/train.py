"""Simplified MCNN pipeline: classic training -> quantization -> evolution -> export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from mcnn import (
    MCNNModel,
    PipelineConfig,
    cap_bias_signals,
    evaluate_model,
    evolutionary_refine,
    quantize_model,
    save_schematic,
    train_classic,
    validate_export_signals,
)


def _valid_shifts(img: np.ndarray, max_shift: int = 2) -> list[tuple[int, int]]:
    active = np.argwhere(img > 0)
    if len(active) == 0:
        return [(0, 0)]
    min_y, min_x = active.min(axis=0)
    max_y, max_x = active.max(axis=0)
    valid = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if 0 <= min_y + dy and max_y + dy < 8 and 0 <= min_x + dx and max_x + dx < 8:
                valid.append((dy, dx))
    if (0, 0) not in valid:
        valid.append((0, 0))
    return valid


def load_curated_dataset(
    validation_fraction: float,
    seed: int,
    augment: bool,
    bitflip_variants: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data = np.load("artifacts/curation/curated_dataset.npz")
    x_all = data["x"].astype(np.float32)
    y_all = data["y"].astype(np.int64)

    train_x, val_x, train_y, val_y = train_test_split(
        x_all,
        y_all,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_all,
    )

    raw_loader = DataLoader(
        TensorDataset(torch.tensor(x_all, dtype=torch.float32), torch.tensor(y_all, dtype=torch.long)),
        batch_size=256,
    )

    if augment:
        rng = np.random.RandomState(seed + 5000)
        aug_x: list[np.ndarray] = []
        aug_y: list[int] = []
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
    else:
        train_x_t = torch.tensor(train_x, dtype=torch.float32)
        train_y_t = torch.tensor(train_y, dtype=torch.long)

    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_x, dtype=torch.float32), torch.tensor(val_y, dtype=torch.long)),
        batch_size=128,
    )
    train_loader = DataLoader(TensorDataset(train_x_t, train_y_t), batch_size=128, shuffle=True)

    return train_loader, val_loader, raw_loader


@dataclass
class RunConfig:
    seed: int = 1
    hidden: list[int] = field(default_factory=lambda: [64])
    hidden_activation: str = "binary_step"

    validation_fraction: float = 0.2
    augment: bool = True
    bitflip_variants: int = 2

    ste_epochs: int = 80
    ste_learning_rate: float = 0.02
    ste_early_stop: int = 10
    signal_penalty_weight: float = 0.05
    signal_penalty_threshold: float = 800.0
    signal_penalty_warmup_epochs: int = 20

    enable_ablation: bool = False
    enable_evolution: bool = False
    evolution_generations: int = 10
    evolution_population: int = 24
    evolution_elites: int = 4
    evolution_weight_mutations: int = 24
    evolution_bias_mutations: int = 4
    evolution_zero_probability: float = 0.15
    evolution_remove_unit_probability: float = 0.1
    refinement_max_val_accuracy_drop: float = 0.0

    output_dir: str = "artifacts/run"
    export_schematic: bool = True


def run(cfg: RunConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cpu")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, raw_loader = load_curated_dataset(
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
        augment=cfg.augment,
        bitflip_variants=cfg.bitflip_variants,
    )

    model = MCNNModel(
        [64, *cfg.hidden, 10],
        hidden_activation=cfg.hidden_activation,
        output_activation="bounded_integer",
        output_min=0,
        output_max=15,
        weight_min=-15,
        weight_max=15,
        bias_min=-1024.0,
    ).to(device)

    pipeline = PipelineConfig(
        ste_epochs=cfg.ste_epochs,
        ste_learning_rate=cfg.ste_learning_rate,
        early_stop_patience=cfg.ste_early_stop,
        evolution_generations=cfg.evolution_generations,
        evolution_population=cfg.evolution_population,
        evolution_elites=cfg.evolution_elites,
        evolution_weight_mutations=cfg.evolution_weight_mutations,
        evolution_bias_mutations=cfg.evolution_bias_mutations,
        evolution_zero_probability=cfg.evolution_zero_probability,
        evolution_remove_unit_probability=cfg.evolution_remove_unit_probability,
        seed=cfg.seed,
        signal_penalty_weight=cfg.signal_penalty_weight,
        signal_penalty_threshold=cfg.signal_penalty_threshold,
        signal_penalty_warmup_epochs=cfg.signal_penalty_warmup_epochs,
    )

    print("\n[1/5] Classic training")
    train_stats = train_classic(model, train_loader, val_loader, device, pipeline)

    print("\n[2/5] Quantization")
    quantized_only = quantize_model(model)
    q_only_train_loss, q_only_train_acc = evaluate_model(
        quantized_only,
        raw_loader,
        device,
        mode="exact",
        loss_name=pipeline.loss,
        target_min=pipeline.target_min,
        target_max=pipeline.target_max,
        margin=pipeline.margin,
    )
    q_only_val_loss, q_only_val_acc = evaluate_model(
        quantized_only,
        val_loader,
        device,
        mode="exact",
        loss_name=pipeline.loss,
        target_min=pipeline.target_min,
        target_max=pipeline.target_max,
        margin=pipeline.margin,
    )

    quantized, bias_edits = cap_bias_signals(quantized_only)
    quantized_signal_check = validate_export_signals(quantized)

    q_capped_train_loss, q_capped_train_acc = evaluate_model(
        quantized,
        raw_loader,
        device,
        mode="exact",
        loss_name=pipeline.loss,
        target_min=pipeline.target_min,
        target_max=pipeline.target_max,
        margin=pipeline.margin,
    )
    q_capped_val_loss, q_capped_val_acc = evaluate_model(
        quantized,
        val_loader,
        device,
        mode="exact",
        loss_name=pipeline.loss,
        target_min=pipeline.target_min,
        target_max=pipeline.target_max,
        margin=pipeline.margin,
    )

    print("\n[3/5] Ablation")
    if cfg.enable_ablation:
        # Placeholder hook: currently no structural ablation pass in consolidated pipeline.
        ablated = quantized
        ablation_stats = {"enabled": True, "applied": False}
        print("ablation enabled, but no-op in current pipeline")
    else:
        ablated = quantized
        ablation_stats = {"enabled": False, "applied": False}
        print("ablation disabled")

    evolved = ablated
    evolution_stats = {
        "score": None,
        "accuracy": q_capped_val_acc,
        "enabled": False,
        "accepted": False,
    }

    print("\n[4/5] Evolutionary refinement")
    if cfg.enable_evolution and cfg.evolution_generations > 0:
        candidate, evo_metrics = evolutionary_refine(ablated, raw_loader, device, pipeline)
        candidate, final_bias_edits = cap_bias_signals(candidate)
        candidate_signal_check = validate_export_signals(candidate)
        _cand_val_loss, cand_val_acc = evaluate_model(
            candidate,
            val_loader,
            device,
            mode="exact",
            loss_name=pipeline.loss,
            target_min=pipeline.target_min,
            target_max=pipeline.target_max,
            margin=pipeline.margin,
        )

        min_allowed = q_capped_val_acc - cfg.refinement_max_val_accuracy_drop
        accepted = cand_val_acc >= min_allowed
        if accepted:
            evolved = candidate
            print(f"evolution accepted: val_acc {q_capped_val_acc:.3f} -> {cand_val_acc:.3f}")
        else:
            evolved = quantized
            print(
                f"evolution rejected by accuracy floor: baseline={q_capped_val_acc:.3f}, "
                f"candidate={cand_val_acc:.3f}, min_allowed={min_allowed:.3f}"
            )

        evolution_stats = {
            "score": float(evo_metrics["score"]),
            "accuracy": float(cand_val_acc),
            "enabled": True,
            "accepted": bool(accepted),
            "post_evolution_bias_edits": int(final_bias_edits),
            "signal_check": candidate_signal_check,
        }
    else:
        final_bias_edits = 0
        print("evolution disabled")

    print("\n[5/5] Final evaluation")
    train_loss, train_acc = evaluate_model(
        evolved,
        raw_loader,
        device,
        mode="exact",
        loss_name=pipeline.loss,
        target_min=pipeline.target_min,
        target_max=pipeline.target_max,
        margin=pipeline.margin,
    )
    val_loss, val_acc = evaluate_model(
        evolved,
        val_loader,
        device,
        mode="exact",
        loss_name=pipeline.loss,
        target_min=pipeline.target_min,
        target_max=pipeline.target_max,
        margin=pipeline.margin,
    )

    final_signal_check = validate_export_signals(evolved)
    if not bool(final_signal_check["ok"]):
        raise RuntimeError(
            "export signal validation failed: "
            f"violations={final_signal_check['violations']} "
            f"max_signal={final_signal_check['max_signal']}"
        )

    summary = {
        "train": {
            "loss": train_loss,
            "accuracy": train_acc,
        },
        "val": {
            "loss": val_loss,
            "accuracy": val_acc,
        },
        "best_epoch": train_stats["best_epoch"],
        "best_val_accuracy": train_stats["best_val_accuracy"],
        "best_exportable_val_accuracy": train_stats["best_exportable_val_accuracy"],
        "history": train_stats["history"],
        "post_quantization_only_train": {"loss": q_only_train_loss, "accuracy": q_only_train_acc},
        "post_quantization_only_val": {"loss": q_only_val_loss, "accuracy": q_only_val_acc},
        "post_quantization_capped_train": {"loss": q_capped_train_loss, "accuracy": q_capped_train_acc},
        "post_quantization_capped_val": {"loss": q_capped_val_loss, "accuracy": q_capped_val_acc},
        "post_quantization_signal_check": quantized_signal_check,
        "ablation": ablation_stats,
        "evolution": evolution_stats,
        "bias_signal_edits": int(bias_edits),
        "post_evolution_bias_edits": int(final_bias_edits),
        "nonzero_weights": int(evolved.nonzero_weights()),
        "max_bias_signal": int(evolved.max_bias_signal_strength()),
        "final_signal_check": final_signal_check,
        "layer_sizes": evolved.layer_sizes(),
    }

    print(f"final train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
    print(f"nonzero_weights={summary['nonzero_weights']} max_bias_signal={summary['max_bias_signal']}")
    print(f"layer_sizes={summary['layer_sizes']}")

    torch.save({"model": evolved.artifact_state()}, out / "model.pt")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "run_config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    if cfg.export_schematic:
        saved = save_schematic(evolved, out / "model")
        print(f"saved schematic={saved}")

    print(f"saved run directory: {out}")


if __name__ == "__main__":
    run(RunConfig())
