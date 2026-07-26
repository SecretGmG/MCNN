"""Evaluate a saved MCNN run and write simple diagnostic plots."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mcnn import model_from_artifact_state


DEFAULT_DATASET = Path("artifacts/curation/curated_dataset.npz")
DEFAULT_RUN_DIR = Path("artifacts/run")
CLASS_COUNT = 10


@dataclass(frozen=True)
class Evaluation:
    labels: np.ndarray
    predictions: np.ndarray
    confusion: np.ndarray

    @property
    def correct(self) -> int:
        return int(np.trace(self.confusion))

    @property
    def total(self) -> int:
        return int(self.confusion.sum())

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def per_class_total(self) -> np.ndarray:
        return self.confusion.sum(axis=1)

    @property
    def per_class_correct(self) -> np.ndarray:
        return self.confusion.diagonal()

    @property
    def per_class_accuracy(self) -> np.ndarray:
        return np.divide(
            self.per_class_correct,
            np.maximum(self.per_class_total, 1),
            dtype=np.float64,
        )

    def to_summary(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
            "per_class": [
                {
                    "class": label,
                    "accuracy": float(self.per_class_accuracy[label]),
                    "correct": int(self.per_class_correct[label]),
                    "total": int(self.per_class_total[label]),
                }
                for label in range(CLASS_COUNT)
            ],
        }


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


def load_model(path: Path) -> torch.nn.Module:
    artifact = torch.load(path, map_location="cpu")
    return model_from_artifact_state(artifact["model"])


def predict(model: torch.nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
            logits = model(batch, mode="exact")
            predictions.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(predictions) if predictions else np.array([], dtype=np.int64)


def build_confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=True):
        confusion[int(label), int(prediction)] += 1
    return confusion


def evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> Evaluation:
    predictions = predict(model, x, batch_size)
    confusion = build_confusion(y, predictions)
    return Evaluation(labels=y, predictions=predictions, confusion=confusion)


def save_confusion_plot(evaluation: Evaluation, path: Path) -> None:
    confusion = evaluation.confusion

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_title(f"Confusion Matrix (accuracy={evaluation.accuracy:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(CLASS_COUNT))
    ax.set_yticks(range(CLASS_COUNT))

    threshold = confusion.max() / 2 if confusion.size else 0
    for row in range(CLASS_COUNT):
        for col in range(CLASS_COUNT):
            value = int(confusion[row, col])
            if value == 0:
                continue
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=7)

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_per_class_plot(evaluation: Evaluation, path: Path) -> None:
    totals = evaluation.per_class_total
    accuracies = evaluation.per_class_accuracy

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(CLASS_COUNT), accuracies, color="#4c72b0")
    ax.set_title("Per-Class Accuracy")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(CLASS_COUNT))

    for bar, accuracy, total in zip(bars, accuracies, totals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{accuracy:.2f}\n(n={int(total)})",
            ha="center",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_training_summary(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_run_config(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run_config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _accuracy(history: dict[str, object], key: str) -> list[float]:
    values = history[key]
    if not isinstance(values, list):
        raise TypeError(f"history field {key!r} must be a list")
    return [_number(value) for value in values]


def _epochs(history: dict[str, object]) -> list[int]:
    values = history["epoch"]
    if not isinstance(values, list):
        raise TypeError("history field 'epoch' must be a list")
    return [int(_number(value)) for value in values]


def _penalty_schedule(epochs: list[int], run_config: dict[str, object]) -> list[float]:
    weight = _number(run_config.get("signal_penalty_weight"))
    warmup_epochs = int(_number(run_config.get("signal_penalty_warmup_epochs")))
    if warmup_epochs <= 0:
        return [weight for _epoch in epochs]
    return [weight * min(1.0, epoch / warmup_epochs) for epoch in epochs]


def _label_stage(ax: plt.Axes, epoch: float, label: str, y: float, color: str) -> None:
    ax.axvline(epoch, color=color, linestyle=":", linewidth=1.2, alpha=0.75)
    ax.text(
        epoch,
        y,
        label,
        rotation=90,
        va="top",
        ha="right",
        color=color,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.8},
    )


def save_accuracy_history_plot(
    summary: dict[str, object],
    run_config: dict[str, object],
    path: Path,
) -> bool:
    history = summary.get("history")
    if not isinstance(history, dict):
        return False

    required = [
        "epoch",
        "train_accuracy",
        "val_accuracy",
        "exportable_train_accuracy",
        "exportable_val_accuracy",
    ]
    if any(name not in history for name in required):
        return False

    epochs = _epochs(history)
    train = _accuracy(history, "train_accuracy")
    val = _accuracy(history, "val_accuracy")
    quantized_train = _accuracy(history, "exportable_train_accuracy")
    quantized_val = _accuracy(history, "exportable_val_accuracy")
    penalty = _penalty_schedule(epochs, run_config)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(epochs, train, marker="o", linewidth=1.8, label="Train")
    ax.plot(epochs, val, marker="o", linewidth=1.8, label="Validation")
    ax.plot(epochs, quantized_train, marker="s", linewidth=1.5, label="Quantized train")
    ax.plot(epochs, quantized_val, marker="s", linewidth=1.5, label="Quantized validation")
    ax.set_title("Training, Quantization, and Physicality Constraints")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    best_epoch = summary.get("best_epoch")
    if isinstance(best_epoch, int | float):
        _label_stage(ax, float(best_epoch), "best exportable", 1.02, "#2f6f3e")

    final_epoch = epochs[-1]
    _label_stage(ax, final_epoch, "quantize + repair", 0.96, "#6a4c93")

    evolution = summary.get("evolution")
    if isinstance(evolution, dict) and evolution.get("enabled"):
        status = "evolution accepted" if evolution.get("accepted") else "evolution rejected"
        _label_stage(ax, final_epoch + 0.35, status, 0.9, "#8a5a00")
    else:
        ax.text(
            0.99,
            0.08,
            "evolution disabled",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    signal_check = summary.get("final_signal_check")
    if isinstance(signal_check, dict):
        max_signal = signal_check.get("max_signal", "?")
        violations = signal_check.get("violations", "?")
        ax.text(
            0.01,
            0.08,
            f"export signal: max={max_signal}, violations={violations}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    penalty_ax = ax.twinx()
    penalty_ax.plot(
        epochs,
        penalty,
        color="#b45f06",
        linestyle="--",
        linewidth=1.6,
        label="Signal penalty weight",
    )
    penalty_ax.set_ylabel("Physicality penalty weight")
    penalty_max = max(penalty) if penalty else 0.0
    penalty_ax.set_ylim(0.0, max(0.01, penalty_max * 1.6))

    signal_threshold = run_config.get("signal_penalty_threshold")
    if signal_threshold is not None:
        penalty_ax.text(
            0.99,
            0.92,
            f"signal threshold={signal_threshold}",
            transform=penalty_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#8a4b08",
        )

    lines, labels = ax.get_legend_handles_labels()
    penalty_lines, penalty_labels = penalty_ax.get_legend_handles_labels()
    ax.legend(lines + penalty_lines, labels + penalty_labels, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.run_dir / "model.pt"

    if not model_path.exists():
        raise SystemExit(f"Model artifact not found: {model_path}")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    model = load_model(model_path)
    x, y = load_dataset(args.dataset)
    evaluation = evaluate(model, x, y, args.batch_size)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_plot(evaluation, args.run_dir / "confusion_matrix.png")
    save_per_class_plot(evaluation, args.run_dir / "per_class_accuracy.png")
    summary = load_training_summary(args.run_dir)
    run_config = load_run_config(args.run_dir)
    saved_history = (
        save_accuracy_history_plot(summary, run_config, args.run_dir / "accuracy_history.png")
        if summary is not None
        else False
    )

    summary = evaluation.to_summary()
    with open(args.run_dir / "evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"accuracy={evaluation.accuracy:.3f} ({evaluation.correct}/{evaluation.total})")
    print(f"saved {args.run_dir / 'confusion_matrix.png'}")
    print(f"saved {args.run_dir / 'per_class_accuracy.png'}")
    if saved_history:
        print(f"saved {args.run_dir / 'accuracy_history.png'}")
    else:
        print(f"skipped accuracy history plot; no training history found in {args.run_dir / 'summary.json'}")
    print(f"saved {args.run_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
