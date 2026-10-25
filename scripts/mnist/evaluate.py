"""Evaluate a saved MCNN digit model and write diagnostic plots."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.common.plotting import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import torch

from mcnn import model_from_artifact_state


DEFAULT_RUN_DIR = Path("artifacts/shift_digits_64_64_10")
DEFAULT_DATASET = Path("artifacts/curation/curated_dataset.npz")
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


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


def load_model(path: Path) -> torch.nn.Module:
    artifact = torch.load(path, map_location="cpu")
    return model_from_artifact_state(artifact["model"])


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def predict(model: torch.nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
            logits = model(batch, mode="exact")
            predictions.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(predictions) if predictions else np.array([], dtype=np.int64)


def evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> Evaluation:
    predictions = predict(model, x, batch_size)
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for label, prediction in zip(y, predictions, strict=True):
        confusion[int(label), int(prediction)] += 1
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


def _show_digit(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image.reshape(8, 8), cmap="Greys", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def save_augmentation_examples_plot(
    x: np.ndarray,
    y: np.ndarray,
    config: dict[str, object],
    path: Path,
) -> bool:
    if not bool(config.get("augment", False)):
        return False

    seed = int(_number(config.get("seed"), 1.0))
    bitflip_variants = int(_number(config.get("bitflip_variants"), 0.0))
    rng = np.random.RandomState(seed + 5000)

    examples: list[tuple[np.ndarray, str]] = []
    source_indices: list[int] = []
    seen_labels: set[int] = set()
    for index, label in enumerate(y.tolist()):
        label = int(label)
        if label in seen_labels:
            continue
        source_indices.append(index)
        seen_labels.add(label)
        if len(source_indices) == 4:
            break

    for source_index in source_indices:
        image = x[source_index].reshape(8, 8)
        label = int(y[source_index])
        examples.append((image.flatten(), f"label={label}\noriginal"))

        shifts = _valid_shifts(image)
        chosen_shifts = [shift for shift in shifts if shift != (0, 0)][:2]
        if len(chosen_shifts) < 2:
            chosen_shifts.extend(shifts[: 2 - len(chosen_shifts)])

        for dy, dx in chosen_shifts[:2]:
            shifted = np.roll(image, (dy, dx), axis=(0, 1)).flatten()
            examples.append((shifted, f"shift\n{dy:+d},{dx:+d}"))

        if bitflip_variants > 0:
            shifted = np.roll(image, chosen_shifts[0] if chosen_shifts else (0, 0), axis=(0, 1)).flatten()
            flipped = shifted.copy()
            flip = rng.randint(64)
            flipped[flip] = 1.0 - flipped[flip]
            examples.append((flipped, "shift +\n1 bit flip"))

    if not examples:
        return False

    cols = 4
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    for ax, (image, title) in zip(axes_array.flatten(), examples, strict=False):
        _show_digit(ax, image, title)
    for ax in axes_array.flatten()[len(examples) :]:
        ax.axis("off")

    fig.suptitle("Augmentation Examples")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def save_mistake_examples_plot(
    x: np.ndarray,
    evaluation: Evaluation,
    path: Path,
    *,
    max_examples: int = 24,
) -> bool:
    mistakes = np.flatnonzero(evaluation.labels != evaluation.predictions)
    if len(mistakes) == 0:
        return False

    chosen = mistakes[:max_examples]
    cols = 6
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.7))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)

    for ax, index in zip(axes_array.flatten(), chosen, strict=False):
        true = int(evaluation.labels[index])
        pred = int(evaluation.predictions[index])
        _show_digit(ax, x[index], f"true={true}\npred={pred}")
    for ax in axes_array.flatten()[len(chosen) :]:
        ax.axis("off")

    fig.suptitle("Mistake Examples")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _series(history: dict[str, object], key: str) -> list[float]:
    values = history[key]
    if not isinstance(values, list):
        raise TypeError(f"history field {key!r} must be a list")
    return [_number(value) for value in values]


def _epochs(history: dict[str, object]) -> list[int]:
    return [int(value) for value in _series(history, "epoch")]


def _penalty_schedule(epochs: list[int], config: dict[str, object]) -> list[float]:
    weight = _number(config.get("signal_penalty_weight"))
    warmup_epochs = int(_number(config.get("signal_penalty_warmup_epochs")))
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
    config: dict[str, object],
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
    train = _series(history, "train_accuracy")
    val = _series(history, "val_accuracy")
    quantized_train = _series(history, "exportable_train_accuracy")
    quantized_val = _series(history, "exportable_val_accuracy")
    penalty = _penalty_schedule(epochs, config)

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
        ax.text(
            0.01,
            0.08,
            f"export signal: max={signal_check.get('max_signal', '?')}, "
            f"violations={signal_check.get('violations', '?')}",
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

    signal_threshold = config.get("signal_penalty_threshold")
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


def save_physicality_history_plot(summary: dict[str, object], path: Path) -> bool:
    history = summary.get("history")
    if not isinstance(history, dict):
        return False
    required = ["epoch", "active_weight_modules", "max_signal"]
    if any(name not in history for name in required):
        return False

    epochs = _epochs(history)
    modules = _series(history, "active_weight_modules")
    max_signal = _series(history, "max_signal")
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.plot(epochs, modules, color="#4c72b0", marker="o", linewidth=1.6, label="Active weight modules")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active weight modules")
    ax.grid(True, axis="y", alpha=0.25)

    signal_ax = ax.twinx()
    signal_ax.plot(epochs, max_signal, color="#b45f06", marker="s", linewidth=1.4, label="Maximum signal")
    signal_ax.set_ylabel("Maximum barrel signal")
    signal_ax.axhline(897, color="#7f7f7f", linestyle=":", linewidth=1, label="Export limit")

    lines, labels = ax.get_legend_handles_labels()
    signal_lines, signal_labels = signal_ax.get_legend_handles_labels()
    ax.legend(lines + signal_lines, labels + signal_labels, loc="best")
    ax.set_title("Physicality During Training")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def save_annealing_history_plot(summary: dict[str, object], path: Path) -> bool:
    history = summary.get("history")
    if not isinstance(history, dict):
        return False
    required = [
        "epoch",
        "exportable_val_accuracy",
        "shifted_val_accuracy",
        "surrogate_temperature",
        "integer_attraction_weight",
    ]
    if any(name not in history for name in required):
        return False

    epochs = _epochs(history)
    val_accuracy = _series(history, "exportable_val_accuracy")
    shifted_accuracy = _series(history, "shifted_val_accuracy")
    temperature = _series(history, "surrogate_temperature")
    integer_weight = _series(history, "integer_attraction_weight")

    fig, (performance_ax, schedule_ax) = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    performance_ax.plot(epochs, val_accuracy, marker="o", linewidth=1.7, label="Centered validation")
    performance_ax.plot(epochs, shifted_accuracy, marker="s", linewidth=1.5, label="Shifted validation")
    performance_ax.set_ylabel("Exact accuracy")
    performance_ax.set_ylim(0.0, 1.05)
    performance_ax.grid(True, axis="y", alpha=0.25)
    performance_ax.legend(loc="lower right")
    performance_ax.set_title("Annealing Schedule and Exportable Model Performance")

    schedule_ax.plot(epochs, temperature, color="#b45f06", linewidth=1.8, label="Surrogate temperature")
    schedule_ax.set_xlabel("Epoch")
    schedule_ax.set_ylabel("Temperature")
    schedule_ax.grid(True, axis="y", alpha=0.25)
    integer_ax = schedule_ax.twinx()
    integer_ax.plot(
        epochs,
        integer_weight,
        color="#6a4c93",
        linestyle="--",
        linewidth=1.6,
        label="Integer attraction weight",
    )
    integer_ax.set_ylabel("Integer attraction weight")
    lines, labels = schedule_ax.get_legend_handles_labels()
    integer_lines, integer_labels = integer_ax.get_legend_handles_labels()
    schedule_ax.legend(lines + integer_lines, labels + integer_labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.run_dir / "config.json")
    dataset = args.dataset or Path(str(config.get("dataset", DEFAULT_DATASET)))
    model_path = args.run_dir / "model.pt"
    plots_dir = args.run_dir / "plots"

    if not model_path.exists():
        raise SystemExit(f"Model artifact not found: {model_path}")
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    model = load_model(model_path)
    x, y = load_dataset(dataset)
    evaluation = evaluate(model, x, y, args.batch_size)

    plots_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_plot(evaluation, plots_dir / "confusion_matrix.png")
    save_per_class_plot(evaluation, plots_dir / "per_class_accuracy.png")
    saved_augmentation_examples = save_augmentation_examples_plot(
        x,
        y,
        config,
        plots_dir / "augmentation_examples.png",
    )
    saved_mistake_examples = save_mistake_examples_plot(
        x,
        evaluation,
        plots_dir / "mistake_examples.png",
    )

    summary = load_json(args.run_dir / "summary.json")
    saved_history = save_accuracy_history_plot(
        summary,
        config,
        plots_dir / "accuracy_history.png",
    )
    saved_physicality_history = save_physicality_history_plot(
        summary,
        plots_dir / "physicality_history.png",
    )
    saved_annealing_history = save_annealing_history_plot(
        summary,
        plots_dir / "annealing_history.png",
    )

    with open(args.run_dir / "evaluation.json", "w") as f:
        json.dump(evaluation.to_summary(), f, indent=2)

    print(f"accuracy={evaluation.accuracy:.3f} ({evaluation.correct}/{evaluation.total})")
    print(f"saved {plots_dir / 'confusion_matrix.png'}")
    print(f"saved {plots_dir / 'per_class_accuracy.png'}")
    if saved_augmentation_examples:
        print(f"saved {plots_dir / 'augmentation_examples.png'}")
    else:
        print("skipped augmentation examples plot; augmentation is disabled")
    if saved_mistake_examples:
        print(f"saved {plots_dir / 'mistake_examples.png'}")
    else:
        print("skipped mistake examples plot; no mistakes found")
    if saved_history:
        print(f"saved {plots_dir / 'accuracy_history.png'}")
    else:
        print(f"skipped accuracy history plot; no training history found in {args.run_dir / 'summary.json'}")
    if saved_physicality_history:
        print(f"saved {plots_dir / 'physicality_history.png'}")
    if saved_annealing_history:
        print(f"saved {plots_dir / 'annealing_history.png'}")
    print(f"saved {args.run_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
