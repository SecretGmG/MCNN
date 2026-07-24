"""Minimal evaluation visuals for the consolidated MCNN pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mcnn import model_from_artifact_state


def load_run_model(run_dir: Path):
    artifact = torch.load(run_dir / "model.pt", map_location="cpu")
    return model_from_artifact_state(artifact["model"])


def load_curated_data() -> tuple[np.ndarray, np.ndarray]:
    data = np.load("artifacts/curation/curated_dataset.npz")
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, classes: int = 10) -> np.ndarray:
    cm = np.zeros((classes, classes), dtype=int)
    for t, p in zip(labels, preds, strict=True):
        cm[int(t), int(p)] += 1
    return cm


def save_confusion_plot(cm: np.ndarray, out_dir: Path) -> None:
    total = cm.sum()
    acc = cm.diagonal().sum() / total if total else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (acc={acc:.3f})")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value:
                color = "white" if value > cm.max() / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


def save_per_class_plot(labels: np.ndarray, preds: np.ndarray, out_dir: Path) -> None:
    totals = np.zeros(10, dtype=int)
    correct = np.zeros(10, dtype=int)

    for t, p in zip(labels, preds, strict=True):
        totals[int(t)] += 1
        if int(t) == int(p):
            correct[int(t)] += 1

    acc = correct / np.maximum(totals, 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(10), acc, color="#4c72b0")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Per-Class Accuracy")
    ax.set_xticks(range(10))

    for bar, score, total in zip(bars, acc, totals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.2f}\n(n={int(total)})",
            ha="center",
            fontsize=7,
        )

    fig.tight_layout()
    path = out_dir / "per_class_accuracy.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/run")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    model = load_run_model(run_dir)
    x_all, y_all = load_curated_data()

    x = torch.tensor(x_all)
    with torch.no_grad():
        logits = model(x, mode="exact")
    preds = logits.argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(y_all, preds)
    save_confusion_plot(cm, run_dir)
    save_per_class_plot(y_all, preds, run_dir)

    accuracy = cm.diagonal().sum() / max(1, cm.sum())
    print(f"accuracy={accuracy:.3f} ({cm.diagonal().sum()}/{cm.sum()})")


if __name__ == "__main__":
    main()
