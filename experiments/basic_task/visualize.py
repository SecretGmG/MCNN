"""Visualize MCNN training results: metrics, confusion matrix, false predictions, debug trace.

Usage:
    .venv/bin/python experiments/basic_task/visualize.py --run-dir artifacts/run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mcnn import DiscreteMCNN, discrete_from_state_dict


def _load_model_and_data(run_dir: Path):
    artifact = torch.load(run_dir / "model.pt", map_location="cpu")
    model = discrete_from_state_dict(artifact["model"])
    return model


def plot_confusion_matrix(model: DiscreteMCNN, labels: np.ndarray, preds: np.ndarray, out_dir: Path):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    total = cm.sum()
    correct = cm.diagonal().sum()
    acc = correct / total if total > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Confusion Matrix (accuracy={acc:.3f}, {total} samples)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"saved {out_dir / 'confusion_matrix.png'}")
    return cm


def plot_false_predictions(model: DiscreteMCNN, images: np.ndarray, labels: np.ndarray, preds: np.ndarray, out_dir: Path, max_examples: int = 20):
    false_mask = preds != labels
    false_indices = np.where(false_mask)[0]
    if len(false_indices) == 0:
        print("No false predictions!")
        return
    rng = np.random.RandomState(42)
    selected = rng.choice(false_indices, size=min(max_examples, len(false_indices)), replace=False)
    n = len(selected)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    for idx, sel in enumerate(selected):
        r, c = idx // cols, idx % cols
        binary = (images[sel] > 4).astype(float).reshape(8, 8)
        axes[r, c].imshow(binary, cmap="gray_r", vmin=0, vmax=1)
        axes[r, c].set_title(f"True:{labels[sel]} Pred:{preds[sel]}", fontsize=9)
        axes[r, c].axis("off")
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")
    fig.suptitle(f"False Predictions ({len(false_indices)} total, showing {n})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "false_predictions.png", dpi=120)
    plt.close(fig)
    print(f"saved {out_dir / 'false_predictions.png'}")


def plot_per_class_accuracy(labels: np.ndarray, preds: np.ndarray, out_dir: Path):
    per_class_correct = np.zeros(10, dtype=int)
    per_class_total = np.zeros(10, dtype=int)
    for t, p in zip(labels, preds):
        per_class_total[t] += 1
        if t == p:
            per_class_correct[t] += 1
    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(10), per_class_acc, color="steelblue", edgecolor="white")
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(10))
    ax.axhline(y=per_class_correct.sum() / per_class_total.sum(), color="r", ls="--", alpha=0.5, label="overall")
    for i, (bar, acc, total) in enumerate(zip(bars, per_class_acc, per_class_total)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{acc:.2f}\n(n={total})", ha="center", fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_accuracy.png", dpi=120)
    plt.close(fig)
    print(f"saved {out_dir / 'per_class_accuracy.png'}")


def export_debug_trace(model: DiscreteMCNN, out_dir: Path):
    """Export one correct input-output pair per class showing hidden activations."""
    data = np.load("artifacts/curation/curated_dataset.npz")
    x_all = data["x"].astype(np.float32)
    y_all = data["y"].astype(np.int64)

    lines = ["MCNN Debug Trace — one correct prediction per class", "=" * 60]

    for digit in range(10):
        mask = y_all == digit
        indices = np.where(mask)[0]
        found = None
        for idx in indices:
            inp = torch.tensor(x_all[idx].reshape(1, 64))
            with torch.no_grad():
                out = model.forward(inp)
            if out.argmax(dim=1).item() == digit:
                found = idx
                break
        if found is None:
            lines.append(f"\nDigit {digit}: no correct prediction found")
            continue

        inp = torch.tensor(x_all[found].reshape(1, 64))
        with torch.no_grad():
            out = model.forward(inp)
        out_clamped = out.round().clamp(0, 15).squeeze().int().tolist()

        lines.append(f"\nDigit {digit} (correct) — sample #{found}")
        lines.append("  Input (8×8 binary):")
        for row in range(8):
            line = "    "
            for col in range(8):
                line += "█" if int(x_all[found, row * 8 + col]) else "·"
            lines.append(line)

        lines.append("  Output (clamped 0–15):")
        for d in range(10):
            marker = " <-- correct" if d == digit else ""
            bar = "█" * int(min(out_clamped[d], 30))
            lines.append(f"    {d}: {out_clamped[d]:2d} {bar}{marker}")

        # Hidden layer activations (binary step: pre > 0 → 1)
        hidden = inp.squeeze().int()
        for layer_idx, layer in enumerate(model.layers):
            w = layer.weight.to(dtype=hidden.dtype)
            b = layer.bias.to(dtype=hidden.dtype)
            pre = (w @ hidden + b).int()
            active = (pre > 0).int()
            lines.append(f"  Hidden layer {layer_idx} ({active.shape[0]} units, {active.sum().item()} active):")
            if active.sum() == 0:
                lines.append("    (all inactive)")
            else:
                shown = 0
                for u in range(active.shape[0]):
                    if active[u].item():
                        if shown < 20:
                            lines.append(f"    h{u:02d}: pre={pre[u].item():+4d} → active")
                        shown += 1
                if shown > 20:
                    lines.append(f"    ... ({shown} total active)")
            hidden = active

    trace = "\n".join(lines)
    (out_dir / "debug_trace.txt").write_text(trace)
    print(f"saved {out_dir / 'debug_trace.txt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/run")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    model = _load_model_and_data(run_dir)

    # Load curated data for evaluation
    data = np.load("artifacts/curation/curated_dataset.npz")
    x_all = data["x"].astype(np.float32)
    y_all = data["y"].astype(np.int64)
    x = torch.tensor(x_all)
    with torch.no_grad():
        outputs = model.forward(x)
    preds = outputs.argmax(dim=1).numpy()

    cm = plot_confusion_matrix(model, y_all, preds, run_dir)
    plot_false_predictions(model, x_all, y_all, preds, run_dir)
    plot_per_class_accuracy(y_all, preds, run_dir)
    export_debug_trace(model, run_dir)

    # Print summary
    total_correct = cm.diagonal().sum()
    total = cm.sum()
    print(f"\nAccuracy: {total_correct}/{total} = {total_correct/total:.3f}")
    print(f"All saved to {run_dir}/")


if __name__ == "__main__":
    main()
