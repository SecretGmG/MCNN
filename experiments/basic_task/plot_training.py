"""Plot training curve from a pipeline log.json.

Usage: .venv/bin/python experiments/basic_task/plot_training.py [--run-dir artifacts/run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGE_COLORS = {
    "ste_train": "#4c72b0",
    "ste_final": "#4c72b0",
    "repair": "#dd8452",
    "ablation": "#55a868",
    "evolution": "#c44e52",
    "final": "#8172b3",
}
STAGE_LABELS = {
    "ste_train": "STE training (epochs)",
    "ste_final": "STE → discrete",
    "repair": "Bias cap repair",
    "ablation": "Structural ablation",
    "evolution": "Evolution",
    "final": "Final",
}


def plot(run_dir: str):
    out = Path(run_dir)

    with open(out / "log.json") as f:
        records = json.load(f)

    # Split: ste_train epochs (continuous line) vs stage markers (vertical bars)
    ste_epochs = [r for r in records if r["stage"] == "ste_train"]
    stages = [r for r in records if r["stage"] != "ste_train"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: STE training curve ---
    if ste_epochs:
        epochs = [r["epoch"] for r in ste_epochs]
        train = [r["train_acc"] for r in ste_epochs]
        val = [r["val_acc"] for r in ste_epochs]

        ax1.plot(epochs, train, color="#4c72b0", alpha=0.3, linewidth=1, label="Train (STE)")
        ax1.plot(epochs, val, color="#4c72b0", linewidth=2, label="Val (exact)")

        best_idx = np.argmax(val)
        ax1.axvline(epochs[best_idx], color="#4c72b0", ls="--", alpha=0.3)
        ax1.annotate(f"best={val[best_idx]:.3f} @ ep {epochs[best_idx]}",
                     xy=(epochs[best_idx], val[best_idx]),
                     xytext=(epochs[best_idx] + 5, val[best_idx] - 0.03),
                     fontsize=8, color="#4c72b0")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("STE Training")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        y_min = max(0.5, min(min(train), min(val)) - 0.05)
        y_max = min(1.0, max(max(train), max(val)) + 0.05)
        ax1.set_ylim(y_min, y_max)
    else:
        ax1.text(0.5, 0.5, "No STE training data", ha="center", va="center")
        ax1.set_title("STE Training")

    # --- Right: post-training stages ---
    stage_names = []
    stage_train = []
    stage_val = []
    stage_colors = []

    for r in stages:
        name = r["stage"]
        label = STAGE_LABELS.get(name, name)
        stage_names.append(label)
        stage_train.append(r["train_acc"])
        stage_val.append(r["val_acc"])
        stage_colors.append(STAGE_COLORS.get(name, "#888888"))

    if stage_names:
        x = np.arange(len(stage_names))
        w = 0.35

        bars1 = ax2.bar(x - w / 2, stage_train, w, label="Train", color=stage_colors,
                        edgecolor="white", alpha=0.85)
        bars2 = ax2.bar(x + w / 2, stage_val, w, label="Val (curated)", color=stage_colors,
                        edgecolor="white")

        for bar, val in zip(bars1, stage_train):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", fontsize=7)
        for bar, val in zip(bars2, stage_val):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", fontsize=7)

        ax2.set_xticks(x)
        ax2.set_xticklabels(stage_names, fontsize=8, rotation=20, ha="right")
        ax2.set_ylabel("Discrete Accuracy")
        ax2.set_title("Post-Training Stages")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")

        all_vals = stage_train + stage_val
        ax2.set_ylim(max(0.0, min(all_vals) - 0.05), min(1.0, max(all_vals) + 0.03))
    else:
        ax2.text(0.5, 0.5, "No stage data", ha="center", va="center")

    fig.suptitle("MCNN Training Pipeline", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", dpi=150)
    plt.close(fig)
    print(f"saved {out / 'training_curve.png'}")

    # Print summary
    if records:
        ste_final = next((r for r in records if r["stage"] == "ste_final"), None)
        final = next((r for r in records if r["stage"] == "final"), None)
        best_ste = max((r["val_acc"] for r in ste_epochs), default=None) if ste_epochs else None

        print("\n=== Pipeline Summary ===")
        if best_ste:
            print(f"Best STE val accuracy: {best_ste:.3f}")
        if ste_final:
            print(f"After discrete conversion: val={ste_final['val_acc']:.3f}")
        for r in stages:
            print(f"  {r['stage']}: train={r['train_acc']:.3f} val={r['val_acc']:.3f}")
        if final:
            print(f"Final: train={final['train_acc']:.3f} val={final['val_acc']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/run")
    args = parser.parse_args()
    plot(args.run_dir)
