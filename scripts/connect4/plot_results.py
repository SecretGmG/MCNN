"""Create the retained Connect Four performance overview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common.plotting import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np


def plot(teacher_path: Path, physical_path: Path, output: Path) -> None:
    teacher = json.loads(teacher_path.read_text())
    physical = json.loads(physical_path.read_text())
    colors = {"Teacher": "#377eb8", "Physical": "#16836f"}
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    metric_keys = (
        ("dataset", "outcome_preservation_rate"),
        ("dataset", "optimal_move_rate"),
        ("tactics", "immediate_win_rate"),
        ("tactics", "mandatory_block_rate"),
    )
    labels = ("Preserve\noutcome", "Exact optimal\nmove", "Immediate\nwin", "Mandatory\nblock")
    x = np.arange(len(labels))
    for offset, name, result in (
        (-0.18, "Teacher", teacher),
        (0.18, "Physical", physical),
    ):
        values = [result[group][key] for group, key in metric_keys]
        axes[0].bar(x + offset, values, 0.36, label=name, color=colors[name])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.04)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Disjoint 10k benchmark")
    axes[0].legend(frameon=False)

    phase_keys = (
        "opening_outcome_preservation",
        "middle_outcome_preservation",
        "late_outcome_preservation",
    )
    phase_labels = ("Opening", "Middle", "Late")
    x = np.arange(3)
    for offset, name, result in (
        (-0.18, "Teacher", teacher),
        (0.18, "Physical", physical),
    ):
        values = [result["dataset"][key] for key in phase_keys]
        axes[1].bar(x + offset, values, 0.36, label=name, color=colors[name])
    axes[1].set_xticks(x, phase_labels)
    axes[1].set_ylim(0, 1.04)
    axes[1].set_ylabel("Outcome preservation")
    axes[1].set_title("Strength by phase")

    categories = ("winning", "drawing", "losing")
    values = []
    for category in categories:
        row = physical["perfect_reply"]["by_category"].get(category, {})
        games = row.get("games", 0)
        if category == "winning":
            values.append(row.get("wins", 0) / max(1, games))
        elif category == "drawing":
            values.append(row.get("draws", 0) / max(1, games))
        else:
            values.append((row.get("wins", 0) + row.get("draws", 0)) / max(1, games))
    axes[2].bar(
        ("Convert win", "Hold draw", "Escape loss"),
        values,
        color=("#4f9557", "#d29b32", "#bd5b4e"),
    )
    axes[2].set_ylim(0, 1.04)
    axes[2].set_ylabel("Rate vs perfect replies")
    axes[2].set_title("Physical model continuations")

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.teacher, args.physical, args.output)


if __name__ == "__main__":
    main()
