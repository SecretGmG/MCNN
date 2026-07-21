"""Interactive sample-by-sample dataset curation for MCNN digits.

Shows each 8x8 handwritten digit at thresholds 2,4,6,8,10 as ASCII art.
Mark each sample:
  a            = keep all thresholds 2-10
  2-6          = keep thresholds 2 through 6 (inclusive)
  2,4,8        = keep specific thresholds
  x            = exclude this sample entirely
  b            = go back one sample
  q            = quit and save progress

Progress is saved to artifacts/curation/curation_progress.json (resumable).
Final curated dataset is exported to artifacts/curation/curated_dataset.npz.

Usage: .venv/bin/python experiments/basic_task/curate.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits


OUT = Path("artifacts/curation")
THRESHOLDS = [2, 4, 6, 8, 10]
ACTIVE = "██"
EMPTY = "░░"


def _ascii_full(img: np.ndarray, thresholds: list[int]) -> str:
    """Render a single image at all thresholds side by side."""
    lines = []
    # Header
    header = "        "
    for t in thresholds:
        header += f"  t>{t:<3} "
    lines.append(header)

    for row in range(8):
        line = "        "
        for t in thresholds:
            binary = (img > t).astype(int)
            for col in range(8):
                line += ACTIVE if binary[row, col] else EMPTY
            line += "  "
        lines.append(line)

    # Footer with active pixel count
    footer = "active: "
    for t in thresholds:
        count = int((img > t).sum())
        footer += f"  {count:<4} "
    lines.append(footer)
    return "\n".join(lines)


def load_progress() -> dict:
    progress_file = OUT / "curation_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"last_index": 0, "decisions": {}, "total": 0}


def save_progress(progress: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "curation_progress.json", "w") as f:
        json.dump(progress, f, indent=2)


def export_curated(progress: dict):
    digits = load_digits()
    images = digits.images.astype(np.float32)
    targets = digits.target

    kept_x, kept_y = [], []
    stats = {}
    for i in range(len(images)):
        key = str(i)
        decision = progress["decisions"].get(key)
        if decision is None or decision == "x":
            continue
        img = images[i]
        label = int(targets[i])
        if decision == "a":
            thr_list = THRESHOLDS
        else:
            thr_list = [int(p.strip()) for p in decision.split(",") if p.strip().isdigit()]
        for t in thr_list:
            binary = (img > t).astype(np.float32)
            kept_x.append(binary.flatten())
            kept_y.append(label)
            stats[label] = stats.get(label, 0) + 1

    kept_x = np.array(kept_x, dtype=np.float32)
    kept_y = np.array(kept_y, dtype=np.int64)
    np.savez(OUT / "curated_dataset.npz", x=kept_x, y=kept_y)

    print(f"\nExported to {OUT / 'curated_dataset.npz'}")
    print(f"Total samples: {len(kept_x)}")
    print(f"Per class: {dict(sorted(stats.items()))}")


def _parse_decision(ans: str) -> str | None:
    """Parse user input into a canonical decision string or None if invalid."""
    ans = ans.strip().lower()
    if ans in ("a", "x", "b", "q"):
        return ans
    if "-" in ans:
        parts = ans.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            lo, hi = int(parts[0]), int(parts[1])
            if lo <= hi:
                return ",".join(str(t) for t in THRESHOLDS if lo <= t <= hi)
    parts = [p.strip() for p in ans.split(",")]
    if all(p.isdigit() and int(p) in THRESHOLDS for p in parts):
        return ",".join(parts)
    return None


def main():
    digits = load_digits()
    images = digits.images.astype(np.float32)
    targets = digits.target
    total = len(images)

    os.makedirs(OUT, exist_ok=True)
    progress = load_progress()
    start = progress["last_index"]

    print(f"MCNN Curation — {len(progress['decisions'])}/{total} done, resuming at #{start}")
    print(f"a=all  x=exclude  #-#=range  #,#=list  b=back  q=quit")
    print()

    i = start
    while i < total:
        key = str(i)
        if key in progress["decisions"]:
            i += 1
            continue

        img = images[i]
        label = int(targets[i])

        print(f"\n─── #{i+1}/{total}  digit={label} ───")
        print(_ascii_full(img, THRESHOLDS))

        try:
            raw = input(f"[a/x/#-#/#,#/b/q] ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "q"

        decision = _parse_decision(raw)
        if decision is None:
            print("  ? — try: a, x, 2-6, 2,4,8, b, q")
            continue

        if decision == "q":
            progress["last_index"] = i
            save_progress(progress)
            print(f"Saved at #{i}. Resume with same command.")
            export_curated(progress)
            return

        if decision == "b":
            i = max(0, i - 1)
            prev = str(i)
            if prev in progress["decisions"]:
                del progress["decisions"][prev]
            continue

        progress["decisions"][key] = decision
        i += 1

    progress["last_index"] = total
    save_progress(progress)
    print(f"\nAll {total} done!")
    export_curated(progress)


if __name__ == "__main__":
    main()
