"""Dataset loading and grouping for the canonical 7x7 Go value pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class GoValueDataset:
    boards: Tensor
    positional_value: Tensor
    action_value: Tensor
    captures: Tensor
    parent: np.ndarray
    game: np.ndarray
    ply: np.ndarray
    move: np.ndarray

    def __len__(self) -> int:
        return len(self.boards)


def _load_part(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        arrays = {name: source[name].copy() for name in source.files}
    inputs = arrays["x"]
    captures = arrays["captures"].astype(np.float32)
    if "positional_value" in arrays:
        if inputs.shape[1:] != (2, 7, 7):
            raise ValueError(f"invalid canonical Go input shape in {path}: {inputs.shape}")
        boards = inputs.astype(np.float32)
        positional_value = arrays["positional_value"].astype(np.float32)
        action_value = arrays["action_value"].astype(np.float32)
    elif inputs.shape[1:] == (3, 7, 7):
        boards = inputs[:, :2].astype(np.float32)
        action_value = arrays["score"].astype(np.float32)
        positional_value = action_value - captures
    elif inputs.shape[1:] == (2, 7, 7):
        # Legacy rows store successors from the next player's perspective.
        boards = inputs[:, ::-1].astype(np.float32)
        positional_value = -arrays["value"].astype(np.float32)
        action_value = positional_value + captures
    else:
        raise ValueError(f"unsupported Go input shape in {path}: {inputs.shape}")
    return {
        "boards": boards,
        "positional_value": positional_value,
        "action_value": action_value,
        "captures": captures,
        "parent": arrays["parent"].astype(np.int64),
        "game": arrays["game"].astype(np.int64),
        "ply": arrays["ply"].astype(np.int16),
        "move": arrays["move"].astype(np.int16),
    }


def load_value_dataset(paths: list[Path]) -> GoValueDataset:
    if not paths:
        raise ValueError("at least one dataset is required")
    parts = [_load_part(path) for path in paths]
    parent_offset = 0
    game_offset = 0
    for part in parts:
        part["parent"] = part["parent"] - int(part["parent"].min()) + parent_offset
        part["game"] = part["game"] - int(part["game"].min()) + game_offset
        parent_offset = int(part["parent"].max()) + 1
        game_offset = int(part["game"].max()) + 1
    merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    return GoValueDataset(
        boards=torch.from_numpy(merged["boards"]),
        positional_value=torch.from_numpy(merged["positional_value"]),
        action_value=torch.from_numpy(merged["action_value"]),
        captures=torch.from_numpy(merged["captures"]),
        parent=merged["parent"],
        game=merged["game"],
        ply=merged["ply"],
        move=merged["move"],
    )


def save_value_dataset(dataset: GoValueDataset, path: Path) -> None:
    """Write the normalized, implementation-independent dataset format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=dataset.boards.cpu().numpy().astype(np.uint8),
        positional_value=dataset.positional_value.cpu().numpy().astype(np.float32),
        action_value=dataset.action_value.cpu().numpy().astype(np.float32),
        captures=dataset.captures.cpu().numpy().astype(np.float32),
        parent=dataset.parent,
        game=dataset.game,
        ply=dataset.ply,
        move=dataset.move,
    )


def split_parent_groups(
    dataset: GoValueDataset, seed: int, validation_fraction: float = 0.2
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    games = np.unique(dataset.game)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    validation_games = set(games[: max(1, round(validation_fraction * len(games)))])
    groups: list[list[np.ndarray]] = []
    for validation in (False, True):
        indices = np.flatnonzero(
            np.asarray([(game in validation_games) == validation for game in dataset.game])
        )
        order = indices[np.argsort(dataset.parent[indices], kind="stable")]
        boundaries = np.flatnonzero(np.diff(dataset.parent[order])) + 1
        groups.append([group for group in np.split(order, boundaries) if len(group) >= 2])
    return groups[0], groups[1]


def d4_augment(inputs: Tensor, rng: np.random.Generator) -> Tensor:
    transformed = []
    for board, code in zip(inputs, rng.integers(0, 8, size=len(inputs)), strict=True):
        turns = int(code) % 4
        value = torch.rot90(board, turns, dims=(-2, -1))
        if int(code) >= 4:
            value = value.flip(-1)
        transformed.append(value)
    return torch.stack(transformed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and merge Go datasets")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_value_dataset(args.input)
    save_value_dataset(dataset, args.output)
    print(f"wrote {len(dataset)} examples to {args.output}")


if __name__ == "__main__":
    main()
