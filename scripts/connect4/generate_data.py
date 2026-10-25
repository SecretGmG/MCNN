"""Generate exactly labeled Connect Four positions from diverse trajectories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mcnn.connect4_rules import COLUMNS, ConnectFourPosition
from mcnn.connect4_solver import ConnectFourSolver


POLICIES = ("optimal", "noisy", "random", "anti")


@dataclass(frozen=True)
class GenerationConfig:
    positions: int = 50_000
    seed: int = 44001
    optimal_fraction: float = 0.10
    noisy_fraction: float = 0.50
    random_fraction: float = 0.25
    anti_fraction: float = 0.15
    noisy_random_probability: float = 0.40
    temperature: float = 2.0

    def probabilities(self) -> np.ndarray:
        values = np.asarray(
            (
                self.optimal_fraction,
                self.noisy_fraction,
                self.random_fraction,
                self.anti_fraction,
            ),
            dtype=np.float64,
        )
        if np.any(values < 0) or not np.isclose(values.sum(), 1):
            raise ValueError("Trajectory fractions must be nonnegative and sum to one")
        return values


def _choose_move(
    scores: np.ndarray,
    legal: np.ndarray,
    policy: str,
    config: GenerationConfig,
    rng: np.random.Generator,
) -> int:
    moves = np.flatnonzero(legal)
    legal_scores = scores[moves].astype(np.float64)
    if policy == "optimal":
        candidates = moves[legal_scores == legal_scores.max()]
        return int(rng.choice(candidates))
    if policy == "random" or (
        policy == "noisy" and rng.random() < config.noisy_random_probability
    ):
        return int(rng.choice(moves))
    if policy == "anti":
        cutoff = np.quantile(legal_scores, 0.35)
        candidates = moves[legal_scores <= cutoff]
        return int(rng.choice(candidates))
    centered = (legal_scores - legal_scores.max()) / config.temperature
    probabilities = np.exp(centered)
    probabilities /= probabilities.sum()
    return int(rng.choice(moves, p=probabilities))


def generate(
    output: Path,
    config: GenerationConfig,
    exclude_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    records: dict[str, list[object]] = {
        name: [] for name in ("x", "scores", "legal", "game", "ply", "source")
    }
    excluded: set[bytes] = set()
    for path in exclude_paths:
        with np.load(path) as data:
            for encoded in data["x"]:
                excluded.add(encoded.tobytes())
                excluded.add(np.ascontiguousarray(encoded[:, :, ::-1]).tobytes())
    seen: set[bytes] = set(excluded)
    game = 0
    suboptimal_moves = 0
    source_games = np.zeros(len(POLICIES), dtype=np.int64)
    source_positions = np.zeros(len(POLICIES), dtype=np.int64)
    with ConnectFourSolver() as solver:
        while len(records["x"]) < config.positions:
            source = int(rng.choice(len(POLICIES), p=config.probabilities()))
            policy = POLICIES[source]
            source_games[source] += 1
            position = ConnectFourPosition.empty()
            while not position.is_terminal and len(records["x"]) < config.positions:
                scores = solver.analyze(position)
                legal = position.legal_mask()
                encoded = position.encode()
                key = encoded.tobytes()
                if key not in seen:
                    seen.add(key)
                    records["x"].append(encoded)
                    records["scores"].append(scores)
                    records["legal"].append(legal)
                    records["game"].append(game)
                    records["ply"].append(len(position.history))
                    records["source"].append(source)
                    source_positions[source] += 1
                move = _choose_move(scores, legal, policy, config, rng)
                if scores[move] < scores[legal].max():
                    suboptimal_moves += 1
                successor = position.play(move)
                assert successor is not None
                position = successor
            game += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=np.asarray(records["x"], dtype=np.uint8),
        scores=np.asarray(records["scores"], dtype=np.int16),
        legal=np.asarray(records["legal"], dtype=np.bool_),
        game=np.asarray(records["game"], dtype=np.int32),
        ply=np.asarray(records["ply"], dtype=np.uint8),
        source=np.asarray(records["source"], dtype=np.uint8),
    )
    summary = {
        "config": asdict(config),
        "positions": len(records["x"]),
        "games": game,
        "unique_positions": len(records["x"]),
        "excluded_positions_and_reflections": len(excluded),
        "exclude_paths": [str(path) for path in exclude_paths],
        "suboptimal_trajectory_moves": suboptimal_moves,
        "source_games": dict(zip(POLICIES, source_games.tolist(), strict=True)),
        "source_positions": dict(zip(POLICIES, source_positions.tolist(), strict=True)),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=GenerationConfig.positions)
    parser.add_argument("--seed", type=int, default=GenerationConfig.seed)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()
    summary = generate(
        args.output,
        GenerationConfig(positions=args.positions, seed=args.seed),
        tuple(args.exclude),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
