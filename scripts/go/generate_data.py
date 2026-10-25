"""Generate scalable successor-value data for the Minecraft Go model."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mcnn.go_match import sample_training_move
from mcnn.go_rules import BLACK, GoPosition
from mcnn.katago import (
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_KATAGO,
    DEFAULT_NETWORK,
    KataGoAnalysis,
    future_score_value,
    history_move,
    query_payload,
)


@dataclass(frozen=True)
class GenerationConfig:
    games: int = 192
    max_moves: int = 96
    root_visits: int = 2
    child_visits: int = 4
    sample_stride: int = 5
    expert_move_fraction: float = 0.35
    expert_temperature: float = 1.5
    min_ply_before_pass: int = 36
    seed: int = 31001


def _worker(
    game_ids: list[int],
    config: GenerationConfig,
    executable: Path,
    engine_config: Path,
    network: Path,
) -> dict[str, np.ndarray]:
    engine = KataGoAnalysis(executable, engine_config, network)
    rows: dict[str, list[object]] = {
        "x": [],
        "positional_value": [],
        "action_value": [],
        "captures": [],
        "parent": [],
        "game": [],
        "ply": [],
        "move": [],
    }
    query_id = 0
    try:
        for game in game_ids:
            rng = np.random.default_rng(config.seed + game)
            position = GoPosition()
            moves: list[list[str]] = []
            for ply in range(config.max_moves):
                root = engine.query(query_payload(moves, config.root_visits, query_id))
                query_id += 1
                if (ply - game) % config.sample_stride == 0:
                    prisoner_count = (
                        position.black_prisoners
                        if position.to_play == BLACK
                        else position.white_prisoners
                    )
                    for move in np.flatnonzero(position.legal_moves()):
                        successor = position.play(int(move))
                        if successor is None:
                            continue
                        response = engine.query(
                            query_payload(
                                [*moves, history_move(position, int(move))],
                                config.child_visits,
                                query_id,
                                include_policy=False,
                            )
                        )
                        query_id += 1
                        next_prisoners = (
                            successor.black_prisoners
                            if position.to_play == BLACK
                            else successor.white_prisoners
                        )
                        captures = next_prisoners - prisoner_count
                        positional = -future_score_value(response, successor)
                        rows["x"].append(successor.encode()[::-1].copy())
                        rows["positional_value"].append(positional)
                        rows["action_value"].append(positional + captures)
                        rows["captures"].append(captures)
                        rows["parent"].append(game * config.max_moves + ply)
                        rows["game"].append(game)
                        rows["ply"].append(ply)
                        rows["move"].append(int(move))
                move = sample_training_move(
                    root,
                    position,
                    rng,
                    ply,
                    expert_move_fraction=config.expert_move_fraction,
                    expert_temperature=config.expert_temperature,
                    min_ply_before_pass=config.min_ply_before_pass,
                )
                successor = position.play(move)
                if successor is None:
                    raise RuntimeError(f"KataGo trajectory selected illegal move {move}")
                moves.append(history_move(position, move))
                position = successor
                if position.is_finished:
                    break
    finally:
        engine.close()
    dtypes = {
        "x": np.float32,
        "positional_value": np.float32,
        "action_value": np.float32,
        "captures": np.int16,
        "parent": np.int32,
        "game": np.int32,
        "ply": np.int16,
        "move": np.int16,
    }
    return {name: np.asarray(values, dtype=dtypes[name]) for name, values in rows.items()}


def generate(
    config: GenerationConfig,
    executable: Path,
    engine_config: Path,
    network: Path,
    workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    groups = [list(range(worker, config.games, workers)) for worker in range(workers)]
    arguments = [
        (group, config, executable, engine_config, network) for group in groups if group
    ]
    if len(arguments) == 1:
        parts = [_worker(*arguments[0])]
    else:
        with mp.get_context("spawn").Pool(len(arguments)) as pool:
            parts = pool.starmap(_worker, arguments)
    dataset = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    order = np.lexsort((dataset["move"], dataset["ply"], dataset["game"]))
    dataset = {name: values[order] for name, values in dataset.items()}
    parents = np.unique(dataset["parent"])
    metadata = {
        "generation": asdict(config),
        "examples": len(dataset["x"]),
        "parents": len(parents),
        "mean_successors": float(len(dataset["x"]) / len(parents)),
        "representation": "post-move candidate/opponent planes",
        "target": "future positional score from the candidate player's perspective",
        "captures": "stored separately and added mechanically during move selection",
        "workers": workers,
    }
    return dataset, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=GenerationConfig.games)
    parser.add_argument("--max-moves", type=int, default=GenerationConfig.max_moves)
    parser.add_argument("--root-visits", type=int, default=GenerationConfig.root_visits)
    parser.add_argument("--child-visits", type=int, default=GenerationConfig.child_visits)
    parser.add_argument("--sample-stride", type=int, default=GenerationConfig.sample_stride)
    parser.add_argument(
        "--expert-move-fraction",
        type=float,
        default=GenerationConfig.expert_move_fraction,
    )
    parser.add_argument(
        "--expert-temperature", type=float, default=GenerationConfig.expert_temperature
    )
    parser.add_argument(
        "--min-ply-before-pass", type=int, default=GenerationConfig.min_ply_before_pass
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=GenerationConfig.seed)
    parser.add_argument("--katago", type=Path, default=DEFAULT_KATAGO)
    parser.add_argument("--engine-config", type=Path, default=DEFAULT_ENGINE_CONFIG)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    args = parser.parse_args()
    config = GenerationConfig(
        games=args.games,
        max_moves=args.max_moves,
        root_visits=args.root_visits,
        child_visits=args.child_visits,
        sample_stride=args.sample_stride,
        expert_move_fraction=args.expert_move_fraction,
        expert_temperature=args.expert_temperature,
        min_ply_before_pass=args.min_ply_before_pass,
        seed=args.seed,
    )
    dataset, metadata = generate(
        config,
        args.katago,
        args.engine_config,
        args.network,
        min(args.workers, config.games),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dataset)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
