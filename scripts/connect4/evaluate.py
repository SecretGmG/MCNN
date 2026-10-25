"""Evaluate Connect Four move quality, tactics, gameplay, and latency."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mcnn.connect4_model import ConnectFourTeacher, PhysicalConnectFourModel
from mcnn.connect4_rules import COLUMNS, ConnectFourPosition
from mcnn.connect4_solver import ConnectFourSolver

from .train_teacher import TrainConfig, evaluate as dataset_evaluate, load_data


def load_model(path: Path) -> nn.Module:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["model_type"] == ConnectFourTeacher.model_type:
        return ConnectFourTeacher.load(path).eval()
    if artifact["model_type"] == PhysicalConnectFourModel.model_type:
        return PhysicalConnectFourModel.load(path).eval()
    raise ValueError(f"Unsupported Connect Four model: {artifact['model_type']}")


@torch.no_grad()
def select_move(model: nn.Module, position: ConnectFourPosition) -> int:
    scores = model(torch.tensor(position.encode()[None], dtype=torch.float32))[0]
    legal = torch.tensor(position.legal_mask())
    return int(scores.masked_fill(~legal, -1e9).argmax())


def _position_from_input(encoded: np.ndarray) -> ConnectFourPosition:
    board = np.zeros((6, 7), dtype=np.int8)
    board[encoded[0, 1:] != 0] = 1
    board[encoded[1, 1:] != 0] = -1
    return ConnectFourPosition(board, to_play=1)


@torch.no_grad()
def tactical_metrics(model: nn.Module, x: torch.Tensor, indices: np.ndarray) -> dict[str, float]:
    immediate_total = 0
    immediate_correct = 0
    block_total = 0
    block_correct = 0
    for index in indices:
        position = _position_from_input(x[index].numpy())
        if position.is_terminal:
            continue
        move = select_move(model, position)
        legal = np.flatnonzero(position.legal_mask())
        immediate = [
            column
            for column in legal
            if (successor := position.play(int(column))) is not None
            and successor.winner == position.to_play
        ]
        if immediate:
            immediate_total += 1
            immediate_correct += move in immediate
            continue
        opponent_position = ConnectFourPosition(position.board, -position.to_play)
        opponent_threats = [
            column
            for column in legal
            if (successor := opponent_position.play(int(column))) is not None
            and successor.winner == opponent_position.to_play
        ]
        if opponent_threats:
            safe = []
            for column in legal:
                successor = position.play(int(column))
                assert successor is not None
                if successor.winner == position.to_play:
                    safe.append(int(column))
                    continue
                replies = np.flatnonzero(successor.legal_mask())
                opponent_can_win = any(
                    (reply := successor.play(int(other))) is not None
                    and reply.winner == successor.to_play
                    for other in replies
                )
                if not opponent_can_win:
                    safe.append(int(column))
            if safe:
                block_total += 1
                block_correct += move in safe
    return {
        "immediate_win_positions": immediate_total,
        "immediate_win_rate": immediate_correct / max(1, immediate_total),
        "mandatory_block_positions": block_total,
        "mandatory_block_rate": block_correct / max(1, block_total),
    }


def _random_prefix(
    rng: np.random.Generator, minimum_ply: int = 4, maximum_ply: int = 26
) -> ConnectFourPosition:
    while True:
        position = ConnectFourPosition.empty()
        target = int(rng.integers(minimum_ply, maximum_ply + 1))
        for _ in range(target):
            if position.is_terminal:
                break
            move = int(rng.choice(np.flatnonzero(position.legal_mask())))
            successor = position.play(move)
            assert successor is not None
            position = successor
        if not position.is_terminal and len(position.history) >= minimum_ply:
            return position


def perfect_reply_matches(
    model: nn.Module,
    solver: ConnectFourSolver,
    scenarios: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    records = []
    attempts = 0
    category_counts: defaultdict[str, int] = defaultdict(int)
    target_per_category = max(1, scenarios // 3)
    while len(records) < scenarios and attempts < scenarios * 100:
        attempts += 1
        position = _random_prefix(rng)
        initial_scores = solver.analyze(position)
        initial_best = int(initial_scores.max())
        category = "winning" if initial_best > 0 else "drawing" if initial_best == 0 else "losing"
        if category_counts[category] >= target_per_category and len(records) < 3 * target_per_category:
            continue
        category_counts[category] += 1
        model_player = position.to_play
        initial_ply = len(position.history)
        model_moves = 0
        model_optimal = 0
        while not position.is_terminal:
            if position.to_play == model_player:
                exact = solver.analyze(position)
                move = select_move(model, position)
                model_moves += 1
                model_optimal += int(exact[move] == exact.max())
            else:
                exact = solver.analyze(position)
                legal = np.flatnonzero(position.legal_mask())
                best = legal[exact[legal] == exact[legal].max()]
                move = int(rng.choice(best))
            successor = position.play(move)
            assert successor is not None
            position = successor
        result = 1 if position.winner == model_player else 0 if position.winner == 0 else -1
        records.append(
            {
                "category": category,
                "initial_score": int(initial_best),
                "initial_ply": initial_ply,
                "result": result,
                "model_moves": int(model_moves),
                "optimal_model_moves": int(model_optimal),
            }
        )
    by_category = {}
    for category in ("winning", "drawing", "losing"):
        rows = [row for row in records if row["category"] == category]
        if rows:
            by_category[category] = {
                "games": len(rows),
                "mean_result": float(np.mean([row["result"] for row in rows])),
                "wins": sum(row["result"] > 0 for row in rows),
                "draws": sum(row["result"] == 0 for row in rows),
            }
    return {
        "games": len(records),
        "mean_result": float(np.mean([row["result"] for row in records])),
        "wins": sum(row["result"] > 0 for row in records),
        "draws": sum(row["result"] == 0 for row in records),
        "losses": sum(row["result"] < 0 for row in records),
        "model_move_optimal_rate": sum(row["optimal_model_moves"] for row in records)
        / max(1, sum(row["model_moves"] for row in records)),
        "by_category": by_category,
        "records": records,
    }


def random_matches(model: nn.Module, games: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    results = []
    for game in range(games):
        position = ConnectFourPosition.empty()
        model_player = 1 if game % 2 == 0 else -1
        while not position.is_terminal:
            if position.to_play == model_player:
                move = select_move(model, position)
            else:
                move = int(rng.choice(np.flatnonzero(position.legal_mask())))
            successor = position.play(move)
            assert successor is not None
            position = successor
        results.append(1 if position.winner == model_player else 0 if position.winner == 0 else -1)
    return {
        "games": games,
        "wins": int(np.sum(np.asarray(results) > 0)),
        "draws": int(np.sum(np.asarray(results) == 0)),
        "losses": int(np.sum(np.asarray(results) < 0)),
        "mean_result": float(np.mean(results)),
    }


@torch.no_grad()
def latency(model: nn.Module, x: torch.Tensor) -> dict[str, float]:
    batch = x[:1024]
    for _ in range(3):
        model(batch)
    start = time.perf_counter()
    repeats = 10
    for _ in range(repeats):
        model(batch)
    elapsed = time.perf_counter() - start
    return {
        "batch_size": len(batch),
        "milliseconds_per_position": 1000 * elapsed / (repeats * len(batch)),
        "positions_per_second": repeats * len(batch) / elapsed,
    }


def run(model_path: Path, dataset_path: Path, scenarios: int, random_games: int, seed: int):
    model = load_model(model_path)
    x, scores, legal, _, ply, source, _, _ = load_data(
        dataset_path, TrainConfig(seed=45001)
    )
    validation_indices = np.arange(len(x))
    result = {
        "model": str(model_path),
        "dataset": dataset_evaluate(
            model, x, scores, legal, validation_indices, ply, source
        ),
        "tactics": tactical_metrics(model, x, validation_indices),
        "random_opponent": random_matches(model, random_games, seed),
        "latency": latency(model, x),
    }
    with ConnectFourSolver() as solver:
        result["perfect_reply"] = perfect_reply_matches(
            model, solver, scenarios, seed + 1000
        )
    if isinstance(model, PhysicalConnectFourModel):
        result["physicality"] = model.physicality_metrics()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=60)
    parser.add_argument("--random-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=47001)
    args = parser.parse_args()
    result = run(args.model, args.dataset, args.scenarios, args.random_games, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "perfect_reply"}, indent=2))
    print(json.dumps({"perfect_reply": {key: value for key, value in result["perfect_reply"].items() if key != "records"}}, indent=2))


if __name__ == "__main__":
    main()
