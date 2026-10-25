"""Shared helpers for reproducible 7x7 Go match benchmarks."""

from __future__ import annotations

import numpy as np

from .go_rules import BLACK, PASS_MOVE, GoPosition, move_from_gtp


def replay_history(moves: list[list[str]]) -> GoPosition:
    """Reconstruct a position from a KataGo-style move history."""
    position = GoPosition()
    for color, coordinate in moves:
        expected = "B" if position.to_play == BLACK else "W"
        if color.upper() != expected:
            raise ValueError(f"Expected {expected} to play, got {color!r}")
        successor = position.play(move_from_gtp(coordinate))
        if successor is None:
            raise ValueError(f"Illegal recorded move: {color} {coordinate}")
        position = successor
    return position


def relative_final_score(position: GoPosition, player: int) -> float:
    """Return the project's final score margin from ``player``'s perspective."""
    black_score, white_score = position.score()
    black_margin = black_score - white_score - 7.5
    return black_margin if player == BLACK else -black_margin


def relative_katago_score(response: dict[str, object], position: GoPosition, player: int) -> float:
    """Convert KataGo's side-to-move score lead to ``player``'s perspective."""
    side_to_move_lead = float(response["rootInfo"]["scoreLead"])
    return side_to_move_lead if position.to_play == player else -side_to_move_lead


def sample_policy_move(
    response: dict[str, object],
    position: GoPosition,
    rng: np.random.Generator,
    *,
    temperature: float,
    noise: float,
    dirichlet_alpha: float,
    suppress_pass: bool = False,
) -> int:
    """Sample a legal move from KataGo policy with explicit Dirichlet noise."""
    if temperature <= 0 or not 0 <= noise <= 1 or dirichlet_alpha <= 0:
        raise ValueError("Invalid policy sampling parameters")
    legal = position.legal_moves()
    if suppress_pass:
        legal[PASS_MOVE] = False
    choices = np.flatnonzero(legal)
    if not len(choices):
        return PASS_MOVE

    raw_policy = np.asarray(response.get("policy", []), dtype=np.float64)
    if raw_policy.shape != (PASS_MOVE + 1,):
        raise RuntimeError("KataGo returned no usable policy")
    policy = np.maximum(raw_policy[choices], 0.0)
    if policy.sum() == 0:
        policy.fill(1.0)
    policy **= 1.0 / temperature
    policy /= policy.sum()
    if noise:
        random_policy = rng.dirichlet(np.full(len(choices), dirichlet_alpha))
        policy = (1.0 - noise) * policy + noise * random_policy
    return int(rng.choice(choices, p=policy / policy.sum()))


def sample_training_move(
    response: dict[str, object],
    position: GoPosition,
    rng: np.random.Generator,
    ply: int,
    *,
    expert_move_fraction: float,
    expert_temperature: float,
    min_ply_before_pass: int,
) -> int:
    """Mix uniform exploration with KataGo's visit distribution."""
    legal = position.legal_moves()
    choices = np.flatnonzero(legal)
    if len(choices) > 1 and position.consecutive_passes == 0 and ply < min_ply_before_pass:
        choices = choices[choices != PASS_MOVE]
    if rng.random() >= expert_move_fraction:
        return int(rng.choice(choices))

    policy = np.zeros(PASS_MOVE + 1, dtype=np.float64)
    for move_info in response.get("moveInfos", []):
        move = move_from_gtp(str(move_info["move"]))
        policy[move] = max(float(move_info.get("visits", 0)), 1.0)
    policy[~legal] = 0.0
    if PASS_MOVE not in choices:
        policy[PASS_MOVE] = 0.0
    if policy.sum() == 0:
        policy[choices] = 1.0
    policy[policy > 0] **= 1.0 / expert_temperature
    policy /= policy.sum()
    return int(rng.choice(PASS_MOVE + 1, p=policy))
