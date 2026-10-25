"""Batched successor inference for the physical 7x7 Go model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .go_model import MinecraftGoModel
from .go_rules import PASS_MOVE, GoPosition


def _select_tied_candidate(scores: torch.Tensor, position: GoPosition) -> int:
    """Resolve exact integer-output ties without a board-index bias."""
    candidates = torch.nonzero(scores == scores.max(), as_tuple=False).flatten().cpu().numpy()
    state = position.board.tobytes() + bytes(
        (
            position.to_play & 0xFF,
            position.black_prisoners & 0xFF,
            position.white_prisoners & 0xFF,
            position.consecutive_passes & 0xFF,
        )
    )
    offset = int.from_bytes(hashlib.blake2b(state, digest_size=8).digest(), "little")
    return int(candidates[offset % len(candidates)])


@torch.no_grad()
def select_successor_move(model: MinecraftGoModel, position: GoPosition) -> int:
    """Score every legal successor and add immediate captures mechanically."""
    if position.consecutive_passes == 1 and position.winner() == position.to_play:
        return PASS_MOVE

    player_prisoners = (
        position.black_prisoners if position.to_play == 1 else position.white_prisoners
    )
    successors: list[tuple[int, np.ndarray, int]] = []
    for move in np.flatnonzero(position.legal_moves()):
        successor = position.play(int(move))
        if successor is None:
            continue
        captures = (
            successor.black_prisoners if position.to_play == 1 else successor.white_prisoners
        ) - player_prisoners
        # The successor's side to move is the opponent, so swap the planes back
        # to the candidate player's perspective before evaluating it.
        successors.append((int(move), successor.encode()[::-1].copy(), captures))
    if not successors:
        raise RuntimeError("position exposes no legal move")

    device = next(model.parameters()).device
    inputs = torch.as_tensor(
        np.stack([encoded for _, encoded, _ in successors]),
        dtype=torch.float32,
        device=device,
    )
    scores = model.score_points(inputs, mode="exact") + torch.tensor(
        [captures for _, _, captures in successors], dtype=torch.float32, device=device
    )
    pass_candidate = next(
        (index for index, (move, _, _) in enumerate(successors) if move == PASS_MOVE), None
    )
    if pass_candidate is not None and scores[pass_candidate] >= scores.max():
        selected = pass_candidate
    else:
        selected = _select_tied_candidate(scores, position)
    return successors[selected][0]


def load_successor_model(path: str | Path) -> MinecraftGoModel:
    return MinecraftGoModel.load(path).eval()
