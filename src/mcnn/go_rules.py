"""Small, deterministic rules implementation for the project's 7x7 Go variant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


BOARD_SIZE = 7
BOARD_POINTS = BOARD_SIZE * BOARD_SIZE
PASS_MOVE = BOARD_POINTS
BLACK = 1
WHITE = -1
KOMI = 7.5


def move_to_gtp(move: int, board_size: int = BOARD_SIZE) -> str:
    """Convert a row-major move index to a GTP coordinate."""
    if move == PASS_MOVE:
        return "pass"
    if not 0 <= move < board_size * board_size:
        raise ValueError(f"Invalid move index: {move}")
    row, column = divmod(move, board_size)
    return f"{chr(ord('A') + column)}{board_size - row}"


def move_from_gtp(coordinate: str, board_size: int = BOARD_SIZE) -> int:
    """Convert a GTP coordinate to a row-major move index."""
    coordinate = coordinate.strip().upper()
    if coordinate == "PASS":
        return PASS_MOVE
    if len(coordinate) < 2 or not "A" <= coordinate[0] <= "Z":
        raise ValueError(f"Invalid GTP coordinate: {coordinate!r}")
    column = ord(coordinate[0]) - ord("A")
    try:
        gtp_row = int(coordinate[1:])
    except ValueError as error:
        raise ValueError(f"Invalid GTP coordinate: {coordinate!r}") from error
    row = board_size - gtp_row
    if not 0 <= row < board_size or not 0 <= column < board_size:
        raise ValueError(f"GTP coordinate outside board: {coordinate!r}")
    return row * board_size + column


def _neighbors(index: int, board_size: int) -> Iterable[int]:
    row, column = divmod(index, board_size)
    if row:
        yield index - board_size
    if row + 1 < board_size:
        yield index + board_size
    if column:
        yield index - 1
    if column + 1 < board_size:
        yield index + 1


def _group_and_liberties(board: np.ndarray, start: int) -> tuple[set[int], set[int]]:
    board_size = board.shape[0]
    color = int(board.flat[start])
    group = {start}
    liberties: set[int] = set()
    pending = [start]
    while pending:
        point = pending.pop()
        for neighbor in _neighbors(point, board_size):
            neighbor_color = int(board.flat[neighbor])
            if neighbor_color == 0:
                liberties.add(neighbor)
            elif neighbor_color == color and neighbor not in group:
                group.add(neighbor)
                pending.append(neighbor)
    return group, liberties


def territory_ownership(board: np.ndarray) -> np.ndarray:
    """Label stones and enclosed empty regions as Black, White, or neutral."""
    board = np.asarray(board, dtype=np.int8)
    if board.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected [{BOARD_SIZE}, {BOARD_SIZE}] board")
    ownership = board.copy()
    visited: set[int] = set()
    for point, value in enumerate(board.flat):
        if value != 0 or point in visited:
            continue
        region = {point}
        bordering_colors: set[int] = set()
        pending = [point]
        visited.add(point)
        while pending:
            current = pending.pop()
            for neighbor in _neighbors(current, BOARD_SIZE):
                neighbor_value = int(board.flat[neighbor])
                if neighbor_value == 0 and neighbor not in visited:
                    visited.add(neighbor)
                    region.add(neighbor)
                    pending.append(neighbor)
                elif neighbor_value:
                    bordering_colors.add(neighbor_value)
        if len(bordering_colors) == 1:
            owner = bordering_colors.pop()
            for member in region:
                ownership.flat[member] = owner
    return ownership


def suicide_attempts_from_encoding(encoded: np.ndarray) -> np.ndarray:
    """Return next-player encodings produced by locally suicidal placements.

    Opponent captures are resolved first. The zero-liberty friendly group is
    deliberately retained so the evaluator can learn that this impossible
    successor must never be selected.
    """
    encoded = np.asarray(encoded)
    if encoded.shape != (2, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected [2, {BOARD_SIZE}, {BOARD_SIZE}] encoding")
    board = np.where(encoded[0] > 0, BLACK, np.where(encoded[1] > 0, WHITE, 0)).astype(np.int8)
    attempts: list[np.ndarray] = []
    for move in np.flatnonzero(board.ravel() == 0):
        attempted = board.copy()
        attempted.flat[move] = BLACK
        for neighbor in _neighbors(int(move), BOARD_SIZE):
            if attempted.flat[neighbor] != WHITE:
                continue
            group, liberties = _group_and_liberties(attempted, neighbor)
            if not liberties:
                for point in group:
                    attempted.flat[point] = 0
        _, liberties = _group_and_liberties(attempted, int(move))
        if not liberties:
            attempts.append(np.stack((attempted == WHITE, attempted == BLACK)).astype(np.float32))
    if not attempts:
        return np.empty((0, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    return np.stack(attempts)


@dataclass(frozen=True)
class GoPosition:
    """Immutable position using positional superko and territory scoring."""

    board: np.ndarray = field(default_factory=lambda: np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8))
    to_play: int = BLACK
    black_prisoners: int = 0
    white_prisoners: int = 0
    consecutive_passes: int = 0
    history: frozenset[bytes] | None = None

    def __post_init__(self) -> None:
        board = np.asarray(self.board, dtype=np.int8)
        if board.ndim != 2 or board.shape[0] != board.shape[1]:
            raise ValueError("Board must be square")
        if self.to_play not in (BLACK, WHITE):
            raise ValueError("to_play must be BLACK or WHITE")
        board = board.copy()
        board.setflags(write=False)
        object.__setattr__(self, "board", board)
        if self.history is None:
            object.__setattr__(self, "history", frozenset({board.tobytes()}))

    @property
    def board_size(self) -> int:
        return self.board.shape[0]

    @property
    def is_finished(self) -> bool:
        return self.consecutive_passes >= 2

    def encode(self) -> np.ndarray:
        """Return me/opponent planes from the perspective of the player to move."""
        return np.stack(
            (self.board == self.to_play, self.board == -self.to_play), axis=0
        ).astype(np.float32)

    def legal_moves(self) -> np.ndarray:
        legal = np.zeros(BOARD_POINTS + 1, dtype=bool)
        for move in range(self.board_size * self.board_size):
            legal[move] = self.play(move) is not None
        legal[PASS_MOVE] = True
        return legal

    def play(self, move: int) -> GoPosition | None:
        """Return the successor, or ``None`` when a move is illegal."""
        if self.is_finished or not 0 <= move <= self.board_size * self.board_size:
            return None
        if move == self.board_size * self.board_size:
            return GoPosition(
                board=self.board,
                to_play=-self.to_play,
                black_prisoners=self.black_prisoners,
                white_prisoners=self.white_prisoners,
                consecutive_passes=self.consecutive_passes + 1,
                history=self.history,
            )
        if self.board.flat[move] != 0:
            return None

        board = self.board.copy()
        board.flat[move] = self.to_play
        captured = 0
        for neighbor in _neighbors(move, self.board_size):
            if board.flat[neighbor] != -self.to_play:
                continue
            group, liberties = _group_and_liberties(board, neighbor)
            if not liberties:
                for point in group:
                    board.flat[point] = 0
                captured += len(group)

        _, own_liberties = _group_and_liberties(board, move)
        if not own_liberties:  # Suicide is not legal in the chosen ruleset.
            return None
        board_key = board.tobytes()
        if board_key in self.history:  # Positional superko.
            return None

        return GoPosition(
            board=board,
            to_play=-self.to_play,
            black_prisoners=self.black_prisoners + (captured if self.to_play == BLACK else 0),
            white_prisoners=self.white_prisoners + (captured if self.to_play == WHITE else 0),
            history=self.history | {board_key},
        )

    def score(self) -> tuple[int, int]:
        """Return integer territory scores, excluding the fixed 7.5 komi."""
        black_score = self.black_prisoners
        white_score = self.white_prisoners
        visited: set[int] = set()
        for point, value in enumerate(self.board.flat):
            if value != 0 or point in visited:
                continue
            region = {point}
            bordering_colors: set[int] = set()
            pending = [point]
            visited.add(point)
            while pending:
                current = pending.pop()
                for neighbor in _neighbors(current, self.board_size):
                    neighbor_value = int(self.board.flat[neighbor])
                    if neighbor_value == 0 and neighbor not in visited:
                        visited.add(neighbor)
                        region.add(neighbor)
                        pending.append(neighbor)
                    elif neighbor_value:
                        bordering_colors.add(neighbor_value)
            if bordering_colors == {BLACK}:
                black_score += len(region)
            elif bordering_colors == {WHITE}:
                white_score += len(region)
        return black_score, white_score

    def winner(self) -> int:
        """Return BLACK when Black wins by the project's 7.5-komi convention."""
        black_score, white_score = self.score()
        return BLACK if black_score >= white_score + 8 else WHITE
