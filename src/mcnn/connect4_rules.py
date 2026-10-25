"""Immutable standard Connect Four rules with a padded 2x7x7 encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ROWS = 6
COLUMNS = 7
CONNECT = 4


def _winner(board: np.ndarray) -> int:
    for row in range(ROWS):
        for column in range(COLUMNS):
            player = int(board[row, column])
            if player == 0:
                continue
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_y = row + (CONNECT - 1) * dy
                end_x = column + (CONNECT - 1) * dx
                if not (0 <= end_y < ROWS and 0 <= end_x < COLUMNS):
                    continue
                if all(
                    board[row + step * dy, column + step * dx] == player
                    for step in range(1, CONNECT)
                ):
                    return player
    return 0


@dataclass(frozen=True)
class ConnectFourPosition:
    board: np.ndarray
    to_play: int = 1
    history: tuple[int, ...] = ()
    winner: int = 0

    def __post_init__(self) -> None:
        board = np.asarray(self.board, dtype=np.int8)
        if board.shape != (ROWS, COLUMNS):
            raise ValueError(f"Expected a {ROWS}x{COLUMNS} board")
        if self.to_play not in (-1, 1):
            raise ValueError("to_play must be -1 or 1")
        if not np.isin(board, (-1, 0, 1)).all():
            raise ValueError("Board cells must be -1, 0, or 1")
        board = board.copy()
        board.setflags(write=False)
        object.__setattr__(self, "board", board)
        actual_winner = _winner(board)
        if self.winner not in (-1, 0, 1) or self.winner != actual_winner:
            object.__setattr__(self, "winner", actual_winner)

    @classmethod
    def empty(cls) -> ConnectFourPosition:
        return cls(np.zeros((ROWS, COLUMNS), dtype=np.int8))

    @classmethod
    def from_moves(cls, moves: tuple[int, ...] | list[int]) -> ConnectFourPosition:
        position = cls.empty()
        for column in moves:
            successor = position.play(int(column))
            if successor is None:
                raise ValueError(f"Illegal move sequence at column {column}")
            position = successor
        return position

    @property
    def is_full(self) -> bool:
        return bool(np.all(self.board[0] != 0))

    @property
    def is_terminal(self) -> bool:
        return self.winner != 0 or self.is_full

    def legal_mask(self) -> np.ndarray:
        return (self.board[0] == 0) & (not self.is_terminal)

    def play(self, column: int) -> ConnectFourPosition | None:
        if self.is_terminal or not 0 <= column < COLUMNS or self.board[0, column] != 0:
            return None
        occupied = np.flatnonzero(self.board[:, column] != 0)
        row = ROWS - 1 if len(occupied) == 0 else int(occupied[0]) - 1
        board = self.board.copy()
        board[row, column] = self.to_play
        return ConnectFourPosition(
            board,
            -self.to_play,
            (*self.history, column),
            _winner(board),
        )

    def encode(self) -> np.ndarray:
        """Return me/opponent planes with an unused zero row at index zero."""
        encoded = np.zeros((2, 7, 7), dtype=np.uint8)
        encoded[0, 1:] = self.board == self.to_play
        encoded[1, 1:] = self.board == -self.to_play
        return encoded

    def reflected(self) -> ConnectFourPosition:
        return ConnectFourPosition(
            np.ascontiguousarray(self.board[:, ::-1]),
            self.to_play,
            tuple(COLUMNS - 1 - column for column in self.history),
            self.winner,
        )

    def move_string(self) -> str:
        return "".join(str(column + 1) for column in self.history)
