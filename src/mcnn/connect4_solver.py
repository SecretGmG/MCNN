"""Persistent interface to Pascal Pons' exact Connect Four solver."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .connect4_rules import COLUMNS, ConnectFourPosition


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOLVER = ROOT / "tools/connect4-solver/c4solver"
DEFAULT_BOOK = ROOT / "tools/connect4-solver/7x6.book"
ILLEGAL_SCORE = -1000


class ConnectFourSolver:
    def __init__(
        self,
        executable: Path = DEFAULT_SOLVER,
        opening_book: Path = DEFAULT_BOOK,
    ) -> None:
        if not executable.is_file():
            raise FileNotFoundError(f"Connect Four solver not found: {executable}")
        if not opening_book.is_file():
            raise FileNotFoundError(f"Connect Four opening book not found: {opening_book}")
        self.process = subprocess.Popen(
            [str(executable), "-a", "-b", str(opening_book)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def analyze(self, position: ConnectFourPosition) -> np.ndarray:
        if position.is_terminal:
            raise ValueError("Cannot analyze a terminal position")
        assert self.process.stdin is not None and self.process.stdout is not None
        sequence = position.move_string()
        self.process.stdin.write(sequence + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Connect Four solver terminated unexpectedly")
        fields = line.split()
        values = fields if sequence == "" else fields[1:]
        if len(values) != COLUMNS:
            raise RuntimeError(f"Unexpected solver response: {line.rstrip()}")
        scores = np.asarray([int(value) for value in values], dtype=np.int16)
        scores[~position.legal_mask()] = ILLEGAL_SCORE
        return scores

    def close(self) -> None:
        if self.process.poll() is None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            self.process.wait(timeout=5)

    def __enter__(self) -> ConnectFourSolver:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
