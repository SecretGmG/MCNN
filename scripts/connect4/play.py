"""Serve a local human-vs-model Connect Four application."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from torch import nn

from mcnn.connect4_rules import ConnectFourPosition
from scripts.common.server import JSONAppHandler, serve_app

from .evaluate import load_model, select_move


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
DEFAULT_MODEL = ROOT / "artifacts/connect4/final/model.pt"


def _position_from_moves(moves: list[int]) -> ConnectFourPosition:
    position = ConnectFourPosition.empty()
    for move in moves:
        if isinstance(move, bool) or not isinstance(move, int):
            raise ValueError("Moves must be integer column indices")
        successor = position.play(move)
        if successor is None:
            raise ValueError(f"Illegal move sequence at column {move}")
        position = successor
    return position


def _state(
    position: ConnectFourPosition, moves: list[int], human_player: int
) -> dict[str, Any]:
    if position.is_terminal:
        if position.winner == 0:
            status = "Draw"
        elif position.winner == human_player:
            status = "You win"
        else:
            status = "Model wins"
    else:
        status = "Your turn" if position.to_play == human_player else "Model thinking"
    return {
        "board": position.board.tolist(),
        "moves": moves,
        "humanPlayer": human_player,
        "toPlay": position.to_play,
        "legal": position.legal_mask().tolist(),
        "finished": position.is_terminal,
        "winner": position.winner,
        "lastMove": moves[-1] if moves else None,
        "status": status,
    }


class ConnectFourHandler(JSONAppHandler):
    model: nn.Module
    web_root = WEB_ROOT
    index_file = "connect4_play.html"

    def do_POST(self) -> None:  # noqa: N802
        try:
            endpoint = urlparse(self.path).path
            payload = self._request_json()
            human_player = payload.get("humanPlayer", 1)
            if human_player not in (-1, 1):
                raise ValueError("humanPlayer must be 1 or -1")
            if endpoint == "/api/new":
                position = ConnectFourPosition.empty()
                moves: list[int] = []
                if human_player == -1:
                    with self.model_lock:
                        move = select_move(self.model, position)
                    successor = position.play(move)
                    assert successor is not None
                    moves.append(move)
                    position = successor
                self._json(HTTPStatus.OK, _state(position, moves, human_player))
                return
            if endpoint != "/api/play":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
                return
            moves = payload.get("moves", [])
            if not isinstance(moves, list):
                raise ValueError("moves must be a list")
            position = _position_from_moves(moves)
            if position.is_terminal or position.to_play != human_player:
                raise ValueError("It is not the human player's turn")
            move = payload.get("move")
            if isinstance(move, bool) or not isinstance(move, int):
                raise ValueError("move must be an integer column index")
            successor = position.play(move)
            if successor is None:
                raise ValueError("That column is full")
            moves = [*moves, move]
            position = successor
            if not position.is_terminal:
                with self.model_lock:
                    model_move = select_move(self.model, position)
                successor = position.play(model_move)
                if successor is None:
                    raise RuntimeError("Model selected an illegal column")
                moves.append(model_move)
                position = successor
            self._json(HTTPStatus.OK, _state(position, moves, human_player))
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model artifact not found: {args.model}")
    ConnectFourHandler.model = load_model(args.model).eval()
    serve_app(ConnectFourHandler, args.host, args.port, "Connect Four")


if __name__ == "__main__":
    main()
