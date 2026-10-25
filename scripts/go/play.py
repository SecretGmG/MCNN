"""Serve a local browser UI for a human-vs-model 7x7 Go game."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from torch import nn

from mcnn.go_inference import load_successor_model, select_successor_move
from mcnn.go_rules import BLACK, GoPosition
from scripts.common.server import JSONAppHandler, serve_app


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
DEFAULT_MODEL = ROOT / "artifacts/go7x7/model.pt"


def _position_from_moves(moves: list[int]) -> GoPosition:
    position = GoPosition()
    for move in moves:
        if isinstance(move, bool) or not isinstance(move, int):
            raise ValueError("Moves must be integer board indices")
        successor = position.play(move)
        if successor is None:
            raise ValueError(f"Move {move} is illegal in this game")
        position = successor
    return position


def _state(position: GoPosition, moves: list[int]) -> dict[str, Any]:
    black_score, white_score = position.score()
    if position.is_finished:
        status = "Black wins" if position.winner() == BLACK else "White wins"
    else:
        status = "Your turn" if position.to_play == BLACK else "Model thinking"
    return {
        "board": position.board.tolist(),
        "moves": moves,
        "to_play": position.to_play,
        "legal": position.legal_moves().tolist(),
        "finished": position.is_finished,
        "last_move": moves[-1] if moves else None,
        "score": {"black": black_score, "white": white_score + 7.5},
        "status": status,
    }


class GoAppHandler(JSONAppHandler):
    model: nn.Module
    web_root = WEB_ROOT
    index_file = "go_play.html"

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/api/new":
            self._json(HTTPStatus.OK, _state(GoPosition(), []))
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        try:
            endpoint = urlparse(self.path).path
            payload = self._request_json()
            moves = payload.get("moves", [])
            if not isinstance(moves, list):
                raise ValueError("moves must be a list")
            position = _position_from_moves(moves)
            if endpoint == "/api/state":
                self._json(HTTPStatus.OK, _state(position, moves))
                return
            if endpoint != "/api/play":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
                return
            if position.is_finished or position.to_play != BLACK:
                raise ValueError("It is not the human player's turn")
            move = payload.get("move")
            if isinstance(move, bool) or not isinstance(move, int):
                raise ValueError("move must be an integer board index")
            successor = position.play(move)
            if successor is None:
                raise ValueError("That move is illegal")
            moves = [*moves, move]
            position = successor
            if not position.is_finished:
                with self.model_lock:
                    model_move = select_successor_move(self.model, position)
                successor = position.play(model_move)
                if successor is None:
                    raise RuntimeError("Model selected an illegal move")
                moves.append(model_move)
                position = successor
            self._json(HTTPStatus.OK, _state(position, moves))
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:  # Keep a browser-friendly response for local experimentation.
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model artifact not found: {args.model}")
    model = load_successor_model(str(args.model))
    model.eval()
    GoAppHandler.model = model
    serve_app(GoAppHandler, args.host, args.port, "Go")


if __name__ == "__main__":
    main()
