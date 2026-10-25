import numpy as np

from mcnn.go_match import relative_final_score, relative_katago_score, replay_history
from mcnn.go_rules import (
    BLACK,
    WHITE,
    GoPosition,
    move_from_gtp,
    move_to_gtp,
    suicide_attempts_from_encoding,
    territory_ownership,
)


def test_gtp_coordinates_round_trip() -> None:
    for move in range(50):
        assert move_from_gtp(move_to_gtp(move)) == move


def test_capture_and_suicide_are_enforced() -> None:
    board = np.zeros((7, 7), dtype=np.int8)
    board[2, 3] = BLACK
    board[1, 3] = WHITE
    board[2, 2] = WHITE
    board[2, 4] = WHITE
    captured = GoPosition(board=board, to_play=WHITE).play(3 * 7 + 3)
    assert captured is not None
    assert captured.board[2, 3] == 0

    board = np.zeros((7, 7), dtype=np.int8)
    board[1, 3] = BLACK
    board[2, 2] = BLACK
    board[2, 4] = BLACK
    board[3, 3] = BLACK
    assert GoPosition(board=board, to_play=WHITE).play(2 * 7 + 3) is None


def test_territory_ownership_labels_enclosed_and_neutral_regions() -> None:
    board = np.zeros((7, 7), dtype=np.int8)
    board[0, 1] = BLACK
    board[1, 0] = BLACK
    board[1, 2] = BLACK
    board[2, 1] = BLACK
    board[6, 6] = WHITE
    ownership = territory_ownership(board)
    assert ownership[1, 1] == BLACK
    assert ownership[6, 6] == WHITE
    assert ownership[4, 4] == 0


def test_match_scores_preserve_player_perspective() -> None:
    position = replay_history([["B", "D4"], ["W", "pass"]])
    response = {"rootInfo": {"scoreLead": 3.5}}
    assert position.to_play == BLACK
    assert relative_katago_score(response, position, BLACK) == 3.5
    assert relative_katago_score(response, position, WHITE) == -3.5
    assert relative_final_score(GoPosition(), BLACK) == -7.5


def test_suicide_attempts_can_be_generated_as_negative_examples() -> None:
    board = np.zeros((7, 7), dtype=np.int8)
    board[1, 3] = BLACK
    board[2, 2] = BLACK
    board[2, 4] = BLACK
    board[3, 3] = BLACK
    attempts = suicide_attempts_from_encoding(
        GoPosition(board=board, to_play=WHITE).encode()
    )
    assert len(attempts) == 1
    assert attempts[0, 1, 2, 3] == 1
