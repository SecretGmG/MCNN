import numpy as np
import torch

from mcnn.connect4_model import (
    ConnectFourTeacher,
    PhysicalConnectFourModel,
    expand_connect4_teacher,
    tactical_moves,
    winning_windows,
)
from mcnn.connect4_rules import ConnectFourPosition
from mcnn.connect4_solver import ConnectFourSolver, ILLEGAL_SCORE


def test_gravity_win_and_terminal_state() -> None:
    position = ConnectFourPosition.from_moves([0, 1, 0, 1, 0, 1, 0])
    assert position.winner == 1
    assert position.is_terminal
    assert not position.legal_mask().any()
    assert position.play(2) is None


def test_requested_encoding_and_reflection() -> None:
    position = ConnectFourPosition.from_moves([0, 1, 0])
    encoded = position.encode()
    assert encoded.shape == (2, 7, 7)
    assert not encoded[:, 0].any()
    assert encoded.sum() == 3
    reflected = position.reflected()
    np.testing.assert_array_equal(reflected.encode(), encoded[:, :, ::-1])
    assert reflected.history == (6, 5, 6)


def test_solver_opening_and_illegal_column() -> None:
    with ConnectFourSolver() as solver:
        opening = solver.analyze(ConnectFourPosition.empty())
        assert opening.argmax() == 3
        assert opening[3] > 0
        full_column = ConnectFourPosition.from_moves([3, 3, 3, 3, 3, 3])
        scores = solver.analyze(full_column)
        assert scores[3] == ILLEGAL_SCORE


def test_winning_windows_and_model_equivariance() -> None:
    windows = winning_windows()
    assert windows.shape == (69, 4)
    assert len({tuple(window.tolist()) for window in windows}) == 69
    model = ConnectFourTeacher().eval()
    inputs = torch.randint(0, 2, (8, 2, 7, 7), dtype=torch.float32)
    inputs[:, :, 0] = 0
    with torch.no_grad():
        scores = model(inputs)
        reflected = model(inputs.flip(-1)).flip(1)
    torch.testing.assert_close(scores, reflected)


def test_physical_expansion_is_integer_and_equivariant() -> None:
    physical = expand_connect4_teacher(ConnectFourTeacher()).eval()
    inputs = torch.randint(0, 2, (4, 2, 7, 7), dtype=torch.float32)
    inputs[:, :, 0] = 0
    with torch.no_grad():
        scores = physical(inputs)
        reflected = physical(inputs.flip(-1)).flip(1)
    torch.testing.assert_close(scores, reflected)
    assert scores.min() >= 0 and scores.max() <= 15
    for parameter in physical.parameters():
        torch.testing.assert_close(parameter, parameter.round())


def test_physical_round_trip(tmp_path) -> None:
    model = PhysicalConnectFourModel()
    path = tmp_path / "model.pt"
    model.save(path)
    restored = PhysicalConnectFourModel.load(path)
    for before, after in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(before, after)


def test_physical_timing_distinguishes_streaming_from_end_to_end() -> None:
    metrics = PhysicalConnectFourModel().physicality_metrics()
    assert metrics["longest_convolution_lane_redstone_ticks"] == 72
    assert metrics["longest_convolution_lane_seconds"] == 7.2
    assert (
        metrics["end_to_end_latency"]
        == "unknown until a concrete schematic is audited"
    )


def test_tactical_branch_finds_wins_and_blocks() -> None:
    winning = ConnectFourPosition.from_moves([0, 6, 1, 6, 2, 5])
    blocking = ConnectFourPosition.from_moves([6, 0, 6, 1, 5, 2])
    inputs = torch.tensor(
        np.stack((winning.encode(), blocking.encode())), dtype=torch.float32
    )
    wins, blocks = tactical_moves(inputs, winning_windows())
    assert wins[0, 3] and not blocks[0].any()
    assert blocks[1, 3] and not wins[1].any()
    scores = ConnectFourTeacher().eval()(inputs)
    assert int(scores[0].argmax()) == 3
    assert int(scores[1].argmax()) == 3
