import numpy as np
import torch

from mcnn.go_model import MinecraftGoConfig, MinecraftGoModel
from mcnn.go_inference import load_successor_model, select_successor_move
from mcnn.go_rules import BLACK, WHITE, GoPosition
from mcnn.minecraft_network import (
    MinecraftLayer,
    MinecraftLayerConfig,
    cyclic_cell_mask,
    cyclic_diagonal_mask,
    cyclic_diagonal_order,
)


def test_cyclic_diagonal_order_starts_four_down_and_ends_on_diagonal() -> None:
    assert cyclic_diagonal_order(8) == (4, 1, 5, 2, 6, 3, 7, 0)
    order = cyclic_diagonal_order(24)
    assert order[:5] == (4, 8, 12, 16, 20)
    assert order[-1] == 0
    assert sorted(order) == list(range(24))


def test_cyclic_cell_mask_uses_the_physical_diagonal_strip() -> None:
    mask = cyclic_cell_mask(24, 24, shift_depth=1)
    assert mask.shape == (24, 24)
    # A value first reaches the neuron four rows down; the diagonal is last.
    assert mask[4, 0]
    assert not mask[0, 0]
    assert torch.equal(mask.sum(1), torch.full((24,), 16))
    assert cyclic_diagonal_mask(24, 24, 24).all()


def test_exact_layer_uses_ternary_weights_and_threshold_count() -> None:
    layer = MinecraftLayer(
        8,
        MinecraftLayerConfig(output_size=8, shift_depth=0),
        initial_weight_std=0.0,
        surrogate_temperature=1.0,
    )
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0, :3] = torch.tensor([0.6, -0.6, 0.4])
        layer.threshold[0] = torch.arange(-2, 13)
    inputs = torch.tensor([[5.0, 2.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # Rounded weights are +1, -1, 0, so the preactivation is three.
    assert layer(inputs, mode="exact")[0, 0] == 5


def test_go_model_is_exact_and_within_accumulator_limit(tmp_path) -> None:
    config = MinecraftGoConfig(
        hidden_sizes=(16, 16),
        shift_depths=(1, 1, 1),
    )
    model = MinecraftGoModel(config).project_().eval()
    inputs = torch.randint(0, 2, (4, 2, 7, 7), dtype=torch.float32)
    before = model.score_units(inputs, mode="exact")
    assert torch.equal(before, before.round())
    assert model.output_lines(inputs, mode="exact").min() >= 0
    assert model.output_lines(inputs, mode="exact").max() <= 15
    assert model.physicality()["within_accumulator_limit"]

    path = tmp_path / "model.pt"
    model.save(path)
    loaded = MinecraftGoModel.load(path)
    assert torch.equal(before, loaded.score_units(inputs, mode="exact"))
    assert isinstance(load_successor_model(str(path)), MinecraftGoModel)
    assert 0 <= select_successor_move(loaded, GoPosition()) <= 49


def test_move_selection_adds_captures_mechanically() -> None:
    board = np.zeros((7, 7), dtype=np.int8)
    board[2, 3] = BLACK
    board[1, 3] = WHITE
    board[2, 2] = WHITE
    board[2, 4] = WHITE
    position = GoPosition(board=board, to_play=WHITE)
    model = MinecraftGoModel(
        MinecraftGoConfig(hidden_sizes=(16,), shift_depths=(1, 1))
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    assert select_successor_move(model, position) == 3 * 7 + 3


def test_pass_wins_an_exact_model_tie() -> None:
    model = MinecraftGoModel(
        MinecraftGoConfig(hidden_sizes=(16,), shift_depths=(1, 1))
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    assert select_successor_move(model, GoPosition()) == 49


def test_network_timing_separates_latency_from_stream_throughput() -> None:
    model = MinecraftGoModel(
        MinecraftGoConfig(hidden_sizes=(16, 16), shift_depths=(1, 1, 1))
    )
    timing = model.timing(batch_size=4)
    # Three layers, each processing two connected cells: 3 * (128 + 120),
    # with two eight-tick inter-layer handoffs.
    assert timing["first_result_redstone_ticks"] == 760
    assert timing["batch_tail_redstone_ticks"] == 6
    assert timing["last_result_redstone_ticks"] == 766
    assert timing["steady_state_evaluations_per_second"] == 5
