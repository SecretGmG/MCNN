import numpy as np
import torch

from mcnn.go_model import MinecraftGoConfig, MinecraftGoModel
from mcnn.go_inference import load_successor_model, select_successor_move
from mcnn.go_rules import BLACK, WHITE, GoPosition
from mcnn.minecraft_network import (
    MinecraftLayer,
    MinecraftLayerConfig,
    cyclic_cell_mask,
)


def test_cyclic_cell_mask_wraps_downward() -> None:
    mask = cyclic_cell_mask(24, 24, shift_depth=1)
    assert mask.shape == (24, 24)
    # Output cell zero receives input cells zero and two through wraparound.
    assert mask[0, :8].all()
    assert not mask[0, 8:16].any()
    assert mask[0, 16:24].all()
    # Every output neuron sees two complete cells.
    assert torch.equal(mask.sum(1), torch.full((24,), 16))


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
