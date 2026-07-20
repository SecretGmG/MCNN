from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import nbtlib
import mcschematic
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcnn.discrete import DiscreteLayer, DiscreteMCNN  # noqa: E402
from mcnn.export import (  # noqa: E402
    MAX_BIAS_SIGNAL_STRENGTH,
    SchematicExportConfig,
    build_module_atlas,
    build_schematic,
    comparator_subtractions,
    encode_comparator_weight,
    export_plan_from_discrete,
    is_representable_bias_signal,
    save_module_atlas,
    save_schematic,
    signal_strength_barrel,
    signal_strength_furnace,
)


def minecraft_outputs(model: DiscreteMCNN, inputs: torch.Tensor) -> torch.Tensor:
    plan = export_plan_from_discrete(model)
    x = inputs.to(torch.float32)
    for layer_index, layer in enumerate(plan.layers):
        outputs = []
        for row in x.to(torch.int16):
            layer_outputs = []
            for neuron in layer.neurons:
                subtracted = sum(
                    connection.active_subtraction
                    if int(row[connection.input_index].item())
                    else connection.inactive_subtraction
                    for connection in neuron.connections
                )
                layer_outputs.append(neuron.physical_bias - subtracted)
            outputs.append(layer_outputs)
        discrete_layer = model.layers[layer_index]
        x = torch.tensor(outputs, dtype=torch.float32).round().clamp(discrete_layer.output_min, discrete_layer.output_max)
    return x


def minecraft_layer_outputs(model: DiscreteMCNN, inputs: torch.Tensor) -> torch.Tensor:
    plan = export_plan_from_discrete(model)
    outputs = []
    for x in inputs.to(torch.int16):
        layer_outputs = []
        for neuron in plan.layers[0].neurons:
            subtracted = sum(
                connection.active_subtraction if int(x[connection.input_index].item()) else connection.inactive_subtraction
                for connection in neuron.connections
            )
            layer_outputs.append(neuron.physical_bias - subtracted)
        outputs.append(layer_outputs)
    layer = model.layers[0]
    return torch.tensor(outputs, dtype=torch.float32).round().clamp(layer.output_min, layer.output_max)


def load_sponge_schematic(path: Path) -> tuple[tuple[int, int, int], dict[tuple[int, int, int], str]]:
    base = nbtlib.load(path, gzipped=True)["Schematic"]
    width = int(base["Width"])
    height = int(base["Height"])
    length = int(base["Length"])
    palette = {int(value): block for block, value in base["Blocks"]["Palette"].items()}
    data = list(base["Blocks"]["Data"])

    blocks = {}
    for y in range(height):
        for z in range(length):
            for x in range(width):
                block = palette[data[(y * length + z) * width + x]]
                if block != "minecraft:air":
                    blocks[(x, y, z)] = block
    return (width, height, length), blocks


class SingleLayerExportTest(unittest.TestCase):
    def test_module_atlas_exports_isolated_editing_modules(self) -> None:
        atlas = build_module_atlas()
        self.assertEqual(atlas.getBlockStateAt((0, 0, 0)), "minecraft:stone")

        occupied = [
            (x, y, z)
            for x in range(180)
            for y in range(20)
            for z in range(-2, 16)
            if atlas.getBlockStateAt((x, y, z)) != "minecraft:air"
        ]
        max_corner = tuple(max(position[index] for position in occupied) for index in range(3))
        self.assertEqual(atlas.getBlockStateAt(max_corner), "minecraft:stone")

        stone_positions = [position for position in occupied if atlas.getBlockStateAt(position) == "minecraft:stone"]
        self.assertEqual(sorted(stone_positions), [(0, 0, 0), max_corner])
        self.assertGreaterEqual(
            sum(
                1
                for position in occupied
                if atlas.getBlockStateAt(position).startswith("minecraft:repeater")
            ),
            8,
        )
        self.assertTrue(
            any(
                atlas.getBlockStateAt((x, y, z)) == "minecraft:air"
                for x in range(10, 18)
                for y in range(1, 18)
                for z in range(1, 8)
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = save_module_atlas(Path(tmpdir) / "module_atlas.schem")
            self.assertTrue(output_path.exists())

    def test_exports_single_output_layer_with_a_b_c_weight_slots(self) -> None:
        model = DiscreteMCNN(
            [
                DiscreteLayer(
                    weight=torch.tensor([[5, -4, 0]], dtype=torch.int16),
                    bias=torch.tensor([7], dtype=torch.int16),
                    output_min=0,
                    output_max=15,
                )
            ]
        )

        plan = export_plan_from_discrete(model)
        layer = plan.layers[0]
        neuron = layer.neurons[0]

        self.assertEqual(plan.layer_sizes, (3, 1))
        self.assertEqual(layer.input_size, 3)
        self.assertEqual(layer.output_size, 1)
        self.assertEqual(neuron.effective_bias, 7)
        self.assertEqual(neuron.physical_bias, 7 + 15 + 11)
        self.assertEqual(
            [(connection.input_index, connection.encoding.mode, connection.encoding.value) for connection in neuron.connections],
            [(0, "A", 10), (1, "B", 11), (2, "C", 0)],
        )

        schematic = build_schematic(plan)
        (width, height, length), correct_blocks = load_sponge_schematic(ROOT / "artifacts" / "correct.schem")
        for position in ((1, 0, 2), (3, 0, 2)):
            correct_blocks.pop(position)
        correct_blocks[(0, 1, 0)] = "minecraft:barrel"
        correct_blocks[(1, 1, 2)] = "minecraft:light_gray_concrete"
        correct_blocks[(1, 1, 3)] = "minecraft:furnace"
        correct_blocks[(3, 1, 2)] = "minecraft:light_gray_concrete"
        correct_blocks[(3, 1, 3)] = "minecraft:furnace"
        correct_blocks[(3, 1, 5)] = "minecraft:light_gray_concrete"
        generated_blocks = {
            (x, y, z): schematic.getBlockStateAt((x, y, z))
            for y in range(height)
            for z in range(length)
            for x in range(width)
            if schematic.getBlockStateAt((x, y, z)) != "minecraft:air"
        }
        self.assertEqual(generated_blocks, correct_blocks)
        self.assertEqual(schematic.getBlockDataAt((0, 1, 0)), signal_strength_barrel(33))
        self.assertEqual(schematic.getBlockDataAt((1, 1, 3)), signal_strength_furnace(10))
        self.assertEqual(schematic.getBlockDataAt((3, 1, 3)), signal_strength_furnace(11))
        self.assertIn('id:"minecraft:water_bucket"', signal_strength_barrel(33))
        self.assertIn('id:"minecraft:water_bucket"', signal_strength_barrel(777))
        self.assertTrue(is_representable_bias_signal(777))
        self.assertTrue(is_representable_bias_signal(MAX_BIAS_SIGNAL_STRENGTH))
        self.assertTrue(all(is_representable_bias_signal(signal) for signal in range(MAX_BIAS_SIGNAL_STRENGTH + 1)))
        with self.assertRaises(ValueError):
            signal_strength_furnace(-1)
        with self.assertRaises(ValueError):
            signal_strength_barrel(MAX_BIAS_SIGNAL_STRENGTH + 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = save_schematic(plan, Path(tmpdir) / "single_layer_abc.schem")
            self.assertTrue(output_path.exists())

    def test_weight_encoding_excludes_zero_signal_furnaces(self) -> None:
        self.assertEqual((encode_comparator_weight(14).mode, encode_comparator_weight(14).value), ("A", 1))
        self.assertEqual((encode_comparator_weight(-14).mode, encode_comparator_weight(-14).value), ("B", 1))
        self.assertEqual((encode_comparator_weight(0).mode, encode_comparator_weight(0).value), ("C", 0))
        self.assertEqual(comparator_subtractions(14).inactive, 15)
        self.assertEqual(comparator_subtractions(14).active, 1)
        self.assertEqual(comparator_subtractions(0).inactive, 0)
        self.assertEqual(comparator_subtractions(0).active, 0)
        self.assertEqual(comparator_subtractions(-14).inactive, 1)
        self.assertEqual(comparator_subtractions(-14).active, 15)
        for weight in range(-14, 15):
            subtractions = comparator_subtractions(weight)
            self.assertEqual(subtractions.inactive - subtractions.active, weight)
            if weight == 0:
                self.assertEqual(subtractions.encoding.value, 0)
            else:
                self.assertGreaterEqual(subtractions.encoding.value, 1)
                self.assertLessEqual(subtractions.encoding.value, 14)
        with self.assertRaises(ValueError):
            encode_comparator_weight(15)
        with self.assertRaises(ValueError):
            encode_comparator_weight(-15)

    def test_export_plan_matches_discrete_model_for_binary_inputs(self) -> None:
        model = DiscreteMCNN(
            [
                DiscreteLayer(
                    weight=torch.tensor(
                        [
                            [2, 4, 8, 0, 0],
                            [-3, -3, -3, -3, -2],
                            [3, 3, 3, 3, 3],
                            [5, -4, 3, -2, 1],
                            [0, 0, 0, 0, 14],
                        ],
                        dtype=torch.int16,
                    ),
                    bias=torch.tensor([1, 15, 2, 5, 1], dtype=torch.int16),
                    output_min=0,
                    output_max=15,
                )
            ]
        )
        inputs = torch.tensor(
            [[(value >> bit) & 1 for bit in range(5)] for value in range(32)],
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(minecraft_layer_outputs(model, inputs), model.forward(inputs)))
        self.assertEqual(model.physical_biases(), [46, 76, 77, 74, 16])

    def test_multilayer_export_plan_matches_discrete_model_for_binary_inputs(self) -> None:
        model = DiscreteMCNN(
            [
                DiscreteLayer(
                    weight=torch.tensor(
                        [
                            [1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1],
                        ],
                        dtype=torch.int16,
                    ),
                    bias=torch.tensor([0, 0, 0, 0], dtype=torch.int16),
                    output_min=0,
                    output_max=1,
                ),
                DiscreteLayer(
                    weight=torch.tensor(
                        [
                            [14, 0, 0, 0],
                            [0, -14, 0, 0],
                            [6, 0, -4, 0],
                            [0, 0, 0, 12],
                            [5, -4, 3, -2],
                        ],
                        dtype=torch.int16,
                    ),
                    bias=torch.tensor([1, 15, 2, 3, 5], dtype=torch.int16),
                    output_min=0,
                    output_max=15,
                ),
            ]
        )
        inputs = torch.tensor(
            [[(value >> bit) & 1 for bit in range(5)] for value in range(32)],
            dtype=torch.float32,
        )

        self.assertTrue(torch.equal(minecraft_outputs(model, inputs), model.forward(inputs)))
        self.assertEqual(model.physical_biases(), [15, 15, 15, 30, 16, 16, 28, 18, 59])

    def test_discrete_binary_step_hidden_layers_use_positive_threshold(self) -> None:
        model = DiscreteMCNN(
            [
                DiscreteLayer(
                    weight=torch.tensor([[1, -1], [-1, 1]], dtype=torch.int16),
                    bias=torch.tensor([0, 0], dtype=torch.int16),
                    output_min=0,
                    output_max=1,
                    activation="binary_step",
                ),
                DiscreteLayer(
                    weight=torch.tensor([[5, -3]], dtype=torch.int16),
                    bias=torch.tensor([2], dtype=torch.int16),
                    output_min=0,
                    output_max=15,
                ),
            ]
        )
        inputs = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)

        self.assertEqual(model.forward(inputs).to(torch.int16).tolist(), [[2], [7], [0], [2]])
        state = model.state_dict()
        self.assertEqual(state["layers"][0]["activation"], "binary_step")

    def test_export_rejects_neurons_over_physical_bias_cap(self) -> None:
        model = DiscreteMCNN(
            [
                DiscreteLayer(
                    weight=torch.full((1, 64), 14, dtype=torch.int16),
                    bias=torch.tensor([1], dtype=torch.int16),
                    output_min=0,
                    output_max=15,
                )
            ]
        )

        self.assertGreater(model.max_physical_bias(), MAX_BIAS_SIGNAL_STRENGTH)
        with self.assertRaises(ValueError):
            export_plan_from_discrete(model)

    def test_next_layer_bias_continues_diagonal_grid(self) -> None:
        model = DiscreteMCNN(
            [
                DiscreteLayer(
                    weight=torch.tensor([[1, 0]], dtype=torch.int16),
                    bias=torch.tensor([3], dtype=torch.int16),
                    output_min=0,
                    output_max=15,
                ),
                DiscreteLayer(
                    weight=torch.tensor([[0], [1]], dtype=torch.int16),
                    bias=torch.tensor([5, 6], dtype=torch.int16),
                    output_min=0,
                    output_max=15,
                ),
            ]
        )

        schematic = build_schematic(export_plan_from_discrete(model))

        self.assertEqual(schematic.getBlockStateAt((0, 1, 0)), "minecraft:barrel")
        self.assertEqual(schematic.getBlockDataAt((0, 1, 0)), signal_strength_barrel(18))
        for pos, signal, west_block in (
            ((4, 1, 0), 5, "minecraft:comparator[facing=west,mode=subtract,powered=false]"),
            ((5, 2, 0), 21, "minecraft:air"),
        ):
            self.assertEqual(schematic.getBlockStateAt(pos), "minecraft:barrel")
            self.assertEqual(schematic.getBlockDataAt(pos), signal_strength_barrel(signal))
            self.assertEqual(schematic.getBlockStateAt((pos[0] - 1, pos[1], pos[2])), west_block)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = save_schematic(export_plan_from_discrete(model), Path(tmpdir) / "two_layer.schem")
            saved = mcschematic.MCSchematic(str(output_path))

        self.assertEqual(saved.getBlockStateAt((0, 1, 0)), "minecraft:barrel")
        self.assertEqual(saved.getBlockStateAt((4, 1, 0)), "minecraft:barrel")
        self.assertEqual(saved.getBlockStateAt((5, 2, 0)), "minecraft:barrel")

    def test_export_modes_isolate_weights_subtraction_and_input_wiring(self) -> None:
        model = DiscreteMCNN(
            [
                DiscreteLayer(
                    weight=torch.tensor([[5, -4, 0]], dtype=torch.int16),
                    bias=torch.tensor([7], dtype=torch.int16),
                    output_min=0,
                    output_max=15,
                )
            ]
        )
        plan = export_plan_from_discrete(model)

        weights_only = build_schematic(
            plan,
            SchematicExportConfig(emit_weights=True, emit_subtraction=False, emit_input_wiring=False),
        )
        self.assertEqual(weights_only.getBlockStateAt((0, 1, 0)), "minecraft:barrel")
        self.assertEqual(weights_only.getBlockStateAt((1, 1, 2)), "minecraft:light_gray_concrete")
        self.assertEqual(weights_only.getBlockStateAt((1, 1, 3)), "minecraft:furnace")
        self.assertEqual(weights_only.getBlockStateAt((1, 1, 0)), "minecraft:air")

        subtraction_only = build_schematic(
            plan,
            SchematicExportConfig(emit_weights=False, emit_subtraction=True, emit_input_wiring=False),
        )
        self.assertEqual(subtraction_only.getBlockStateAt((0, 1, 0)), "minecraft:magenta_concrete")
        self.assertEqual(subtraction_only.getBlockStateAt((1, 1, 2)), "minecraft:light_gray_concrete")
        self.assertEqual(subtraction_only.getBlockStateAt((1, 1, 3)), "minecraft:purple_concrete")
        self.assertEqual(
            subtraction_only.getBlockStateAt((1, 1, 0)),
            "minecraft:comparator[facing=west,mode=subtract,powered=false]",
        )

        wiring_model = DiscreteMCNN(
            [
                DiscreteLayer(
                    weight=torch.zeros((16, 4), dtype=torch.int16),
                    bias=torch.zeros(16, dtype=torch.int16),
                    output_min=0,
                    output_max=15,
                )
            ]
        )
        wiring_only = build_schematic(
            export_plan_from_discrete(wiring_model),
            SchematicExportConfig(emit_weights=False, emit_subtraction=False, emit_input_wiring=True),
        )
        for pos in ((1, 2, 4), (3, 2, 4), (5, 2, 4), (7, 2, 4)):
            self.assertEqual(wiring_only.getBlockStateAt(pos), "minecraft:target[power=0]")
        for pos in ((0, 1, 4), (2, 1, 4), (4, 1, 4), (6, 1, 4)):
            self.assertEqual(wiring_only.getBlockStateAt(pos), "minecraft:target[power=0]")
        self.assertEqual(
            wiring_only.getBlockStateAt((1, 1, 5)),
            "minecraft:smooth_stone_slab[type=top,waterlogged=false]",
        )
        self.assertEqual(
            wiring_only.getBlockStateAt((15, 15, 5)),
            "minecraft:smooth_stone_slab[type=top,waterlogged=false]",
        )
        self.assertEqual(
            wiring_only.getBlockStateAt((15, 16, 6)),
            "minecraft:redstone_wire[east=none,north=side,power=1,south=side,west=side]",
        )
        self.assertEqual(
            wiring_only.getBlockStateAt((17, 16, 6)),
            "minecraft:repeater[delay=1,facing=north,locked=false,powered=true]",
        )
        self.assertEqual(
            wiring_only.getBlockStateAt((17, 15, 6)),
            "minecraft:smooth_stone_slab[type=top,waterlogged=false]",
        )
        self.assertEqual(wiring_only.getBlockStateAt((15, 15, 7)), "minecraft:black_concrete")
        self.assertEqual(
            wiring_only.getBlockStateAt((18, 17, 6)),
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=true]",
        )
        self.assertEqual(wiring_only.getBlockStateAt((17, 17, 5)), "minecraft:white_concrete")
        self.assertEqual(wiring_only.getBlockStateAt((18, 18, 7)), "minecraft:air")
        self.assertEqual(wiring_only.getBlockStateAt((16, 18, 8)), "minecraft:air")
        self.assertEqual(wiring_only.getBlockStateAt((0, 1, 0)), "minecraft:air")

        output_wiring_only = build_schematic(
            export_plan_from_discrete(wiring_model),
            SchematicExportConfig(
                emit_weights=False,
                emit_subtraction=False,
                emit_input_wiring=False,
                emit_output_wiring=True,
            ),
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((7, 1, 1)),
            "minecraft:air",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((8, 1, 0)),
            "minecraft:air",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((8, 1, -2)),
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((8, 1, -3)),
            "minecraft:redstone_wire[east=side,north=none,power=0,south=side,west=none]",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((8, 1, -1)),
            "minecraft:redstone_wire[east=none,north=side,power=0,south=side,west=none]",
        )
        self.assertEqual(output_wiring_only.getBlockStateAt((7, 1, 0)), "minecraft:air")
        self.assertEqual(
            output_wiring_only.getBlockStateAt((8, 0, -2)),
            "minecraft:smooth_stone_slab[type=top,waterlogged=false]",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((9, 0, -3)),
            "minecraft:redstone_wire[east=side,north=none,power=0,south=none,west=up]",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((11, 0, -2)),
            "minecraft:redstone_wire[east=side,north=none,power=0,south=none,west=up]",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((41, 0, -3)),
            "minecraft:air",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((24, 15, -3)),
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((23, 14, -4)),
            "minecraft:repeater[delay=1,facing=north,locked=false,powered=false]",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((24, 15, -2)),
            "minecraft:redstone_wire[east=none,north=side,power=0,south=none,west=up]",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((24, 15, -4)),
            "minecraft:redstone_wire[east=side,north=none,power=0,south=side,west=none]",
        )
        self.assertEqual(
            output_wiring_only.getBlockStateAt((25, 13, -4)),
            "minecraft:light_gray_concrete",
        )
        self.assertEqual(output_wiring_only.getBlockStateAt((23, 15, -5)), "minecraft:air")
        self.assertEqual(output_wiring_only.getBlockStateAt((25, 15, -4)), "minecraft:air")
        self.assertEqual(output_wiring_only.getBlockStateAt((23, 15, -3)), "minecraft:light_gray_concrete")
        self.assertEqual(output_wiring_only.getBlockStateAt((25, 15, -2)), "minecraft:light_gray_concrete")
        floor_redstone_count = sum(
            1
            for x in range(80)
            for z in range(-6, 1)
            if output_wiring_only.getBlockStateAt((x, 0, z)).startswith("minecraft:redstone_wire")
        )
        self.assertEqual(floor_redstone_count, 16)
        output_repeater_count = sum(
            1
            for x in range(80)
            for y in range(20)
            for z in range(-6, 2)
            if output_wiring_only.getBlockStateAt((x, y, z))
            == "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]"
        )
        self.assertEqual(output_repeater_count, 18)
        output_north_repeater_count = sum(
            1
            for x in range(80)
            for y in range(20)
            for z in range(-6, 2)
            if output_wiring_only.getBlockStateAt((x, y, z))
            == "minecraft:repeater[delay=1,facing=north,locked=false,powered=false]"
        )
        self.assertEqual(output_north_repeater_count, 2)
        output_comparator_count = sum(
            1
            for x in range(80)
            for y in range(20)
            for z in range(-6, 2)
            if output_wiring_only.getBlockStateAt((x, y, z))
            == "minecraft:comparator[facing=south,mode=compare,powered=false]"
        )
        self.assertEqual(output_comparator_count, 0)
        self.assertEqual(
            sum(
                1
                for x in range(80)
                for y in range(20)
                for z in range(-6, 2)
                if output_wiring_only.getBlockStateAt((x, y, z))
                == "minecraft:comparator[facing=west,mode=compare,powered=false]"
            ),
            0,
        )
        output_gray_count = sum(
            1
            for x in range(80)
            for y in range(20)
            for z in range(-6, 1)
            if output_wiring_only.getBlockStateAt((x, y, z)) == "minecraft:light_gray_concrete"
        )
        self.assertEqual(output_gray_count, 9)
        self.assertEqual(
            sum(
                1
                for x in range(80)
                for y in range(20)
                for z in range(-6, 1)
                if output_wiring_only.getBlockStateAt((x, y, z))
                == "minecraft:smooth_stone_slab[type=top,waterlogged=false]"
            ),
            10,
        )
        for position in ((22, 15, -1), (26, 15, -3), (27, 15, -2), (30, 15, -3), (32, 15, -1)):
            self.assertEqual(output_wiring_only.getBlockStateAt(position), "minecraft:air")
        self.assertEqual(output_wiring_only.getBlockStateAt((0, 1, 0)), "minecraft:air")

        tall_output_wiring = build_schematic(
            export_plan_from_discrete(
                DiscreteMCNN(
                    [
                        DiscreteLayer(
                            weight=torch.zeros((40, 4), dtype=torch.int16),
                            bias=torch.zeros(40, dtype=torch.int16),
                            output_min=0,
                            output_max=15,
                        )
                    ]
                )
            ),
            SchematicExportConfig(
                emit_weights=False,
                emit_subtraction=False,
                emit_input_wiring=False,
                emit_output_wiring=True,
            ),
        )
        self.assertEqual(tall_output_wiring.getBlockStateAt((38, 30, -3)), "minecraft:black_concrete")
        self.assertEqual(tall_output_wiring.getBlockStateAt((40, 30, -2)), "minecraft:white_concrete")
        self.assertEqual(tall_output_wiring.getBlockStateAt((42, 30, -3)), "minecraft:black_concrete")
        self.assertEqual(
            tall_output_wiring.getBlockStateAt((42, 28, -3)),
            "minecraft:smooth_stone_slab[type=top,waterlogged=false]",
        )

        odd_input_network_wiring = build_schematic(
            export_plan_from_discrete(
                DiscreteMCNN(
                    [
                        DiscreteLayer(
                            weight=torch.zeros((8, 5), dtype=torch.int16),
                            bias=torch.zeros(8, dtype=torch.int16),
                            output_min=0,
                            output_max=15,
                        ),
                        DiscreteLayer(
                            weight=torch.zeros((5, 8), dtype=torch.int16),
                            bias=torch.zeros(5, dtype=torch.int16),
                            output_min=0,
                            output_max=15,
                        ),
                    ]
                )
            ),
            SchematicExportConfig(
                emit_weights=False,
                emit_subtraction=False,
                emit_input_wiring=True,
                emit_output_wiring=True,
                emit_layer_io_connectors=True,
            ),
        )
        self.assertEqual(odd_input_network_wiring.getBlockStateAt((2, 2, 6)), "minecraft:black_concrete")
        self.assertEqual(odd_input_network_wiring.getBlockStateAt((10, 2, 6)), "minecraft:black_concrete")
        self.assertEqual(odd_input_network_wiring.getBlockStateAt((12, 2, 5)), "minecraft:white_concrete")
        self.assertEqual(odd_input_network_wiring.getBlockStateAt((12, 2, 6)), "minecraft:air")
        self.assertEqual(odd_input_network_wiring.getBlockStateAt((14, 2, 6)), "minecraft:black_concrete")
        self.assertEqual(odd_input_network_wiring.getBlockStateAt((10, 0, -2)), "minecraft:white_concrete")
        self.assertEqual(odd_input_network_wiring.getBlockStateAt((12, 0, -3)), "minecraft:black_concrete")
        for position in ((10, 1, -1), (11, 2, -2), (26, 1, -1), (27, 2, -2)):
            self.assertEqual(
                odd_input_network_wiring.getBlockStateAt(position),
                "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]",
            )
        for position in ((11, 2, -1), (27, 2, -1)):
            self.assertEqual(
                odd_input_network_wiring.getBlockStateAt(position),
                "minecraft:redstone_wire[east=none,north=side,power=0,south=side,west=none]",
            )
        for x, z, block in (
            (11, -1, "minecraft:white_concrete"),
            (13, -2, "minecraft:black_concrete"),
            (15, -1, "minecraft:white_concrete"),
            (17, -2, "minecraft:black_concrete"),
            (19, -1, "minecraft:white_concrete"),
            (21, -2, "minecraft:black_concrete"),
            (23, -1, "minecraft:white_concrete"),
            (25, -2, "minecraft:black_concrete"),
        ):
            self.assertEqual(odd_input_network_wiring.getBlockStateAt((x, -2, z)), block)
        self.assertEqual(odd_input_network_wiring.getBlockStateAt((11, -2, 1)), "minecraft:white_concrete")
        self.assertEqual(
            odd_input_network_wiring.getBlockStateAt((11, 2, 5)),
            "minecraft:redstone_wire[east=up,north=side,power=15,south=none,west=none]",
        )
        for input_size, hidden_size, expected in (
            (
                4,
                6,
                (
                    (9, -2, "minecraft:black_concrete"),
                    (11, -1, "minecraft:white_concrete"),
                    (13, -2, "minecraft:black_concrete"),
                    (15, -1, "minecraft:white_concrete"),
                    (17, -2, "minecraft:black_concrete"),
                    (19, -1, "minecraft:white_concrete"),
                ),
            ),
            (
                5,
                6,
                (
                    (11, -1, "minecraft:white_concrete"),
                    (13, -2, "minecraft:black_concrete"),
                    (15, -1, "minecraft:white_concrete"),
                    (17, -2, "minecraft:black_concrete"),
                    (19, -1, "minecraft:white_concrete"),
                    (21, -2, "minecraft:black_concrete"),
                ),
            ),
        ):
            phased_wiring = build_schematic(
                export_plan_from_discrete(
                    DiscreteMCNN(
                        [
                            DiscreteLayer(
                                weight=torch.zeros((hidden_size, input_size), dtype=torch.int16),
                                bias=torch.zeros(hidden_size, dtype=torch.int16),
                                output_min=0,
                                output_max=15,
                            ),
                            DiscreteLayer(
                                weight=torch.zeros((2, hidden_size), dtype=torch.int16),
                                bias=torch.zeros(2, dtype=torch.int16),
                                output_min=0,
                                output_max=15,
                            ),
                        ]
                    )
                ),
                SchematicExportConfig(
                    emit_weights=False,
                    emit_subtraction=False,
                    emit_input_wiring=True,
                    emit_output_wiring=True,
                    emit_layer_io_connectors=True,
                ),
            )
            for x, z, block in expected:
                self.assertEqual(phased_wiring.getBlockStateAt((x, -2, z)), block)
        full_connector_wiring = build_schematic(
            export_plan_from_discrete(
                DiscreteMCNN(
                    [
                        DiscreteLayer(
                            weight=torch.zeros((3, 2), dtype=torch.int16),
                            bias=torch.zeros(3, dtype=torch.int16),
                            output_min=0,
                            output_max=15,
                        ),
                        DiscreteLayer(
                            weight=torch.zeros((2, 3), dtype=torch.int16),
                            bias=torch.zeros(2, dtype=torch.int16),
                            output_min=0,
                            output_max=15,
                        ),
                    ]
                )
            ),
            SchematicExportConfig(
                emit_weights=True,
                emit_subtraction=True,
                emit_input_wiring=True,
                emit_output_wiring=True,
                emit_layer_io_connectors=True,
            ),
        )
        self.assertEqual(full_connector_wiring.getBlockStateAt((4, 1, 0)), "minecraft:barrel")
        self.assertEqual(
            full_connector_wiring.getBlockStateAt((3, 1, 0)),
            "minecraft:comparator[facing=west,mode=subtract,powered=false]",
        )
        with self.assertRaises(ValueError):
            build_schematic(
                export_plan_from_discrete(
                    DiscreteMCNN(
                        [
                            DiscreteLayer(
                                weight=torch.zeros((1, 1), dtype=torch.int16),
                                bias=torch.zeros(1, dtype=torch.int16),
                                output_min=0,
                                output_max=15,
                            ),
                            DiscreteLayer(
                                weight=torch.zeros((1, 1), dtype=torch.int16),
                                bias=torch.zeros(1, dtype=torch.int16),
                                output_min=0,
                                output_max=15,
                            ),
                        ]
                    )
                ),
                SchematicExportConfig(emit_layer_io_connectors=True),
            )


if __name__ == "__main__":
    unittest.main()
