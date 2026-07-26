from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mcnn import MCNNModel, build_schematic, save_schematic


class ExportTest(unittest.TestCase):
    def test_single_layer_exports_and_saves(self) -> None:
        model = MCNNModel.from_discrete_layers(
            [
                {
                    "weight": torch.tensor([[5, -4, 0]], dtype=torch.int16),
                    "bias": torch.tensor([7], dtype=torch.int16),
                    "activation": "bounded_integer",
                    "output_min": 0,
                    "output_max": 15,
                }
            ],
            bias_min=0.0,
        )

        schematic = build_schematic(model)
        self.assertTrue(schematic.getBlockStateAt((-1, 1, 0)).startswith("minecraft:barrel"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_schematic(model, Path(tmpdir) / "test.schem")
            self.assertTrue(path.exists())

    def test_zero_weight_keeps_module_footprint_without_weight_barrel(self) -> None:
        model = MCNNModel.from_discrete_layers(
            [
                {
                    "weight": torch.tensor([[0]], dtype=torch.int16),
                    "bias": torch.tensor([1], dtype=torch.int16),
                    "activation": "bounded_integer",
                    "output_min": 0,
                    "output_max": 15,
                }
            ],
            bias_min=0.0,
        )

        schematic = build_schematic(model)
        self.assertEqual(
            schematic.getBlockStateAt((0, 0, 0)),
            "minecraft:light_gray_concrete",
        )
        self.assertEqual(schematic.getBlockStateAt((0, 1, 1)), "minecraft:air")
        self.assertEqual(schematic.getBlockStateAt((0, 1, 3)), "minecraft:air")

    def test_last_layer_adds_output_circuit(self) -> None:
        model = MCNNModel.from_discrete_layers(
            [
                {
                    "weight": torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int16),
                    "bias": torch.tensor([1, 1, 1], dtype=torch.int16),
                    "activation": "bounded_integer",
                    "output_min": 0,
                    "output_max": 15,
                }
            ],
            bias_min=0.0,
        )

        schematic = build_schematic(model)
        self.assertEqual(
            schematic.getBlockStateAt((3, 0, -1)),
            "minecraft:green_concrete",
        )
        self.assertEqual(
            schematic.getBlockStateAt((3, 1, -1)),
            "minecraft:redstone_wire",
        )
        self.assertEqual(
            schematic.getBlockStateAt((3, 2, -1)),
            "minecraft:green_concrete",
        )
        self.assertEqual(
            schematic.getBlockStateAt((3, 0, -2)),
            "minecraft:oak_wall_sign[facing=north,waterlogged=false]",
        )


if __name__ == "__main__":
    unittest.main()
