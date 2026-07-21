"""Tests for the nn_module-based schematic export."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mcnn.discrete import DiscreteLayer, DiscreteMCNN
from mcnn.export import build_schematic, save_schematic


class NewExportTest(unittest.TestCase):
    def test_single_neuron(self) -> None:
        model = DiscreteMCNN(
            [DiscreteLayer(
                weight=torch.tensor([[5, -4, 0]], dtype=torch.int16),
                bias=torch.tensor([7], dtype=torch.int16),
                output_min=0, output_max=15,
            )]
        )
        schematic = build_schematic(model)

        # Bias barrel at (0,0,0) for output 0
        self.assertTrue(schematic.getBlockStateAt((0, 0, 0)).startswith("minecraft:barrel"))

        # Module for weight 5 at (0,0)
        self.assertEqual(
            schematic.getBlockStateAt((0, 1, 0)),
            "minecraft:comparator[facing=west,mode=subtract,powered=false]",
        )
        # Module for weight -4 at x=2 (i=1)
        self.assertEqual(
            schematic.getBlockStateAt((2, 1, 0)),
            "minecraft:comparator[facing=west,mode=subtract,powered=false]",
        )
        # Weight 0: no module at x=4
        self.assertEqual(schematic.getBlockStateAt((4, 1, 0)), "minecraft:air")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_schematic(model, Path(tmpdir) / "test.schem")
            self.assertTrue(path.exists())

    def test_checkerboard_z1(self) -> None:
        """Redstone blocks at z=1 should form R.R.R checkerboard."""
        model = DiscreteMCNN(
            [DiscreteLayer(
                weight=torch.tensor([[1, 1], [1, 1]], dtype=torch.int16),
                bias=torch.tensor([0, 0], dtype=torch.int16),
                output_min=0, output_max=15,
            )]
        )
        schematic = build_schematic(model)

        # Module at (o=0,i=0)→(0,0): R at (0,0,1) and (1,1,1)
        self.assertEqual(schematic.getBlockStateAt((0, 0, 1)), "minecraft:redstone_block")
        self.assertEqual(schematic.getBlockStateAt((1, 1, 1)), "minecraft:redstone_block")

        # Module at (o=0,i=1)→(2,0): R at (2,0,1) and (3,1,1)
        self.assertEqual(schematic.getBlockStateAt((2, 0, 1)), "minecraft:redstone_block")

        # Module at (o=1,i=0)→(1,1): R at (1,1,1) [overlap!] and (2,2,1)
        # (1,1,1) already placed by module (0,0)
        self.assertEqual(schematic.getBlockStateAt((2, 2, 1)), "minecraft:redstone_block")

        # Air at even-parity positions not covered
        self.assertEqual(schematic.getBlockStateAt((1, 0, 1)), "minecraft:air")

    def test_zero_weight_omitted(self) -> None:
        model = DiscreteMCNN(
            [DiscreteLayer(
                weight=torch.tensor([[0]], dtype=torch.int16),
                bias=torch.tensor([1], dtype=torch.int16),
                output_min=0, output_max=15,
            )]
        )
        schematic = build_schematic(model)
        # Only the bias barrel, no module blocks
        self.assertTrue(schematic.getBlockStateAt((0, 0, 0)).startswith("minecraft:barrel"))
        self.assertEqual(schematic.getBlockStateAt((0, 1, 0)), "minecraft:air")


if __name__ == "__main__":
    unittest.main()
