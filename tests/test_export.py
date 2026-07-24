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


if __name__ == "__main__":
    unittest.main()
