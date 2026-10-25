from __future__ import annotations

import unittest

import torch

from mcnn import MCNNModel, cap_bias_signals, physicality_metrics, quantize_model, remove_hidden_unit


class ModelPipelineTest(unittest.TestCase):
    def test_forward_exact_binary_hidden_and_bounded_output(self) -> None:
        model = MCNNModel.from_discrete_layers(
            [
                {
                    "weight": torch.tensor([[1, -1], [-1, 1]], dtype=torch.int16),
                    "bias": torch.tensor([0, 0], dtype=torch.int16),
                    "activation": "binary_step",
                    "output_min": 0,
                    "output_max": 1,
                },
                {
                    "weight": torch.tensor([[3, 1]], dtype=torch.int16),
                    "bias": torch.tensor([2], dtype=torch.int16),
                    "activation": "bounded_integer",
                    "output_min": 0,
                    "output_max": 15,
                },
            ],
            bias_min=0.0,
        )

        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = model(x, mode="exact")
        expected = torch.tensor([[5.0], [3.0]])
        self.assertTrue(torch.equal(y, expected))

    def test_temperature_surrogate_keeps_binary_forward_exact(self) -> None:
        model = MCNNModel.from_discrete_layers(
            [
                {
                    "weight": torch.tensor([[1]], dtype=torch.int16),
                    "bias": torch.tensor([0], dtype=torch.int16),
                    "activation": "binary_step",
                    "output_min": 0,
                    "output_max": 1,
                }
            ],
            bias_min=0.0,
        )
        x = torch.tensor([[1.0], [-1.0]])
        model.set_binary_surrogate_temperature_(6.0)
        self.assertTrue(torch.equal(model(x, mode="ste"), torch.tensor([[1.0], [0.0]])))
        self.assertTrue(torch.equal(model(x, mode="exact"), torch.tensor([[1.0], [0.0]])))

    def test_quantize_model_produces_integer_snapshot(self) -> None:
        model = MCNNModel([2, 2], hidden_activation="bounded_integer", output_activation="bounded_integer")
        quantized = quantize_model(model)
        layers = quantized.discrete_layers()

        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0].weight.dtype, torch.int16)
        self.assertEqual(layers[0].bias.dtype, torch.int16)

    def test_remove_hidden_unit_compacts_adjacent_layers(self) -> None:
        model = MCNNModel.from_discrete_layers(
            [
                {
                    "weight": torch.tensor(
                        [[1, 2], [3, 4], [5, 6]],
                        dtype=torch.int16,
                    ),
                    "bias": torch.tensor([1, 2, 3], dtype=torch.int16),
                    "activation": "binary_step",
                    "output_min": 0,
                    "output_max": 1,
                },
                {
                    "weight": torch.tensor([[1, 2, 3]], dtype=torch.int16),
                    "bias": torch.tensor([1], dtype=torch.int16),
                    "activation": "bounded_integer",
                    "output_min": 0,
                    "output_max": 15,
                },
            ],
            bias_min=0.0,
        )

        pruned = remove_hidden_unit(model, layer_index=0, unit_index=1)
        self.assertEqual(pruned.layer_sizes(), [2, 2, 1])

    def test_cap_bias_signals_keeps_export_range(self) -> None:
        model = MCNNModel.from_discrete_layers(
            [
                {
                    "weight": torch.full((1, 64), 14, dtype=torch.int16),
                    "bias": torch.tensor([900], dtype=torch.int16),
                    "activation": "bounded_integer",
                    "output_min": 0,
                    "output_max": 15,
                }
            ],
            bias_min=0.0,
        )

        capped, edits = cap_bias_signals(model)
        self.assertGreater(edits, 0)
        self.assertLessEqual(capped.max_bias_signal_strength(), 897)

    def test_physicality_metrics_report_exact_module_and_signal_cost(self) -> None:
        model = MCNNModel.from_discrete_layers(
            [
                {
                    "weight": torch.tensor([[1, 0], [-2, 3]], dtype=torch.int16),
                    "bias": torch.tensor([2, 4], dtype=torch.int16),
                    "activation": "bounded_integer",
                    "output_min": 0,
                    "output_max": 15,
                }
            ],
            bias_min=0.0,
        )

        metrics = physicality_metrics(model, signal_target=20)
        self.assertEqual(metrics["total_weight_slots"], 4)
        self.assertEqual(metrics["active_weight_modules"], 3)
        self.assertEqual(metrics["max_signal"], 34)
        self.assertEqual(metrics["signal_headroom"], 863)


if __name__ == "__main__":
    unittest.main()
