"""Tests for the discrete model runtime."""

from __future__ import annotations

import unittest

import torch

from mcnn.discrete import (
    DiscreteLayer,
    DiscreteMCNN,
    discrete_from_state_dict,
    remove_hidden_unit,
)
from mcnn.encoding import inactive_subtraction


class DiscreteLayerTest(unittest.TestCase):
    """Tests for DiscreteLayer construction and clone."""

    def test_basic_construction(self) -> None:
        weight = torch.tensor([[1, -2], [3, 0]], dtype=torch.int16)
        bias = torch.tensor([5, 7], dtype=torch.int16)
        layer = DiscreteLayer(
            weight=weight,
            bias=bias,
            output_min=0.0,
            output_max=15.0,
            activation="bounded_integer",
        )
        self.assertTrue(torch.equal(layer.weight, weight))
        self.assertTrue(torch.equal(layer.bias, bias))
        self.assertEqual(layer.output_min, 0.0)
        self.assertEqual(layer.output_max, 15.0)
        self.assertEqual(layer.activation, "bounded_integer")

    def test_clone_produces_independent_copy(self) -> None:
        weight = torch.tensor([[1, -2], [3, 0]], dtype=torch.int16)
        bias = torch.tensor([5, 7], dtype=torch.int16)
        layer = DiscreteLayer(weight=weight, bias=bias, output_min=0.0, output_max=15.0)
        cloned = layer.clone()
        self.assertTrue(torch.equal(cloned.weight, layer.weight))
        self.assertTrue(torch.equal(cloned.bias, layer.bias))
        self.assertEqual(cloned.output_min, layer.output_min)
        self.assertEqual(cloned.output_max, layer.output_max)
        self.assertEqual(cloned.activation, layer.activation)
        # Mutate clone, original should be unchanged
        cloned.weight[0, 0] = 99
        self.assertNotEqual(layer.weight[0, 0].item(), 99)


class DiscreteMCNNForwardTest(unittest.TestCase):
    """Tests for DiscreteMCNN forward pass."""

    def test_single_layer_forward_matches_manual(self) -> None:
        # One layer: 2 inputs -> 2 outputs
        weight = torch.tensor([[2, -1], [0, 3]], dtype=torch.int16)
        bias = torch.tensor([1, 2], dtype=torch.int16)
        layer = DiscreteLayer(weight=weight, bias=bias, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer])

        x = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
        y = model.forward(x)

        # Manual: y[0] = clamp(round(2*1 + (-1)*2 + 1), 0, 15) = clamp(round(1), 0, 15) = 1
        #          y[1] = clamp(round(0*1 + 3*2 + 2), 0, 15) = clamp(round(8), 0, 15) = 8
        #          y[2] = clamp(round(2*3 + (-1)*0 + 1), 0, 15) = clamp(round(7), 0, 15) = 7
        #          y[3] = clamp(round(0*3 + 3*0 + 2), 0, 15) = clamp(round(2), 0, 15) = 2
        expected = torch.tensor([[1.0, 8.0], [7.0, 2.0]])
        self.assertTrue(torch.equal(y, expected))

    def test_multilayer_forward_matches_manual(self) -> None:
        # Layer 0: 2 -> 2, Layer 1: 2 -> 1
        w0 = torch.tensor([[1, 0], [0, 1]], dtype=torch.int16)
        b0 = torch.tensor([0, 0], dtype=torch.int16)
        w1 = torch.tensor([[2, -1]], dtype=torch.int16)
        b1 = torch.tensor([1], dtype=torch.int16)

        layer0 = DiscreteLayer(weight=w0, bias=b0, output_min=0.0, output_max=15.0)
        layer1 = DiscreteLayer(weight=w1, bias=b1, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer0, layer1])

        x = torch.tensor([[3.0, 5.0]])
        y = model.forward(x)

        # h = clamp(round([3, 5]), 0, 15) = [3, 5]
        # y = clamp(round(2*3 + (-1)*5 + 1), 0, 15) = clamp(round(2), 0, 15) = 2
        expected = torch.tensor([[2.0]])
        self.assertTrue(torch.equal(y, expected))

    def test_forward_clamps_output(self) -> None:
        weight = torch.tensor([[10]], dtype=torch.int16)
        bias = torch.tensor([0], dtype=torch.int16)
        layer = DiscreteLayer(weight=weight, bias=bias, output_min=0.0, output_max=5.0)
        model = DiscreteMCNN([layer])

        # 10 * 2 + 0 = 20, round -> 20, clamp to 5
        x = torch.tensor([[2.0]])
        y = model.forward(x)
        self.assertEqual(y.item(), 5.0)

        # 10 * (-1) + 0 = -10, round -> -10, clamp to 0
        x = torch.tensor([[-1.0]])
        y = model.forward(x)
        self.assertEqual(y.item(), 0.0)


class DiscreteMCNNBinaryStepTest(unittest.TestCase):
    """Tests for DiscreteMCNN forward with binary_step activation."""

    def test_binary_step_threshold_at_positive(self) -> None:
        weight = torch.tensor([[1, -1]], dtype=torch.int16)
        bias = torch.tensor([0], dtype=torch.int16)
        layer = DiscreteLayer(
            weight=weight,
            bias=bias,
            output_min=0.0,
            output_max=1.0,
            activation="binary_step",
        )
        model = DiscreteMCNN([layer])

        # pre_activation = 1*1 + (-1)*0 + 0 = 1 > 0 -> 1
        x = torch.tensor([[1.0, 0.0]])
        y = model.forward(x)
        self.assertEqual(y.item(), 1.0)

        # pre_activation = 1*0 + (-1)*1 + 0 = -1 <= 0 -> 0
        x = torch.tensor([[0.0, 1.0]])
        y = model.forward(x)
        self.assertEqual(y.item(), 0.0)

        # pre_activation = 0 -> not > 0 -> 0
        x = torch.tensor([[0.0, 0.0]])
        y = model.forward(x)
        self.assertEqual(y.item(), 0.0)

    def test_binary_step_hidden_with_bounded_integer_output(self) -> None:
        # Hidden layer: binary_step, Output layer: bounded_integer
        w0 = torch.tensor([[2, 0], [0, -1]], dtype=torch.int16)
        b0 = torch.tensor([-1, 2], dtype=torch.int16)
        w1 = torch.tensor([[3, 1]], dtype=torch.int16)
        b1 = torch.tensor([0], dtype=torch.int16)

        layer0 = DiscreteLayer(
            weight=w0,
            bias=b0,
            output_min=0.0,
            output_max=1.0,
            activation="binary_step",
        )
        layer1 = DiscreteLayer(
            weight=w1,
            bias=b1,
            output_min=0.0,
            output_max=15.0,
            activation="bounded_integer",
        )
        model = DiscreteMCNN([layer0, layer1])

        # Input [1, 0]:
        #   h0 = (2*1 + 0*0 + (-1) = 1) > 0 -> 1
        #   h1 = (0*1 + (-1)*0 + 2 = 2) > 0 -> 1
        #   y = round(3*1 + 1*1 + 0) = 4
        x = torch.tensor([[1.0, 0.0]])
        y = model.forward(x)
        self.assertEqual(y.item(), 4.0)

    def test_unknown_activation_raises(self) -> None:
        layer = DiscreteLayer(
            weight=torch.zeros(1, 1, dtype=torch.int16),
            bias=torch.zeros(1, dtype=torch.int16),
            output_min=0.0,
            output_max=1.0,
            activation="relu",
        )
        model = DiscreteMCNN([layer])
        with self.assertRaises(ValueError):
            model.forward(torch.tensor([[1.0]]))


class DiscreteMCNNLayerSizesTest(unittest.TestCase):
    """Tests for DiscreteMCNN.layer_sizes()."""

    def test_single_layer(self) -> None:
        weight = torch.zeros(3, 4, dtype=torch.int16)
        bias = torch.zeros(3, dtype=torch.int16)
        layer = DiscreteLayer(weight=weight, bias=bias, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer])
        self.assertEqual(model.layer_sizes(), [4, 3])

    def test_multiple_layers(self) -> None:
        w0 = torch.zeros(5, 3, dtype=torch.int16)
        b0 = torch.zeros(5, dtype=torch.int16)
        w1 = torch.zeros(2, 5, dtype=torch.int16)
        b1 = torch.zeros(2, dtype=torch.int16)
        layer0 = DiscreteLayer(weight=w0, bias=b0, output_min=0.0, output_max=15.0)
        layer1 = DiscreteLayer(weight=w1, bias=b1, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer0, layer1])
        self.assertEqual(model.layer_sizes(), [3, 5, 2])

    def test_empty_model(self) -> None:
        model = DiscreteMCNN([])
        self.assertEqual(model.layer_sizes(), [])


class DiscreteMCNNNNonzeroWeightsTest(unittest.TestCase):
    """Tests for DiscreteMCNN.nonzero_weights()."""

    def test_counts_nonzero_correctly(self) -> None:
        w0 = torch.tensor([[1, 0, -2], [0, 0, 3]], dtype=torch.int16)
        b0 = torch.zeros(2, dtype=torch.int16)
        w1 = torch.tensor([[0, 1], [-1, 0]], dtype=torch.int16)
        b1 = torch.zeros(2, dtype=torch.int16)
        layer0 = DiscreteLayer(weight=w0, bias=b0, output_min=0.0, output_max=15.0)
        layer1 = DiscreteLayer(weight=w1, bias=b1, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer0, layer1])
        # w0 has 3 nonzeros, w1 has 2 nonzeros -> 5 total
        self.assertEqual(model.nonzero_weights(), 5)

    def test_all_zero_weights(self) -> None:
        w = torch.zeros(2, 3, dtype=torch.int16)
        b = torch.zeros(2, dtype=torch.int16)
        layer = DiscreteLayer(weight=w, bias=b, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer])
        self.assertEqual(model.nonzero_weights(), 0)


class DiscreteMCNNPhysicalBiasesTest(unittest.TestCase):
    """Tests for DiscreteMCNN.physical_biases()."""

    def test_matches_manual_computation(self) -> None:
        # Single layer with 2 outputs, 3 inputs
        weight = torch.tensor([[2, -1, 0], [0, 3, -2]], dtype=torch.int16)
        bias = torch.tensor([5, 7], dtype=torch.int16)
        layer = DiscreteLayer(weight=weight, bias=bias, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer])

        physical = model.physical_biases()

        # Output 0: bias=5 + inactive(2) + inactive(-1) + inactive(0)
        #   inactive(2) = 13 (mode A: 15-2=13)
        #   inactive(-1) = 14 (mode B: -1+15=14)
        #   inactive(0) = 0
        #   total = 5 + 13 + 14 + 0 = 32
        expected_0 = (
            5
            + inactive_subtraction(2)
            + inactive_subtraction(-1)
            + inactive_subtraction(0)
        )

        # Output 1: bias=7 + inactive(0) + inactive(3) + inactive(-2)
        #   inactive(0) = 0
        #   inactive(3) = 12 (mode A: 15-3=12)
        #   inactive(-2) = 13 (mode B: -2+15=13)
        #   total = 7 + 0 + 12 + 13 = 32
        expected_1 = (
            7
            + inactive_subtraction(0)
            + inactive_subtraction(3)
            + inactive_subtraction(-2)
        )

        self.assertEqual(physical, [expected_0, expected_1])

    def test_multilayer_physical_biases(self) -> None:
        w0 = torch.tensor([[1, 0]], dtype=torch.int16)
        b0 = torch.tensor([2], dtype=torch.int16)
        w1 = torch.tensor([[-1], [2]], dtype=torch.int16)
        b1 = torch.tensor([3, 4], dtype=torch.int16)

        layer0 = DiscreteLayer(weight=w0, bias=b0, output_min=0.0, output_max=15.0)
        layer1 = DiscreteLayer(weight=w1, bias=b1, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer0, layer1])

        physical = model.physical_biases()

        # Layer 0, output 0: 2 + inactive(1) + inactive(0) = 2 + 14 + 0 = 16
        expected_0 = 2 + inactive_subtraction(1) + inactive_subtraction(0)
        # Layer 1, output 0: 3 + inactive(-1) = 3 + 14 = 17
        expected_1 = 3 + inactive_subtraction(-1)
        # Layer 1, output 1: 4 + inactive(2) = 4 + 13 = 17
        expected_2 = 4 + inactive_subtraction(2)

        self.assertEqual(physical, [expected_0, expected_1, expected_2])


class DiscreteMCNNMaxPhysicalBiasTest(unittest.TestCase):
    """Tests for DiscreteMCNN.max_physical_bias()."""

    def test_returns_max_of_physical_biases(self) -> None:
        weight = torch.tensor([[2, -1], [0, 3]], dtype=torch.int16)
        bias = torch.tensor([5, 7], dtype=torch.int16)
        layer = DiscreteLayer(weight=weight, bias=bias, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer])

        physical = model.physical_biases()
        self.assertEqual(model.max_physical_bias(), max(physical))

    def test_empty_model_returns_zero(self) -> None:
        model = DiscreteMCNN([])
        self.assertEqual(model.max_physical_bias(), 0)


class DiscreteMCNNHiddenUnitsTest(unittest.TestCase):
    """Tests for DiscreteMCNN.hidden_units()."""

    def test_single_layer_has_no_hidden_units(self) -> None:
        weight = torch.zeros(3, 4, dtype=torch.int16)
        bias = torch.zeros(3, dtype=torch.int16)
        layer = DiscreteLayer(weight=weight, bias=bias, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer])
        self.assertEqual(model.hidden_units(), 0)

    def test_multilayer_sums_hidden_sizes(self) -> None:
        # 3 -> 5 -> 2 -> 4: hidden sizes are 5 + 2 = 7
        w0 = torch.zeros(5, 3, dtype=torch.int16)
        b0 = torch.zeros(5, dtype=torch.int16)
        w1 = torch.zeros(2, 5, dtype=torch.int16)
        b1 = torch.zeros(2, dtype=torch.int16)
        w2 = torch.zeros(4, 2, dtype=torch.int16)
        b2 = torch.zeros(4, dtype=torch.int16)

        layers = [
            DiscreteLayer(weight=w0, bias=b0, output_min=0.0, output_max=15.0),
            DiscreteLayer(weight=w1, bias=b1, output_min=0.0, output_max=15.0),
            DiscreteLayer(weight=w2, bias=b2, output_min=0.0, output_max=15.0),
        ]
        model = DiscreteMCNN(layers)
        self.assertEqual(model.hidden_units(), 7)


class DiscreteMCNNStateDictRoundtripTest(unittest.TestCase):
    """Tests for state_dict / discrete_from_state_dict roundtrip."""

    def test_roundtrip_preserves_model(self) -> None:
        w0 = torch.tensor([[2, -1], [0, 3]], dtype=torch.int16)
        b0 = torch.tensor([5, 7], dtype=torch.int16)
        w1 = torch.tensor([[1, -2]], dtype=torch.int16)
        b1 = torch.tensor([3], dtype=torch.int16)

        layer0 = DiscreteLayer(
            weight=w0,
            bias=b0,
            output_min=0.0,
            output_max=15.0,
            activation="binary_step",
        )
        layer1 = DiscreteLayer(
            weight=w1,
            bias=b1,
            output_min=0.0,
            output_max=15.0,
            activation="bounded_integer",
        )
        original = DiscreteMCNN([layer0, layer1])

        state = original.state_dict()
        restored = discrete_from_state_dict(state)

        self.assertEqual(len(restored.layers), len(original.layers))
        for orig_layer, rest_layer in zip(original.layers, restored.layers):
            self.assertTrue(torch.equal(orig_layer.weight, rest_layer.weight))
            self.assertTrue(torch.equal(orig_layer.bias, rest_layer.bias))
            self.assertEqual(orig_layer.output_min, rest_layer.output_min)
            self.assertEqual(orig_layer.output_max, rest_layer.output_max)
            self.assertEqual(orig_layer.activation, rest_layer.activation)

        # Forward pass should match
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(torch.equal(original.forward(x), restored.forward(x)))


class RemoveHiddenUnitTest(unittest.TestCase):
    """Tests for remove_hidden_unit."""

    def test_removes_unit_and_compacts_matrices(self) -> None:
        # 3 -> 4 -> 2 model, remove unit 1 from layer 0
        w0 = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],  # this unit will be removed
                [7, 8, 9],
                [10, 11, 12],
            ],
            dtype=torch.int16,
        )
        b0 = torch.tensor([1, 2, 3, 4], dtype=torch.int16)
        w1 = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=torch.int16,
        )
        b1 = torch.tensor([1, 2], dtype=torch.int16)

        layer0 = DiscreteLayer(weight=w0, bias=b0, output_min=0.0, output_max=15.0)
        layer1 = DiscreteLayer(weight=w1, bias=b1, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer0, layer1])

        pruned = remove_hidden_unit(model, layer_index=0, unit_index=1)

        # Layer 0 should now be 3x3 (removed row 1)
        expected_w0 = torch.tensor(
            [
                [1, 2, 3],
                [7, 8, 9],
                [10, 11, 12],
            ],
            dtype=torch.int16,
        )
        expected_b0 = torch.tensor([1, 3, 4], dtype=torch.int16)
        self.assertTrue(torch.equal(pruned.layers[0].weight, expected_w0))
        self.assertTrue(torch.equal(pruned.layers[0].bias, expected_b0))

        # Layer 1 should now be 2x3 (removed column 1)
        expected_w1 = torch.tensor(
            [
                [1, 3, 4],
                [5, 7, 8],
            ],
            dtype=torch.int16,
        )
        self.assertTrue(torch.equal(pruned.layers[1].weight, expected_w1))
        self.assertTrue(torch.equal(pruned.layers[1].bias, b1))

        # Original model should be unchanged
        self.assertEqual(model.layers[0].weight.shape[0], 4)

    def test_raises_for_output_layer(self) -> None:
        weight = torch.zeros(2, 3, dtype=torch.int16)
        bias = torch.zeros(2, dtype=torch.int16)
        layer = DiscreteLayer(weight=weight, bias=bias, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer])
        with self.assertRaises(ValueError):
            remove_hidden_unit(model, layer_index=0, unit_index=0)

    def test_raises_for_out_of_range_unit(self) -> None:
        w0 = torch.zeros(3, 2, dtype=torch.int16)
        b0 = torch.zeros(3, dtype=torch.int16)
        w1 = torch.zeros(1, 3, dtype=torch.int16)
        b1 = torch.zeros(1, dtype=torch.int16)
        layer0 = DiscreteLayer(weight=w0, bias=b0, output_min=0.0, output_max=15.0)
        layer1 = DiscreteLayer(weight=w1, bias=b1, output_min=0.0, output_max=15.0)
        model = DiscreteMCNN([layer0, layer1])
        with self.assertRaises(ValueError):
            remove_hidden_unit(model, layer_index=0, unit_index=5)


if __name__ == "__main__":
    unittest.main()
