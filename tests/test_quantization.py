"""Tests for quantization and encoding modules."""

from __future__ import annotations

import unittest

import torch

from mcnn.encoding import (
    ComparatorEncoding,
    ComparatorSubtractions,
    comparator_subtractions,
    encode_comparator_weight,
    inactive_subtraction,
)
from mcnn.quantization import (
    bounded_integer_ste,
    exact_bounded_integer,
    exact_quantize_weight,
    positive_bias,
    quantize_weight,
)


class BoundedIntegerSTETest(unittest.TestCase):
    """Tests for bounded_integer_ste: rounding, clamping, and STE gradient."""

    def test_rounds_and_clamps_within_bounds(self) -> None:
        x = torch.tensor([0.2, 1.7, 3.5, 4.9, -0.3, 5.1])
        result = bounded_integer_ste(x, lower=0.0, upper=5.0)
        expected = torch.tensor([0.0, 2.0, 4.0, 5.0, 0.0, 5.0])
        self.assertTrue(torch.equal(result, expected))

    def test_ste_gradient_passthrough(self) -> None:
        x = torch.tensor([1.3, 2.7, 3.1], requires_grad=True)
        result = bounded_integer_ste(x, lower=0.0, upper=5.0)
        loss = result.sum()
        loss.backward()
        # STE should pass gradient through to x unchanged
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_raises_value_error_when_lower_gt_upper(self) -> None:
        with self.assertRaises(ValueError):
            bounded_integer_ste(torch.tensor([1.0]), lower=5.0, upper=0.0)


class ExactBoundedIntegerTest(unittest.TestCase):
    """Tests for exact_bounded_integer: rounding and clamping without STE."""

    def test_rounds_and_clamps_within_bounds(self) -> None:
        x = torch.tensor([0.2, 1.7, 3.5, 4.9, -0.3, 5.1])
        result = exact_bounded_integer(x, lower=0.0, upper=5.0)
        expected = torch.tensor([0.0, 2.0, 4.0, 5.0, 0.0, 5.0])
        self.assertTrue(torch.equal(result, expected))

    def test_no_gradient_passthrough(self) -> None:
        x = torch.tensor([1.3, 2.7, 3.1], requires_grad=True)
        result = exact_bounded_integer(x, lower=0.0, upper=5.0)
        # exact_bounded_integer does not use STE, so gradients are zero
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(x.grad == 0))

    def test_raises_value_error_when_lower_gt_upper(self) -> None:
        with self.assertRaises(ValueError):
            exact_bounded_integer(torch.tensor([1.0]), lower=5.0, upper=0.0)


class QuantizeWeightTest(unittest.TestCase):
    """Tests for quantize_weight: sigmoid mapping + STE quantization."""

    def test_maps_through_sigmoid_and_quantizes(self) -> None:
        # Very large positive raw_weight -> sigmoid ~1 -> near upper bound
        raw = torch.tensor([100.0])
        result = quantize_weight(raw, lower=-14.0, upper=14.0)
        self.assertEqual(result.item(), 14.0)

        # Very large negative raw_weight -> sigmoid ~0 -> near lower bound
        raw = torch.tensor([-100.0])
        result = quantize_weight(raw, lower=-14.0, upper=14.0)
        self.assertEqual(result.item(), -14.0)

        # raw_weight = 0 -> sigmoid(0) = 0.5 -> midpoint
        raw = torch.tensor([0.0])
        result = quantize_weight(raw, lower=-14.0, upper=14.0)
        self.assertEqual(result.item(), 0.0)

    def test_ste_gradient_passthrough(self) -> None:
        raw = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
        result = quantize_weight(raw, lower=-14.0, upper=14.0)
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(raw.grad)
        # Gradient should flow back (STE)
        self.assertEqual(raw.grad.shape, raw.shape)

    def test_raises_value_error_when_lower_gt_upper(self) -> None:
        with self.assertRaises(ValueError):
            quantize_weight(torch.tensor([1.0]), lower=5.0, upper=0.0)


class ExactQuantizeWeightTest(unittest.TestCase):
    """Tests for exact_quantize_weight: sigmoid mapping without STE."""

    def test_maps_through_sigmoid_and_quantizes(self) -> None:
        raw = torch.tensor([100.0])
        result = exact_quantize_weight(raw, lower=-14.0, upper=14.0)
        self.assertEqual(result.item(), 14.0)

        raw = torch.tensor([-100.0])
        result = exact_quantize_weight(raw, lower=-14.0, upper=14.0)
        self.assertEqual(result.item(), -14.0)

        raw = torch.tensor([0.0])
        result = exact_quantize_weight(raw, lower=-14.0, upper=14.0)
        self.assertEqual(result.item(), 0.0)

    def test_no_gradient_passthrough(self) -> None:
        raw = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
        result = exact_quantize_weight(raw, lower=-14.0, upper=14.0)
        # exact_quantize_weight does not use STE, so gradients are zero
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.all(raw.grad == 0))

    def test_raises_value_error_when_lower_gt_upper(self) -> None:
        with self.assertRaises(ValueError):
            exact_quantize_weight(torch.tensor([1.0]), lower=5.0, upper=0.0)


class PositiveBiasTest(unittest.TestCase):
    """Tests for positive_bias: softplus + minimum, always positive."""

    def test_output_is_always_positive(self) -> None:
        raw = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        result = positive_bias(raw)
        self.assertTrue(torch.all(result > 0))

    def test_respects_custom_minimum(self) -> None:
        raw = torch.tensor([-100.0, 0.0, 100.0])
        result = positive_bias(raw, minimum=2.0)
        self.assertTrue(torch.all(result >= 2.0))

    def test_default_minimum_is_positive(self) -> None:
        raw = torch.tensor([-1000.0])
        result = positive_bias(raw)
        self.assertGreater(result.item(), 0.0)


class EncodeComparatorWeightTest(unittest.TestCase):
    """Tests for encode_comparator_weight: all weights map correctly."""

    def test_positive_weights_use_mode_a(self) -> None:
        for w in range(1, 15):
            encoding = encode_comparator_weight(w)
            self.assertEqual(encoding.mode, "A")
            self.assertEqual(encoding.value, 15 - w)
            self.assertGreaterEqual(encoding.value, 1)
            self.assertLessEqual(encoding.value, 14)

    def test_negative_weights_use_mode_b(self) -> None:
        for w in range(-14, 0):
            encoding = encode_comparator_weight(w)
            self.assertEqual(encoding.mode, "B")
            self.assertEqual(encoding.value, w + 15)
            self.assertGreaterEqual(encoding.value, 1)
            self.assertLessEqual(encoding.value, 14)

    def test_zero_weight_uses_mode_c(self) -> None:
        encoding = encode_comparator_weight(0)
        self.assertEqual(encoding.mode, "C")
        self.assertEqual(encoding.value, 0)

    def test_all_nonzero_values_in_range_1_to_14(self) -> None:
        for w in range(-14, 15):
            if w == 0:
                continue
            encoding = encode_comparator_weight(w)
            self.assertGreaterEqual(encoding.value, 1)
            self.assertLessEqual(encoding.value, 14)

    def test_raises_value_error_for_out_of_range_positive(self) -> None:
        with self.assertRaises(ValueError):
            encode_comparator_weight(15)

    def test_raises_value_error_for_out_of_range_negative(self) -> None:
        with self.assertRaises(ValueError):
            encode_comparator_weight(-15)


class ComparatorSubtractionsTest(unittest.TestCase):
    """Tests for comparator_subtractions: inactive - active == weight."""

    def test_inactive_minus_active_equals_weight(self) -> None:
        for w in range(-14, 15):
            subs = comparator_subtractions(w)
            self.assertEqual(
                subs.inactive - subs.active,
                w,
                f"weight={w}: inactive={subs.inactive}, active={subs.active}",
            )

    def test_mode_a_subtractions(self) -> None:
        subs = comparator_subtractions(5)
        self.assertEqual(subs.encoding.mode, "A")
        self.assertEqual(subs.inactive, 15)
        self.assertEqual(subs.active, 10)

    def test_mode_b_subtractions(self) -> None:
        subs = comparator_subtractions(-5)
        self.assertEqual(subs.encoding.mode, "B")
        self.assertEqual(subs.inactive, 10)
        self.assertEqual(subs.active, 15)

    def test_mode_c_subtractions(self) -> None:
        subs = comparator_subtractions(0)
        self.assertEqual(subs.encoding.mode, "C")
        self.assertEqual(subs.inactive, 0)
        self.assertEqual(subs.active, 0)


class InactiveSubtractionTest(unittest.TestCase):
    """Tests for inactive_subtraction: matches comparator_subtractions.inactive."""

    def test_matches_comparator_subtractions_inactive(self) -> None:
        for w in range(-14, 15):
            self.assertEqual(
                inactive_subtraction(w),
                comparator_subtractions(w).inactive,
                f"weight={w}",
            )


if __name__ == "__main__":
    unittest.main()
