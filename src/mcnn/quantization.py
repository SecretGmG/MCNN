"""Quantization and straight-through estimators for MCNN models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .constants import DEFAULT_BIAS_MIN, DEFAULT_WEIGHT_MAX, DEFAULT_WEIGHT_MIN


def _ste(forward_value: Tensor, gradient_value: Tensor) -> Tensor:
    """Use ``forward_value`` in the forward pass and ``gradient_value`` for grads."""

    return gradient_value + (forward_value - gradient_value).detach()


@dataclass(frozen=True)
class ComparatorEncoding:
    """Minecraft comparator encoding for one effective connection weight."""

    mode: str
    value: int


@dataclass(frozen=True)
class ComparatorSubtractions:
    """Subtractions made by one physical connection module.

    ``inactive`` is the amount subtracted when the binary input is 0.
    ``active`` is the amount subtracted when the binary input is 1.
    The effective model weight is therefore ``inactive - active``.
    """

    encoding: ComparatorEncoding
    inactive: int
    active: int


def encode_comparator_weight(weight: int) -> ComparatorEncoding:
    """Encode an effective integer weight as a comparator mode and value.

    Mode A subtracts 15 by default and ``value`` when active, giving
    ``weight = 15 - value``. Mode B subtracts ``value`` by default and 15 when
    active, giving ``weight = value - 15``. ``value`` must be nonzero because a
    zero-signal furnace behaves like a normal block in the redstone circuit.
    Mode C is the zero-weight no-op: no comparator connection module is emitted,
    reducing the required bias signal.
    """

    if not -14 <= weight <= 14:
        raise ValueError(f"weight {weight} is outside the Minecraft range [-14, 14]")
    if weight > 0:
        return ComparatorEncoding(mode="A", value=15 - weight)
    if weight < 0:
        return ComparatorEncoding(mode="B", value=weight + 15)
    return ComparatorEncoding(mode="C", value=0)


def comparator_subtractions(weight: int) -> ComparatorSubtractions:
    """Return the exact Minecraft subtraction behavior for a model weight."""

    encoding = encode_comparator_weight(weight)
    if encoding.mode == "A":
        return ComparatorSubtractions(encoding=encoding, inactive=15, active=encoding.value)
    if encoding.mode == "B":
        return ComparatorSubtractions(encoding=encoding, inactive=encoding.value, active=15)
    if encoding.mode == "C":
        return ComparatorSubtractions(encoding=encoding, inactive=0, active=0)
    raise ValueError(f"unknown comparator mode {encoding.mode!r}")


def positive_bias(raw_bias: Tensor, minimum: float = DEFAULT_BIAS_MIN) -> Tensor:
    """Map unconstrained bias parameters to strictly positive biases."""

    return F.softplus(raw_bias) + minimum


def bounded_integer_ste(x: Tensor, lower: float, upper: float) -> Tensor:
    """Round and clamp to integer values in [lower, upper] with STE."""

    if lower > upper:
        raise ValueError(f"lower bound {lower} must be <= upper bound {upper}")
    quantized = x.round().clamp(lower, upper)
    return _ste(quantized, x)


def exact_bounded_integer(x: Tensor, lower: float, upper: float) -> Tensor:
    """Round and clamp to integer values in [lower, upper] without STE."""

    if lower > upper:
        raise ValueError(f"lower bound {lower} must be <= upper bound {upper}")
    return x.round().clamp(lower, upper)


def quantize_weight(
    raw_weight: Tensor,
    lower: float = DEFAULT_WEIGHT_MIN,
    upper: float = DEFAULT_WEIGHT_MAX,
) -> Tensor:
    """Quantize latent weights to integer values in [lower, upper] with STE.

    The latent value is passed through a sigmoid before quantization. This keeps
    the trainable parameter unconstrained while making the exact forward pass
    obey the configured Minecraft weight range.
    """

    if lower > upper:
        raise ValueError(f"lower bound {lower} must be <= upper bound {upper}")
    continuous = lower + (upper - lower) * torch.sigmoid(raw_weight)
    return bounded_integer_ste(continuous, lower, upper)


def exact_quantize_weight(
    raw_weight: Tensor,
    lower: float = DEFAULT_WEIGHT_MIN,
    upper: float = DEFAULT_WEIGHT_MAX,
) -> Tensor:
    """Quantize latent weights without STE behavior."""

    if lower > upper:
        raise ValueError(f"lower bound {lower} must be <= upper bound {upper}")
    continuous = lower + (upper - lower) * torch.sigmoid(raw_weight)
    return exact_bounded_integer(continuous, lower, upper)
