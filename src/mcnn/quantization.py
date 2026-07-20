"""Quantization and straight-through estimators for MCNN models."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .constants import DEFAULT_BIAS_MIN, DEFAULT_WEIGHT_MAX, DEFAULT_WEIGHT_MIN


def _ste(forward_value: Tensor, gradient_value: Tensor) -> Tensor:
    """Use ``forward_value`` in the forward pass and ``gradient_value`` for grads."""

    return gradient_value + (forward_value - gradient_value).detach()


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
