"""PyTorch layers that follow the MCNN inference constraints."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .constants import (
    DEFAULT_OUTPUT_MAX,
    DEFAULT_OUTPUT_MIN,
    DEFAULT_WEIGHT_MAX,
    DEFAULT_WEIGHT_MIN,
)
from .quantization import (
    bounded_integer_ste,
    exact_bounded_integer,
    exact_quantize_weight,
    positive_bias,
    quantize_weight,
)


class BoundedIntegerActivation(nn.Module):
    """Bounded integer activation with optional exact inference."""

    kind = "bounded_integer"

    def __init__(
        self,
        lower: float = DEFAULT_OUTPUT_MIN,
        upper: float = DEFAULT_OUTPUT_MAX,
    ) -> None:
        super().__init__()
        if lower > upper:
            raise ValueError(f"lower bound {lower} must be <= upper bound {upper}")
        self.lower = lower
        self.upper = upper

    def forward(self, x: Tensor, *, exact: bool = False) -> Tensor:
        if exact:
            return exact_bounded_integer(x, self.lower, self.upper)
        return bounded_integer_ste(x, self.lower, self.upper)


class BinaryStepActivation(nn.Module):
    """Binary hidden activation matching Minecraft active/inactive wiring."""

    kind = "binary_step"
    lower = 0.0
    upper = 1.0

    def forward(self, x: Tensor, *, exact: bool = False) -> Tensor:
        activated = (x > 0).to(x.dtype)
        if exact:
            return activated
        return x + (activated - x).detach()


class MCNNLayer(nn.Module):
    """A constrained linear layer followed by bounded integer activation.

    Forward inference uses quantized integer weights, positive biases, and
    bounded integer outputs. During training the weight and output quantizers use
    straight-through estimators.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight_min: float = DEFAULT_WEIGHT_MIN,
        weight_max: float = DEFAULT_WEIGHT_MAX,
        output_min: float = DEFAULT_OUTPUT_MIN,
        output_max: float = DEFAULT_OUTPUT_MAX,
        activation: str = "bounded_integer",
    ) -> None:
        super().__init__()
        if weight_min > weight_max:
            raise ValueError(f"weight_min {weight_min} must be <= weight_max {weight_max}")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.raw_bias = nn.Parameter(torch.empty(out_features))
        if activation == "bounded_integer":
            self.activation = BoundedIntegerActivation(output_min, output_max)
        elif activation == "binary_step":
            self.activation = BinaryStepActivation()
        else:
            raise ValueError(f"unknown activation {activation!r}")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.raw_weight, mean=0.0, std=0.5)
        nn.init.constant_(self.raw_bias, 1.0)

    def quantized_weight(self, *, exact: bool = False) -> Tensor:
        if exact:
            return exact_quantize_weight(self.raw_weight, self.weight_min, self.weight_max)
        return quantize_weight(self.raw_weight, self.weight_min, self.weight_max)

    def positive_bias(self) -> Tensor:
        return positive_bias(self.raw_bias)

    def forward(self, x: Tensor, *, exact: bool = False) -> Tensor:
        weight = self.quantized_weight(exact=exact)
        bias = self.positive_bias()
        pre_activation = torch.nn.functional.linear(x, weight, bias)
        return self.activation(pre_activation, exact=exact)
