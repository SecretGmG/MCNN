"""Trainable MCNN module — the single model representation.

``TrainableMCNN`` is a thin ``nn.Module`` wrapper that holds latent
(floating-point) parameters for each layer and exposes two modes:

- ``forward()`` — STE-quantized training pass (gradients flow through quantization)
- ``discrete()`` — returns an exact ``DiscreteMCNN`` snapshot of the current
  quantized state

The ``DiscreteMCNN`` is the canonical representation used by evolution, export,
and all post-training stages.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .constants import (
    DEFAULT_BIAS_MIN,
    DEFAULT_OUTPUT_MAX,
    DEFAULT_OUTPUT_MIN,
    DEFAULT_WEIGHT_MAX,
    DEFAULT_WEIGHT_MIN,
)
from .discrete import DiscreteLayer, DiscreteMCNN
from .quantization import (
    _ste,
    bounded_integer_ste,
    exact_bounded_integer,
    exact_quantize_weight,
    positive_bias,
    quantize_weight,
)


class TrainableMCNN(nn.Module):
    """Trainable multi-layer MCNN with STE quantization.

    Each layer has ``raw_weight`` and ``raw_bias`` parameters. The forward
    pass quantizes them and applies the activation. Call ``.discrete()`` to
    get an exact ``DiscreteMCNN`` snapshot for evolution or export.
    """

    def __init__(
        self,
        input_features: int,
        hidden_sizes: Sequence[int],
        output_features: int,
        *,
        weight_min: float = DEFAULT_WEIGHT_MIN,
        weight_max: float = DEFAULT_WEIGHT_MAX,
        output_min: float = DEFAULT_OUTPUT_MIN,
        output_max: float = DEFAULT_OUTPUT_MAX,
        hidden_activation: str = "bounded_integer",
        output_activation: str = "bounded_integer",
    ) -> None:
        super().__init__()
        if weight_min > weight_max:
            raise ValueError(
                f"weight_min {weight_min} must be <= weight_max {weight_max}"
            )
        if output_min > output_max:
            raise ValueError(
                f"output_min {output_min} must be <= output_max {output_max}"
            )

        sizes = [input_features, *hidden_sizes, output_features]
        layer_params: list[dict[str, object]] = []
        for layer_index, (in_features, out_features) in enumerate(
            zip(sizes[:-1], sizes[1:], strict=True)
        ):
            is_output = layer_index == len(sizes) - 2
            activation = output_activation if is_output else hidden_activation
            layer_params.append(
                {
                    "in_features": in_features,
                    "out_features": out_features,
                    "activation": activation,
                    "output_min": output_min,
                    "output_max": output_max,
                }
            )

        self.layer_params = layer_params
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.raw_weights = nn.ParameterList()
        self.raw_biases = nn.ParameterList()
        for param in layer_params:
            self.raw_weights.append(
                nn.Parameter(torch.empty(param["out_features"], param["in_features"]))
            )
            self.raw_biases.append(nn.Parameter(torch.empty(param["out_features"])))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for raw_weight, raw_bias in zip(self.raw_weights, self.raw_biases):
            nn.init.normal_(raw_weight, mean=0.0, std=0.5)
            nn.init.constant_(raw_bias, 1.0)

    def _quantized_weight(self, layer_index: int, *, exact: bool = False) -> Tensor:
        raw = self.raw_weights[layer_index]
        if exact:
            return exact_quantize_weight(raw, self.weight_min, self.weight_max)
        return quantize_weight(raw, self.weight_min, self.weight_max)

    def _positive_bias(self, layer_index: int) -> Tensor:
        return positive_bias(self.raw_biases[layer_index])

    def _apply_activation(
        self,
        x: Tensor,
        activation: str,
        output_min: float,
        output_max: float,
        *,
        exact: bool,
    ) -> Tensor:
        if activation == "binary_step":
            activated = (x > 0).to(x.dtype)
            if exact:
                return activated
            return _ste(activated, x)
        if activation == "bounded_integer":
            if exact:
                return exact_bounded_integer(x, output_min, output_max)
            return bounded_integer_ste(x, output_min, output_max)
        raise ValueError(f"unknown activation {activation!r}")

    def forward(self, x: Tensor, *, exact: bool = False) -> Tensor:
        for layer_index, param in enumerate(self.layer_params):
            weight = self._quantized_weight(layer_index, exact=exact)
            bias = self._positive_bias(layer_index)
            x = torch.nn.functional.linear(x, weight, bias)
            x = self._apply_activation(
                x,
                param["activation"],
                param["output_min"],
                param["output_max"],
                exact=exact,
            )
        return x

    def quantized_weights(self, *, exact: bool = True) -> list[Tensor]:
        """Return the quantized weight matrices for every layer."""
        return [
            self._quantized_weight(i, exact=exact)
            for i in range(len(self.layer_params))
        ]

    def positive_biases(self) -> list[Tensor]:
        """Return the positive bias vectors for every layer."""
        return [self._positive_bias(i) for i in range(len(self.layer_params))]

    def discrete(self) -> DiscreteMCNN:
        """Return an exact ``DiscreteMCNN`` snapshot of the current quantized state."""
        layers: list[DiscreteLayer] = []
        for layer_index, param in enumerate(self.layer_params):
            weight = (
                self._quantized_weight(layer_index, exact=True)
                .detach()
                .cpu()
                .round()
                .to(torch.int16)
            )
            bias = (
                self._positive_bias(layer_index)
                .detach()
                .cpu()
                .round()
                .clamp_min(1)
                .to(torch.int16)
            )
            activation = param["activation"]
            output_min = (
                0.0 if activation == "binary_step" else float(param["output_min"])
            )
            output_max = (
                1.0 if activation == "binary_step" else float(param["output_max"])
            )
            layers.append(
                DiscreteLayer(
                    weight=weight,
                    bias=bias,
                    output_min=output_min,
                    output_max=output_max,
                    activation=activation,
                )
            )
        return DiscreteMCNN(layers)
