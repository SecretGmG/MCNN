"""Small MCNN model definitions."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from .constants import (
    DEFAULT_OUTPUT_MAX,
    DEFAULT_OUTPUT_MIN,
    DEFAULT_WEIGHT_MAX,
    DEFAULT_WEIGHT_MIN,
)
from .layers import MCNNLayer


class MCNNClassifier(nn.Module):
    """Fully connected classifier built only from constrained MCNN layers."""

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
        sizes = [input_features, *hidden_sizes, output_features]
        layers = []
        for layer_index, (in_features, out_features) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            is_output_layer = layer_index == len(sizes) - 2
            layers.append(
                MCNNLayer(
                in_features,
                out_features,
                weight_min=weight_min,
                weight_max=weight_max,
                output_min=output_min,
                output_max=output_max,
                    activation=output_activation if is_output_layer else hidden_activation,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, *, exact: bool = False) -> Tensor:
        for layer in self.layers:
            x = layer(x, exact=exact)
        return x
