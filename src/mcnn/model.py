from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

ForwardMode = Literal["ste", "exact"]


@dataclass(frozen=True)
class DiscreteLayerState:
    weight: Tensor
    bias: Tensor
    activation: str
    output_min: float
    output_max: float


def _ste(discrete: Tensor, continuous: Tensor) -> Tensor:
    return continuous + (discrete - continuous).detach()


def _bounded_integer(x: Tensor, minimum: float, maximum: float, *, ste: bool) -> Tensor:
    discrete = x.round().clamp(minimum, maximum)
    return _ste(discrete, x) if ste else discrete


class MCNNModel(nn.Module):
    """Single MCNN model representation used by all stages.

    The same model instance can be used for:
    - classic STE training (forward mode="ste")
    - exact/discrete evaluation and export (forward mode="exact")
    - post-training transformations (quantization/evolution)
    """

    def __init__(
        self,
        sizes: Sequence[int],
        *,
        hidden_activation: str = "binary_step",
        output_activation: str = "bounded_integer",
        output_min: float = 0.0,
        output_max: float = 15.0,
        weight_min: int = -15,
        weight_max: int = 15,
        bias_min: float = 1.0,
    ) -> None:
        super().__init__()
        if len(sizes) < 2:
            raise ValueError("MCNNModel requires at least [input, output]")

        self.weight_min = int(weight_min)
        self.weight_max = int(weight_max)
        self.bias_min = float(bias_min)

        self.layer_activations: list[str] = []
        self.layer_output_mins: list[float] = []
        self.layer_output_maxs: list[float] = []

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for layer_index, (in_features, out_features) in enumerate(
            zip(sizes[:-1], sizes[1:], strict=True)
        ):
            is_output = layer_index == len(sizes) - 2
            activation = output_activation if is_output else hidden_activation
            out_min = 0.0 if activation == "binary_step" else float(output_min)
            out_max = 1.0 if activation == "binary_step" else float(output_max)

            self.weights.append(nn.Parameter(torch.empty(out_features, in_features)))
            self.biases.append(nn.Parameter(torch.empty(out_features)))
            self.layer_activations.append(activation)
            self.layer_output_mins.append(out_min)
            self.layer_output_maxs.append(out_max)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weights, self.biases, strict=True):
            nn.init.normal_(weight, mean=0.0, std=0.5)
            nn.init.constant_(bias, 1.0)

    def _quantized_weight(self, layer_index: int, *, ste: bool) -> Tensor:
        weight = self.weights[layer_index]
        discrete = weight.round().clamp(self.weight_min, self.weight_max)
        return _ste(discrete, weight) if ste else discrete

    def _quantized_bias(self, layer_index: int, *, ste: bool) -> Tensor:
        bias = self.biases[layer_index]
        discrete = bias.round().clamp_min(self.bias_min)
        return _ste(discrete, bias) if ste else discrete

    def _apply_activation(self, x: Tensor, layer_index: int, *, ste: bool) -> Tensor:
        activation = self.layer_activations[layer_index]
        if activation == "binary_step":
            discrete = (x > 0).to(x.dtype)
            return _ste(discrete, x) if ste else discrete
        if activation == "bounded_integer":
            return _bounded_integer(
                x,
                self.layer_output_mins[layer_index],
                self.layer_output_maxs[layer_index],
                ste=ste,
            )
        raise ValueError(f"unknown activation {activation!r}")

    def forward(self, x: Tensor, *, mode: ForwardMode = "ste") -> Tensor:
        ste = mode == "ste"
        if mode not in ("ste", "exact"):
            raise ValueError("mode must be one of {'ste', 'exact'}")

        for layer_index in range(len(self.weights)):
            weight = self._quantized_weight(layer_index, ste=ste)
            bias = self._quantized_bias(layer_index, ste=ste)
            x = torch.nn.functional.linear(x, weight, bias)
            x = self._apply_activation(x, layer_index, ste=ste)
        return x

    def clone(self) -> "MCNNModel":
        return deepcopy(self)

    # ------------------------------------------------------------------
    # Discrete snapshots / serialization
    # ------------------------------------------------------------------

    def discrete_layers(self) -> list[DiscreteLayerState]:
        layers: list[DiscreteLayerState] = []
        for layer_index in range(len(self.weights)):
            layers.append(
                DiscreteLayerState(
                    weight=(
                        self._quantized_weight(layer_index, ste=False)
                        .detach()
                        .cpu()
                        .to(torch.int16)
                    ),
                    bias=(
                        self._quantized_bias(layer_index, ste=False)
                        .detach()
                        .cpu()
                        .to(torch.int16)
                    ),
                    activation=self.layer_activations[layer_index],
                    output_min=self.layer_output_mins[layer_index],
                    output_max=self.layer_output_maxs[layer_index],
                )
            )
        return layers

    def apply_discrete_layers_(self, layers: Sequence[DiscreteLayerState]) -> "MCNNModel":
        if len(layers) != len(self.weights):
            raise ValueError("layer count mismatch")

        with torch.no_grad():
            for layer_index, layer in enumerate(layers):
                self.weights[layer_index] = nn.Parameter(layer.weight.to(torch.float32).clone())
                self.biases[layer_index] = nn.Parameter(layer.bias.to(torch.float32).clone())
                self.layer_activations[layer_index] = layer.activation
                self.layer_output_mins[layer_index] = float(layer.output_min)
                self.layer_output_maxs[layer_index] = float(layer.output_max)
        return self

    def artifact_state(self) -> dict[str, object]:
        return {
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "bias_min": self.bias_min,
            "layers": [
                {
                    "weight": layer.weight.to(torch.int16),
                    "bias": layer.bias.to(torch.int16),
                    "activation": layer.activation,
                    "output_min": layer.output_min,
                    "output_max": layer.output_max,
                }
                for layer in self.discrete_layers()
            ],
        }

    @classmethod
    def from_discrete_layers(
        cls,
        layers: Sequence[dict[str, object]],
        *,
        weight_min: int = -15,
        weight_max: int = 15,
        bias_min: float = 0.0,
    ) -> "MCNNModel":
        if not layers:
            raise ValueError("at least one layer is required")
        sizes = [int(cast_tensor(layers[0]["weight"]).shape[1])]
        sizes.extend(int(cast_tensor(layer["weight"]).shape[0]) for layer in layers)

        model = cls(
            sizes,
            hidden_activation="bounded_integer",
            output_activation="bounded_integer",
            output_min=0.0,
            output_max=15.0,
            weight_min=weight_min,
            weight_max=weight_max,
            bias_min=bias_min,
        )

        discrete = [
            DiscreteLayerState(
                weight=cast_tensor(layer["weight"]).to(torch.int16),
                bias=cast_tensor(layer["bias"]).to(torch.int16),
                activation=str(layer.get("activation", "bounded_integer")),
                output_min=to_float(layer.get("output_min", 0.0)),
                output_max=to_float(layer.get("output_max", 15.0)),
            )
            for layer in layers
        ]
        model.apply_discrete_layers_(discrete)
        return model

    @classmethod
    def from_artifact_state(cls, state: dict[str, object]) -> "MCNNModel":
        layers = state.get("layers")
        if not isinstance(layers, list):
            raise TypeError("artifact state must include a list under 'layers'")
        return cls.from_discrete_layers(
            layers,
            weight_min=to_int(state.get("weight_min", -15)),
            weight_max=to_int(state.get("weight_max", 15)),
            bias_min=to_float(state.get("bias_min", 0.0)),
        )

    # ------------------------------------------------------------------
    # Structure/stats used by refinement and reporting
    # ------------------------------------------------------------------

    def layer_sizes(self) -> list[int]:
        if not self.weights:
            return []
        sizes = [int(self.weights[0].shape[1])]
        sizes.extend(int(weight.shape[0]) for weight in self.weights)
        return sizes

    def hidden_units(self) -> int:
        sizes = self.layer_sizes()
        return sum(sizes[1:-1])

    def nonzero_weights(self) -> int:
        return sum(int(layer.weight.count_nonzero().item()) for layer in self.discrete_layers())

    def bias_signal_strengths(self) -> list[int]:
        """Bias-barrel signal strengths following export.py source-of-truth logic."""
        strengths: list[int] = []
        for layer in self.discrete_layers():
            output_size, input_size = layer.weight.shape
            for output_index in range(output_size):
                signal = int(layer.bias[output_index].item())
                flip = output_index % 2 == 0
                for input_index in range(input_size):
                    weight = int(layer.weight[output_index, input_index].item())
                    if weight == 0:
                        continue
                    network_signal = 15 + weight * (1 if flip else -1)
                    if flip:
                        signal += network_signal
                    else:
                        signal += 15
                strengths.append(signal)
        return strengths

    def max_bias_signal_strength(self) -> int:
        strengths = self.bias_signal_strengths()
        return max(strengths) if strengths else 0


def cast_tensor(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("expected torch.Tensor")
    return value


def to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError("expected numeric value")


def to_int(value: object) -> int:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    raise TypeError("expected integer-like value")


def model_from_artifact_state(state: dict[str, object]) -> MCNNModel:
    return MCNNModel.from_artifact_state(state)
