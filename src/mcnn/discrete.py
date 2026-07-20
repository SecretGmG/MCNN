"""Exact discrete MCNN runtime and post-training refinement."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from .quantization import comparator_subtractions
from .layers import MCNNLayer
from .models import MCNNClassifier


@dataclass
class DiscreteLayer:
    weight: Tensor
    bias: Tensor
    output_min: float
    output_max: float
    activation: str = "bounded_integer"

    def clone(self) -> "DiscreteLayer":
        return DiscreteLayer(
            weight=self.weight.clone(),
            bias=self.bias.clone(),
            output_min=self.output_min,
            output_max=self.output_max,
            activation=self.activation,
        )


@dataclass
class DiscreteMCNN:
    layers: list[DiscreteLayer]

    def clone(self) -> "DiscreteMCNN":
        return DiscreteMCNN([layer.clone() for layer in self.layers])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            weight = layer.weight.to(device=x.device, dtype=x.dtype)
            bias = layer.bias.to(device=x.device, dtype=x.dtype)
            x = torch.nn.functional.linear(x, weight, bias)
            if layer.activation == "binary_step":
                x = (x > 0).to(x.dtype)
            elif layer.activation == "bounded_integer":
                x = x.round().clamp(layer.output_min, layer.output_max)
            else:
                raise ValueError(f"unknown discrete activation {layer.activation!r}")
        return x

    def nonzero_weights(self) -> int:
        return sum(int(layer.weight.count_nonzero().item()) for layer in self.layers)

    def total_weights(self) -> int:
        return sum(layer.weight.numel() for layer in self.layers)

    def layer_sizes(self) -> list[int]:
        if not self.layers:
            return []
        sizes = [self.layers[0].weight.shape[1]]
        sizes.extend(layer.weight.shape[0] for layer in self.layers)
        return [int(size) for size in sizes]

    def hidden_units(self) -> int:
        sizes = self.layer_sizes()
        return sum(sizes[1:-1])

    def physical_biases(self) -> list[int]:
        return [
            int(layer.bias[output_index].item()) + sum(
                inactive_subtraction(int(weight))
                for weight in layer.weight[output_index].tolist()
            )
            for layer in self.layers
            for output_index in range(layer.weight.shape[0])
        ]

    def max_physical_bias(self) -> int:
        return max(self.physical_biases(), default=0)

    def state_dict(self) -> dict[str, object]:
        return {
            "layers": [
                {
                    "weight": layer.weight.to(torch.int16),
                    "bias": layer.bias.to(torch.int16),
                    "output_min": layer.output_min,
                    "output_max": layer.output_max,
                    "activation": layer.activation,
                }
                for layer in self.layers
            ],
        }


def discrete_from_model(model: MCNNClassifier) -> DiscreteMCNN:
    layers: list[DiscreteLayer] = []
    for module in model.layers:
        if not isinstance(module, MCNNLayer):
            raise TypeError(f"expected MCNNLayer, got {type(module).__name__}")
        layers.append(
            DiscreteLayer(
                weight=module.quantized_weight(exact=True).detach().cpu().round().to(torch.int16),
                bias=module.positive_bias().detach().cpu().round().clamp_min(1).to(torch.int16),
                output_min=module.activation.lower,
                output_max=module.activation.upper,
                activation=module.activation.kind,
            )
        )
    return DiscreteMCNN(layers)


def discrete_from_state_dict(state: dict[str, object]) -> DiscreteMCNN:
    """Load a discrete model from the compact artifact state."""

    layers = []
    for raw_layer in state["layers"]:
        if not isinstance(raw_layer, dict):
            raise TypeError("each serialized layer must be a dictionary")
        layers.append(
            DiscreteLayer(
                weight=raw_layer["weight"].detach().cpu().to(torch.int16),
                bias=raw_layer["bias"].detach().cpu().to(torch.int16),
                output_min=float(raw_layer["output_min"]),
                output_max=float(raw_layer["output_max"]),
                activation=str(raw_layer.get("activation", "bounded_integer")),
            )
        )
    return DiscreteMCNN(layers)


def inactive_subtraction(weight: int) -> int:
    return comparator_subtractions(weight).inactive


def remove_hidden_unit(model: DiscreteMCNN, layer_index: int, unit_index: int) -> DiscreteMCNN:
    """Remove one hidden unit and compact adjacent layer matrices."""

    if layer_index < 0 or layer_index >= len(model.layers) - 1:
        raise ValueError("only hidden layers can be structurally pruned")

    pruned = model.clone()
    layer = pruned.layers[layer_index]
    next_layer = pruned.layers[layer_index + 1]

    if unit_index < 0 or unit_index >= layer.weight.shape[0]:
        raise ValueError(f"unit_index {unit_index} is out of range for layer {layer_index}")

    keep_outputs = torch.ones(layer.weight.shape[0], dtype=torch.bool)
    keep_outputs[unit_index] = False
    keep_inputs = torch.ones(next_layer.weight.shape[1], dtype=torch.bool)
    keep_inputs[unit_index] = False

    layer.weight = layer.weight[keep_outputs].clone()
    layer.bias = layer.bias[keep_outputs].clone()
    next_layer.weight = next_layer.weight[:, keep_inputs].clone()
    return pruned
