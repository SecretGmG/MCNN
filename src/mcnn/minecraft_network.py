"""Exact and trainable models for the streamed Minecraft neural network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn


CELL_SIZE = 8
ACTIVATION_STEPS = 15
ACCUMULATOR_LIMIT = 693
ForwardMode = Literal["exact", "ste", "relaxed"]


@dataclass(frozen=True)
class MinecraftLayerConfig:
    """One cyclic block-banded layer of eight-neuron cells."""

    output_size: int
    shift_depth: int

    def __post_init__(self) -> None:
        if self.output_size <= 0 or self.output_size % CELL_SIZE:
            raise ValueError("output_size must be a positive multiple of eight")
        if self.shift_depth < 0:
            raise ValueError("shift_depth must be nonnegative")


@dataclass(frozen=True)
class MinecraftNetworkConfig:
    """Physical network dimensions and training-time surrogate settings."""

    input_size: int
    layers: tuple[MinecraftLayerConfig, ...]
    activation_steps: int = ACTIVATION_STEPS
    accumulator_limit: int = ACCUMULATOR_LIMIT
    initial_weight_std: float = 0.8
    surrogate_temperature: float = 2.0

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.input_size % CELL_SIZE:
            raise ValueError("input_size must be a positive multiple of eight")
        if not self.layers:
            raise ValueError("at least one layer is required")
        if self.activation_steps != ACTIVATION_STEPS:
            raise ValueError("the physical circuit always uses 15 activation steps")
        if self.accumulator_limit <= 0:
            raise ValueError("accumulator_limit must be positive")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["layers"] = [asdict(layer) for layer in self.layers]
        return result

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> MinecraftNetworkConfig:
        data = dict(values)
        data["layers"] = tuple(
            MinecraftLayerConfig(**layer) for layer in data["layers"]  # type: ignore[arg-type]
        )
        return cls(**data)  # type: ignore[arg-type]


def cyclic_cell_mask(input_size: int, output_size: int, shift_depth: int) -> Tensor:
    """Return the exact modular downward-shift connectivity mask."""
    if input_size % CELL_SIZE or output_size % CELL_SIZE:
        raise ValueError("layer sizes must be multiples of eight")
    input_cells = input_size // CELL_SIZE
    if shift_depth >= input_cells:
        raise ValueError("shift_depth must be smaller than the input cell count")

    input_cell = torch.arange(input_size) // CELL_SIZE
    output_cell = torch.arange(output_size) // CELL_SIZE
    distance = (output_cell[:, None] - input_cell[None, :]) % input_cells
    return distance <= shift_depth


def _ste(hard: Tensor, soft: Tensor) -> Tensor:
    return soft + (hard - soft).detach()


class MinecraftLayer(nn.Module):
    """Ternary accumulation followed by a 15-threshold count activation."""

    def __init__(
        self,
        input_size: int,
        config: MinecraftLayerConfig,
        *,
        initial_weight_std: float,
        surrogate_temperature: float,
        input_max: int = ACTIVATION_STEPS,
    ) -> None:
        super().__init__()
        input_cells = input_size // CELL_SIZE
        if config.shift_depth >= input_cells:
            raise ValueError(
                f"shift_depth {config.shift_depth} exceeds {input_cells - 1} for {input_size} inputs"
            )
        self.input_size = input_size
        self.output_size = config.output_size
        self.shift_depth = config.shift_depth
        self.surrogate_temperature = float(surrogate_temperature)
        self.quantization_progress = 1.0
        self.weight = nn.Parameter(
            torch.empty(config.output_size, input_size).normal_(std=initial_weight_std)
        )
        self.threshold = nn.Parameter(torch.empty(config.output_size, ACTIVATION_STEPS))
        self.register_buffer(
            "connection_mask",
            cyclic_cell_mask(input_size, config.output_size, config.shift_depth),
        )
        self.reset_thresholds(input_max)

    def reset_thresholds(self, input_max: int) -> None:
        fan_in = CELL_SIZE * (self.shift_depth + 1)
        spread = max(4.0, fan_in**0.5 * input_max * 0.6)
        initial = torch.linspace(-spread, spread, ACTIVATION_STEPS)
        with torch.no_grad():
            self.threshold.copy_(initial.expand(self.output_size, -1))

    def discrete_weight(self) -> Tensor:
        return self.weight.clamp(-1, 1).round() * self.connection_mask

    def ordered_threshold(self, *, rounded: bool) -> Tensor:
        threshold = self.threshold.sort(dim=1).values
        return threshold.round() if rounded else threshold

    def forward(self, inputs: Tensor, mode: ForwardMode = "ste") -> Tensor:
        clipped_weight = self.weight.clamp(-1, 1) * self.connection_mask
        hard_weight = clipped_weight.round()
        if mode == "exact":
            weight = hard_weight
        elif mode == "ste":
            weight = clipped_weight + self.quantization_progress * (
                hard_weight - clipped_weight
            ).detach()
        elif mode == "relaxed":
            weight = clipped_weight
        else:
            raise ValueError(f"unknown forward mode: {mode}")

        preactivation = inputs @ weight.T
        soft_threshold = self.ordered_threshold(rounded=False)
        hard_threshold = soft_threshold.round()
        if mode == "exact":
            return (preactivation.unsqueeze(-1) > hard_threshold).sum(-1).to(inputs.dtype)

        temperature = max(self.surrogate_temperature, 1e-3)
        soft = torch.sigmoid(
            (preactivation.unsqueeze(-1) - soft_threshold) / temperature
        ).sum(-1)
        if mode == "relaxed":
            return soft
        hard = (preactivation.unsqueeze(-1) > hard_threshold).sum(-1).to(inputs.dtype)
        return soft + self.quantization_progress * (hard - soft).detach()

    def physicality(self, input_max: int) -> dict[str, int | float]:
        weight = self.discrete_weight()
        active = weight != 0
        positive_fan_in = (weight > 0).sum(1)
        negative_fan_in = (weight < 0).sum(1)
        maximum_signal = int(
            torch.maximum(positive_fan_in, negative_fan_in).max().item() * input_max
        )
        thresholds = self.ordered_threshold(rounded=True)
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "shift_depth": self.shift_depth,
            "connection_slots": int(self.connection_mask.sum().item()),
            "active_connections": int(active.sum().item()),
            "active_fraction": float(active.sum().item() / self.connection_mask.sum().item()),
            "max_active_fan_in": int(active.sum(1).max().item()),
            "maximum_signal": maximum_signal,
            "threshold_min": int(thresholds.min().item()),
            "threshold_max": int(thresholds.max().item()),
        }


class MinecraftNetwork(nn.Module):
    """A stack of exact streamed Minecraft layers."""

    def __init__(self, config: MinecraftNetworkConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        input_size = config.input_size
        input_max = 1
        for layer_config in config.layers:
            layers.append(
                MinecraftLayer(
                    input_size,
                    layer_config,
                    initial_weight_std=config.initial_weight_std,
                    surrogate_temperature=config.surrogate_temperature,
                    input_max=input_max,
                )
            )
            input_size = layer_config.output_size
            input_max = ACTIVATION_STEPS
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: Tensor, mode: ForwardMode = "ste") -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.config.input_size:
            raise ValueError(f"expected [batch, {self.config.input_size}] inputs")
        values = inputs
        for layer in self.layers:
            values = layer(values, mode=mode)
        return values

    @torch.no_grad()
    def project_(self) -> MinecraftNetwork:
        for layer in self.layers:
            layer.weight.copy_(layer.discrete_weight())
            layer.threshold.copy_(layer.ordered_threshold(rounded=True))
        return self

    def physicality(self) -> dict[str, object]:
        rows = []
        input_max = 1
        for layer in self.layers:
            rows.append(layer.physicality(input_max))
            input_max = ACTIVATION_STEPS
        return {
            "cell_size": CELL_SIZE,
            "activation_steps": ACTIVATION_STEPS,
            "accumulator_limit": self.config.accumulator_limit,
            "layers": rows,
            "active_connections": sum(int(row["active_connections"]) for row in rows),
            "connection_slots": sum(int(row["connection_slots"]) for row in rows),
            "maximum_signal": max(int(row["maximum_signal"]) for row in rows),
            "within_accumulator_limit": all(
                int(row["maximum_signal"]) <= self.config.accumulator_limit for row in rows
            ),
        }
