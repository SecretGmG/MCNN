"""Exact and trainable models for the streamed Minecraft neural network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn


CELL_SIZE = 8
ACTIVATION_STEPS = 15
ACCUMULATOR_LIMIT = 693
REDSTONE_TICK_SECONDS = 0.1
STREAM_INTERVAL_REDSTONE_TICKS = 2
SHIFTER_FIRST_SLOT_REDSTONE_TICKS = 3
SHIFTER_CELL_REDSTONE_TICKS = 8
ForwardMode = Literal["exact", "ste", "relaxed"]


@dataclass(frozen=True)
class MinecraftLayerConfig:
    """One module-aligned cyclic-diagonal layer of eight-neuron cells."""

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


def cyclic_diagonal_order(input_size: int) -> tuple[int, ...]:
    """Return the physical downward-shifter diagonal order.

    A four-value temporal slice moves to the next two-block slice first. At
    each end, the delay-balanced wrap advances to the next temporal phase.
    The unshifted diagonal is therefore reached last.
    """
    if input_size <= 0 or input_size % CELL_SIZE:
        raise ValueError("input_size must be a positive multiple of eight")
    slice_count = input_size // 4
    order = [4 * step for step in range(1, slice_count)]
    for phase in range(1, 4):
        order.extend(phase + 4 * step for step in range(slice_count))
    order.append(0)
    if len(order) != input_size or len(set(order)) != input_size:
        raise RuntimeError("invalid physical diagonal traversal")
    return tuple(order)


def cyclic_diagonal_mask(
    input_size: int, output_size: int, diagonal_width: int
) -> Tensor:
    """Return the first ``diagonal_width`` physical cyclic diagonals."""
    if input_size % CELL_SIZE or output_size % CELL_SIZE:
        raise ValueError("layer sizes must be multiples of eight")
    if not 1 <= diagonal_width <= input_size:
        raise ValueError("diagonal_width must be in [1, input_size]")

    selected = torch.tensor(cyclic_diagonal_order(input_size)[:diagonal_width])
    input_index = torch.arange(input_size)
    output_index = torch.arange(output_size)[:, None] % input_size
    distance = (output_index - input_index[None, :]) % input_size
    return (distance[..., None] == selected).any(-1)


def cyclic_cell_mask(input_size: int, output_size: int, shift_depth: int) -> Tensor:
    """Return the module-aligned physical diagonal strip.

    ``shift_depth`` retains the architecture's eight-neuron module unit. Each
    included module contributes eight consecutive diagonals in the physical
    traversal order.
    """
    diagonal_width = CELL_SIZE * (shift_depth + 1)
    return cyclic_diagonal_mask(input_size, output_size, min(input_size, diagonal_width))


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
        self.diagonal_width = CELL_SIZE * (config.shift_depth + 1)
        self.surrogate_temperature = float(surrogate_temperature)
        self.quantization_progress = 1.0
        self.weight = nn.Parameter(
            torch.empty(config.output_size, input_size).normal_(std=initial_weight_std)
        )
        self.threshold = nn.Parameter(torch.empty(config.output_size, ACTIVATION_STEPS))
        self.register_buffer(
            "connection_mask",
            cyclic_cell_mask(input_size, config.output_size, config.shift_depth),
            persistent=False,
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
            "diagonal_width": self.diagonal_width,
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

    def timing(
        self,
        *,
        batch_size: int = 1,
        include_input_handoff: bool = False,
        include_output_handoff: bool = False,
    ) -> dict[str, object]:
        """Return exact scheduled timing for the streamed physical network.

        Inputs are assumed to be preloaded and final outputs read in place by
        default. Handoffs between layers are mandatory. Additional examples
        follow the first one at the two-redstone-tick stream interval.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        layers: list[dict[str, int]] = []
        first_result = SHIFTER_CELL_REDSTONE_TICKS if include_input_handoff else 0
        last_layer = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            connected_cells = layer.shift_depth + 1
            diagonal_width = layer.diagonal_width
            weight_words = 4 * diagonal_width
            accumulation = STREAM_INTERVAL_REDSTONE_TICKS * weight_words
            activation_words = 4 * ACTIVATION_STEPS
            activation = STREAM_INTERVAL_REDSTONE_TICKS * activation_words
            handoff = (
                SHIFTER_CELL_REDSTONE_TICKS
                if index < last_layer or include_output_handoff
                else 0
            )
            layer_total = accumulation + activation + handoff
            first_result += layer_total
            layers.append(
                {
                    "connected_cells": connected_cells,
                    "diagonal_width": diagonal_width,
                    "weight_words": weight_words,
                    "accumulation_redstone_ticks": accumulation,
                    "activation_words": activation_words,
                    "activation_redstone_ticks": activation,
                    "handoff_redstone_ticks": handoff,
                    "total_redstone_ticks": layer_total,
                }
            )
        batch_tail = STREAM_INTERVAL_REDSTONE_TICKS * (batch_size - 1)
        last_result = first_result + batch_tail
        return {
            "redstone_tick_seconds": REDSTONE_TICK_SECONDS,
            "stream_interval_redstone_ticks": STREAM_INTERVAL_REDSTONE_TICKS,
            "steady_state_evaluations_per_second": 1
            / (STREAM_INTERVAL_REDSTONE_TICKS * REDSTONE_TICK_SECONDS),
            "batch_size": batch_size,
            "layers": layers,
            "first_result_redstone_ticks": first_result,
            "first_result_seconds": first_result * REDSTONE_TICK_SECONDS,
            "batch_tail_redstone_ticks": batch_tail,
            "last_result_redstone_ticks": last_result,
            "last_result_seconds": last_result * REDSTONE_TICK_SECONDS,
        }
