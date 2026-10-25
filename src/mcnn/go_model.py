"""7x7 Go value model implemented by the streamed Minecraft network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .minecraft_network import (
    ForwardMode,
    MinecraftLayerConfig,
    MinecraftNetwork,
    MinecraftNetworkConfig,
)


BOARD_INPUTS = 2 * 7 * 7
PADDED_INPUTS = 104


@dataclass(frozen=True)
class MinecraftGoConfig:
    """Configuration for a successor-position Go evaluator."""

    hidden_sizes: tuple[int, ...] = (128, 128, 128, 128)
    shift_depths: tuple[int, ...] = (12, 4, 4, 4, 4)
    score_unit_points: float = 1.25
    initial_weight_std: float = 0.8
    surrogate_temperature: float = 2.0

    def __post_init__(self) -> None:
        if len(self.shift_depths) != len(self.hidden_sizes) + 1:
            raise ValueError("shift_depths must contain one value per hidden and output layer")
        if any(size <= 0 or size % 8 for size in self.hidden_sizes):
            raise ValueError("hidden sizes must be positive multiples of eight")
        if self.score_unit_points <= 0:
            raise ValueError("score_unit_points must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> MinecraftGoConfig:
        data = dict(values)
        data["hidden_sizes"] = tuple(data["hidden_sizes"])  # type: ignore[arg-type]
        data["shift_depths"] = tuple(data["shift_depths"])  # type: ignore[arg-type]
        return cls(**data)  # type: ignore[arg-type]


class MinecraftGoModel(nn.Module):
    """Score successor boards using exact Minecraft-compatible operations."""

    model_type = "minecraft_go_value"

    def __init__(self, config: MinecraftGoConfig = MinecraftGoConfig()) -> None:
        super().__init__()
        self.config = config
        sizes = (*config.hidden_sizes, 8)
        network_config = MinecraftNetworkConfig(
            input_size=PADDED_INPUTS,
            layers=tuple(
                MinecraftLayerConfig(output_size=size, shift_depth=shift)
                for size, shift in zip(sizes, config.shift_depths, strict=True)
            ),
            initial_weight_std=config.initial_weight_std,
            surrogate_temperature=config.surrogate_temperature,
        )
        self.network = MinecraftNetwork(network_config)

    def output_lines(self, inputs: Tensor, mode: ForwardMode = "ste") -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (2, 7, 7):
            raise ValueError("expected [batch, 2, 7, 7] board inputs")
        flat = inputs.flatten(start_dim=1)
        padded = torch.nn.functional.pad(flat, (0, PADDED_INPUTS - BOARD_INPUTS))
        return self.network(padded, mode=mode)

    def score_units(self, inputs: Tensor, mode: ForwardMode = "ste") -> Tensor:
        lines = self.output_lines(inputs, mode=mode)
        return lines[:, :4].sum(1) - lines[:, 4:].sum(1)

    def score_points(self, inputs: Tensor, mode: ForwardMode = "ste") -> Tensor:
        return self.score_units(inputs, mode=mode) * self.config.score_unit_points

    def forward(self, inputs: Tensor, mode: ForwardMode = "ste") -> Tensor:
        return self.score_points(inputs, mode=mode)

    @torch.no_grad()
    def project_(self) -> MinecraftGoModel:
        self.network.project_()
        return self

    def physicality(self) -> dict[str, object]:
        return self.network.physicality()

    def timing(self, *, batch_size: int = 1) -> dict[str, object]:
        return self.network.timing(batch_size=batch_size)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_type": self.model_type,
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> MinecraftGoModel:
        artifact = torch.load(path, map_location=map_location, weights_only=True)
        if artifact.get("model_type") != cls.model_type:
            raise ValueError(f"unsupported model type: {artifact.get('model_type')!r}")
        model = cls(MinecraftGoConfig.from_dict(artifact["config"]))
        state = {
            name: value
            for name, value in artifact["state_dict"].items()
            if not name.endswith("connection_mask")
        }
        model.load_state_dict(state)
        return model
