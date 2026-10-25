"""Reflection-equivariant streaming line models for Connect Four."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .connect4_rules import COLUMNS, CONNECT, ROWS


def winning_windows() -> Tensor:
    """Return the 69 length-four lines as indices into a padded 7x7 plane."""
    windows: list[list[int]] = []
    for row in range(ROWS):
        for column in range(COLUMNS - CONNECT + 1):
            windows.append([(row + 1) * 7 + column + step for step in range(CONNECT)])
    for row in range(ROWS - CONNECT + 1):
        for column in range(COLUMNS):
            windows.append([(row + step + 1) * 7 + column for step in range(CONNECT)])
    for row in range(ROWS - CONNECT + 1):
        for column in range(COLUMNS - CONNECT + 1):
            windows.append(
                [(row + step + 1) * 7 + column + step for step in range(CONNECT)]
            )
            windows.append(
                [(row + step + 1) * 7 + column + CONNECT - 1 - step for step in range(CONNECT)]
            )
    if len(windows) != 69:
        raise RuntimeError(f"Expected 69 winning windows, got {len(windows)}")
    return torch.tensor(windows, dtype=torch.long)


def tactical_moves(inputs: Tensor, windows: Tensor) -> tuple[Tensor, Tensor]:
    """Return immediate winning and mandatory-block columns for each board."""
    if inputs.ndim != 4 or inputs.shape[1:] != (2, 7, 7):
        raise ValueError("Expected [batch, 2, 7, 7] inputs")
    occupied = inputs.sum(dim=1)
    heights = occupied[:, 1:].sum(dim=1).long()
    legal = heights < ROWS
    landing_rows = 6 - heights.clamp(max=ROWS - 1)
    landing_indices = landing_rows * 7 + torch.arange(
        COLUMNS, device=inputs.device
    )[None]
    additions = torch.zeros(
        (len(inputs), COLUMNS, 49), dtype=inputs.dtype, device=inputs.device
    )
    additions.scatter_(2, landing_indices[:, :, None], legal[:, :, None].to(inputs.dtype))

    def completes_four(plane: Tensor) -> Tensor:
        candidates = plane.flatten(start_dim=1)[:, None] + additions
        lines = candidates[:, :, windows]
        return (lines.sum(dim=-1) == CONNECT).any(dim=-1) & legal

    return completes_four(inputs[:, 0]), completes_four(inputs[:, 1])


def apply_tactical_priority(scores: Tensor, inputs: Tensor, windows: Tensor) -> Tensor:
    wins, blocks = tactical_moves(inputs, windows)
    any_win = wins.any(dim=1, keepdim=True)
    any_block = blocks.any(dim=1, keepdim=True)
    forced_win_scores = torch.where(wins, 15.0, 0.0)
    forced_block_scores = torch.where(blocks, 15.0, scores.clamp(max=14))
    return torch.where(any_win, forced_win_scores, torch.where(any_block, forced_block_scores, scores))


@dataclass(frozen=True)
class ConnectFourTeacherConfig:
    pattern_units: int = 16
    global_units: int = 64
    mixing_units: int = 32
    initial_weight_std: float = 0.0


class ConnectFourTeacher(nn.Module):
    """Continuous line-convolution teacher with four canonical output heads."""

    model_type = "connect4_line_teacher"

    def __init__(self, config: ConnectFourTeacherConfig = ConnectFourTeacherConfig()) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("windows", winning_windows())
        self.pattern = nn.Linear(2 * CONNECT, config.pattern_units)
        self.global_layer = nn.Linear(69 * config.pattern_units, config.global_units)
        self.mixing = nn.Linear(config.global_units, config.mixing_units)
        self.output = nn.Linear(config.mixing_units, 4)
        for layer in (self.pattern, self.global_layer, self.mixing, self.output):
            if config.initial_weight_std > 0:
                nn.init.normal_(layer.weight, std=config.initial_weight_std)
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def line_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (2, 7, 7):
            raise ValueError("Expected [batch, 2, 7, 7] inputs")
        flat = inputs.flatten(start_dim=2)
        values = flat[:, :, self.windows].permute(0, 2, 1, 3).flatten(start_dim=2)
        return torch.sigmoid(self.pattern(values))

    def _canonical_scores(self, inputs: Tensor) -> Tensor:
        patterns = self.line_features(inputs).flatten(start_dim=1)
        global_values = torch.sigmoid(self.global_layer(patterns))
        mixed = torch.sigmoid(self.mixing(global_values))
        return self.output(mixed)

    def forward(self, inputs: Tensor) -> Tensor:
        original = self._canonical_scores(inputs)
        reflected = self._canonical_scores(inputs.flip(-1))
        center = (original[:, 3] + reflected[:, 3]) / 2
        scores = torch.cat(
            (original[:, :3], center[:, None], reflected[:, :3].flip(1)), dim=1
        )
        return apply_tactical_priority(scores, inputs, self.windows)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_type": self.model_type,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> ConnectFourTeacher:
        artifact = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(ConnectFourTeacherConfig(**artifact["config"]))
        model.load_state_dict(artifact["state_dict"])
        return model


def _ste(discrete: Tensor, continuous: Tensor) -> Tensor:
    return continuous + (discrete - continuous).detach()


def _quantize(values: Tensor, low: int, high: int, mode: str) -> Tensor:
    clipped = values.clamp(low, high)
    rounded = clipped.round()
    if mode == "exact":
        return rounded
    if mode == "ste":
        return _ste(rounded, clipped)
    raise ValueError(f"Unknown physical mode: {mode}")


@dataclass(frozen=True)
class PhysicalConnectFourConfig:
    pattern_units: int = 16
    global_units: int = 64
    mixing_units: int = 32
    pattern_levels: int = 2
    global_levels: int = 2
    mixing_levels: int = 2
    weight_min: int = -15
    weight_max: int = 15
    bias_min: int = -127
    bias_max: int = 127


class PhysicalConnectFourModel(nn.Module):
    """Integer-weight, binary-activation implementation of the line model."""

    model_type = "connect4_physical_line"

    def __init__(
        self, config: PhysicalConnectFourConfig = PhysicalConnectFourConfig()
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("windows", winning_windows())
        self.pattern_weight = nn.Parameter(
            torch.zeros(config.pattern_units, config.pattern_levels, 2 * CONNECT)
        )
        self.pattern_bias = nn.Parameter(
            torch.zeros(config.pattern_units, config.pattern_levels)
        )
        self.global_weight = nn.Parameter(
            torch.zeros(
                config.global_units,
                config.global_levels,
                69,
                config.pattern_units,
                config.pattern_levels,
            )
        )
        self.global_bias = nn.Parameter(
            torch.zeros(config.global_units, config.global_levels)
        )
        self.mixing_weight = nn.Parameter(
            torch.zeros(
                config.mixing_units,
                config.mixing_levels,
                config.global_units,
                config.global_levels,
            )
        )
        self.mixing_bias = nn.Parameter(
            torch.zeros(config.mixing_units, config.mixing_levels)
        )
        self.output_weight = nn.Parameter(
            torch.zeros(4, config.mixing_units, config.mixing_levels)
        )
        self.output_bias = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _binary(values: Tensor, mode: str) -> Tensor:
        discrete = (values > 0).to(values.dtype)
        if mode == "exact":
            return discrete
        surrogate = torch.sigmoid(values * 0.5)
        return _ste(discrete, surrogate)

    def _window_values(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (2, 7, 7):
            raise ValueError("Expected [batch, 2, 7, 7] inputs")
        flat = inputs.flatten(start_dim=2)
        return flat[:, :, self.windows].permute(0, 2, 1, 3).flatten(start_dim=2)

    def _canonical_scores(self, inputs: Tensor, mode: str) -> Tensor:
        cfg = self.config
        pattern_weight = _quantize(
            self.pattern_weight, cfg.weight_min, cfg.weight_max, mode
        )
        pattern_bias = _quantize(
            self.pattern_bias, cfg.bias_min, cfg.bias_max, mode
        )
        patterns = self._binary(
            torch.einsum("bwi,pli->bwpl", self._window_values(inputs), pattern_weight)
            + pattern_bias,
            mode,
        )
        global_weight = _quantize(
            self.global_weight, cfg.weight_min, cfg.weight_max, mode
        )
        global_bias = _quantize(
            self.global_bias, cfg.bias_min, cfg.bias_max, mode
        )
        global_values = self._binary(
            torch.einsum("bwpl,grwpl->bgr", patterns, global_weight)
            + global_bias,
            mode,
        )
        mixing_weight = _quantize(
            self.mixing_weight, cfg.weight_min, cfg.weight_max, mode
        )
        mixing_bias = _quantize(
            self.mixing_bias, cfg.bias_min, cfg.bias_max, mode
        )
        mixed = self._binary(
            torch.einsum("bgr,mqgr->bmq", global_values, mixing_weight)
            + mixing_bias,
            mode,
        )
        output_weight = _quantize(
            self.output_weight, cfg.weight_min, cfg.weight_max, mode
        )
        output_bias = _quantize(
            self.output_bias, cfg.bias_min, cfg.bias_max, mode
        )
        raw = torch.einsum("bmq,cmq->bc", mixed, output_weight) + output_bias
        bounded = raw.clamp(0, 15)
        return bounded if mode == "exact" else _ste(bounded, raw)

    def forward(self, inputs: Tensor, mode: str = "exact") -> Tensor:
        original = self._canonical_scores(inputs, mode)
        reflected = self._canonical_scores(inputs.flip(-1), mode)
        center = (original[:, 3] + reflected[:, 3]) / 2
        scores = torch.cat(
            (original[:, :3], center[:, None], reflected[:, :3].flip(1)), dim=1
        )
        return apply_tactical_priority(scores, inputs, self.windows).clamp(0, 15)

    @torch.no_grad()
    def quantize_(self) -> None:
        for name, parameter in self.named_parameters():
            low, high = (
                (self.config.bias_min, self.config.bias_max)
                if "bias" in name
                else (self.config.weight_min, self.config.weight_max)
            )
            parameter.copy_(parameter.round().clamp(low, high))

    def physicality_metrics(self) -> dict[str, object]:
        weights = (
            self.pattern_weight,
            self.global_weight,
            self.mixing_weight,
            self.output_weight,
        )
        connection_slots = sum(weight.numel() for weight in weights)
        active = sum(int((weight.detach().round() != 0).sum()) for weight in weights)

        def max_active_fan_in(weight: Tensor, start_dim: int) -> int:
            active_inputs = (weight.detach().round() != 0).flatten(start_dim=start_dim)
            return int(active_inputs.sum(dim=-1).max())

        active_fan_in = {
            "pattern": max_active_fan_in(self.pattern_weight, 2),
            "global": max_active_fan_in(self.global_weight, 2),
            "mixing": max_active_fan_in(self.mixing_weight, 2),
            "output": max_active_fan_in(self.output_weight, 1),
        }
        dense_fan_in = {
            "pattern": 2 * CONNECT,
            "global": 69 * self.config.pattern_units * self.config.pattern_levels,
            "mixing": self.config.global_units * self.config.global_levels,
            "output": self.config.mixing_units * self.config.mixing_levels,
        }
        active_serial_depth = sum(active_fan_in.values())
        dense_serial_depth = sum(dense_fan_in.values())
        return {
            "connection_slots": connection_slots,
            "active_connection_slots": active,
            "active_fraction": active / connection_slots,
            "fixed_tactical_patterns": 8,
            "fixed_tactical_candidate_checks": 56,
            "stream_windows": 69,
            "windows_by_direction": [24, 21, 12, 12],
            "assumed_window_interval_redstone_ticks": 3,
            "longest_convolution_lane_redstone_ticks": 72,
            "longest_convolution_lane_seconds": 7.2,
            "max_active_fan_in": active_fan_in,
            "dense_fan_in": dense_fan_in,
            "naive_active_serial_comparator_depth": active_serial_depth,
            "naive_active_serial_seconds_before_routing": active_serial_depth * 0.1,
            "naive_dense_serial_comparator_depth": dense_serial_depth,
            "naive_dense_serial_seconds_before_routing": dense_serial_depth * 0.1,
            "end_to_end_latency": "unknown until a concrete schematic is audited",
            "stored_output_heads": 4,
            "logical_outputs": 7,
        }

    def save(self, path: str | Path) -> None:
        self.quantize_()
        torch.save(
            {
                "model_type": self.model_type,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> PhysicalConnectFourModel:
        artifact = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(PhysicalConnectFourConfig(**artifact["config"]))
        model.load_state_dict(artifact["state_dict"])
        return model


def _thresholds(levels: int, reference: Tensor) -> Tensor:
    probabilities = (
        torch.arange(levels, dtype=reference.dtype, device=reference.device) + 0.5
    ) / levels
    return torch.logit(probabilities)


@torch.no_grad()
def expand_connect4_teacher(
    teacher: ConnectFourTeacher,
    *,
    pattern_levels: int = 2,
    global_levels: int = 2,
    mixing_levels: int = 2,
) -> PhysicalConnectFourModel:
    cfg = teacher.config
    student = PhysicalConnectFourModel(
        PhysicalConnectFourConfig(
            pattern_units=cfg.pattern_units,
            global_units=cfg.global_units,
            mixing_units=cfg.mixing_units,
            pattern_levels=pattern_levels,
            global_levels=global_levels,
            mixing_levels=mixing_levels,
        )
    )

    def neuron_scales(weight: Tensor, bias: Tensor, thresholds: Tensor) -> Tensor:
        flat = weight.flatten(start_dim=1)
        weight_limit = 15 / flat.abs().amax(dim=1).clamp_min(1e-6)
        bias_delta = bias[:, None] - thresholds[None]
        bias_limit = 127 / bias_delta.abs().clamp_min(1e-6)
        return torch.minimum(weight_limit[:, None], bias_limit)

    pattern_thresholds = _thresholds(pattern_levels, teacher.pattern.bias)
    pattern_scales = neuron_scales(
        teacher.pattern.weight, teacher.pattern.bias, pattern_thresholds
    )
    student.pattern_weight.copy_(
        teacher.pattern.weight[:, None] * pattern_scales[:, :, None]
    )
    student.pattern_bias.copy_(
        (teacher.pattern.bias[:, None] - pattern_thresholds[None]) * pattern_scales
    )

    global_weight = teacher.global_layer.weight.reshape(
        cfg.global_units, 69, cfg.pattern_units
    )
    global_base = global_weight / pattern_levels
    global_thresholds = _thresholds(global_levels, teacher.global_layer.bias)
    global_scales = neuron_scales(
        global_base, teacher.global_layer.bias, global_thresholds
    )
    student.global_weight.copy_(
        (
            global_base[:, None, :, :, None]
            * global_scales[:, :, None, None, None]
        ).expand(-1, global_levels, -1, -1, pattern_levels)
    )
    student.global_bias.copy_(
        (
            teacher.global_layer.bias[:, None]
            - global_thresholds[None]
        )
        * global_scales
    )

    mixing_base = teacher.mixing.weight / global_levels
    mixing_thresholds = _thresholds(mixing_levels, teacher.mixing.bias)
    mixing_scales = neuron_scales(
        mixing_base, teacher.mixing.bias, mixing_thresholds
    )
    student.mixing_weight.copy_(
        (
            mixing_base[:, None, :, None]
            * mixing_scales[:, :, None, None]
        ).expand(-1, mixing_levels, -1, global_levels)
    )
    student.mixing_bias.copy_(
        (
            teacher.mixing.bias[:, None]
            - mixing_thresholds[None]
        )
        * mixing_scales
    )

    output_base = teacher.output.weight / mixing_levels
    output_scale = min(
        4.0, 15.0 / max(float(output_base.abs().max()), 1e-6)
    )
    student.output_weight.copy_(
        (
            output_base[:, :, None] * output_scale
        ).expand(-1, -1, mixing_levels)
    )
    student.output_bias.copy_(teacher.output.bias * output_scale + 7.5)
    student.quantize_()
    return student
